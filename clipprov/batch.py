"""Resumable deterministic batch matching and offline reports."""

from __future__ import annotations

import base64
import csv
import hashlib
import html
import io
import json
import os
import resource
import shutil
import tempfile
import time
import uuid
from pathlib import Path

from clipprov._io import atomic_json, atomic_text, digest, finite_number, json_snapshot
from clipprov.confidence import apply_calibration
from clipprov.config import Config, resolve_speed_profile
from clipprov.eval import adapt_product, product_result
from clipprov.evidence import generate_evidence
from clipprov.index import IndexError as IndexReadError
from clipprov.index import _pin_source, config_from_manifest, load_index
from clipprov.match import match_clip_path

BATCH_SCHEMA_VERSION = 1
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
MEMORY_CEILING_BYTES = 1024 * 1024 * 1024
THROUGHPUT_TARGET_CLIPS_PER_SEC = 0.05


class BatchError(Exception):
    """Raised when durable batch state is unsafe to resume."""


def discover_clips(root: Path) -> list[dict]:
    """Return a deterministic, digest-bound inventory without following symlinks."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise BatchError(f"clips directory not found: {root}")
    rows = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(dirpath)
        dirnames[:] = sorted(name for name in dirnames if not (base / name).is_symlink())
        for name in sorted(filenames):
            path = base / name
            if path.is_symlink() or not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                raise BatchError(f"clip escapes root: {path}")
            rows.append({
                "relpath": resolved.relative_to(root).as_posix(),
                "path": str(resolved),
                "size_bytes": resolved.stat().st_size,
                "sha256": digest(resolved),
            })
    rows.sort(key=lambda row: row["relpath"])
    return rows


def _json_digest(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_calibration(path: Path | None):
    if path is None:
        return None, None
    path = Path(path)
    try:
        value, file_sha = json_snapshot(path)
    except (OSError, ValueError) as exc:
        raise BatchError(f"cannot read calibration: {exc}") from exc
    if isinstance(value, dict) and "confidence" in value:
        confidence = value["confidence"]
        # Type guard: confidence must be a dict with calibration key
        if not isinstance(confidence, dict):
            raise BatchError("invalid calibration receipt")
        calibration = confidence.get("calibration")
        if calibration is None:
            raise BatchError("invalid calibration receipt")
        value = calibration
    if not isinstance(value, dict):
        raise BatchError("invalid calibration receipt")
    # Allow tolerance_seconds (new in development) but reject arbitrary extras.
    expected_keys = {
        "method", "semantics", "development_rows", "eligible_correct_alignments",
        "thresholds", "calibration_sha256",
    }
    if "tolerance_seconds" in value:
        expected_keys.add("tolerance_seconds")
    if set(value) != expected_keys:
        raise BatchError("invalid calibration receipt")
    # Validate tolerance semantics before any digest recomputation so NaN or
    # Infinity cannot reach allow_nan=False serialization as a raw ValueError.
    # Legacy six-field receipts remain valid and carry an implicit 1.5-second
    # tolerance at every derive_tiers/measure_heldout call site.
    if "tolerance_seconds" in value:
        tolerance = value["tolerance_seconds"]
        if not finite_number(tolerance) or tolerance < 0:
            raise BatchError("invalid calibration tolerance_seconds")
    expected = dict(value)
    receipt = expected.pop("calibration_sha256")
    if receipt != _json_digest(expected):
        raise BatchError("calibration receipt digest mismatch")
    if value["method"] != "development_medians_v1" or value["semantics"] != "diagnostic_not_probability":
        raise BatchError("unsupported calibration method or semantics")
    for key in ("development_rows", "eligible_correct_alignments"):
        if type(value[key]) is not int or value[key] < 1:
            raise BatchError(f"invalid calibration {key}")
    if value["eligible_correct_alignments"] > value["development_rows"]:
        raise BatchError("eligible calibration rows exceed development rows")
    thresholds = value["thresholds"]
    threshold_keys = {
        "strong_score", "strong_margin", "focused_support_fraction",
        "focused_profile_width_sec",
    }
    if not isinstance(thresholds, dict) or set(thresholds) != threshold_keys:
        raise BatchError("invalid calibration thresholds")
    for key, number in thresholds.items():
        if not finite_number(number):
            raise BatchError(f"invalid calibration threshold: {key}")
    if thresholds["strong_margin"] < 0 or thresholds["focused_profile_width_sec"] < 0:
        raise BatchError("calibration thresholds must be non-negative")
    if not 0 <= thresholds["focused_support_fraction"] <= 1:
        raise BatchError("focused_support_fraction must be in [0, 1]")
    return value, file_sha


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value * 1024)  # Linux ru_maxrss is KiB.


def _implementation_receipt(root: Path | None = None) -> dict:
    from clipprov.harness import _git_receipt

    root = Path(root) if root is not None else Path(__file__).resolve().parent
    names = (
        "__init__.py", "_io.py", "appearance.py", "audio.py", "batch.py", "confidence.py",
        "config.py", "core.py", "eval.py", "evidence.py", "index.py",
        "index_batch.py", "match.py", "media.py", "motion.py",
    )
    files = [{"path": f"clipprov/{name}", "sha256": digest(root / name)} for name in names]
    return {
        "files": files,
        "sha256": _json_digest(files),
        "git": _git_receipt(root.parent),
    }


def _source_summaries(rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in rows:
        result = row["result"]
        source = result.get("best_master") or "unidentified"
        summary = grouped.setdefault(source, {
            "source": source,
            "clips": [],
            "matched": 0,
            "uncertain": 0,
            "ambiguous": 0,
            "unmatched": 0,
        })
        decision = result["confidence"]["decision"]
        summary[decision] += 1
        summary["clips"].append(row["clip"])
    for summary in grouped.values():
        summary["clips"].sort()
    return [grouped[key] for key in sorted(grouped)]


def _queue_rows(rows: list[dict]) -> dict[str, list[dict]]:
    queues: dict[str, list[dict]] = {}
    for row in rows:
        confidence = row["result"]["confidence"]
        keys = {
            f"decision-{confidence['decision']}",
            f"source-tier-{confidence['source']['tier']}",
            f"timing-tier-{confidence['timing']['tier']}",
        }
        for key in keys:
            queues.setdefault(key, []).append({
                "clip": row["clip"],
                "decision": confidence["decision"],
                "best_master": row["result"].get("best_master"),
                "best_score": row["result"].get("best_score"),
            })
    return {key: sorted(value, key=lambda item: item["clip"]) for key, value in sorted(queues.items())}


def _evidence_html(result: dict, out_dir: Path) -> str:
    evidence = result.get("evidence")
    if not evidence:
        return ""
    directory = Path(evidence["directory"])
    panels = []
    for frame in evidence["frames"]:
        clip_path = directory / frame["clip_frame"]
        master_path = directory / frame["master_frame"]
        clip_src = "data:image/jpeg;base64," + base64.b64encode(clip_path.read_bytes()).decode()
        master_src = "data:image/jpeg;base64," + base64.b64encode(master_path.read_bytes()).decode()
        caption = f"clip {frame['clip_sec']:.3f}s · master {frame['master_sec']:.3f}s"
        panels.append(
            '<figure class="pair">'
            f'<img src="{html.escape(clip_src, quote=True)}" alt="clip evidence">'
            f'<img src="{html.escape(master_src, quote=True)}" alt="master evidence">'
            f'<figcaption>{html.escape(caption)}</figcaption></figure>'
        )
    return '<div class="evidence">' + "".join(panels) + "</div>"


def _render_html(report: dict, out_dir: Path) -> str:
    table_rows = []
    for row in report["results"]:
        result = row["result"]
        if result.get("schema_version") == 2:
            adapt_product(result)
        confidence = result["confidence"]
        searchable = " ".join([
            row["clip"], confidence["decision"], str(result.get("best_master") or ""),
            confidence["source"]["tier"], confidence["timing"]["tier"],
        ])
        table_rows.append(
            f'<tr data-search="{html.escape(searchable.lower(), quote=True)}">'
            f'<td>{html.escape(row["clip"])}</td>'
            f'<td>{html.escape(confidence["decision"])}</td>'
            f'<td>{html.escape(str(result.get("best_master") or ""))}</td>'
            f'<td>{result.get("best_score", 0):.6f}</td>'
            f'<td>{html.escape(confidence["source"]["tier"])}</td>'
            f'<td>{html.escape(confidence["timing"]["tier"])}</td>'
            f'<td>{_evidence_html(result, out_dir)}</td></tr>'
        )
    for failure in report["failures"]:
        text = f"{failure['type']}: {failure['message']}"
        table_rows.append(
            f'<tr class="error" data-search="{html.escape((failure["clip"] + " " + text).lower(), quote=True)}">'
            f'<td>{html.escape(failure["clip"])}</td><td>error</td><td></td><td></td>'
            f'<td></td><td></td><td>{html.escape(text)}</td></tr>'
        )
    return """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Clip Sleuth batch report</title><style>
body{font-family:system-ui,sans-serif;margin:1rem}input{padding:.5rem;width:min(32rem,90%)}
table{border-collapse:collapse;width:100%;margin-top:1rem}th,td{border:1px solid #bbb;padding:.4rem;vertical-align:top}
th{background:#eee;cursor:pointer}.error{background:#fee}.evidence{display:flex;gap:.5rem;flex-wrap:wrap}
.pair{margin:0}.pair img{max-width:150px;display:inline-block}.pair figcaption{font-size:.75rem}
</style></head><body><h1>Clip Sleuth batch report</h1>
<p>Offline report. Confidence tiers are diagnostic, not probabilities.</p>
<input id="q" placeholder="Search clips, decisions, sources, or tiers" aria-label="Search">
<table id="report"><thead><tr><th>Clip</th><th>Decision</th><th>Source</th><th>Score</th><th>Source tier</th><th>Timing tier</th><th>Evidence / error</th></tr></thead><tbody>""" + "".join(table_rows) + """</tbody></table>
<script>const q=document.getElementById('q');q.addEventListener('input',()=>{const s=q.value.toLowerCase();document.querySelectorAll('tbody tr').forEach(r=>r.hidden=!r.dataset.search.includes(s));});document.querySelectorAll('th').forEach((h,i)=>h.addEventListener('click',()=>{const b=h.closest('table').tBodies[0];[...b.rows].sort((a,c)=>a.cells[i].textContent.localeCompare(c.cells[i].textContent,undefined,{numeric:true})).forEach(r=>b.appendChild(r));}));</script>
</body></html>"""


def _write_artifact_manifest(out_dir: Path) -> None:
    rows = []
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file() or path.name in {
            "artifact-manifest.json", ".batch-progress.json", ".batch.lock", ".evidence.lock",
        }:
            continue
        rows.append({
            "path": path.relative_to(out_dir).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": digest(path),
        })
    atomic_json(out_dir / "artifact-manifest.json", {"schema_version": 1, "files": rows})


def _csv_safe(value):
    """Neutralize spreadsheet formulas while preserving machine JSON unchanged."""
    if not isinstance(value, str):
        return value
    candidate = value.lstrip(" \t\r\n")
    if candidate.startswith(("=", "+", "-", "@")) or value.startswith(("\t", "\r", "\n")):
        return "'" + value
    return value


def _render_report_files(report: dict, out_dir: Path) -> None:
    """Write one complete immutable report generation."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(out_dir / "report.json", report)

    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["clip", "status", "decision", "best_master", "best_score", "source_tier", "timing_tier", "error_type", "error_message"])
    failures = {row["clip"]: row for row in report["failures"]}
    results = {row["clip"]: row["result"] for row in report["results"]}
    for clip in sorted(set(failures) | set(results)):
        if clip in failures:
            failure = failures[clip]
            writer.writerow([
                _csv_safe(clip), "error", "", "", "", "", "",
                _csv_safe(failure["type"]), _csv_safe(failure["message"]),
            ])
        else:
            result = results[clip]
            confidence = result["confidence"]
            writer.writerow([
                _csv_safe(clip), "result", confidence["decision"],
                _csv_safe(result.get("best_master") or ""),
                result.get("best_score"), confidence["source"]["tier"],
                confidence["timing"]["tier"], "", "",
            ])
    atomic_text(out_dir / "report.csv", buffer.getvalue())

    summaries = _source_summaries(report["results"])
    atomic_json(out_dir / "source-summary.json", {"schema_version": 1, "sources": summaries})
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["source", "matched", "uncertain", "ambiguous", "unmatched", "clips"])
    for row in summaries:
        writer.writerow([
            _csv_safe(row["source"]), row["matched"], row["uncertain"],
            row["ambiguous"], row["unmatched"], _csv_safe("|".join(row["clips"])),
        ])
    atomic_text(out_dir / "source-summary.csv", buffer.getvalue())

    queues_dir = out_dir / "queues"
    queues_dir.mkdir(exist_ok=True)
    queues = _queue_rows(report["results"])
    for path in queues_dir.glob("*.json"):
        if path.stem not in queues:
            path.unlink()
    for name, rows in queues.items():
        atomic_json(queues_dir / f"{name}.json", {"schema_version": 1, "queue": name, "rows": rows})
    atomic_text(out_dir / "report.html", _render_html(report, out_dir))
    _write_artifact_manifest(out_dir)


def _verify_artifact_manifest(directory: Path) -> None:
    try:
        value, _ = json_snapshot(directory / "artifact-manifest.json")
    except (OSError, ValueError) as exc:
        raise BatchError(f"invalid artifact manifest: {exc}") from exc
    if not isinstance(value, dict) or set(value) != {"schema_version", "files"}:
        raise BatchError("invalid artifact manifest schema")
    if value["schema_version"] != 1 or not isinstance(value["files"], list):
        raise BatchError("invalid artifact manifest contents")
    listed = set()
    for row in value["files"]:
        if not isinstance(row, dict) or set(row) != {"path", "size_bytes", "sha256"}:
            raise BatchError("invalid artifact record")
        relative = row["path"]
        if (
            not isinstance(relative, str)
            or not relative
            or "\\" in relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or relative in listed
        ):
            raise BatchError("unsafe or duplicate artifact path")
        listed.add(relative)
        path = directory / relative
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(directory):
            raise BatchError(f"published artifact missing or unsafe: {relative}")
        if path.stat().st_size != row["size_bytes"] or digest(path) != row["sha256"]:
            raise BatchError(f"published artifact failed integrity check: {relative}")
    # Admit every entry before file-only coverage comparison, including symlinks
    # whose targets are missing or directories. Never follow those entries.
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise BatchError(f"published generation contains unlisted symlink: {path.relative_to(directory)}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise BatchError(f"published generation contains non-regular entry: {path.relative_to(directory)}")
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name != "artifact-manifest.json"
    }
    if actual != listed:
        raise BatchError("artifact manifest does not cover the complete report generation")


def current_report_dir(out_dir: Path) -> Path:
    """Resolve and verify the one atomically published report generation."""
    out_dir = Path(out_dir).resolve()
    pointer = out_dir / "current.json"
    try:
        value, _ = json_snapshot(pointer)
    except (OSError, ValueError) as exc:
        raise BatchError(f"invalid report pointer: {exc}") from exc
    if not isinstance(value, dict) or set(value) != {
        "schema_version", "generation", "report_sha256", "artifact_manifest_sha256",
    }:
        raise BatchError("invalid report pointer schema")
    generation = value["generation"]
    if (
        value["schema_version"] != 1
        or not isinstance(generation, str)
        or len(generation) != 32
        or any(char not in "0123456789abcdef" for char in generation)
    ):
        raise BatchError("invalid report generation")
    directory = out_dir / "generations" / generation
    if not directory.is_dir() or not directory.resolve().is_relative_to(out_dir):
        raise BatchError("published report generation is missing")
    if digest(directory / "report.json") != value["report_sha256"]:
        raise BatchError("published report digest mismatch")
    if digest(directory / "artifact-manifest.json") != value["artifact_manifest_sha256"]:
        raise BatchError("published artifact manifest digest mismatch")
    _verify_artifact_manifest(directory)
    return directory


def _sync_directory(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def render_reports(report: dict, out_dir: Path) -> Path:
    """Publish the complete report set through one atomic generation pointer."""
    import fcntl

    out_dir = Path(out_dir).resolve()
    generations = out_dir / "generations"
    generations.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(str(out_dir / ".report.lock"), os.O_RDWR | os.O_CREAT, 0o644)
    generation = uuid.uuid4().hex
    staging = generations / f".{generation}.tmp"
    final = generations / generation
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        staging.mkdir()
        _render_report_files(report, staging)
        _verify_artifact_manifest(staging)
        queues_dir = staging / "queues"
        if queues_dir.is_dir():
            _sync_directory(queues_dir)
        _sync_directory(staging)
        os.replace(staging, final)
        _sync_directory(generations)
        pointer_value = {
            "schema_version": 1,
            "generation": generation,
            "report_sha256": digest(final / "report.json"),
            "artifact_manifest_sha256": digest(final / "artifact-manifest.json"),
        }
        try:
            atomic_json(out_dir / "current.json", pointer_value)
        except BaseException:
            try:
                current = json.loads((out_dir / "current.json").read_text())
            except (OSError, json.JSONDecodeError):
                current = None
            if not isinstance(current, dict) or current.get("generation") != generation:
                shutil.rmtree(final, ignore_errors=True)
            raise
        return current_report_dir(out_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)


def _validate_batch_journal(journal: dict, identity: dict, run_digest: str, index_manifest: dict) -> dict:
    required = {"schema_version", "run_digest", "identity", "completed"}
    if not isinstance(journal, dict) or set(journal) != required:
        raise BatchError("invalid resume journal schema")
    if (
        journal["schema_version"] != BATCH_SCHEMA_VERSION
        or journal["run_digest"] != run_digest
        or journal["identity"] != identity
    ):
        raise BatchError("resume journal does not match inputs, index, implementation, or options")
    completed = journal["completed"]
    if not isinstance(completed, dict):
        raise BatchError("invalid resume journal completed records")
    inventory = {row["relpath"]: row for row in identity["inventory"]}
    if not set(completed).issubset(inventory):
        raise BatchError("resume journal contains a clip outside the input inventory")
    # Source binding follows result.best_master/source_record.id into the exact
    # manifest master, never the clip relpath: a clip filename has nothing to do
    # with the master's source_relpath.
    manifest_sources = {
        m["id"]: m
        for m in index_manifest.get("masters", [])
        if isinstance(m, dict) and isinstance(m.get("id"), str)
    }
    for relpath, result in completed.items():
        if not isinstance(result, dict) or result.get("clip") != inventory[relpath]["path"]:
            raise BatchError(f"resume result clip identity disagrees with inventory: {relpath}")
        source_record = result.get("source_record")
        if source_record is not None:
            if not isinstance(source_record, dict):
                raise BatchError(f"resume result has invalid source_record: {relpath}")
            source_id = result.get("best_master") or source_record.get("id")
            manifest_entry = manifest_sources.get(source_id)
            if manifest_entry is None:
                raise BatchError(f"resume result references unknown source in manifest: {relpath}")
            for field in (
                "id", "source_relpath", "display_name", "original_root_hint", "content_digest",
            ):
                if source_record.get(field) != manifest_entry.get(field):
                    raise BatchError(
                        f"resume result source_record {field} changed for {relpath}"
                    )
        # Malformed source associations, digests, and decisions are caught here.
        try:
            adapt_product(result)  # Also verifies evidence files when present.
        except (TypeError, ValueError) as exc:
            raise BatchError(f"invalid resumed result for {relpath}: {exc}") from exc
        receipt = result.get("matching_receipt")
        options = identity["options"]
        phase6_policy = (
            options["speed_profile"]["name"] != "v1-default"
            or options["geometry_mode"] != "identity"
            or options["audio_enabled"]
        )
        if phase6_policy and (
            result.get("schema_version") != 3
            or not isinstance(receipt, dict)
            or receipt.get("schema_version") != 2
        ):
            raise BatchError(
                f"resume result is missing the current Phase 6 policy receipt: {relpath}"
            )
        if receipt is not None:
            inputs = receipt["input_receipt"]
            if inputs["clip_sha256"] != inventory[relpath]["sha256"]:
                raise BatchError(f"resume result receipt clip digest changed for {relpath}")
            if inputs["index_manifest_sha256"] != identity["index_manifest_sha256"]:
                raise BatchError(f"resume result receipt index digest changed for {relpath}")
            if receipt["speed_profile"] != options["speed_profile"]:
                raise BatchError(f"resume result receipt speed profile changed for {relpath}")
            if receipt["schema_version"] == 2 and (
                receipt["geometry_mode"] != options["geometry_mode"]
                or receipt["audio_enabled"] != options["audio_enabled"]
            ):
                raise BatchError(f"resume result receipt match policy changed for {relpath}")
    return dict(completed)


def _match_directory_locked(
    clips_root: Path,
    index_dir: Path,
    out_dir: Path,
    *,
    resume: bool = True,
    continue_on_error: bool = True,
    evidence: bool = True,
    calibration_path: Path | None = None,
    progress=None,
    interrupt_after: int | None = None,
    speed_profile: str | None = None,
    config: Config | None = None,
    geometry_mode: str | None = None,
    audio_enabled: bool | None = None,
) -> dict:
    """Match a directory with durable per-clip checkpoints and deterministic exports."""
    clips_root = Path(clips_root).resolve()
    index_dir = Path(index_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    inventory = discover_clips(clips_root)
    if not inventory:
        raise BatchError(f"no supported videos in {clips_root}")
    index_manifest, masters = load_index(index_dir)
    try:
        index_config = config_from_manifest(index_manifest)
    except IndexReadError:
        # Test doubles and legacy in-process callers may supply a reduced
        # manifest. The default path still delegates policy selection to the
        # matcher exactly as before; an opt-in profile requires a valid index.
        if speed_profile is not None:
            raise
        index_config = Config()
    base_config = config if config is not None else index_config
    resolved_config, speed_profile_receipt = resolve_speed_profile(
        speed_profile, base_config
    )
    effective_geometry = geometry_mode or resolved_config.match.geometry_mode
    effective_audio = (
        resolved_config.match.audio_enabled if audio_enabled is None else audio_enabled
    )
    resolved_config = resolved_config.model_copy(update={
        "match": resolved_config.match.model_copy(update={
            "geometry_mode": effective_geometry,
            "audio_enabled": effective_audio,
        })
    })
    match_config = (
        resolved_config
        if (
            speed_profile is not None
            or config is not None
            or effective_geometry != "identity"
            or effective_audio
        )
        else None
    )
    # Batch identity binds to the loaded snapshot's manifest_sha256 attribute
    # (set by _ManifestSnapshot at parse time), never to a digest of whatever
    # manifest.json bytes happen to sit at the path now.
    manifest_snapshot_sha = getattr(index_manifest, "manifest_sha256", None)
    if (
        not isinstance(manifest_snapshot_sha, str)
        or len(manifest_snapshot_sha) != 64
        or any(char not in "0123456789abcdef" for char in manifest_snapshot_sha)
    ):
        raise BatchError("loaded index manifest does not carry a snapshot digest")
    calibration, calibration_file_digest = _load_calibration(calibration_path)
    options = {
        "evidence": evidence,
        "calibration_sha256": calibration.get("calibration_sha256") if calibration else None,
        "calibration_file_sha256": calibration_file_digest,
        "speed_profile": speed_profile_receipt,
        "geometry_mode": effective_geometry,
        "audio_enabled": effective_audio,
        "media_config": resolved_config.media.model_dump(mode="json"),
        "match_config": resolved_config.match.model_dump(mode="json"),
    }
    implementation = _implementation_receipt()
    identity = {
        "schema_version": BATCH_SCHEMA_VERSION,
        "clips_root": str(clips_root),
        "index_dir": str(index_dir),
        "index_manifest_sha256": manifest_snapshot_sha,
        "implementation_sha256": implementation["sha256"],
        "inventory": inventory,
        "options": options,
    }
    run_digest = _json_digest(identity)
    journal_path = out_dir / ".batch-progress.json"
    completed = {}
    if journal_path.exists():
        if not resume:
            journal_path.unlink()
        else:
            try:
                journal, _ = json_snapshot(journal_path)
            except (OSError, ValueError) as exc:
                raise BatchError(f"corrupt resume journal: {exc}") from exc
            completed = _validate_batch_journal(journal, identity, run_digest, index_manifest)
    resumed = len(completed)
    failures = []
    started = time.monotonic()
    processed_this_run = 0
    evidence_root = out_dir / "evidence"

    for position, item in enumerate(inventory, start=1):
        relpath = item["relpath"]
        if relpath in completed:
            if progress:
                progress({"event": "resumed", "clip": relpath, "position": position, "total": len(inventory)})
            continue
        if progress:
            progress({"event": "matching", "clip": relpath, "position": position, "total": len(inventory)})
        clip_path = Path(item["path"])
        pin = None
        stage_dir = None
        try:
            # Consume verified immutable staged bytes for BOTH matching and
            # evidence: pin the descriptor, bind it to the frozen inventory
            # digest/size, then stage a private read-only snapshot. All decode
            # paths see the snapshot even if the original is replaced and
            # restored (or replaced outright) while the call is in flight.
            pin = _pin_source(clips_root, clip_path)
            if pin.digest != item["sha256"] or os.fstat(pin.fd).st_size != item["size_bytes"]:
                raise BatchError(f"clip changed since inventory: {relpath}")
            stage_dir = tempfile.TemporaryDirectory(prefix=".clip-", dir=out_dir)
            staged_clip = pin.stage(Path(stage_dir.name))
            diag, manifest = match_clip_path(
                staged_clip,
                index_dir,
                match_config,
                loaded_index=(index_manifest, masters),
            )
            matching_receipt = None
            if speed_profile is not None or effective_geometry != "identity" or effective_audio:
                matching_receipt = {
                    "schema_version": 2,
                    "speed_profile": speed_profile_receipt,
                    "geometry_mode": effective_geometry,
                    "audio_enabled": effective_audio,
                    "input_receipt": {
                        "clip_sha256": item["sha256"],
                        "index_manifest_sha256": identity["index_manifest_sha256"],
                    },
                }
            result = product_result(
                str(clip_path), diag, manifest, matching_receipt=matching_receipt
            )
            if calibration is not None:
                result = apply_calibration(result, calibration)
            if evidence and result["source_record"] is not None and result["confidence"]["timing"]["aligned_timestamps"]:
                evidence_id = hashlib.sha256(relpath.encode()).hexdigest()[:20]
                source = result["source_record"]
                master_path = Path(source["original_root_hint"]) / source["source_relpath"]
                artifact = generate_evidence(
                    staged_clip,
                    master_path,
                    master_id=source["id"],
                    source_digest=source["content_digest"],
                    aligned_timestamps=result["confidence"]["timing"]["aligned_timestamps"],
                    output_dir=evidence_root / evidence_id,
                )
                # Persist the original absolute inventory path, never the
                # transient staged snapshot path.
                artifact["clip_path"] = str(clip_path)
                result = product_result(
                    str(clip_path),
                    diag,
                    manifest,
                    evidence=artifact,
                    matching_receipt=matching_receipt,
                )
                if calibration is not None:
                    result = apply_calibration(result, calibration)
            # Recheck the original after decoding and before checkpointing: a
            # replaced-then-restored file is safe because every consumer saw
            # the staged snapshot, but bytes that still disagree are rejected.
            if clip_path.stat().st_size != item["size_bytes"] or digest(clip_path) != item["sha256"]:
                raise BatchError(f"clip changed during matching: {relpath}")
            adapt_product(result)  # Verifies evidence against original bytes.
            completed[relpath] = result
        except Exception as exc:  # Per-item containment is the batch contract.
            failures.append({
                "clip": relpath,
                "type": type(exc).__name__,
                "message": str(exc),
            })
            if not continue_on_error:
                raise
        finally:
            if pin is not None:
                pin.close()
            if stage_dir is not None:
                stage_dir.cleanup()
        processed_this_run += 1
        atomic_json(journal_path, {
            "schema_version": BATCH_SCHEMA_VERSION,
            "run_digest": run_digest,
            "identity": identity,
            "completed": completed,
        })
        if interrupt_after is not None and processed_this_run >= interrupt_after:
            raise KeyboardInterrupt("injected batch interruption")

    # Before publication, every inventory row is rechecked against the frozen
    # identity so a report can never vouch for stale inventory; restored bytes
    # pass and persistently changed bytes abort publication.
    for item in inventory:
        current_path = Path(item["path"])
        try:
            stale = (
                current_path.stat().st_size != item["size_bytes"]
                or digest(current_path) != item["sha256"]
            )
        except OSError:
            stale = True
        if stale:
            raise BatchError(f"inventory changed during run: {item['relpath']}")

    elapsed = time.monotonic() - started
    results = [
        {"clip": item["relpath"], "result": completed[item["relpath"]]}
        for item in inventory
        if item["relpath"] in completed
    ]
    failures.sort(key=lambda row: row["clip"])
    report = {
        "schema_version": 1,
        "run_receipt": {
            "run_digest": run_digest,
            "clips_root": str(clips_root),
            "index_manifest_sha256": identity["index_manifest_sha256"],
            "options": options,
            "speed_profile": speed_profile_receipt,
            "input_receipt": {
                "inventory_digest": _json_digest(inventory),
                "index_manifest_digest": identity["index_manifest_sha256"],
            },
            "inventory_count": len(inventory),
            "result_count": len(results),
            "failure_count": len(failures),
            "resumed_count": resumed,
            "implementation": implementation,
            "artifact_manifest": "artifact-manifest.json",
        },
        "inventory": inventory,
        "results": results,
        "failures": failures,
        "performance": {
            "elapsed_sec": elapsed,
            "processed_this_run": processed_this_run,
            "clips_per_sec": processed_this_run / elapsed if elapsed else None,
            "peak_rss_bytes": _peak_rss_bytes(),
            "memory_ceiling_bytes": MEMORY_CEILING_BYTES,
            "throughput_target_clips_per_sec": THROUGHPUT_TARGET_CLIPS_PER_SEC,
            "one_loaded_index_reused": True,
        },
    }
    render_reports(report, out_dir)
    if progress:
        progress({"event": "complete", "results": len(results), "failures": len(failures)})
    return report


def match_directory(
    clips_root: Path,
    index_dir: Path,
    out_dir: Path,
    *,
    resume: bool = True,
    continue_on_error: bool = True,
    evidence: bool = True,
    calibration_path: Path | None = None,
    progress=None,
    interrupt_after: int | None = None,
    speed_profile: str | None = None,
    config: Config | None = None,
    geometry_mode: str | None = None,
    audio_enabled: bool | None = None,
) -> dict:
    """Serialize writers to one output directory, then run the resumable batch."""
    import fcntl

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(str(out_dir / ".batch.lock"), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        return _match_directory_locked(
            clips_root,
            index_dir,
            out_dir,
            resume=resume,
            continue_on_error=continue_on_error,
            evidence=evidence,
            calibration_path=calibration_path,
            progress=progress,
            interrupt_after=interrupt_after,
            speed_profile=speed_profile,
            config=config,
            geometry_mode=geometry_mode,
            audio_enabled=audio_enabled,
        )
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)
