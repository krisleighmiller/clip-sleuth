"""Generate and validate visual evidence for reported clip alignments."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path

from clipprov._io import finite_number
from clipprov.config import MediaConfig
from clipprov.media import MediaError, probe


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame(tool: str, source: Path, timestamp: float, output: Path) -> None:
    proc = subprocess.run(
        [tool, "-v", "error", "-y", "-ss", f"{timestamp:.6f}", "-i", str(source),
         "-frames:v", "1", "-q:v", "2", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
        raise MediaError(f"evidence frame extraction failed for {source}: {proc.stderr.strip()}")


def generate_evidence(
    clip_path: Path,
    master_path: Path,
    *,
    master_id: str,
    source_digest: str,
    aligned_timestamps: list[dict],
    output_dir: Path,
    config: MediaConfig | None = None,
) -> dict:
    """Publish frames from verified immutable inputs, retaining logical source paths."""
    from clipprov.index import _staged_source_snapshot

    clip_path, master_path = Path(clip_path).resolve(), Path(master_path).resolve()
    with (
        _staged_source_snapshot(clip_path) as (staged_clip, clip_digest),
        _staged_source_snapshot(master_path) as (staged_master, master_digest),
    ):
        if master_digest != source_digest:
            raise ValueError("master content digest disagrees with index source identity")
        artifact = _generate_evidence_snapshots(
            staged_clip, staged_master, master_id=master_id, source_digest=source_digest,
            aligned_timestamps=aligned_timestamps, output_dir=output_dir, config=config,
        )
        artifact.update(
            clip_path=str(clip_path), master_path=str(master_path),
            clip_sha256=clip_digest, master_sha256=master_digest,
        )
        return artifact


def _generate_evidence_snapshots(
    clip_path: Path,
    master_path: Path,
    *,
    master_id: str,
    source_digest: str,
    aligned_timestamps: list[dict],
    output_dir: Path,
    config: MediaConfig | None = None,
) -> dict:
    """Atomically produce first/middle/last aligned frame pairs from private copies."""
    clip_path = Path(clip_path).resolve()
    master_path = Path(master_path).resolve()
    clip_info = probe(clip_path, config)
    master_info = probe(master_path, config)
    actual_digest = _sha256(master_path)
    if actual_digest != source_digest:
        raise ValueError("master content digest disagrees with index source identity")
    pairs = aligned_timestamps
    if not pairs:
        raise ValueError("cannot generate evidence without aligned timestamps")
    selected = sorted({0, len(pairs) // 2, len(pairs) - 1})
    for pair in pairs:
        if set(pair) != {"clip_sec", "master_sec"}:
            raise ValueError("invalid aligned timestamp fields")
        if not 0 <= pair["clip_sec"] < clip_info.duration_s:
            raise ValueError("clip evidence timestamp outside timeline")
        if not 0 <= pair["master_sec"] < master_info.duration_s:
            raise ValueError("master evidence timestamp outside timeline")
    ffmpeg_name = config.ffmpeg if config else "ffmpeg"
    ffmpeg = shutil.which(ffmpeg_name)
    if ffmpeg is None:
        raise MediaError(f"required tool not found on PATH: {ffmpeg_name}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generations = output_dir / "generations"
    generations.mkdir(exist_ok=True)
    import fcntl
    import json

    lock_fd = os.open(str(output_dir / ".evidence.lock"), os.O_RDWR | os.O_CREAT, 0o644)
    generation = uuid.uuid4().hex
    staging = generations / f".{generation}.tmp"
    final_dir = generations / generation
    pointer_tmp = output_dir / f".current.{generation}.tmp"
    frames = []
    promoted = False
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        staging.mkdir()
        for sequence, pair_index in enumerate(selected):
            pair = pairs[pair_index]
            clip_frame = staging / f"{sequence:02d}-clip.jpg"
            master_frame = staging / f"{sequence:02d}-master.jpg"
            _frame(ffmpeg, clip_path, pair["clip_sec"], clip_frame)
            _frame(ffmpeg, master_path, pair["master_sec"], master_frame)
            frames.append({
                "clip_sec": pair["clip_sec"],
                "master_sec": pair["master_sec"],
                "clip_frame": clip_frame.name,
                "master_frame": master_frame.name,
                "clip_frame_sha256": _sha256(clip_frame),
                "master_frame_sha256": _sha256(master_frame),
            })
        os.replace(staging, final_dir)
        pointer_tmp.write_text(json.dumps({"generation": generation}) + "\n")
        with pointer_tmp.open("rb") as pointer_handle:
            os.fsync(pointer_handle.fileno())
        os.replace(pointer_tmp, output_dir / "current.json")
        # From this point current.json names final_dir. Never delete that
        # generation during exception cleanup, even if durability fsync fails.
        promoted = True
        parent_fd = os.open(str(output_dir), os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        pointer_tmp.unlink(missing_ok=True)
        if final_dir.exists() and not promoted:
            shutil.rmtree(final_dir, ignore_errors=True)
        raise
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)
    return {
        "schema_version": 1,
        "directory": str(final_dir.resolve()),
        "clip_path": str(clip_path),
        "clip_sha256": _sha256(clip_path),
        "clip_duration_sec": clip_info.duration_s,
        "master_path": str(master_path),
        "master_id": master_id,
        "master_sha256": actual_digest,
        "master_duration_sec": master_info.duration_s,
        "frames": frames,
    }


def _reported_path_matches(reported, actual: Path) -> bool:
    """Evidence requires an exact absolute clip identity, never a basename guess."""
    if not isinstance(reported, str) or not reported:
        return False
    try:
        reported_path = Path(reported)
        return reported_path.is_absolute() and reported_path.resolve() == actual.resolve()
    except (OSError, ValueError, RuntimeError):
        return False


def _image_decodes(path: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    proc = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(path), "-frames:v", "1", "-f", "null", "-"],
        capture_output=True,
        check=False,
    )
    return proc.returncode == 0


def validate_evidence(evidence: dict, result: dict) -> list[str]:
    """Return consistency failures for an evidence object and product result."""
    problems = []
    if not isinstance(result, dict):
        return ["invalid result schema"]
    required = {
        "schema_version", "directory", "clip_path", "clip_sha256", "clip_duration_sec",
        "master_path", "master_id", "master_sha256", "master_duration_sec", "frames",
    }
    if not isinstance(evidence, dict) or evidence.keys() != required:
        return ["invalid evidence schema"]
    if evidence["schema_version"] != 1:
        return ["invalid evidence schema version"]
    text_fields = ("directory", "clip_path", "clip_sha256", "master_path", "master_id", "master_sha256")
    if any(not isinstance(evidence[key], str) or not evidence[key] for key in text_fields):
        return ["invalid evidence text fields"]
    if not re.fullmatch(r"[0-9a-f]{64}", evidence["clip_sha256"]) or not re.fullmatch(
        r"[0-9a-f]{64}", evidence["master_sha256"]
    ):
        return ["invalid evidence source digest"]
    for key in ("clip_duration_sec", "master_duration_sec"):
        value = evidence[key]
        if not finite_number(value) or value <= 0:
            return [f"invalid {key}"]
    if not isinstance(evidence["frames"], list) or not evidence["frames"]:
        return ["invalid evidence frames"]
    directory = Path(evidence["directory"])
    source = result.get("source_record")
    clip_path = Path(str(evidence.get("clip_path", "")))
    master_path = Path(str(evidence.get("master_path", "")))
    if not _reported_path_matches(result.get("clip"), clip_path):
        problems.append("clip path disagrees with result")

    # Validate source_record: must be complete dict with all 5 required fields, all non-empty strings
    if source is not None:
        required_source_fields = ("id", "source_relpath", "display_name", "original_root_hint", "content_digest")
        if (
            not isinstance(source, dict)
            or set(source.keys()) != set(required_source_fields)
            or any(not isinstance(source[f], str) or not source[f] for f in required_source_fields)
        ):
            problems.append("source_record must be a complete dict with all required fields")
            source = None

    if source is None or evidence["master_id"] != source.get("id"):
        problems.append("master identity disagrees with result")
    if source is None or evidence["master_sha256"] != source.get("content_digest"):
        problems.append("master digest disagrees with result")
    if source is not None:
        try:
            expected_master = Path(source["original_root_hint"]) / source["source_relpath"]
            master_matches = expected_master.resolve() == master_path.resolve()
        except (OSError, ValueError, RuntimeError):
            problems.append("invalid master path")
        else:
            if not master_matches:
                problems.append("master path disagrees with result")
    for path_key, digest_key in (("clip_path", "clip_sha256"), ("master_path", "master_sha256")):
        path = Path(evidence[path_key])
        if not path.is_file():
            problems.append(f"missing {path_key}")
        elif _sha256(path) != evidence[digest_key]:
            problems.append(f"changed {path_key}")
    confidence = result.get("confidence")
    if not isinstance(confidence, dict):
        return problems + ["invalid result confidence"]
    timing = confidence.get("timing")
    if not isinstance(timing, dict):
        return problems + ["invalid result timing"]
    reported = timing.get("aligned_timestamps")
    if not isinstance(reported, list):
        return problems + ["aligned_timestamps must be a list"]
    from clipprov.eval import alignment_timestamp_problems  # deferred: avoids circular import

    timestamp_problems = alignment_timestamp_problems(result)
    problems.extend(timestamp_problems)
    # Only build hash keys after the shared validator has admitted every pair.
    reported_pairs = set() if timestamp_problems else {
        (row["clip_sec"], row["master_sec"]) for row in reported
    }
    frame_fields = {
        "clip_sec", "master_sec", "clip_frame", "master_frame",
        "clip_frame_sha256", "master_frame_sha256",
    }
    for frame in evidence["frames"]:
        if not isinstance(frame, dict) or frame.keys() != frame_fields:
            problems.append("invalid evidence frame schema")
            continue
        timestamps_valid = True
        for key in ("clip_sec", "master_sec"):
            value = frame[key]
            if not finite_number(value) or value < 0:
                problems.append(f"invalid {key}")
                timestamps_valid = False
        if not timestamps_valid:
            continue
        pair = (frame["clip_sec"], frame["master_sec"])
        if pair not in reported_pairs:
            problems.append("evidence timestamp absent from result")
        if not 0 <= frame.get("clip_sec", -1) < evidence["clip_duration_sec"]:
            problems.append("clip frame outside timeline")
        if not 0 <= frame.get("master_sec", -1) < evidence["master_duration_sec"]:
            problems.append("master frame outside timeline")
        for name_key, digest_key in (
            ("clip_frame", "clip_frame_sha256"),
            ("master_frame", "master_frame_sha256"),
        ):
            name = frame.get(name_key, "")
            if not isinstance(name, str) or not name or Path(name).name != name:
                problems.append(f"invalid {name_key}")
                continue
            path = directory / name
            digest_value = frame.get(digest_key)
            if not isinstance(digest_value, str) or not re.fullmatch(r"[0-9a-f]{64}", digest_value):
                problems.append(f"invalid {digest_key}")
                continue
            if not path.is_file():
                problems.append(f"missing {name_key}")
            else:
                if _sha256(path) != frame.get(digest_key):
                    problems.append(f"changed {name_key}")
                if not _image_decodes(path):
                    problems.append(f"undecodable {name_key}")
    return problems
