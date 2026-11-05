"""Frozen controls adapted from Astra validation and Kimi synthesis.

Only manifest-listed clips are evaluated. The production match library is used
unchanged; exhaustive checks are diagnostic and bounded, never another matcher.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from clipprov import __version__
from clipprov._io import atomic_json, digest, json_snapshot
from clipprov.eval import (
    adapt_product,
    load_product_results,
    load_truth,
    product_result,
    score_product,
)
from clipprov.index import load_index
from clipprov.match import audit_shortlist, match_clip_path
from clipprov.media import probe
from clipprov.transforms import audit_decode, number, transform, validate_recipe

GROUPS = ("development", "holdout", "source_withheld")
RECIPE_VERSION = 3
VISUAL_AUDIT_METHOD = "rgb32_normalized_timeline_mae_v1"
VISUAL_DISTANCE_THRESHOLD = 0.06
GENERATOR_IMPLEMENTATION_FILES = (
    "clipprov/__init__.py",
    "clipprov/_io.py",
    "clipprov/cli.py",
    "clipprov/config.py",
    "clipprov/harness.py",
    "clipprov/media.py",
    "clipprov/transforms.py",
)
EVALUATION_IMPLEMENTATION_FILES = (
    "clipprov/__init__.py",
    "clipprov/_io.py",
    "clipprov/appearance.py",
    "clipprov/cli.py",
    "clipprov/config.py",
    "clipprov/confidence.py",
    "clipprov/core.py",
    "clipprov/eval.py",
    "clipprov/evidence.py",
    "clipprov/harness.py",
    "clipprov/index.py",
    "clipprov/match.py",
    "clipprov/media.py",
    "clipprov/motion.py",
    "clipprov/transforms.py",
)
# Frozen numeric values are disjoint across development and holdout. Both audio
# modes appear in each group; their replacement frequencies are also disjoint.
RECIPES = {
    "development": [
        ("baseline", {"audio": "keep"}),
        ("reencode", {"crf": 24, "audio": "keep"}),
        ("crop", {"crop": [0.92, 0.1, 0.9], "audio": "keep"}),
        ("speed", {"speed": 0.984, "audio": "keep"}),
        ("watermark", {"watermark": [0.05, 0.1, 0.4], "audio": "keep"}),
        ("audio_keep", {"audio": "keep", "crf": 25}),
        ("grayscale", {"grayscale": True, "audio": "keep"}),
        ("resize", {"resize": [88, 58], "audio": "keep"}),
        ("brightness", {"brightness": 0.08, "audio": "keep"}),
        ("saturation", {"saturation": 0.72, "audio": "keep"}),
        ("padding", {"padding": [6, 4], "audio": "keep"}),
        ("audio_remove", {"audio": "remove"}),
        ("audio_replace", {"audio": "replace", "audio_frequency": 560}),
        ("crop_speed", {"crop": [0.88, 0.2, 0.8], "speed": 1.016, "audio": "keep"}),
        ("compound_stress", {
            "crop": [0.9, 0, 1],
            "resize": [64, 48],
            "crf": 26,
            "watermark": [0.05, 0.1, 0.4],
            "brightness": 0.04,
            "saturation": 0.65,
            "padding": [4, 4],
            "speed": 0.98,
            "audio": "remove",
            "audio_frequency": 420,
        }),
    ],
    "holdout": [
        ("baseline", {"audio": "keep"}),
        ("reencode", {"crf": 27, "audio": "keep"}),
        ("crop", {"crop": [0.87, 0.3, 0.7], "audio": "keep"}),
        ("speed", {"speed": 0.976, "audio": "keep"}),
        ("watermark", {"watermark": [0.2, 0.15, 0.6], "audio": "keep"}),
        ("audio_keep", {"audio": "keep", "crf": 28}),
        ("grayscale", {"grayscale": True, "audio": "keep"}),
        ("resize", {"resize": [74, 54], "audio": "keep"}),
        ("brightness", {"brightness": -0.07, "audio": "keep"}),
        ("saturation", {"saturation": 1.35, "audio": "keep"}),
        ("padding", {"padding": [10, 8], "audio": "keep"}),
        ("audio_remove", {"audio": "remove"}),
        ("audio_replace", {"audio": "replace", "audio_frequency": 860}),
        ("crop_speed", {"crop": [0.83, 0.7, 0.3], "speed": 1.024, "audio": "keep"}),
        ("compound_stress", {
            "crop": [0.85, 0.25, 0.75],
            "resize": [80, 60],
            "crf": 31,
            "watermark": [0.2, 0.15, 0.6],
            "brightness": -0.03,
            "saturation": 1.2,
            "padding": [8, 6],
            "speed": 0.972,
            "audio": "replace",
            "audio_frequency": 740,
        }),
    ],
}
RECIPES["source_withheld"] = RECIPES["development"]


def _generator_implementation(package_root: Path | None = None) -> dict[str, Any]:
    """Digest the ordered set of installed files that can affect generation."""
    root = Path(package_root or Path(__file__).resolve().parent.parent)
    files = []
    for relative in GENERATOR_IMPLEMENTATION_FILES:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"generator implementation file is unavailable: {relative}")
        files.append({"path": relative, "sha256": digest(path)})
    frozen = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return {"files": files, "sha256": hashlib.sha256(frozen.encode()).hexdigest()}


def _evaluation_implementation(package_root: Path | None = None) -> dict[str, Any]:
    """Digest the ordered installed files that can affect evaluation reports."""
    root = Path(package_root or Path(__file__).resolve().parent.parent)
    files = []
    for relative in EVALUATION_IMPLEMENTATION_FILES:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"evaluation implementation file is unavailable: {relative}")
        files.append({"path": relative, "sha256": digest(path)})
    frozen = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return {"files": files, "sha256": hashlib.sha256(frozen.encode()).hexdigest()}


def _git_receipt(package_root: Path | None = None) -> dict[str, Any]:
    """Return source Git identity only when the package root is that repository root.

    Installed packages may live below an unrelated parent repository.  Such a
    repository is not provenance for these installed bytes and must not leak
    into the receipt.
    """
    root = Path(package_root or Path(__file__).resolve().parent.parent).resolve()
    try:
        repository = Path(subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], cwd=root,
            capture_output=True, text=True, check=True,
        ).stdout.strip()).resolve()
        if repository != root:
            return {"commit": "unavailable", "dirty": None}
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=root,
            capture_output=True, text=True, check=True,
        ).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError, ValueError):
        return {"commit": "unavailable", "dirty": None}


def _generation_receipt():
    """Record the implementation and runtime that produced control bytes."""
    try:
        ffmpeg = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, check=True
        ).stdout.splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError) as exc:
        raise ValueError(f"cannot record ffmpeg version: {exc}") from exc
    git = _git_receipt()
    import numpy as np

    implementation = _generator_implementation()
    return {
        "package_version": __version__,
        "git": git,
        "ffmpeg_version": ffmpeg,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        # Keep the legacy aggregate field while exposing all of its inputs.
        "generator_sha256": implementation["sha256"],
        "generator_implementation": implementation,
        "recipe_version": RECIPE_VERSION,
    }


def _evaluation_receipt():
    """Record the installed implementation and runtime that produced a report."""
    try:
        ffmpeg = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, check=True
        ).stdout.splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError) as exc:
        raise ValueError(f"cannot record ffmpeg version: {exc}") from exc
    git = _git_receipt()
    import numpy as np

    implementation = _evaluation_implementation()
    return {
        "package_version": __version__,
        "git": git,
        "ffmpeg_version": ffmpeg,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "evaluation_sha256": implementation["sha256"],
        "evaluation_implementation": implementation,
    }


def _valid_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _validate_generation_receipt(receipt):
    if not isinstance(receipt, dict):
        raise ValueError("generation receipt must be an object")  # noqa: TRY004
    if any(not isinstance(receipt.get(key), str) or not receipt[key] for key in (
        "package_version", "ffmpeg_version", "python_version", "numpy_version"
    )):
        raise ValueError("generation receipt runtime schema mismatch")
    git = receipt.get("git")
    if (
        not isinstance(git, dict)
        or set(git) != {"commit", "dirty"}
        or not isinstance(git["commit"], str)
        or (git["dirty"] is not None and type(git["dirty"]) is not bool)
        or receipt.get("recipe_version") != RECIPE_VERSION
    ):
        raise ValueError("generation receipt provenance schema mismatch")
    implementation = receipt.get("generator_implementation")
    if not isinstance(implementation, dict) or set(implementation) != {"files", "sha256"}:
        raise ValueError("generator implementation identity schema mismatch")
    files = implementation["files"]
    if (
        not isinstance(files, list)
        or [row.get("path") if isinstance(row, dict) else None for row in files]
        != list(GENERATOR_IMPLEMENTATION_FILES)
        or any(
            set(row) != {"path", "sha256"} or not _valid_sha256(row["sha256"])
            for row in files
        )
    ):
        raise ValueError("generator implementation file identity mismatch")
    frozen = json.dumps(files, sort_keys=True, separators=(",", ":"))
    aggregate = hashlib.sha256(frozen.encode()).hexdigest()
    if implementation["sha256"] != aggregate or receipt.get("generator_sha256") != aggregate:
        raise ValueError("generator implementation aggregate mismatch")


def _metadata(info):
    return {
        k: getattr(info, k)
        for k in ("duration_s", "width", "height", "fps", "frame_count", "has_video", "has_audio")
    }


def _relative(root, name):
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("invalid manifest relative path")
    p = Path(name)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError("unsafe manifest path")
    resolved = (root / p).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("manifest path escapes root")
    return resolved


def _visual_signature(path):
    """Return deterministic, fixed-size RGB timeline frames and their identity."""
    import numpy as np

    run = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-vf",
            "fps=2,scale=32:32:flags=area,format=rgb24", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-",
        ],
        capture_output=True,
        check=False,
    )
    if run.returncode:
        raise ValueError(
            f"visual signature decode failed for {path}: "
            f"{run.stderr.decode(errors='replace').strip()}"
        )
    frame_size = 32 * 32 * 3
    count = len(run.stdout) // frame_size
    if count == 0:
        raise ValueError(f"visual signature decoded zero frames from {path}")
    frames = np.frombuffer(run.stdout[: count * frame_size], dtype=np.uint8).reshape(
        count, 32, 32, 3
    )
    indices = np.linspace(0, count - 1, 8).round().astype(int)
    normalized = np.ascontiguousarray(frames[indices])
    return hashlib.sha256(normalized.tobytes()).hexdigest(), normalized


def _visual_distance(left, right):
    import numpy as np

    return float(np.abs(left.astype(np.int16) - right.astype(np.int16)).mean() / 255.0)


def _assign_sources(library, *, return_audit=False):
    paths = sorted(library.rglob("*.mp4"))
    if len({p.stem for p in paths}) != len(paths):
        raise ValueError("duplicate source basenames")
    sources = []
    paths_by_digest = {}
    signatures = []
    for path in paths:
        sha = digest(path)
        relative = path.relative_to(library).as_posix()
        previous = paths_by_digest.get(sha)
        if previous is not None:
            raise ValueError(
                "duplicate source content is not allowed: "
                f"{previous} and {relative} share SHA-256 {sha}"
            )
        paths_by_digest[sha] = relative
        info = probe(path)
        audit_decode(path)
        signature, frames = _visual_signature(path)
        signatures.append(frames)
        sources.append(
            {
                "source_relpath": relative,
                "sha256": sha,
                "visual_signature": signature,
                "metadata": _metadata(info),
            }
        )

    parents = list(range(len(sources)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parents[max(left, right)] = min(left, right)

    near_pairs = []
    comparison_count = 0
    for left in range(len(sources)):
        for right in range(left + 1, len(sources)):
            comparison_count += 1
            distance = _visual_distance(signatures[left], signatures[right])
            if distance <= VISUAL_DISTANCE_THRESHOLD:
                union(left, right)
                near_pairs.append(
                    {
                        "left": sources[left]["source_relpath"],
                        "right": sources[right]["source_relpath"],
                        "distance": round(distance, 8),
                    }
                )
    components = {}
    for index in range(len(sources)):
        components.setdefault(find(index), []).append(index)
    ordered = sorted(components.values(), key=lambda members: sources[min(members)]["source_relpath"])
    if len(ordered) < 3:
        raise ValueError("at least three visually distinct source families are required")
    families = []
    for family_index, members in enumerate(ordered):
        family_id = f"family_{family_index:04d}"
        group = GROUPS[family_index % len(GROUPS)]
        for member in members:
            sources[member]["source_family_id"] = family_id
            sources[member]["group"] = group
        families.append(
            {
                "id": family_id,
                "group": group,
                "members": [sources[member]["source_relpath"] for member in members],
            }
        )
    audit = {
        "method": VISUAL_AUDIT_METHOD,
        "threshold": VISUAL_DISTANCE_THRESHOLD,
        "comparison_count": comparison_count,
        "pairs": near_pairs,
        "family_count": len(families),
        "families": families,
    }
    return (sources, audit) if return_audit else sources


def _timing(meta, ri):
    fps = number(meta["fps"], "fps", 0.01, 1000)
    total = math.floor(meta["duration_s"] * fps + 1e-6)
    if meta["frame_count"] is not None:
        total = min(total, meta["frame_count"])
    if total < 4:
        raise ValueError("control source must contain at least four frames")
    span = max(4, min(round(8 * fps), math.floor(total * 0.65)))
    first = (total - span) // 3 if ri == 0 else total - span
    return first / fps, span / fps


def _truth(case):
    return {
        "clip": Path(case["clip"]).name,
        "master": None if case["group"] == "source_withheld" else Path(case["source_relpath"]).name,
        "offset_sec": None if case["group"] == "source_withheld" else case["offset_sec"],
    }


def generate_demo_corpus(out_dir: Path) -> dict[str, Any]:
    """Create the installed three-source smoke corpus used by the documented demo."""
    out_dir = Path(out_dir).resolve()
    created = []
    for i, (width, height, fps, duration) in enumerate(
        ((96, 64, 12, 2), (80, 72, 16, 2.5), (64, 96, 10, 3))
    ):
        relative = Path(f"nested {i}") / f"source {i}.mp4"
        path = out_dir / "sources" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        run = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "lavfi", "-i",
                f"testsrc2=s={width}x{height}:r={fps}:d={duration}", "-f", "lavfi", "-i",
                f"sine=frequency={300 + i * 100}:duration={duration}", "-vf", f"hue=h={i * 75}",
                "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p", "-c:a", "aac",
                "-shortest", str(path),
            ], capture_output=True, text=True, check=False,
        )
        if run.returncode:
            raise ValueError(f"ffmpeg demo generation failed: {run.stderr.strip()}")
        audit_decode(path)
        created.append({"source_relpath": relative.as_posix(), "sha256": digest(path)})
        if i < 2:
            selected = out_dir / "indexed sources" / relative
            selected.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, selected)
    receipt = {"schema_version": 1, "sources": created, "generation": _generation_receipt()}
    atomic_json(out_dir / "demo-receipt.json", receipt)
    return receipt


def _generate_synthetic_negatives(stage: Path) -> list[dict[str, Any]]:
    specs = {
        "unrelated_visual": (
            "nullsrc=s=96x64:r=12:d=2,geq=r='X*3+N*9':g='Y*4+N*5':b='(X+Y)*2+N*13'",
            {"kind": "independent_synthetic"},
        ),
        "synthetic_noise": (
            "nullsrc=s=96x64:r=12:d=2,geq=random(1)*255:128:128",
            {"kind": "independent_synthetic"},
        ),
        "static_color": (
            "color=c=royalblue:s=96x64:r=12:d=2",
            {"kind": "independent_synthetic"},
        ),
        "repeated_pattern": (
            "color=c=black:s=96x64:r=12:d=2,drawgrid=w=12:h=12:t=3:c=white",
            {"kind": "independent_synthetic"},
        ),
        "independent_test_pattern_variant": (
            "smptebars=s=96x64:r=12:d=2,hflip",
            {"kind": "procedural_variant_of", "category": "unrelated_visual"},
        ),
    }
    negatives = []
    for category, (source, relation) in specs.items():
        name = f"negative_{category}.mp4"
        run = subprocess.run(
            ["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "lavfi", "-i", source,
             "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p", str(stage / name)],
            capture_output=True, text=True, check=False,
        )
        if run.returncode:
            raise ValueError(f"ffmpeg {category} generation failed: {run.stderr.strip()}")
        audit_decode(stage / name)
        negatives.append({
            "clip": name,
            "category": category,
            "sha256": digest(stage / name),
            "relation": relation,
        })
    if len({n["clip"] for n in negatives}) != len(negatives) or len(
        {n["sha256"] for n in negatives}
    ) != len(negatives):
        raise ValueError("generated negative clips must be unique")
    return negatives


def _annotate_negative_separation(library, sources, output, negatives):
    positives = []
    for source in sources:
        if source["group"] == "source_withheld":
            continue
        signature, frames = _visual_signature(library / source["source_relpath"])
        if signature != source["visual_signature"]:
            raise ValueError(f"visual signature changed for {source['source_relpath']}")
        positives.append((signature, frames))
    negative_frames = []
    for negative in negatives:
        signature, frames = _visual_signature(output / negative["clip"])
        distances = [
            (_visual_distance(frames, candidate), candidate_id)
            for candidate_id, candidate in positives
        ]
        nearest_distance, nearest_signature = min(distances)
        if nearest_distance <= VISUAL_DISTANCE_THRESHOLD:
            raise ValueError(
                f"foreign procedural negative {negative['clip']} is visually near an indexed "
                f"positive ({nearest_distance:.8f} <= {VISUAL_DISTANCE_THRESHOLD})"
            )
        negative["visual_signature"] = signature
        negative["visual_separation"] = {
            "method": VISUAL_AUDIT_METHOD,
            "threshold": VISUAL_DISTANCE_THRESHOLD,
            "minimum_distance": round(nearest_distance, 8),
            "nearest_signature": nearest_signature,
        }
        negative_frames.append(frames)
    for left in range(len(negatives)):
        for right in range(left + 1, len(negatives)):
            if _visual_distance(negative_frames[left], negative_frames[right]) <= VISUAL_DISTANCE_THRESHOLD:
                raise ValueError(
                    "foreign procedural negatives are not visually distinct: "
                    f"{negatives[left]['clip']} and {negatives[right]['clip']}"
                )


def generate_controls(library: Path, out_dir: Path) -> dict[str, Any]:
    """Generate frozen deterministic recipes and publish manifest.json last.

    Sources are sorted by relative path; exact duplicate bytes are rejected.
    The library may be relocated at evaluation using the library override.
    """
    library, out_dir = Path(library).resolve(), Path(out_dir).resolve()
    if not library.is_dir():
        raise ValueError("library must be a directory")
    if out_dir.is_relative_to(library):
        raise ValueError("control output must be outside the input library")
    sources, visual_audit = _assign_sources(library, return_audit=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(dir=out_dir, prefix=".generate-"))
    try:
        cases = []
        seen = set()
        for source in sources:
            if source["sha256"] in seen:
                continue
            seen.add(source["sha256"])
            meta, group = source["metadata"], source["group"]
            for ri, (diagnostic, recipe) in enumerate(RECIPES[group]):
                start, duration = _timing(meta, ri)
                name = f"{group}_{len(cases):04d}.mp4"
                info = transform(
                    library / source["source_relpath"],
                    stage / name,
                    start=start,
                    duration=duration,
                    recipe=recipe,
                )
                cases.append(
                    {
                        "clip": name,
                        "group": group,
                        "recipe_index": ri,
                        "diagnostic": diagnostic,
                        "recipe": recipe,
                        "source_relpath": source["source_relpath"],
                        "source_sha256": source["sha256"],
                        "source_family_id": source["source_family_id"],
                        "offset_sec": start,
                        "source_span_sec": duration,
                        "source_end_sec": start + duration,
                        "source_metadata": meta,
                        "clip_metadata": _metadata(info),
                        "clip_sha256": digest(stage / name),
                        "audits": {"decode_ok": True, "source_end_ok": True},
                    }
                )
        truth = [_truth(case) for case in cases]
        negatives = _generate_synthetic_negatives(stage)
        _annotate_negative_separation(library, sources, stage, negatives)
        seed = {
            "schema_version": 1,
            "recipe_version": RECIPE_VERSION,
            "sources": sources,
            "cases": cases,
            "truth": truth,
            "negatives": negatives,
            "visual_audit": visual_audit,
            "generation_receipt": _generation_receipt(),
        }
        generation = hashlib.sha256(json.dumps(seed, sort_keys=True, allow_nan=False).encode())
        rel = Path("generations") / generation.hexdigest()[:24]
        dest = out_dir / rel
        dest.parent.mkdir(exist_ok=True)
        if dest.exists():
            artifacts = [
                (case["clip"], case["clip_sha256"]) for case in cases
            ] + [(negative["clip"], negative["sha256"]) for negative in negatives]
            for clip, expected_sha256 in artifacts:
                path = dest / clip
                if not path.is_file() or digest(path) != expected_sha256:
                    raise ValueError("existing generation content hash mismatch")
        else:
            os.rename(stage, dest)
        for case in cases:
            case["clip"] = (rel / case["clip"]).as_posix()
        for negative in negatives:
            negative["clip"] = (rel / negative["clip"]).as_posix()
        manifest = {**seed, "library": str(library)}
        atomic_json(out_dir / "manifest.json", manifest)
        return manifest
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _audit_manifest(manifest, root, library, indexed):
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
        or manifest.get("recipe_version") != RECIPE_VERSION
    ):
        raise ValueError("unsupported controls manifest version")
    _validate_generation_receipt(manifest.get("generation_receipt"))
    sources, cases = manifest.get("sources"), manifest.get("cases")
    if not isinstance(sources, list) or not sources or not isinstance(cases, list) or not cases:
        raise ValueError("manifest must contain sources and cases")
    stored_audit = manifest.get("visual_audit")
    if not isinstance(stored_audit, dict) or not isinstance(stored_audit.get("families"), list):
        raise ValueError("source-family visual audit mismatch")  # noqa: TRY004
    stored_groups = {}
    for source in sources:
        if (
            not isinstance(source, dict)
            or not isinstance(source.get("source_relpath"), str)
            or source.get("group") not in GROUPS
        ):
            raise ValueError("source-family visual audit mismatch")
        stored_groups[source["source_relpath"]] = source["group"]
    family_members = []
    for family in stored_audit["families"]:
        if (
            not isinstance(family, dict)
            or not isinstance(family.get("members"), list)
            or not all(isinstance(member, str) for member in family["members"])
        ):
            raise ValueError("source-family visual audit mismatch")
        family_members.extend(family["members"])
        groups = {stored_groups.get(member) for member in family["members"]}
        if groups != {family.get("group")}:
            raise ValueError("source family would span frozen groups")
    if sorted(family_members) != sorted(stored_groups):
        raise ValueError("source-family visual audit mismatch")
    # Recompute hashes, decode and frozen assignments instead of trusting stored audit flags.
    actual_sources, actual_visual_audit = _assign_sources(library, return_audit=True)
    if actual_sources != sources:
        raise ValueError("source hash, metadata or frozen membership mismatch")
    if manifest.get("visual_audit") != actual_visual_audit:
        raise ValueError("source-family visual audit mismatch")
    by_path = {s["source_relpath"]: s for s in sources}
    entries = indexed["masters"]
    expected_candidates = sorted(
        (
            {"source_relpath": s["source_relpath"], "content_digest": s["sha256"]}
            for s in sources
            if s["group"] in {"development", "holdout"}
        ),
        key=lambda row: (row["source_relpath"], row["content_digest"]),
    )
    actual_candidates = sorted(
        (
            {"source_relpath": m["source_relpath"], "content_digest": m["content_digest"]}
            for m in entries
        ),
        key=lambda row: (row["source_relpath"], row["content_digest"]),
    )
    if actual_candidates != expected_candidates:
        expected_set = {(r["source_relpath"], r["content_digest"]) for r in expected_candidates}
        actual_set = {(r["source_relpath"], r["content_digest"]) for r in actual_candidates}
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        raise ValueError(
            f"candidate universe mismatch (including withheld membership); "
            f"missing={missing}; extra={extra}"
        )
    for source in sources:
        rel, sha = source["source_relpath"], source["sha256"]
        matches = [m for m in entries if m["source_relpath"] == rel]
        if source["group"] == "source_withheld":
            if any(
                m["source_relpath"] == rel
                or m["content_digest"] == sha
                or m["id"] == Path(rel).stem
                for m in entries
            ):
                raise ValueError("source-withheld membership violation")
        elif len(matches) != 1 or matches[0]["content_digest"] != sha:
            raise ValueError(f"index source membership mismatch: {rel}")
    names, observed, truths = set(), set(), {}
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("case must be an object")  # noqa: TRY004
        clip = case.get("clip")
        path = _relative(root, clip)
        if Path(clip).name in names:
            raise ValueError("duplicate control clip")
        names.add(Path(clip).name)
        source = by_path.get(case.get("source_relpath"))
        if source is None or case.get("source_sha256") != source["sha256"]:
            raise ValueError("case source hash mismatch")
        group, ri = case.get("group"), case.get("recipe_index")
        if (
            group != source["group"]
            or case.get("source_family_id") != source["source_family_id"]
            or type(ri) is not int
            or not 0 <= ri < len(RECIPES[group])
            or case.get("diagnostic") != RECIPES[group][ri][0]
            or case.get("recipe") != RECIPES[group][ri][1]
        ):
            raise ValueError("case frozen recipe or membership mismatch")
        key = (source["sha256"], ri)
        if key in observed:
            raise ValueError("duplicate source recipe case")
        observed.add(key)
        info = probe(library / source["source_relpath"])
        validate_recipe(case["recipe"], info)
        start = number(case.get("offset_sec"), "offset", 0, info.duration_s)
        span = number(case.get("source_span_sec"), "span", 1 / info.fps, info.duration_s)
        end = number(case.get("source_end_sec"), "source-end boundary", 0, info.duration_s)
        if abs(start + span - end) > 1e-8 or abs(start * info.fps - round(start * info.fps)) > 1e-6:
            raise ValueError("source-end boundary or frame alignment mismatch")
        if (start, span) != _timing(source["metadata"], ri):
            raise ValueError("case timing disagrees with frozen recipe")
        if case.get("source_metadata") != source["metadata"]:
            raise ValueError("case source metadata mismatch")
        if not path.is_file() or digest(path) != case.get("clip_sha256"):
            raise ValueError("control clip hash mismatch")
        actual = probe(path)
        if case.get("clip_metadata") != _metadata(actual):
            raise ValueError("control clip metadata mismatch")
        if abs(actual.duration_s - span / case["recipe"].get("speed", 1.0)) > 2 / info.fps + 0.05:
            raise ValueError("control duration exceeds source-end boundary allowance")
        audit_decode(path)
        truths[Path(clip).name] = _truth(case)
    if manifest.get("truth") != list(truths.values()):
        raise ValueError("manifest ground truth mismatch")
    expected_order = []
    seen = set()
    for source in sources:
        if source["sha256"] not in seen:
            expected_order.extend((source["sha256"], i) for i in range(len(RECIPES[source["group"]])))
            seen.add(source["sha256"])
    if [(c["source_sha256"], c["recipe_index"]) for c in cases] != expected_order:
        raise ValueError("frozen case order mismatch")
    expected = {
        (s["sha256"], i) for s in sources for i in range(len(RECIPES[s["group"]]))
    }
    if observed != expected or {c["group"] for c in cases} != set(GROUPS):
        raise ValueError("incomplete frozen control coverage")
    negatives = manifest.get("negatives")
    expected_negative_categories = {
        "unrelated_visual", "synthetic_noise", "static_color", "repeated_pattern",
        "independent_test_pattern_variant",
    }
    if (
        not isinstance(negatives, list)
        or len(negatives) != len(expected_negative_categories)
        or any(not isinstance(n, dict) for n in negatives)
        or {n.get("category") for n in negatives} != expected_negative_categories
    ):
        raise ValueError("incomplete frozen negative categories")
    negative_clips = [n.get("clip") for n in negatives]
    negative_hashes = [n.get("sha256") for n in negatives]
    case_clips = {c["clip"] for c in cases}
    case_hashes = {c["clip_sha256"] for c in cases}
    if (
        len(negative_clips) != len(negatives)
        or any(not isinstance(clip, str) for clip in negative_clips)
        or any(not _valid_sha256(sha256) for sha256 in negative_hashes)
        or len(set(negative_clips)) != len(negatives)
        or len(set(negative_hashes)) != len(negatives)
        or set(negative_clips) & case_clips
        or set(negative_hashes) & case_hashes
    ):
        raise ValueError("frozen negative clips and content must be unique")
    expected_relations = {
        category: {"kind": "independent_synthetic"}
        for category in expected_negative_categories - {"independent_test_pattern_variant"}
    }
    expected_relations["independent_test_pattern_variant"] = {
        "kind": "procedural_variant_of", "category": "unrelated_visual"
    }
    for negative in negatives:
        if negative.get("relation") != expected_relations[negative["category"]]:
            raise ValueError("frozen negative relation mismatch")
        path = _relative(root, negative.get("clip"))
        if not path.is_file() or digest(path) != negative.get("sha256"):
            raise ValueError("frozen negative hash mismatch")
        audit_decode(path)
    stored_visual = [
        (negative.get("visual_signature"), negative.get("visual_separation"))
        for negative in negatives
    ]
    _annotate_negative_separation(library, sources, root, negatives)
    if stored_visual != [
        (negative.get("visual_signature"), negative.get("visual_separation"))
        for negative in negatives
    ]:
        raise ValueError("frozen negative visual separation audit mismatch")
    frozen = json.dumps(expected_candidates, sort_keys=True, separators=(",", ":"))
    return truths, {
        "identities": expected_candidates,
        "sha256": hashlib.sha256(frozen.encode()).hexdigest(),
    }


def _stratified_shortlist_samples(manifest, sample_count):
    population = []
    for case in manifest["cases"]:
        population.append(
            {
                "clip": case["clip"],
                "group": case["group"],
                "class": "negative" if case["group"] == "source_withheld" else "positive",
                "source_id": case["source_relpath"],
                "recipe_family": case["diagnostic"],
                "truth_master": None
                if case["group"] == "source_withheld"
                else Path(case["source_relpath"]).stem,
            }
        )
    for negative in manifest["negatives"]:
        population.append(
            {
                "clip": negative["clip"],
                "group": "foreign_procedural",
                "class": "negative",
                "source_id": f"foreign:{negative['category']}",
                "recipe_family": negative["category"],
                "truth_master": None,
            }
        )
    dimensions = ("group", "class", "source_id", "recipe_family")
    counts = {dimension: Counter() for dimension in dimensions}
    remaining = list(enumerate(population))
    selected = []
    while remaining and len(selected) < sample_count:
        def score(item):
            index, sample = item
            unseen = sum(counts[dimension][sample[dimension]] == 0 for dimension in dimensions)
            balance = sum(
                1 / (1 + counts[dimension][sample[dimension]]) for dimension in dimensions
            )
            return unseen, balance, -index

        chosen = max(remaining, key=score)
        remaining.remove(chosen)
        sample = chosen[1]
        selected.append(sample)
        for dimension in dimensions:
            counts[dimension][sample[dimension]] += 1
    return selected


def _sample_composition(samples):
    names = {
        "group": "groups",
        "class": "classes",
        "source_id": "source_ids",
        "recipe_family": "recipe_families",
    }
    return {
        output: dict(sorted(Counter(sample[dimension] for sample in samples).items()))
        for dimension, output in names.items()
    }


def evaluate_controls(
    manifest_path,
    index_dir,
    out_path,
    *,
    library=None,
    benchmark_results=None,
    benchmark_truth=None,
    shortlist_samples=8,
    tolerance=1.5,
):
    """Audit and match manifest controls; emit combined benchmark and group metrics.

    Missing benchmark artifacts are represented as null, never invented numbers.
    Exhaustive checks use deterministic stratification, at most 64 clips/64 masters.
    """
    if type(shortlist_samples) is not int or not 0 <= shortlist_samples <= 64:
        raise ValueError("shortlist_samples must be in [0, 64]")
    number(tolerance, "tolerance", 0, 1e9)
    if (benchmark_results is None) != (benchmark_truth is None):
        raise ValueError("supply both benchmark results and truth")
    manifest_path = Path(manifest_path)
    manifest, manifest_sha = json_snapshot(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be an object")  # noqa: TRY004
    try:
        loaded = load_index(Path(index_dir))
    except Exception as exc:  # index validation errors are contract failures here
        raise ValueError(f"index source membership invalid: {exc}") from exc
    if shortlist_samples and len(loaded[1]) > 64:
        raise ValueError("exhaustive shortlist scoring is bounded to 64 masters")
    truths, candidate_universe = _audit_manifest(
        manifest, manifest_path.parent, Path(library or manifest["library"]), loaded[0]
    )
    results, checks, grouped, raw_grouped = [], [], {}, {}
    samples = _stratified_shortlist_samples(manifest, shortlist_samples)
    sampled_by_clip = {sample["clip"]: sample for sample in samples}
    sample_rank = {sample["clip"]: rank for rank, sample in enumerate(samples)}
    for case in manifest["cases"]:
        clip = manifest_path.parent / case["clip"]
        if case["clip"] in sampled_by_clip:
            sample = sampled_by_clip[case["clip"]]
            diag, index_manifest, audit = audit_shortlist(clip, index_dir, loaded_index=loaded)
            truth_mid = sample["truth_master"]
            audit["truth_master_in_shortlist"] = (
                truth_mid in diag.shortlist if truth_mid is not None else None
            )
            audit["sample"] = {
                key: value for key, value in sample.items() if key != "truth_master"
            }
            checks.append(audit)
        else:
            diag, index_manifest = match_clip_path(clip, index_dir, loaded_index=loaded)
        raw_result = product_result(Path(case["clip"]).name, diag, index_manifest)
        result = product_result(Path(case["clip"]).name, diag, index_manifest, schema_version=1)
        results.append(result)
        grouped.setdefault(case["group"], []).append(adapt_product(result))
        raw_grouped.setdefault(case["group"], []).append(raw_result)
    negative_rows = {"source_withheld": grouped["source_withheld"]}
    for negative in manifest["negatives"]:
        clip = manifest_path.parent / negative["clip"]
        if negative["clip"] in sampled_by_clip:
            sample = sampled_by_clip[negative["clip"]]
            diag, index_manifest, audit = audit_shortlist(clip, index_dir, loaded_index=loaded)
            audit["truth_master_in_shortlist"] = None
            audit["sample"] = {
                key: value for key, value in sample.items() if key != "truth_master"
            }
            checks.append(audit)
        else:
            diag, index_manifest = match_clip_path(clip, index_dir, loaded_index=loaded)
        row = adapt_product(product_result(Path(negative["clip"]).name, diag, index_manifest))
        negative_rows.setdefault(negative["category"], []).append(row)
    checks.sort(key=lambda row: sample_rank[row["sample"]["clip"]])
    valid = [c for c in checks if c["winner_in_shortlist"] is not None]
    genuine = [c for c in checks if c["truth_master_in_shortlist"] is not None]
    benchmark = None
    benchmark_results_sha = benchmark_truth_sha = None
    if benchmark_results is not None:
        benchmark_rows, benchmark_results_sha = load_product_results(benchmark_results, with_digest=True)
        benchmark_expected, benchmark_truth_sha = load_truth(benchmark_truth, with_digest=True)
        benchmark = score_product(
            benchmark_rows, benchmark_expected, tolerance,
            benchmark_results_sha, benchmark_truth_sha,
        )
    from clipprov.confidence import apply_calibration, derive_tiers, measure_heldout

    calibration = derive_tiers(raw_grouped.get("development", []), truths, tolerance)
    calibrated_groups = {
        group: [apply_calibration(row, calibration) for row in rows]
        for group, rows in sorted(raw_grouped.items())
        if group in {"development", "holdout"}
    }
    heldout_tiers = measure_heldout(
        calibrated_groups.get("holdout", []), truths, calibration, tolerance
    )
    report = {
        "evaluation_schema_version": 2,
        "run_receipt": {
            "generation": manifest["generation_receipt"],
            "evaluation": _evaluation_receipt(),
        },
        "candidate_universe": candidate_universe,
        "benchmark": benchmark,
        "confidence": {
            "calibration_split": "development_only",
            "calibration": calibration,
            "calibrated_results": calibrated_groups,
            "heldout_behavior": heldout_tiers,
        },
        "groups": {
            g: score_product(rows, {r["clip"]: truths[r["clip"]] for r in rows}, tolerance)
            for g, rows in sorted(grouped.items())
        },
        "negative_categories": {
            category: score_product(
                rows,
                {
                    row["clip"]: {
                        "clip": row["clip"], "master": None, "offset_sec": None,
                    }
                    for row in rows
                },
                tolerance,
            )
            for category, rows in sorted(negative_rows.items())
        },
        "shortlist": {
            "samples": len(checks),
            "valid_alignments": len(valid),
            "exhaustive_winner_recall": sum(c["winner_in_shortlist"] for c in valid) / len(valid)
            if valid
            else None,
            "genuine_truth_recall": sum(c["truth_master_in_shortlist"] for c in genuine)
            / len(genuine)
            if genuine
            else None,
            "rows": checks,
            "composition": _sample_composition(samples),
        },
        "audits": {
            "source_membership_ok": True,
            "content_hashes_ok": True,
            "decode_ok": True,
            "source_end_ok": True,
        },
        "input_digests": {
            "manifest": manifest_sha,
            "index_manifest": loaded[0].manifest_sha256,
            "benchmark_results": benchmark_results_sha,
            "benchmark_truth": benchmark_truth_sha,
        },
        "results": results,
    }
    atomic_json(out_path, report)
    return report


def run_harness(library: Path, index_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Convenience API; the supplied index must exclude frozen negative sources."""
    generate_controls(library, out_dir)
    return evaluate_controls(
        Path(out_dir) / "manifest.json", index_dir, Path(out_dir) / "report.json"
    )
