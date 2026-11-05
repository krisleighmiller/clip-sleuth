"""Review-contract regressions for evidence grid math, CLI errors, JSON determinism.

Fixtures follow the actual frame-grid rule from appearance.search_diagnostics
(FPS=8): with raw offset O and speed s, clip frame k pairs with
master_sec = O + round(s * k) / FPS. Ties round to even, like np.rint.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from clipprov.core import MatchDiagnostics
from clipprov.eval import (
    PRODUCT_FIELDS,
    adapt_product,
    alignment_timestamp_problems,
    product_result,
)
from clipprov.evidence import generate_evidence, validate_evidence
from clipprov.motion import FPS


def _video(path: Path, pattern: str) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", f"{pattern}=size=96x64:rate=8", "-t", "3", "-pix_fmt", "yuv420p",
         str(path)],
        check=True,
    )


GRID = (2, 6, 10, 14, 18)  # sample frame indices; last pair stays inside 3 s media


def _grid_pairs(raw: float, speed: float, idxs=GRID):
    return tuple((k / FPS, raw + round(speed * k) / FPS) for k in idxs)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _diag(**overrides) -> MatchDiagnostics:
    """Grid-consistent MatchDiagnostics (raw 0.25 s, speed 1.0) by default."""
    raw, speed = 0.25, 1.0
    values = {
        "best_score": 0.91, "second_score": 0.60, "margin": 0.31,
        "best_master": "master_00", "raw_alignment_offset_sec": raw,
        "source_offset_sec": max(0.0, raw), "speed": speed, "accepted": True,
        "accepted_master": "master_00", "accepted_offset_sec": max(0.0, raw),
        "accepted_speed": speed, "reason": "matched",
        "ranked": [(0.91, "master_00", raw, speed)], "shortlist": ["master_00"],
        "acceptance_floor": 0.35, "acceptance_margin": 0.12,
        "supporting_samples": 5, "sample_count": 5, "support_fraction": 1.0,
        "temporal_residual_sec": 0.0, "offset_profile_width_sec": 0.25,
        "offset_ambiguity": 0.20, "reframe_index": 0, "reframe": (1.0, 0.5, 0.5),
        "aligned_timestamps": _grid_pairs(raw, speed), "decision": "matched",
        "source_confidence": "uncalibrated", "timing_confidence": "uncalibrated",
    }
    return MatchDiagnostics(**{**values, **overrides})


def _manifest(master: Path) -> dict:
    return {"masters": [{
        "id": "master_00", "source_relpath": master.name, "display_name": master.name,
        "original_root_hint": str(master.parent), "content_digest": _digest(master),
    }]}


def _result(clip, diag, master, **kwargs):
    return product_result(str(clip), diag, _manifest(master), **kwargs)


def _row(raw, speed, pairs, clip_sec_key="clip_sec"):
    confidence = {"timing": {"aligned_timestamps": [
        {clip_sec_key: c, "master_sec": m} for c, m in pairs
    ]}}
    return {"raw_alignment_offset_sec": raw, "speed": speed, "confidence": confidence}


# ---------------------------------------------------------------- grid math

def test_helper_accepts_grid_consistent_pairs():
    for raw, speed in ((0.25, 1.0), (0.0, 1.02), (-0.125, 1.0)):
        pairs = _grid_pairs(raw, speed)
        if raw < 0:  # only pairs with non-negative masters are publishable
            pairs = tuple(p for p in pairs if p[1] >= 0)
        assert alignment_timestamp_problems(_row(raw, speed, pairs)) == []


def test_helper_rejects_offset_and_grid_contradictions():
    raw, speed = 0.25, 1.0
    bad_master = [(c, m + 0.5) for c, m in _grid_pairs(raw, speed)]
    assert any("disagrees with raw offset" in p
               for p in alignment_timestamp_problems(_row(raw, speed, bad_master)))
    assert any("off the sample grid" in p
               for p in alignment_timestamp_problems(_row(raw, speed, [(0.03, 0.25)])))


def test_helper_reports_malformed_types_before_arithmetic():
    raw, speed = 0.25, 1.0
    good = _grid_pairs(raw, speed)
    for broken_row, needle in (
        (_row(True, speed, good), "number"),
        (_row(raw, "1.0", good), "number"),
        (_row(raw, 0.0, good), "positive"),
        (_row(raw, speed, [(float("nan"), good[0][1])]), "finite"),
        (_row(raw, speed, [(good[0][0], "0.5")]), "number"),
    ):
        assert any(needle in p for p in alignment_timestamp_problems(broken_row)), needle


def test_v2_without_evidence_rejects_contradictory_timestamps(tmp_path):
    master = tmp_path / "master_00.mp4"
    master.write_bytes(b"identity only")
    raw, speed = 0.25, 1.0
    pairs = _grid_pairs(raw, speed)
    diag = _diag(aligned_timestamps=tuple((c, m + 0.5) for c, m in pairs))
    with pytest.raises(ValueError, match="aligned timestamps disagree"):
        adapt_product(_result(tmp_path / "clip.mp4", diag, master))


def test_negative_raw_offset_with_nonnegative_masters_is_valid(tmp_path):
    master = tmp_path / "master_00.mp4"
    master.write_bytes(b"identity only")
    raw, speed = -0.125, 1.0
    pairs = tuple(p for p in _grid_pairs(raw, speed) if p[1] >= 0)
    diag = _diag(raw_alignment_offset_sec=raw, source_offset_sec=0.0,
                 accepted_offset_sec=0.0, aligned_timestamps=pairs)
    adapted = adapt_product(_result(tmp_path / "clip.mp4", diag, master))
    assert adapted["candidate_offset"] == 0.0


def test_product_result_field_order_is_seed_independent():
    master = Path(__file__)  # any existing file; only its digest bytes matter
    row = _result(Path("clip.mp4"), _diag(), master, schema_version=1)
    assert list(row) == sorted(PRODUCT_FIELDS - {"schema_version", "clip", "source_record"}) + [
        "schema_version", "clip", "source_record",
    ]


# ---------------------------------------------------------------- evidence

@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_real_evidence_passes_then_rejects_mutated_offsets(tmp_path):
    clip, master = tmp_path / "clip.mp4", tmp_path / "master_00.mp4"
    _video(clip, "testsrc2")
    _video(master, "testsrc")
    result = _result(clip, _diag(), master)
    evidence = generate_evidence(
        clip, master, master_id="master_00", source_digest=_digest(master),
        aligned_timestamps=result["confidence"]["timing"]["aligned_timestamps"],
        output_dir=tmp_path / "evidence",
    )
    result = _result(clip, _diag(), master, evidence=evidence)
    assert validate_evidence(evidence, result) == []

    mutated = copy.deepcopy(result)
    for field in ("raw_alignment_offset_sec", "source_offset_sec", "accepted_offset_sec"):
        mutated[field] = 5.0
    assert any("disagrees" in p for p in validate_evidence(evidence, mutated))
    with pytest.raises(ValueError, match="aligned timestamps disagree"):
        adapt_product(mutated)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_evidence_frames_must_match_grid_rule_and_types(tmp_path):
    clip, master = tmp_path / "clip.mp4", tmp_path / "master_00.mp4"
    _video(clip, "testsrc2")
    _video(master, "testsrc")
    raw, speed = 0.0, 1.02
    diag = _diag(raw_alignment_offset_sec=raw, source_offset_sec=raw,
                 accepted_offset_sec=raw, speed=speed, accepted_speed=speed,
                 aligned_timestamps=_grid_pairs(raw, speed))
    result = _result(clip, diag, master)
    evidence = generate_evidence(
        clip, master, master_id="master_00", source_digest=_digest(master),
        aligned_timestamps=result["confidence"]["timing"]["aligned_timestamps"],
        output_dir=tmp_path / "evidence",
    )
    result = _result(clip, diag, master, evidence=evidence)
    assert validate_evidence(evidence, result) == []
    assert adapt_product(result)["candidate_offset"] == 0.0

    bad = copy.deepcopy(evidence)
    bad["frames"][0]["clip_sec"] = "not_a_number"
    assert any("invalid" in p or "absent" in p for p in validate_evidence(bad, result))
    off_grid = copy.deepcopy(result)
    off_grid["confidence"]["timing"]["aligned_timestamps"][0]["master_sec"] += 0.031
    assert any("disagrees" in p for p in validate_evidence(evidence, off_grid))


# ------------------------------------------------------------------- CLI

def _run_cli(args, env_seed=None):
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    if env_seed is not None:
        env["PYTHONHASHSEED"] = str(env_seed)
    return subprocess.run(
        [sys.executable, "-m", "clipprov.cli", *args],
        capture_output=True, text=True, env=env, timeout=300, check=False,
    )


def _assert_clean_error(proc):
    assert proc.returncode == 2, proc.stderr
    assert "Traceback" not in proc.stderr
    assert any(line.startswith("error:") for line in proc.stderr.splitlines())


def test_index_corrupt_library_exits_2_without_traceback(tmp_path):
    lib = tmp_path / "library"
    lib.mkdir()
    (lib / "corrupt.mp4").write_bytes(b"not valid media data")
    _assert_clean_error(_run_cli(["index", str(lib), "--out", str(tmp_path / "idx")]))


def test_match_dir_invalid_manifest_exits_2_without_traceback(tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "clip.mp4").write_bytes(b"fake")
    index = tmp_path / "index"
    index.mkdir()
    (index / "manifest.json").write_text("{}")
    _assert_clean_error(_run_cli(["match-dir", str(clips), "--index", str(index),
                                  "--out", str(tmp_path / "out"), "--no-evidence"]))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_match_stdout_and_artifact_bytes_stable_across_hashseeds(tmp_path):
    lib = tmp_path / "library"
    lib.mkdir()
    master = lib / "master_00.mp4"
    _video(master, "testsrc")
    clip = tmp_path / "clip.mp4"
    shutil.copy2(master, clip)
    index = tmp_path / "index"
    assert _run_cli(["index", str(lib), "--out", str(index)]).returncode == 0

    outputs = []
    for seed in (1, 2):
        artifact = tmp_path / f"result-{seed}.json"
        proc = _run_cli(["match", str(clip), "--index", str(index), "--out", str(artifact)],
                        env_seed=seed)
        assert proc.returncode == 0, proc.stderr
        outputs.append((proc.stdout, artifact.read_bytes()))
    assert outputs[0] == outputs[1], "match output varies with PYTHONHASHSEED"
    published = json.loads(outputs[0][1])
    assert list(published) == sorted(published)
