from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from clipprov.appearance import geometry_hypotheses
from clipprov.batch import BatchError, match_directory
from clipprov.config import Config, MatchConfig
from clipprov.core import MatchDiagnostics, match_clip
from clipprov.eval import adapt_product, product_result
from clipprov.index import _ManifestSnapshot
from clipprov.motion import GRID


def _normalized_grids(seed: int, frames: int = 6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    grids = rng.standard_normal((frames, GRID, GRID)).astype(np.float32)
    grids -= grids.mean(axis=(1, 2), keepdims=True)
    grids /= np.sqrt((grids * grids).sum(axis=(1, 2), keepdims=True))
    return grids


def _diag(*, mode="identity", hypothesis="identity") -> MatchDiagnostics:
    return MatchDiagnostics(
        best_score=0.9, second_score=0.2, margin=0.7, best_master="master_00",
        raw_alignment_offset_sec=0.0, source_offset_sec=0.0, speed=1.0,
        accepted=True, accepted_master="master_00", accepted_offset_sec=0.0,
        accepted_speed=1.0, reason="matched", ranked=[(0.9, "master_00", 0.0, 1.0)],
        shortlist=["master_00"], acceptance_floor=0.35, acceptance_margin=0.12,
        supporting_samples=6, sample_count=6, support_fraction=1.0,
        temporal_residual_sec=0.0, offset_profile_width_sec=0.0,
        offset_ambiguity=1.0, reframe_index=0, reframe=(1.0, 0.5, 0.5),
        aligned_timestamps=((0.0, 0.0),), decision="matched",
        source_confidence="uncalibrated", timing_confidence="uncalibrated",
        geometry_mode=mode, geometry_hypothesis=hypothesis,
    )


def _manifest(master: Path) -> dict:
    return {"masters": [{
        "id": "master_00", "source_relpath": master.name, "display_name": master.name,
        "original_root_hint": str(master.parent),
        "content_digest": hashlib.sha256(master.read_bytes()).hexdigest(),
    }]}


def test_geometry_mode_is_strictly_opt_in_and_validated():
    assert MatchConfig().geometry_mode == "identity"
    assert MatchConfig(geometry_mode="d4").geometry_mode == "d4"
    with pytest.raises(ValueError):
        MatchConfig.model_validate({"geometry_mode": "perspective"})


def test_d4_hypotheses_are_the_eight_unique_square_symmetries():
    grids = np.arange(2 * GRID * GRID, dtype=np.float32).reshape(2, GRID * GRID)
    hypotheses = geometry_hypotheses(grids, "d4")
    assert [name for name, _ in hypotheses] == [
        "identity", "rotate-90-ccw", "rotate-180", "rotate-270-ccw",
        "mirror-horizontal", "mirror-horizontal-rotate-90-ccw",
        "mirror-horizontal-rotate-180", "mirror-horizontal-rotate-270-ccw",
    ]
    arrays = [value.reshape(2, GRID, GRID) for _, value in hypotheses]
    assert np.array_equal(arrays[1], np.rot90(grids.reshape(2, GRID, GRID), 1, axes=(1, 2)))
    assert np.array_equal(arrays[4], np.flip(grids.reshape(2, GRID, GRID), axis=2))
    assert len({array.tobytes() for array in arrays}) == 8
    assert [name for name, _ in geometry_hypotheses(grids, "identity")] == ["identity"]


def test_d4_matching_restores_quarter_turn_and_reports_winning_hypothesis():
    master = _normalized_grids(11)
    clip = np.rot90(master, -1, axes=(1, 2)).reshape(6, -1)
    distractor = _normalized_grids(12).reshape(6, -1)
    idx = np.arange(6)
    mm = {"correct": np.zeros(6), "distractor": np.zeros(6)}
    views = {"correct": [master.reshape(6, -1)], "distractor": [distractor]}

    baseline = match_clip(clip, idx, np.zeros(6), mm, views, config=Config())
    enabled = match_clip(
        clip, idx, np.zeros(6), mm, views,
        config=Config(match=MatchConfig(geometry_mode="d4")),
    )

    assert not baseline.accepted
    assert enabled.accepted and enabled.best_master == "correct"
    assert enabled.geometry_mode == "d4"
    assert enabled.geometry_hypothesis == "rotate-90-ccw"
    assert enabled.best_score == pytest.approx(1.0, abs=1e-6)
    assert len(enabled.ranked) == 2  # margin is across sources, not hypotheses


def test_opt_in_geometry_diagnostic_is_schema_bound_but_legacy_v2_stays_readable(tmp_path):
    master = tmp_path / "master_00.mp4"
    master.write_bytes(b"master")
    manifest = _manifest(master)

    legacy = product_result("clip.mp4", _diag(), manifest)
    assert "selected_geometry_hypothesis" not in legacy["confidence"]["timing"]
    assert adapt_product(legacy)["clip"] == "clip.mp4"

    enabled = product_result(
        "clip.mp4",
        _diag(mode="d4", hypothesis="mirror-horizontal"),
        manifest,
        matching_receipt={
            "schema_version": 2,
            "speed_profile": {
                "schema_version": 1, "name": "v1-default", "speed_min": 0.968,
                "speed_max": 1.032, "speed_steps": 33, "measured_range": None,
            },
            "geometry_mode": "d4",
            "audio_enabled": False,
            "input_receipt": {
                "clip_sha256": "1" * 64,
                "index_manifest_sha256": "2" * 64,
            },
        },
    )
    assert enabled["schema_version"] == 3
    assert enabled["confidence"]["timing"]["selected_geometry_hypothesis"] == (
        "mirror-horizontal"
    )
    assert adapt_product(enabled)["clip"] == "clip.mp4"
    enabled["confidence"]["timing"]["selected_geometry_hypothesis"] = "perspective"
    with pytest.raises(ValueError, match="geometry hypothesis"):
        adapt_product(enabled)


def test_batch_receipt_binds_geometry_mode_for_resume(tmp_path, monkeypatch):
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "clip.mp4").write_bytes(b"clip")
    index = tmp_path / "index"
    index.mkdir()
    manifest_path = index / "manifest.json"
    manifest_path.write_text("{}")
    master = tmp_path / "master.mp4"
    master.write_bytes(b"master")
    manifest = _ManifestSnapshot(
        _manifest(master), hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    )
    monkeypatch.setattr("clipprov.batch.load_index", lambda _path: (manifest, {}))
    seen = []

    def fake_match(_path, _index, config=None, **_kwargs):
        seen.append(config.match.geometry_mode)
        return _diag(mode=config.match.geometry_mode), manifest

    monkeypatch.setattr("clipprov.batch.match_clip_path", fake_match)
    out = tmp_path / "out"
    config = Config(match=MatchConfig(geometry_mode="d4"))
    report = match_directory(clips, index, out, config=config, evidence=False)
    assert seen == ["d4"]
    assert report["run_receipt"]["options"]["geometry_mode"] == "d4"
    journal = json.loads((out / ".batch-progress.json").read_text())
    assert journal["identity"]["options"]["geometry_mode"] == "d4"

    with pytest.raises(BatchError, match="does not match"):
        match_directory(clips, index, out, config=Config(), evidence=False)
