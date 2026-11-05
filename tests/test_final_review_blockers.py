from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from clipprov.audio import AUDIO_SAMPLE_RATE, corroborate_audio
from clipprov.batch import BatchError, match_directory
from clipprov.config import Config, MatchConfig, MediaConfig
from clipprov.core import MatchDiagnostics
from clipprov.eval import adapt_product, product_result
from clipprov.index import _ManifestSnapshot


def _diag(*, geometry_mode="identity", geometry_hypothesis="identity") -> MatchDiagnostics:
    return MatchDiagnostics(
        best_score=0.9,
        second_score=0.2,
        margin=0.7,
        best_master="master_00",
        raw_alignment_offset_sec=0.0,
        source_offset_sec=0.0,
        speed=1.0,
        accepted=True,
        accepted_master="master_00",
        accepted_offset_sec=0.0,
        accepted_speed=1.0,
        reason="matched",
        ranked=[(0.9, "master_00", 0.0, 1.0)],
        shortlist=["master_00"],
        acceptance_floor=0.35,
        acceptance_margin=0.12,
        supporting_samples=6,
        sample_count=6,
        support_fraction=1.0,
        temporal_residual_sec=0.0,
        offset_profile_width_sec=0.0,
        offset_ambiguity=1.0,
        reframe_index=0,
        reframe=(1.0, 0.5, 0.5),
        aligned_timestamps=((0.0, 0.0),),
        decision="matched",
        source_confidence="uncalibrated",
        timing_confidence="uncalibrated",
        geometry_mode=geometry_mode,
        geometry_hypothesis=geometry_hypothesis,
    )


def _manifest(master: Path) -> _ManifestSnapshot:
    raw = b"manifest snapshot"
    return _ManifestSnapshot(
        {
            "masters": [
                {
                    "id": "master_00",
                    "source_relpath": master.name,
                    "display_name": master.name,
                    "original_root_hint": str(master.parent),
                    "content_digest": hashlib.sha256(master.read_bytes()).hexdigest(),
                }
            ]
        },
        hashlib.sha256(raw).hexdigest(),
    )


def test_default_product_v2_keeps_exact_phase5_confidence_shape(tmp_path):
    master = tmp_path / "master_00.mp4"
    master.write_bytes(b"master")
    diag = _diag(geometry_mode="d4", geometry_hypothesis="rotate-90-ccw")
    diag.audio = {"unexpected": "phase6 diagnostic"}

    result = product_result("clip.mp4", diag, _manifest(master))

    assert result["schema_version"] == 2
    assert set(result["confidence"]) == {
        "semantics",
        "calibration",
        "decision",
        "source",
        "timing",
    }
    assert set(result["confidence"]["timing"]) == {
        "tier",
        "supporting_samples",
        "sample_count",
        "support_fraction",
        "temporal_residual_sec",
        "offset_profile_width_sec",
        "offset_ambiguity",
        "selected_reframe",
        "speed_ratio",
        "aligned_timestamps",
    }
    assert adapt_product(result)["clip"] == "clip.mp4"


def test_phase6_diagnostics_require_v3_policy_and_input_receipt(tmp_path):
    master = tmp_path / "master_00.mp4"
    master.write_bytes(b"master")
    diag = _diag(geometry_mode="d4", geometry_hypothesis="rotate-90-ccw")
    receipt = {
        "schema_version": 2,
        "speed_profile": {
            "schema_version": 1,
            "name": "v1-default",
            "speed_min": 0.968,
            "speed_max": 1.032,
            "speed_steps": 33,
            "measured_range": None,
        },
        "geometry_mode": "d4",
        "audio_enabled": False,
        "input_receipt": {
            "clip_sha256": "a" * 64,
            "index_manifest_sha256": "b" * 64,
        },
    }

    result = product_result("clip.mp4", diag, _manifest(master), matching_receipt=receipt)

    assert result["schema_version"] == 3
    assert result["confidence"]["timing"]["selected_geometry_hypothesis"] == "rotate-90-ccw"
    assert "audio" not in result["confidence"]
    assert adapt_product(result)["clip"] == "clip.mp4"


def _batch_fixture(tmp_path: Path, monkeypatch):
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "clip.mp4").write_bytes(b"clip")
    index = tmp_path / "index"
    index.mkdir()
    master = tmp_path / "master_00.mp4"
    master.write_bytes(b"master")
    manifest = _manifest(master)
    monkeypatch.setattr("clipprov.batch.load_index", lambda _path: (manifest, {}))

    def fake_match(_path, _index, config=None, **_kwargs):
        mode = config.match.geometry_mode if config is not None else "identity"
        return _diag(geometry_mode=mode), manifest

    monkeypatch.setattr("clipprov.batch.match_clip_path", fake_match)
    return clips, index


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (
            Config(match=MatchConfig(score_floor=0.35)),
            Config(match=MatchConfig(score_floor=0.95)),
        ),
        (
            Config(media=MediaConfig(max_frames=20)),
            Config(media=MediaConfig(max_frames=21)),
        ),
    ],
    ids=["score-floor", "media-config"],
)
def test_batch_resume_identity_binds_effective_config(tmp_path, monkeypatch, first, second):
    clips, index = _batch_fixture(tmp_path, monkeypatch)
    out = tmp_path / "out"
    match_directory(clips, index, out, config=first, evidence=False)

    with pytest.raises(BatchError, match="does not match"):
        match_directory(clips, index, out, config=second, evidence=False)


@pytest.mark.parametrize(
    ("target", "replacement"),
    [
        ("clip_sha256", "c" * 64),
        ("index_manifest_sha256", "d" * 64),
        ("profile", "wider-speed-v1"),
        ("geometry_mode", "identity"),
        ("audio_enabled", False),
    ],
)
def test_batch_rejects_tampered_v3_receipt(tmp_path, monkeypatch, target, replacement):
    clips, index = _batch_fixture(tmp_path, monkeypatch)
    out = tmp_path / "out"
    config = Config(match=MatchConfig(geometry_mode="d4", audio_enabled=True))
    match_directory(clips, index, out, config=config, evidence=False)
    journal_path = out / ".batch-progress.json"
    journal = json.loads(journal_path.read_text())
    receipt = journal["completed"]["clip.mp4"]["matching_receipt"]
    if target in {"clip_sha256", "index_manifest_sha256"}:
        receipt["input_receipt"][target] = replacement
    elif target == "profile":
        receipt["speed_profile"]["name"] = replacement
        receipt["speed_profile"]["measured_range"] = {"minimum": 0.5, "maximum": 2.0}
        receipt["speed_profile"]["speed_min"] = 0.5
        receipt["speed_profile"]["speed_max"] = 2.0
        receipt["speed_profile"]["speed_steps"] = 401
    else:
        receipt[target] = replacement
    journal_path.write_text(json.dumps(journal))

    with pytest.raises(BatchError, match="receipt"):
        match_directory(clips, index, out, config=config, evidence=False)


@pytest.mark.parametrize("downgrade", ["schema-v2-result", "schema-v1-receipt"])
def test_phase6_resume_rejects_missing_current_policy_receipt(
    tmp_path, monkeypatch, downgrade
):
    clips, index = _batch_fixture(tmp_path, monkeypatch)
    out = tmp_path / "out"
    config = Config(match=MatchConfig(geometry_mode="d4", audio_enabled=True))
    match_directory(clips, index, out, config=config, evidence=False)
    journal_path = out / ".batch-progress.json"
    journal = json.loads(journal_path.read_text())
    result = journal["completed"]["clip.mp4"]

    if downgrade == "schema-v2-result":
        result["schema_version"] = 2
        result.pop("matching_receipt")
        result["confidence"].pop("audio", None)
        result["confidence"]["timing"].pop("selected_geometry_hypothesis", None)
    else:
        receipt = result["matching_receipt"]
        receipt["schema_version"] = 1
        receipt.pop("geometry_mode")
        receipt.pop("audio_enabled")
    journal_path.write_text(json.dumps(journal))

    with pytest.raises(BatchError, match="policy receipt"):
        match_directory(clips, index, out, config=config, evidence=False)


def test_audio_overflow_correlation_is_unavailable_without_score(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    source = tmp_path / "source.mp4"
    clip.write_bytes(b"clip")
    source.write_bytes(b"source")
    samples = np.full(2 * AUDIO_SAMPLE_RATE, 1e308, dtype=np.float64)
    samples[::2] *= -1
    monkeypatch.setattr("clipprov.audio._decode_pcm", lambda *_args: samples.copy())

    result = corroborate_audio(
        clip,
        source,
        hashlib.sha256(source.read_bytes()).hexdigest(),
        0.0,
        1.0,
    )

    assert result["status"] == "unavailable"
    assert result["reason"] == "nonfinite_audio"
    assert result["score"] is None
    json.dumps(result, allow_nan=False)
