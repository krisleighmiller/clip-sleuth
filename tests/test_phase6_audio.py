"""Phase 6 optional audio corroboration contracts."""

from __future__ import annotations

import hashlib
import wave
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from clipprov.core import MatchDiagnostics
from clipprov.eval import adapt_product, product_result


def _wav(path: Path, samples: np.ndarray, sample_rate: int = 8000) -> None:
    pcm = np.clip(samples, -1, 1)
    pcm = np.rint(pcm * 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pcm_audio_corroborates_aligned_verified_snapshot_deterministically(tmp_path):
    from clipprov.audio import corroborate_audio

    rng = np.random.default_rng(20261103)
    master_samples = rng.uniform(-0.8, 0.8, 12 * 8000)
    master = tmp_path / "master.wav"
    clip = tmp_path / "clip.wav"
    _wav(master, master_samples)
    _wav(clip, master_samples[2 * 8000 : 10 * 8000])

    expected_digest = _digest(master)
    first = corroborate_audio(clip, master, expected_digest, source_offset_sec=2.0, speed=1.0)
    second = corroborate_audio(clip, master, expected_digest, source_offset_sec=2.0, speed=1.0)

    assert first == second
    assert first["status"] == "corroborated"
    assert first["score"] > 0.99
    assert first["source_sha256"] == expected_digest
    assert first["source_content_digest"] == expected_digest
    assert first["clip_sha256"] == _digest(clip)
    assert first["algorithm"] == "pcm_ncc_v1"
    assert first["compared_samples"] >= 7 * 8000


def test_replaced_audio_is_diagnostic_only_and_does_not_corroborate(tmp_path):
    from clipprov.audio import corroborate_audio

    rng = np.random.default_rng(7)
    master_samples = rng.uniform(-0.8, 0.8, 12 * 8000)
    replacement = np.random.default_rng(8).uniform(-0.8, 0.8, 8 * 8000)
    master = tmp_path / "master.wav"
    clip = tmp_path / "clip.wav"
    _wav(master, master_samples)
    _wav(clip, replacement)

    result = corroborate_audio(clip, master, _digest(master), source_offset_sec=2.0, speed=1.0)

    assert result["status"] == "not_corroborated"
    assert result["score"] < result["threshold"]


def test_missing_silent_insufficient_and_unverified_audio_are_unavailable(tmp_path):
    from clipprov.audio import corroborate_audio

    rng = np.random.default_rng(9)
    master = tmp_path / "master.wav"
    good_clip = tmp_path / "good.wav"
    silent = tmp_path / "silent.wav"
    short = tmp_path / "short.wav"
    samples = rng.uniform(-0.8, 0.8, 12 * 8000)
    _wav(master, samples)
    _wav(good_clip, samples[2 * 8000 : 10 * 8000])
    _wav(silent, np.zeros(8 * 8000))
    _wav(short, samples[: 8000 // 2])
    source_digest = _digest(master)

    cases = [
        corroborate_audio(tmp_path / "missing.wav", master, source_digest, 2.0, 1.0),
        corroborate_audio(silent, master, source_digest, 2.0, 1.0),
        corroborate_audio(short, master, source_digest, 2.0, 1.0),
        corroborate_audio(good_clip, master, "0" * 64, 2.0, 1.0),
        corroborate_audio(good_clip, master, source_digest, 2.0, 1.1),
    ]

    assert [case["status"] for case in cases] == ["unavailable"] * len(cases)
    assert [case["reason"] for case in cases] == [
        "clip_snapshot_unavailable",
        "silent_audio",
        "insufficient_audio",
        "source_digest_mismatch",
        "unsupported_speed",
    ]
    assert all(case["score"] is None for case in cases)


def test_nonfinite_decoded_pcm_is_unavailable_and_never_breaks_visual_result(tmp_path, monkeypatch):
    from clipprov import audio

    clip = tmp_path / "clip.bin"
    source = tmp_path / "source.bin"
    clip.write_bytes(b"clip")
    source.write_bytes(b"source")
    samples = np.ones(2 * audio.AUDIO_SAMPLE_RATE, dtype=np.float64)
    samples[0] = np.nan
    monkeypatch.setattr(audio, "_decode_pcm", lambda *_args: samples.copy())

    receipt = audio.corroborate_audio(
        clip, source, _digest(source), source_offset_sec=0.0, speed=1.0
    )

    assert receipt["status"] == "unavailable"
    assert receipt["reason"] == "nonfinite_audio"
    assert receipt["score"] is None


def test_audio_receipt_is_optional_and_cannot_change_visual_decision():
    diag = MatchDiagnostics(
        best_score=0.9,
        second_score=0.1,
        margin=0.8,
        best_master="master_00",
        raw_alignment_offset_sec=2.0,
        source_offset_sec=2.0,
        speed=1.0,
        accepted=True,
        accepted_master="master_00",
        accepted_offset_sec=2.0,
        accepted_speed=1.0,
        reason="matched",
        decision="matched",
        acceptance_floor=0.35,
        acceptance_margin=0.12,
    )
    diag.audio = {
        "status": "not_corroborated",
        "reason": "score_below_threshold",
        "score": 0.01,
        "threshold": 0.45,
        "algorithm": "pcm_ncc_v1",
        "sample_rate_hz": 16000,
        "compared_samples": 128000,
        "lag_samples": 0,
        "clip_sha256": "1" * 64,
        "source_sha256": "2" * 64,
        "source_content_digest": "2" * 64,
        "source_offset_sec": 2.0,
    }
    manifest = {
        "masters": [{
            "id": "master_00",
            "source_relpath": "master_00.mp4",
            "display_name": "master_00.mp4",
            "original_root_hint": "/library",
            "content_digest": "2" * 64,
        }]
    }

    v1 = product_result("clip.mp4", diag, manifest, schema_version=1)
    v2 = product_result("clip.mp4", diag, manifest)

    assert "confidence" not in v1
    assert v1["accepted"] is True
    assert v2["accepted"] is True
    assert v2["reason"] == "matched"
    assert "audio" not in v2["confidence"]

    v3 = product_result(
        "clip.mp4",
        diag,
        manifest,
        matching_receipt={
            "schema_version": 2,
            "speed_profile": {
                "schema_version": 1, "name": "v1-default", "speed_min": 0.968,
                "speed_max": 1.032, "speed_steps": 33, "measured_range": None,
            },
            "geometry_mode": "identity",
            "audio_enabled": True,
            "input_receipt": {
                "clip_sha256": "1" * 64,
                "index_manifest_sha256": "3" * 64,
            },
        },
    )
    assert v3["schema_version"] == 3
    assert v3["confidence"]["audio"]["status"] == "not_corroborated"
    assert adapt_product(v3)["tier"] == "accepted"
    assert adapt_product(v2)["tier"] == "accepted"

    for field, replacement in (
        ("source_content_digest", "3" * 64),
        ("clip_sha256", "not-a-digest"),
        ("score", float("nan")),
        ("status", "corroborated"),
    ):
        tampered = deepcopy(v3)
        tampered["confidence"]["audio"][field] = replacement
        with pytest.raises(ValueError):
            adapt_product(tampered)


def test_matching_attachment_uses_index_source_identity_without_penalizing(tmp_path):
    from clipprov.match import attach_audio_corroboration

    rng = np.random.default_rng(10)
    master_samples = rng.uniform(-0.8, 0.8, 12 * 8000)
    master = tmp_path / "master.wav"
    clip = tmp_path / "replaced.wav"
    _wav(master, master_samples)
    _wav(clip, np.random.default_rng(11).uniform(-0.8, 0.8, 8 * 8000))
    diag = MatchDiagnostics(
        best_score=0.9,
        second_score=0.1,
        margin=0.8,
        best_master="master_00",
        raw_alignment_offset_sec=2.0,
        source_offset_sec=2.0,
        speed=1.0,
        accepted=True,
        accepted_master="master_00",
        accepted_offset_sec=2.0,
        accepted_speed=1.0,
        reason="matched",
        decision="matched",
    )
    manifest = {
        "masters": [{
            "id": "master_00",
            "source_relpath": master.name,
            "original_root_hint": str(tmp_path),
            "content_digest": _digest(master),
        }]
    }

    returned = attach_audio_corroboration(diag, clip, manifest)

    assert returned is diag
    assert diag.audio["status"] == "not_corroborated"
    assert diag.accepted is True
    assert diag.decision == "matched"
    assert diag.accepted_master == "master_00"


def test_combined_speed_geometry_keeps_unsupported_audio_positive_only(tmp_path):
    from clipprov.match import attach_audio_corroboration

    master = tmp_path / "master.wav"
    master.write_bytes(b"source")
    diag = MatchDiagnostics(
        best_score=0.9,
        second_score=0.1,
        margin=0.8,
        best_master="master_00",
        raw_alignment_offset_sec=2.0,
        source_offset_sec=2.0,
        speed=0.75,
        accepted=True,
        accepted_master="master_00",
        accepted_offset_sec=2.0,
        accepted_speed=0.75,
        reason="matched",
        decision="matched",
        geometry_mode="d4",
        geometry_hypothesis="rotate-90-ccw",
    )
    manifest = {
        "masters": [{
            "id": "master_00",
            "source_relpath": master.name,
            "original_root_hint": str(tmp_path),
            "content_digest": _digest(master),
        }]
    }

    returned = attach_audio_corroboration(diag, tmp_path / "clip.wav", manifest)

    assert returned is diag
    assert diag.audio["status"] == "unavailable"
    assert diag.audio["reason"] == "unsupported_speed"
    assert diag.audio["score"] is None
    assert diag.accepted is True
    assert diag.decision == "matched"
    assert diag.geometry_mode == "d4"
    assert diag.geometry_hypothesis == "rotate-90-ccw"
