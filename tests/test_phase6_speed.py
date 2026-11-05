from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from clipprov.batch import BatchError, match_directory
from clipprov.cli import main
from clipprov.config import Config, MatchConfig, resolve_speed_profile
from clipprov.core import MatchDiagnostics
from clipprov.eval import adapt_product, product_result
from clipprov.index import _ManifestSnapshot, extractor_config


def _diag(name="master"):
    return MatchDiagnostics(
        best_score=0.9, second_score=0.4, margin=0.5, best_master=name,
        raw_alignment_offset_sec=0.0, source_offset_sec=0.0, speed=0.5,
        accepted=True, accepted_master=name, accepted_offset_sec=0.0,
        accepted_speed=0.5, reason="matched",
        ranked=[(0.9, name, 0.0, 0.5)], shortlist=[name],
        acceptance_floor=0.35, acceptance_margin=0.12,
        supporting_samples=3, sample_count=3, support_fraction=1.0,
        temporal_residual_sec=0.0, offset_profile_width_sec=0.0,
        offset_ambiguity=0.5, reframe_index=0, reframe=(1.0, 0.5, 0.5),
        aligned_timestamps=((0.0, 0.0), (0.5, 0.25), (1.0, 0.5)),
        decision="matched", source_confidence="uncalibrated",
        timing_confidence="uncalibrated",
    )


def _manifest(master: Path):
    base = Config()
    return {
        "extractor_config": extractor_config(base.media, base.match),
        "masters": [{
            "id": "master", "source_relpath": master.name,
            "display_name": master.name, "original_root_hint": str(master.parent),
            "content_digest": hashlib.sha256(master.read_bytes()).hexdigest(),
        }],
    }


def _snapshot(value: dict, path: Path):
    return _ManifestSnapshot(value, hashlib.sha256(path.read_bytes()).hexdigest())


def test_wider_speed_profile_is_explicit_and_v1_defaults_are_unchanged():
    defaults = MatchConfig()
    assert (defaults.speed_min, defaults.speed_max, defaults.speed_steps) == (0.968, 1.032, 33)

    config, receipt = resolve_speed_profile("wider-speed-v1", Config())

    assert (config.match.speed_min, config.match.speed_max, config.match.speed_steps) == (0.5, 2.0, 401)
    assert receipt == {
        "schema_version": 1,
        "name": "wider-speed-v1",
        "speed_min": 0.5,
        "speed_max": 2.0,
        "speed_steps": 401,
        "measured_range": {"minimum": 0.5, "maximum": 2.0},
    }
    assert resolve_speed_profile(None, Config())[1]["name"] == "v1-default"


def test_batch_binds_wider_profile_to_receipts_and_resume_identity(tmp_path, monkeypatch):
    master = tmp_path / "master.mp4"
    master.write_bytes(b"master")
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "clip.mp4").write_bytes(b"clip")
    index = tmp_path / "index"
    index.mkdir()
    manifest_path = index / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest(master)), encoding="utf-8")

    seen = []

    def fake_match(path, index_dir, config, *, loaded_index):
        seen.append((config.match.speed_min, config.match.speed_max, config.match.speed_steps))
        return _diag(), loaded_index[0]

    monkeypatch.setattr("clipprov.batch.match_clip_path", fake_match)
    monkeypatch.setattr(
        "clipprov.batch.load_index",
        lambda _path: (_snapshot(_manifest(master), manifest_path), [object()]),
    )

    output = tmp_path / "out"
    report = match_directory(
        clips, index, output, speed_profile="wider-speed-v1", evidence=False
    )
    receipt = report["run_receipt"]

    assert seen == [(0.5, 2.0, 401)]
    assert receipt["speed_profile"]["name"] == "wider-speed-v1"
    assert receipt["speed_profile"]["schema_version"] == 1
    assert receipt["input_receipt"]["inventory_digest"] == hashlib.sha256(
        json.dumps(report["inventory"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert receipt["input_receipt"]["index_manifest_digest"] == receipt["index_manifest_sha256"]
    result = report["results"][0]["result"]
    assert result["schema_version"] == 3
    assert result["matching_receipt"]["speed_profile"] == receipt["speed_profile"]
    assert result["matching_receipt"]["input_receipt"]["clip_sha256"] == report["inventory"][0]["sha256"]

    with pytest.raises(BatchError, match="resume"):
        match_directory(clips, index, output)


def test_combined_speed_and_geometry_survive_batch_receipt_and_resume(tmp_path, monkeypatch):
    master = tmp_path / "master.mp4"
    master.write_bytes(b"master")
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "clip.mp4").write_bytes(b"clip")
    index = tmp_path / "index"
    index.mkdir()
    manifest_path = index / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest(master)), encoding="utf-8")
    seen = []

    def fake_match(path, index_dir, config, *, loaded_index):
        seen.append(
            (
                config.match.speed_min,
                config.match.speed_max,
                config.match.speed_steps,
                config.match.geometry_mode,
            )
        )
        return _diag(), loaded_index[0]

    monkeypatch.setattr("clipprov.batch.match_clip_path", fake_match)
    monkeypatch.setattr(
        "clipprov.batch.load_index",
        lambda _path: (_snapshot(_manifest(master), manifest_path), [object()]),
    )
    output = tmp_path / "out"

    first = match_directory(
        clips,
        index,
        output,
        speed_profile="wider-speed-v1",
        geometry_mode="d4",
        evidence=False,
    )
    resumed = match_directory(
        clips,
        index,
        output,
        speed_profile="wider-speed-v1",
        geometry_mode="d4",
        evidence=False,
    )

    assert seen == [(0.5, 2.0, 401, "d4")]
    assert first["run_receipt"]["options"]["geometry_mode"] == "d4"
    assert first["run_receipt"]["speed_profile"]["name"] == "wider-speed-v1"
    assert resumed["run_receipt"]["resumed_count"] == 1
    journal = json.loads((output / ".batch-progress.json").read_text())
    assert journal["identity"]["options"]["geometry_mode"] == "d4"
    assert journal["identity"]["options"]["speed_profile"]["name"] == "wider-speed-v1"


def test_cli_forwards_combined_speed_and_geometry_to_batch(tmp_path, monkeypatch):
    clips = tmp_path / "clips"
    index = tmp_path / "index"
    out = tmp_path / "out"
    clips.mkdir()
    index.mkdir()
    captured = {}

    def fake_match_directory(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return {"results": [], "failures": []}

    monkeypatch.setattr("clipprov.cli.match_directory", fake_match_directory)
    monkeypatch.setattr("clipprov.cli.current_report_dir", lambda _out: out / "report")

    result = CliRunner().invoke(
        main,
        [
            "match-dir",
            str(clips),
            "--index",
            str(index),
            "--out",
            str(out),
            "--speed-profile",
            "wider-speed-v1",
            "--geometry",
            "d4",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["kwargs"]["speed_profile"] == "wider-speed-v1"
    assert captured["kwargs"]["geometry_mode"] == "d4"


def test_product_v3_binds_profile_and_input_receipts(tmp_path):
    master = tmp_path / "master.mp4"
    master.write_bytes(b"master")
    matching_receipt = {
        "schema_version": 1,
        "speed_profile": resolve_speed_profile("wider-speed-v1", Config())[1],
        "input_receipt": {
            "clip_sha256": "a" * 64,
            "index_manifest_sha256": "b" * 64,
        },
    }

    result = product_result(
        "clip.mp4", _diag(), _manifest(master), matching_receipt=matching_receipt
    )

    assert result["schema_version"] == 3
    assert result["matching_receipt"] == matching_receipt
    assert adapt_product(result)["native_tier"] == "matched"
    tampered = json.loads(json.dumps(result))
    tampered["matching_receipt"]["speed_profile"]["speed_steps"] = 400
    with pytest.raises(ValueError, match="speed profile"):
        adapt_product(tampered)

    audio_tampered = json.loads(json.dumps(result))
    audio_tampered["confidence"]["audio"] = {
        "status": "corroborated",
        "reason": "score_at_or_above_threshold",
        "score": 0.9,
        "threshold": 0.45,
        "algorithm": "pcm_ncc_v1",
        "sample_rate_hz": 16000,
        "compared_samples": 32000,
        "lag_samples": 0,
        "clip_sha256": "c" * 64,
        "source_sha256": _manifest(master)["masters"][0]["content_digest"],
        "source_content_digest": _manifest(master)["masters"][0]["content_digest"],
        "source_offset_sec": 0.0,
    }
    with pytest.raises(ValueError, match="audio clip digest"):
        adapt_product(audio_tampered)

    speed_tampered = json.loads(json.dumps(result))
    speed_tampered["speed"] = 5.0
    speed_tampered["accepted_speed"] = 5.0
    speed_tampered["confidence"]["timing"]["speed_ratio"] = 5.0
    for pair in speed_tampered["confidence"]["timing"]["aligned_timestamps"]:
        pair["master_sec"] = pair["clip_sec"] * 5.0
    with pytest.raises(ValueError, match="speed profile bounds"):
        adapt_product(speed_tampered)


@pytest.mark.parametrize("command", ["match", "match-dir"])
def test_cli_exposes_explicit_wider_speed_profile(command):
    result = CliRunner().invoke(main, [command, "--help"])

    assert result.exit_code == 0
    assert "--speed-profile [wider-speed-v1]" in result.output
