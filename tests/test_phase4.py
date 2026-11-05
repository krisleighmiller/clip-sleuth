from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from click.testing import CliRunner

from clipprov.appearance import search, search_diagnostics
from clipprov.cli import main
from clipprov.confidence import apply_calibration, derive_tiers, measure_heldout
from clipprov.core import MatchDiagnostics, confidence_labels
from clipprov.eval import adapt_product, product_result
from clipprov.evidence import generate_evidence, validate_evidence


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _diag():
    return MatchDiagnostics(
        best_score=0.91,
        second_score=0.60,
        margin=0.31,
        best_master="master_00",
        raw_alignment_offset_sec=0.5,
        source_offset_sec=0.5,
        speed=1.0,
        accepted=True,
        accepted_master="master_00",
        accepted_offset_sec=0.5,
        accepted_speed=1.0,
        reason="matched",
        ranked=[(0.91, "master_00", 0.5, 1.0), (0.60, "master_01", 0.5, 1.0)],
        shortlist=["master_00", "master_01"],
        acceptance_floor=0.35,
        acceptance_margin=0.12,
        supporting_samples=3,
        sample_count=3,
        support_fraction=1.0,
        temporal_residual_sec=0.0,
        offset_profile_width_sec=0.25,
        offset_ambiguity=0.20,
        reframe_index=0,
        reframe=(1.0, 0.5, 0.5),
        # Frame-grid rule (appearance.search_diagnostics, FPS=8): with raw
        # offset 0.5 and speed 1.0, master_sec = 0.5 + round(1.0 * k) / 8 for
        # clip frame k, so clip 0.5/1.5/2.25 map to master 1.0/2.0/2.75.
        aligned_timestamps=((0.5, 1.0), (1.5, 2.0), (2.25, 2.75)),
        decision="matched",
        source_confidence="uncalibrated",
        timing_confidence="uncalibrated",
    )


def _manifest(master: Path):
    return {
        "masters": [{
            "id": "master_00",
            "source_relpath": master.name,
            "display_name": master.name,
            "original_root_hint": str(master.parent),
            "content_digest": _digest(master),
        }]
    }


def test_search_diagnostics_preserves_tuple_contract_and_exposes_alignment():
    rng = np.random.default_rng(4)
    master = rng.normal(size=(20, 16)).astype(np.float32)
    master /= np.linalg.norm(master, axis=1, keepdims=True)
    idx = np.arange(5)
    clip = master[7:12]
    detail = search_diagnostics(clip, idx, [master], speeds=[1.0])
    assert search(clip, idx, [master], speeds=[1.0]) == (
        detail.score, detail.offset_sec, detail.speed
    )
    assert detail.offset_sec == pytest.approx(7 / 8)
    assert detail.supporting_samples == detail.sample_count == 5
    assert detail.support_fraction == 1.0
    assert detail.reframe == (1.0, 0.5, 0.5)
    assert len(detail.aligned_timestamps) == 5


def test_product_v2_separates_source_and_timing_confidence(tmp_path):
    master = tmp_path / "master_00.mp4"
    master.write_bytes(b"identity only")
    row = product_result("clip.mp4", _diag(), _manifest(master))
    assert row["schema_version"] == 2
    assert row["confidence"]["semantics"] == "diagnostic_not_probability"
    assert row["confidence"]["source"]["competing_master"] == "master_01"
    assert row["confidence"]["timing"]["speed_ratio"] == 1.0
    assert adapt_product(row)["tier"] == "accepted"
    assert adapt_product(row)["native_tier"] == "matched"
    row["confidence"]["timing"]["support_fraction"] = 2.0
    with pytest.raises(ValueError, match="support_fraction"):
        adapt_product(row)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_real_evidence_frames_validate_identity_boundaries_and_consistency(tmp_path):
    clip = tmp_path / "clip.mp4"
    master = tmp_path / "master_00.mp4"
    for path, pattern in ((clip, "testsrc2"), (master, "testsrc")):
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"{pattern}=size=96x64:rate=8",
             "-t", "3", "-pix_fmt", "yuv420p", str(path)],
            check=True,
        )
    manifest = _manifest(master)
    result = product_result(str(clip), _diag(), manifest)
    evidence = generate_evidence(
        clip,
        master,
        master_id="master_00",
        source_digest=_digest(master),
        aligned_timestamps=result["confidence"]["timing"]["aligned_timestamps"],
        output_dir=tmp_path / "evidence",
    )
    result = product_result(str(clip), _diag(), manifest, evidence=evidence)
    assert validate_evidence(evidence, result) == []
    assert len(evidence["frames"]) == 3
    (Path(evidence["directory"]) / evidence["frames"][0]["clip_frame"]).write_bytes(b"tampered")
    assert "changed clip_frame" in validate_evidence(evidence, result)


def _row(name, score, margin, support, width, master="master_00", offset=1.0):
    row = {
        "clip": name,
        "best_master": master,
        "source_offset_sec": offset,
        "best_score": score,
        "margin": margin,
        "confidence": {
            "calibration": {"status": "uncalibrated"},
            "decision": "matched",
            "source": {
                "tier": "uncalibrated",
                "best_score": score,
                "competing_score": score - margin,
                "cross_master_margin": margin,
            },
            "timing": {
                "tier": "uncalibrated",
                "support_fraction": support,
                "offset_profile_width_sec": width,
            },
        },
    }
    return row


def test_tiers_calibrate_on_development_and_measure_holdout_only():
    development = [
        _row("dev-a.mp4", 0.8, 0.3, 0.9, 0.25),
        _row("dev-b.mp4", 0.6, 0.2, 0.7, 0.50),
    ]
    truth = {
        "dev-a.mp4": {"master": "master_00.mp4", "offset_sec": 1.0},
        "dev-b.mp4": {"master": "master_00.mp4", "offset_sec": 1.0},
        "held.mp4": {"master": "master_00.mp4", "offset_sec": 1.0},
    }
    calibration = derive_tiers(development, truth)
    assert calibration["method"] == "development_medians_v1"
    assert calibration["thresholds"]["strong_score"] == pytest.approx(0.7)
    measured = measure_heldout([_row("held.mp4", 0.9, 0.4, 0.95, 0.1)], truth, calibration)
    assert measured["split"] == "holdout_only"
    assert measured["tiers"]["strong"]["joint_accuracy"] == 1.0
    assert measured["semantics"] == "observed_frequency_not_probability"


def test_matcher_does_not_publish_uncalibrated_fixed_tiers(tmp_path):
    master = tmp_path / "master_00.mp4"
    master.write_bytes(b"identity only")
    diag = _diag()
    diag.source_confidence = "uncalibrated"
    diag.timing_confidence = "uncalibrated"
    row = product_result("clip.mp4", diag, _manifest(master))
    assert row["confidence"]["calibration"] == {"status": "uncalibrated"}
    assert row["confidence"]["source"]["tier"] == "uncalibrated"
    assert row["confidence"]["timing"]["tier"] == "uncalibrated"


def test_development_calibration_is_applied_to_published_result_tiers():
    development = [
        _row("dev-a.mp4", 0.8, 0.3, 0.9, 0.25),
        _row("dev-b.mp4", 0.6, 0.2, 0.7, 0.50),
    ]
    truth = {
        "dev-a.mp4": {"master": "master_00.mp4", "offset_sec": 1.0},
        "dev-b.mp4": {"master": "master_00.mp4", "offset_sec": 1.0},
    }
    calibration = derive_tiers(development, truth)
    row = _row("held.mp4", 0.9, 0.4, 0.95, 0.1)
    calibrated = apply_calibration(row, calibration)
    assert calibrated["confidence"]["calibration"] == {
        "status": "applied",
        "sha256": calibration["calibration_sha256"],
    }
    assert calibrated["confidence"]["source"]["tier"] == "strong"
    assert calibrated["confidence"]["timing"]["tier"] == "strong"


def test_v2_rejects_confidence_fields_that_contradict_result(tmp_path):
    master = tmp_path / "master_00.mp4"
    master.write_bytes(b"identity only")
    row = product_result("clip.mp4", _diag(), _manifest(master))
    contradictions = [
        ("best score", lambda r: r["confidence"]["source"].__setitem__("best_score", 0.01)),
        ("reason", lambda r: r.__setitem__("reason", "uncertain")),
        ("speed", lambda r: r["confidence"]["timing"].__setitem__("speed_ratio", 0.5)),
    ]
    for message, mutate in contradictions:
        bad = copy.deepcopy(row)
        mutate(bad)
        with pytest.raises(ValueError, match=message):
            adapt_product(bad)
    arbitrary = copy.deepcopy(row)
    arbitrary["evidence"] = {"anything": "goes"}
    with pytest.raises(ValueError, match="evidence"):
        adapt_product(arbitrary)


def test_timing_uncertainty_is_not_accepted_or_called_matched():
    detail = SimpleNamespace(
        offset_sec=1.0,
        support_fraction=0.2,
        offset_profile_width_sec=2.0,
    )
    decision, source_tier, timing_tier = confidence_labels(
        True, 0.9, 0.4, 0.35, 0.12, detail
    )
    assert (decision, source_tier, timing_tier) == (
        "uncertain", "uncalibrated", "uncalibrated"
    )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_evidence_binds_reported_clip_and_decodes_images(tmp_path):
    clip = tmp_path / "clip.mp4"
    master = tmp_path / "master_00.mp4"
    for path, pattern in ((clip, "testsrc2"), (master, "testsrc")):
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
             f"{pattern}=size=96x64:rate=8", "-t", "3", "-pix_fmt", "yuv420p", str(path)],
            check=True,
        )
    result = product_result(str(clip), _diag(), _manifest(master))
    evidence = generate_evidence(
        clip, master, master_id="master_00", source_digest=_digest(master),
        aligned_timestamps=result["confidence"]["timing"]["aligned_timestamps"],
        output_dir=tmp_path / "evidence",
    )
    result = product_result(str(clip), _diag(), _manifest(master), evidence=evidence)
    wrong_clip = copy.deepcopy(result)
    wrong_clip["clip"] = str(tmp_path / "some-other-clip.mp4")
    assert "clip path disagrees with result" in validate_evidence(evidence, wrong_clip)
    basename_only = copy.deepcopy(result)
    basename_only["clip"] = clip.name
    assert "clip path disagrees with result" in validate_evidence(evidence, basename_only)
    frame = Path(evidence["directory"]) / evidence["frames"][0]["clip_frame"]
    frame.write_bytes(b"not an image")
    evidence["frames"][0]["clip_frame_sha256"] = _digest(frame)
    assert "undecodable clip_frame" in validate_evidence(evidence, result)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_failed_evidence_promotion_preserves_last_good_generation(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    master = tmp_path / "master_00.mp4"
    for path, pattern in ((clip, "testsrc2"), (master, "testsrc")):
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
             f"{pattern}=size=96x64:rate=8", "-t", "3", "-pix_fmt", "yuv420p", str(path)],
            check=True,
        )
    result = product_result(str(clip), _diag(), _manifest(master))
    kwargs = {
        "master_id": "master_00",
        "source_digest": _digest(master),
        "aligned_timestamps": result["confidence"]["timing"]["aligned_timestamps"],
        "output_dir": tmp_path / "evidence",
    }
    first = generate_evidence(clip, master, **kwargs)
    pointer = kwargs["output_dir"] / "current.json"
    before = json.loads(pointer.read_text())
    real_replace = __import__("clipprov.evidence", fromlist=["os"]).os.replace

    def fail_pointer(source, target):
        if Path(target) == pointer:
            raise OSError("injected pointer promotion failure")
        return real_replace(source, target)

    monkeypatch.setattr("clipprov.evidence.os.replace", fail_pointer)
    with pytest.raises(OSError, match="injected"):
        generate_evidence(clip, master, **kwargs)
    assert json.loads(pointer.read_text()) == before
    assert validate_evidence(first, product_result(str(clip), _diag(), _manifest(master), evidence=first)) == []


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_post_pointer_sync_failure_never_leaves_dangling_current(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    master = tmp_path / "master_00.mp4"
    for path, pattern in ((clip, "testsrc2"), (master, "testsrc")):
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
             f"{pattern}=size=96x64:rate=8", "-t", "3", "-pix_fmt", "yuv420p", str(path)],
            check=True,
        )
    result = product_result(str(clip), _diag(), _manifest(master))
    output_dir = tmp_path / "evidence"
    kwargs = {
        "master_id": "master_00",
        "source_digest": _digest(master),
        "aligned_timestamps": result["confidence"]["timing"]["aligned_timestamps"],
        "output_dir": output_dir,
    }
    first = generate_evidence(clip, master, **kwargs)
    first_generation = Path(first["directory"])
    real_fsync = os.fsync

    def fail_directory_sync(fd):
        target = Path(os.readlink(f"/proc/self/fd/{fd}"))
        if target.resolve() == output_dir.resolve():
            raise OSError("injected post-pointer directory sync failure")
        return real_fsync(fd)

    monkeypatch.setattr("clipprov.evidence.os.fsync", fail_directory_sync)
    with pytest.raises(OSError, match="post-pointer"):
        generate_evidence(clip, master, **kwargs)
    current = json.loads((output_dir / "current.json").read_text())
    current_generation = output_dir / "generations" / current["generation"]
    assert current_generation.is_dir()
    assert first_generation.is_dir()
    assert sorted(path.name for path in current_generation.glob("*.jpg")) == [
        "00-clip.jpg", "00-master.jpg", "01-clip.jpg",
        "01-master.jpg", "02-clip.jpg", "02-master.jpg",
    ]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_cli_evidence_canonicalizes_relative_clip_identity(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    master = tmp_path / "master_00.mp4"
    for path, pattern in ((clip, "testsrc2"), (master, "testsrc")):
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
             f"{pattern}=size=96x64:rate=8", "-t", "3", "-pix_fmt", "yuv420p", str(path)],
            check=True,
        )
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    manifest = _manifest(master)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("clipprov.cli.match_clip_path", lambda *_args, **_kwargs: (_diag(), manifest))
    result = CliRunner().invoke(
        main,
        ["match", "clip.mp4", "--index", "index", "--evidence-dir", "evidence"],
    )
    assert result.exit_code == 0, result.output
    row = json.loads(result.output)
    assert row["clip"] == str(clip.resolve())
    assert adapt_product(row)["native_tier"] == "matched"
