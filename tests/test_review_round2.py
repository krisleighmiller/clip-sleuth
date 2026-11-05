"""Regression coverage for the three independent-review findings."""
from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys

import pytest
import test_phase5_integrity as phase5_tests

from clipprov.batch import BatchError, _load_calibration, current_report_dir, match_directory
from clipprov.eval import adapt_product, product_result
from clipprov.evidence import generate_evidence, validate_evidence
from clipprov.match import match_clip_path

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
# Reuse the existing real-media fixture without duplicating setup.
corpus = phase5_tests.corpus


@pytest.fixture(scope="module")
def valid_evidence(corpus, tmp_path_factory):
    root = tmp_path_factory.mktemp("review2-evidence")
    clip = root / "different-clip-name.mp4"
    master = corpus.library / "source_00.mp4"
    shutil.copy2(master, clip)
    diag, manifest = match_clip_path(clip, corpus.index)
    row = product_result(str(clip), diag, manifest)
    source = row["source_record"]
    evidence = generate_evidence(
        clip, master, master_id=source["id"], source_digest=source["content_digest"],
        aligned_timestamps=row["confidence"]["timing"]["aligned_timestamps"],
        output_dir=root / "evidence",
    )
    row = product_result(str(clip), diag, manifest, evidence=evidence)
    assert validate_evidence(evidence, row) == []
    adapt_product(row)
    return row


@pytest.mark.parametrize("confidence", [None, "not-an-object", 42, [], {"calibration": None}])
def test_malformed_calibration_envelope_is_clean_cli_error(tmp_path, corpus, confidence):
    receipt = tmp_path / "bad-calibration.json"
    receipt.write_text(json.dumps({"confidence": confidence}))
    with pytest.raises(BatchError, match="invalid calibration"):
        _load_calibration(receipt)
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}}
    proc = subprocess.run(
        [sys.executable, "-m", "clipprov.cli", "match-dir", str(corpus.library),
         "--index", str(corpus.index), "--out", str(tmp_path / "out"),
         "--calibration", str(receipt), "--no-evidence"],
        capture_output=True, text=True, env=env, timeout=60, check=False,
    )
    assert proc.returncode == 2, proc.stderr
    assert "error:" in proc.stderr and "invalid calibration" in proc.stderr
    assert "Traceback" not in proc.stderr


@pytest.mark.parametrize("bad", [
    "not a pair", None, 3, [],
    {"clip_sec": [0], "master_sec": 0},
    {"clip_sec": {}, "master_sec": 0},
    {"clip_sec": None, "master_sec": 0},
    {"clip_sec": float("nan"), "master_sec": 0},
])
def test_standalone_evidence_reports_malformed_pair(valid_evidence, bad):
    row = copy.deepcopy(valid_evidence)
    row["confidence"]["timing"]["aligned_timestamps"][0] = bad
    problems = validate_evidence(row["evidence"], row)
    assert isinstance(problems, list) and problems
    assert all(isinstance(problem, str) for problem in problems)
    with pytest.raises((ValueError, TypeError)):
        adapt_product(row)


@pytest.mark.parametrize("bad", [None, "not-a-list", {}])
def test_standalone_evidence_reports_malformed_timestamp_container(valid_evidence, bad):
    row = copy.deepcopy(valid_evidence)
    row["confidence"]["timing"]["aligned_timestamps"] = bad
    problems = validate_evidence(row["evidence"], row)
    assert isinstance(problems, list) and problems


@pytest.mark.parametrize("field,bad", [("confidence", None), ("confidence", []), ("timing", None)])
def test_standalone_evidence_reports_malformed_nested_container(valid_evidence, field, bad):
    row = copy.deepcopy(valid_evidence)
    if field == "confidence":
        row[field] = bad
    else:
        row["confidence"][field] = bad
    problems = validate_evidence(row["evidence"], row)
    assert isinstance(problems, list) and problems


@pytest.fixture(scope="module")
def published_report(corpus, tmp_path_factory):
    root = tmp_path_factory.mktemp("review2-report")
    clips = root / "clips"
    clips.mkdir()
    shutil.copy2(corpus.library / "source_00.mp4", clips / "different.mp4")
    out = root / "batch"
    report = match_directory(clips, corpus.index, out, evidence=False)
    assert len(report["results"]) == 1 and not report["failures"]
    assert current_report_dir(out).is_dir()
    return out


@pytest.mark.parametrize("target_kind", ["missing", "directory", "file"])
def test_unlisted_symlink_is_rejected(published_report, tmp_path, target_kind):
    out = tmp_path / "copy"
    shutil.copytree(published_report, out)
    generation = current_report_dir(out)
    target = tmp_path / "target"
    if target_kind == "directory":
        target.mkdir()
    elif target_kind == "file":
        target.write_text("ordinary file")
    (generation / "unlisted-link").symlink_to(target)
    with pytest.raises(BatchError, match="symlink"):
        current_report_dir(out)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFO required")
def test_unlisted_nonregular_entry_is_rejected(published_report, tmp_path):
    out = tmp_path / "copy"
    shutil.copytree(published_report, out)
    generation = current_report_dir(out)
    os.mkfifo(generation / "unlisted-pipe")
    with pytest.raises(BatchError, match="non-regular"):
        current_report_dir(out)


def test_valid_generation_and_evidence_still_pass(published_report, valid_evidence):
    assert current_report_dir(published_report).is_dir()
    assert validate_evidence(valid_evidence["evidence"], valid_evidence) == []
