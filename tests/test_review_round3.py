"""Oversized JSON numbers and malformed evidence source records are domain errors."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys

import pytest
import test_phase5_integrity as phase5_tests
import test_review_round2 as round2_tests

from clipprov import _io
from clipprov.batch import BatchError, _load_calibration, match_directory
from clipprov.confidence import derive_tiers
from clipprov.eval import (
    _canonical_master,
    _number_or_none,
    adapt_product,
    alignment_timestamp_problems,
    score_product,
)
from clipprov.evidence import validate_evidence
from clipprov.index import IndexError as IndexReadError
from clipprov.index import _finite_number, load_index
from clipprov.transforms import number

corpus = phase5_tests.corpus
valid_evidence = round2_tests.valid_evidence
BIG = 10**400


@pytest.mark.parametrize("value", [BIG, -BIG, float("inf"), float("nan"), True, None, [], {}])
def test_finite_number_rejects_unrepresentable_or_non_numeric_values(value):
    assert _io.finite_number(value) is False
    assert _finite_number(value) is False


@pytest.mark.parametrize("value", [0, -1, 1.5, 1e308])
def test_finite_number_preserves_valid_numbers(value):
    assert _io.finite_number(value) is True
    assert _number_or_none(value) == float(value)


@pytest.mark.parametrize("value", [BIG, -BIG], ids=["oversized-positive", "oversized-negative"])
def test_numeric_admission_returns_domain_errors(value):
    with pytest.raises(ValueError):
        _number_or_none(value)
    with pytest.raises(ValueError):
        _canonical_master(value)
    with pytest.raises(ValueError):
        score_product([], {}, value)
    with pytest.raises(ValueError):
        derive_tiers([], {}, value)
    with pytest.raises(ValueError):
        number(value, "speed", 0.9, 1.1)


@pytest.mark.parametrize("location", [
    "pair_clip", "pair_master", "frame_clip", "frame_master",
    "clip_duration", "master_duration", "raw_offset", "speed",
])
def test_oversized_evidence_numbers_return_problems(valid_evidence, location):
    row = copy.deepcopy(valid_evidence)
    if location.startswith("pair_"):
        row["confidence"]["timing"]["aligned_timestamps"][0][location[5:] + "_sec"] = BIG
    elif location.startswith("frame_"):
        row["evidence"]["frames"][0][location[6:] + "_sec"] = BIG
    elif location.endswith("duration"):
        row["evidence"][location + "_sec"] = BIG
    elif location == "raw_offset":
        row["raw_alignment_offset_sec"] = BIG
    else:
        row["speed"] = BIG
    problems = validate_evidence(row["evidence"], row)
    assert isinstance(problems, list) and problems
    with pytest.raises(ValueError):
        adapt_product(row)


@pytest.mark.parametrize("clip_sec,speed,raw", [
    (1e308, 1.0, 0.0),
    (12.5, 1e308, 0.0),
    (12.5, 1e307, 1e308),
])
def test_finite_operands_with_overflowing_intermediates_are_diagnostics(
    valid_evidence, clip_sec, speed, raw,
):
    row = copy.deepcopy(valid_evidence)
    row["speed"] = speed
    row["raw_alignment_offset_sec"] = raw
    row["confidence"]["timing"]["aligned_timestamps"] = [{"clip_sec": clip_sec, "master_sec": 0}]
    problems = alignment_timestamp_problems(row)
    assert isinstance(problems, list) and problems


@pytest.mark.parametrize("source", [None, [], {}, {"id": "bad"}, 4, "bad"])
def test_malformed_source_record_returns_problems(valid_evidence, source):
    row = copy.deepcopy(valid_evidence)
    row["source_record"] = source
    problems = validate_evidence(row["evidence"], row)
    assert isinstance(problems, list) and problems


@pytest.mark.parametrize("field", ["id", "source_relpath", "display_name", "original_root_hint", "content_digest"])
def test_non_string_source_fields_return_problems(valid_evidence, field):
    row = copy.deepcopy(valid_evidence)
    row["source_record"][field] = []
    assert validate_evidence(row["evidence"], row)


@pytest.mark.parametrize("where", ["source_root", "reported_clip", "evidence_clip"])
def test_invalid_path_text_returns_problems(valid_evidence, where):
    row = copy.deepcopy(valid_evidence)
    if where == "source_root":
        row["source_record"]["original_root_hint"] = "\0"
    elif where == "reported_clip":
        row["clip"] = "/invalid/\0"
    else:
        row["evidence"]["clip_path"] = "/invalid/\0"
    assert validate_evidence(row["evidence"], row)


@pytest.mark.parametrize("result", [None, [], "bad"])
def test_non_object_result_returns_problems(valid_evidence, result):
    problems = validate_evidence(valid_evidence["evidence"], result)
    assert isinstance(problems, list) and problems


@pytest.mark.parametrize("field", ["tolerance_seconds", "strong_score", "strong_margin", "focused_support_fraction", "focused_profile_width_sec"])
def test_oversized_calibration_numbers_are_domain_errors(valid_evidence, tmp_path, field):
    truth = {valid_evidence["clip"]: {
        "master": valid_evidence["best_master"] + ".mp4",
        "offset_sec": valid_evidence["source_offset_sec"],
    }}
    receipt = derive_tiers([valid_evidence], truth)
    if field == "tolerance_seconds":
        receipt[field] = BIG
    else:
        receipt["thresholds"][field] = BIG
    body = {k: v for k, v in receipt.items() if k != "calibration_sha256"}
    receipt["calibration_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(receipt))
    with pytest.raises(BatchError):
        _load_calibration(path)


def test_oversized_index_configuration_is_index_error(corpus, tmp_path):
    index = tmp_path / "index"
    shutil.copytree(corpus.index, index)
    path = index / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["extractor_config"]["score_floor"] = BIG
    path.write_text(json.dumps(manifest))
    with pytest.raises(IndexReadError):
        load_index(index)


def test_resume_oversized_timestamp_exits_2_without_traceback(corpus, tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    shutil.copy2(corpus.library / "source_00.mp4", clips / "clip.mp4")
    out = tmp_path / "run"
    report = match_directory(clips, corpus.index, out, evidence=False)
    assert len(report["results"]) == 1 and not report["failures"]
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}}
    args = [sys.executable, "-m", "clipprov.cli", "match-dir", str(clips),
            "--index", str(corpus.index), "--out", str(out), "--no-evidence"]
    normal = subprocess.run(args, capture_output=True, text=True, env=env, timeout=60, check=False)
    assert normal.returncode == 0, normal.stderr
    path = out / ".batch-progress.json"
    journal = json.loads(path.read_text())
    journal["completed"]["clip.mp4"]["confidence"]["timing"]["aligned_timestamps"][0]["clip_sec"] = BIG
    path.write_text(json.dumps(journal))
    malformed = subprocess.run(args, capture_output=True, text=True, env=env, timeout=60, check=False)
    assert malformed.returncode == 2, malformed.stderr
    assert "error:" in malformed.stderr and "Traceback" not in malformed.stderr


def test_valid_evidence_control_remains_valid(valid_evidence):
    assert validate_evidence(valid_evidence["evidence"], valid_evidence) == []
    adapt_product(valid_evidence)
