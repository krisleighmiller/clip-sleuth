"""Every evaluation receipt hashes the exact JSON snapshot it consumed."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil

import pytest
import test_phase3_harness as harness_tests
import test_phase5_integrity as phase5_tests
from click.testing import CliRunner

import clipprov.eval as evaluation
from clipprov import harness
from clipprov._io import atomic_json, atomic_text
from clipprov.cli import main
from clipprov.eval import product_result
from clipprov.match import match_clip_path

media_corpus = harness_tests.media_corpus
corpus = phase5_tests.corpus


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _publish_reformatted(path):
    value = json.loads(path.read_bytes())
    atomic_text(path, json.dumps(value, indent=4, sort_keys=True) + "\n")


@pytest.fixture
def saved_pair(corpus, tmp_path):
    diag, manifest = match_clip_path(corpus.library / "source_00.mp4", corpus.index)
    row = product_result("clip.mp4", diag, manifest)
    results, truth = tmp_path / "results.json", tmp_path / "truth.json"
    atomic_json(results, [row])
    atomic_json(truth, [{"clip": "clip.mp4", "master": row["best_master"] + ".mp4",
                         "offset_sec": row["source_offset_sec"]}])
    return results, truth


def test_calibration_file_receipt_binds_consumed_bytes(tmp_path, monkeypatch):
    from pathlib import Path

    from clipprov.batch import _load_calibration

    receipt = {
        "method": "development_medians_v1", "semantics": "diagnostic_not_probability",
        "development_rows": 1, "eligible_correct_alignments": 1,
        "thresholds": {"strong_score": 0.8, "strong_margin": 0.1,
                       "focused_support_fraction": 0.6, "focused_profile_width_sec": 1.0},
        "tolerance_seconds": 1.5,
    }
    receipt["calibration_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = tmp_path / "calibration.json"
    atomic_json(path, receipt)
    expected = _sha(path)
    replacement = json.dumps(receipt, indent=4, sort_keys=True) + "\n"
    real_open = Path.open
    real_fdopen = os.fdopen
    original_stat = path.stat()
    seen = {"published": False}

    class PublishAfterRead:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def read(self, *args):
            data = self.handle.read(*args)
            if not seen["published"]:
                seen["published"] = True
                atomic_text(path, replacement)
            return data

    def open_and_publish(candidate, *args, **kwargs):
        handle = real_open(candidate, *args, **kwargs)
        return PublishAfterRead(handle) if candidate == path else handle

    def fdopen_and_publish(fd, *args, **kwargs):
        current = os.fstat(fd)
        handle = real_fdopen(fd, *args, **kwargs)
        if (current.st_dev, current.st_ino) == (original_stat.st_dev, original_stat.st_ino):
            return PublishAfterRead(handle)
        return handle

    monkeypatch.setattr(Path, "open", open_and_publish)
    monkeypatch.setattr(os, "fdopen", fdopen_and_publish)
    loaded, sha = _load_calibration(path)
    assert seen["published"] and _sha(path) != expected
    assert loaded == receipt
    assert sha == expected


@pytest.mark.parametrize("kind", ["results", "truth"])
def test_json_loader_digest_is_optional_and_exact(saved_pair, kind):
    path = saved_pair[0 if kind == "results" else 1]
    loader = evaluation.load_product_results if kind == "results" else evaluation.load_truth
    legacy = loader(path)
    data, sha = loader(path, with_digest=True)
    assert data == legacy
    assert sha == _sha(path)
    assert isinstance(legacy, list if kind == "results" else dict)


@pytest.mark.parametrize("kind", ["results", "truth"])
def test_saved_eval_receipt_survives_post_load_publication(saved_pair, tmp_path, monkeypatch, kind):
    results, truth = saved_pair
    expected = {"results": _sha(results), "truth": _sha(truth)}
    name = "load_product_results" if kind == "results" else "load_truth"
    real = getattr(evaluation, name)
    observed = {}

    def load_then_publish(path, *args, **kwargs):
        loaded = real(path, *args, **kwargs)
        _publish_reformatted(path)
        observed["published"] = _sha(path)
        return loaded

    monkeypatch.setattr(evaluation, name, load_then_publish)
    output = tmp_path / "score.json"
    result = CliRunner().invoke(main, ["eval", "--results", str(results),
                                     "--truth", str(truth), "--out", str(output)])
    assert result.exit_code == 0, result.output
    report = json.loads(output.read_bytes())
    assert expected[kind] != observed["published"]
    assert report["results_digest"] == expected["results"]
    assert report["truth_digest"] == expected["truth"]


@pytest.fixture(scope="module")
def controls_case(media_corpus, tmp_path_factory):
    _, library = media_corpus
    root = tmp_path_factory.mktemp("receipt-controls")
    controls = root / "controls"
    manifest = harness.generate_controls(library, controls)
    _, index = harness_tests.controls_index(library, manifest, root)
    return controls, index


@pytest.mark.parametrize("kind", ["index", "manifest"])
def test_control_evaluation_binds_consumed_snapshots(controls_case, tmp_path, monkeypatch, kind):
    controls, original_index = controls_case
    local_controls, index = tmp_path / "controls", tmp_path / "index"
    shutil.copytree(controls, local_controls)
    shutil.copytree(original_index, index)
    manifest_path = local_controls / "manifest.json"
    before = {"manifest": _sha(manifest_path), "index": _sha(index / "manifest.json")}
    real = harness.load_index
    observed = {}

    def load_then_publish(path, *args, **kwargs):
        loaded = real(path, *args, **kwargs)
        observed["loaded"] = loaded[0].manifest_sha256
        if kind == "index":
            replacement = copy.deepcopy(dict(loaded[0]))
            replacement["extractor_config"]["score_floor"] = 0.99
            atomic_json(index / "manifest.json", replacement)
            observed["published"] = _sha(index / "manifest.json")
        else:
            _publish_reformatted(manifest_path)
            observed["published"] = _sha(manifest_path)
        return loaded

    monkeypatch.setattr(harness, "load_index", load_then_publish)
    report = harness.evaluate_controls(manifest_path, index, tmp_path / "report.json", shortlist_samples=0)
    assert observed["published"] != before[kind]
    assert report["input_digests"]["manifest"] == before["manifest"]
    assert report["input_digests"]["index_manifest"] == observed["loaded"] == before["index"]
    if kind == "index":
        assert real(index)[0]["extractor_config"]["score_floor"] == 0.99
        assert report["confidence"]["calibrated_results"]["development"][0]["confidence"]["source"]["acceptance_floor"] == 0.35


@pytest.mark.parametrize("kind", ["results", "truth"])
def test_combined_benchmark_receipts_use_consumed_json(
    controls_case, saved_pair, tmp_path, monkeypatch, kind,
):
    controls, index = controls_case
    results, truth = saved_pair
    expected = {"results": _sha(results), "truth": _sha(truth)}
    name = "load_product_results" if kind == "results" else "load_truth"
    real = getattr(harness, name)
    observed = {}

    def load_then_publish(path, *args, **kwargs):
        loaded = real(path, *args, **kwargs)
        if path == (results if kind == "results" else truth):
            _publish_reformatted(path)
            observed["published"] = _sha(path)
        return loaded

    monkeypatch.setattr(harness, name, load_then_publish)
    report = harness.evaluate_controls(
        controls / "manifest.json", index, tmp_path / "combined.json", shortlist_samples=0,
        benchmark_results=results, benchmark_truth=truth,
    )
    assert observed["published"] != expected[kind]
    assert report["input_digests"]["benchmark_results"] == expected["results"]
    assert report["input_digests"]["benchmark_truth"] == expected["truth"]
    assert report["benchmark"]["results_digest"] == expected["results"]
    assert report["benchmark"]["truth_digest"] == expected["truth"]
