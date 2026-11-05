"""Matching and JPEG generation must consume immutable, verified media bytes."""
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
from click.testing import CliRunner

import clipprov.index as indexing
from clipprov import cli, evidence
from clipprov.eval import product_result
from clipprov.match import match_clip_path

corpus = phase5_tests.corpus


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_snapshot_is_immutable_and_cleaned(corpus, tmp_path):
    original = tmp_path / "clip.mp4"
    shutil.copy2(corpus.library / "source_00.mp4", original)
    before = original.read_bytes()
    with indexing._staged_source_snapshot(original) as (snapshot, digest):
        original.write_bytes(corpus.replacement.read_bytes())
        assert snapshot != original
        assert snapshot.read_bytes() == before
        assert digest == hashlib.sha256(before).hexdigest()
    assert not snapshot.exists()


def test_snapshot_cleanup_on_interruption(corpus, tmp_path):
    original = tmp_path / "clip.mp4"
    shutil.copy2(corpus.library / "source_00.mp4", original)
    with pytest.raises(KeyboardInterrupt), indexing._staged_source_snapshot(original) as (snapshot, _):
        raise KeyboardInterrupt("injected interruption")
    assert not snapshot.exists()


@pytest.mark.parametrize("target", ["clip", "master"])
def test_evidence_records_the_bytes_its_frames_use(corpus, tmp_path, monkeypatch, target):
    clip, master = tmp_path / "clip.mp4", tmp_path / "source_00.mp4"
    shutil.copy2(corpus.library / "source_00.mp4", clip)
    shutil.copy2(corpus.library / "source_00.mp4", master)
    diag, manifest = match_clip_path(clip, corpus.index)
    row = product_result(str(clip), diag, manifest)
    row["source_record"] = copy.deepcopy(row["source_record"])
    row["source_record"]["original_root_hint"] = str(tmp_path)
    points = row["confidence"]["timing"]["aligned_timestamps"]
    timestamp = points[0][target + "_sec"]
    original = clip if target == "clip" else master
    original_bytes = original.read_bytes()
    real_frame = evidence._frame
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    reference_a, reference_b = tmp_path / "A.jpg", tmp_path / "B.jpg"
    real_frame(ffmpeg, original, timestamp, reference_a)
    real_frame(ffmpeg, corpus.replacement, timestamp, reference_b)
    assert _sha(reference_a) != _sha(reference_b)
    seen = []

    def swap_original_during_frame(tool, source, at, output):
        if output.name.endswith("-" + target + ".jpg"):
            original.write_bytes(corpus.replacement.read_bytes())
            seen.append(source)
            try:
                return real_frame(tool, source, at, output)
            finally:
                original.write_bytes(original_bytes)
        return real_frame(tool, source, at, output)

    monkeypatch.setattr(evidence, "_frame", swap_original_during_frame)
    artifact = evidence.generate_evidence(
        clip, master, master_id=row["source_record"]["id"],
        source_digest=row["source_record"]["content_digest"],
        aligned_timestamps=points, output_dir=tmp_path / "evidence",
    )
    row["evidence"] = artifact
    assert artifact["clip_path"] == str(clip)
    assert artifact["master_path"] == str(master)
    assert artifact["clip_sha256"] == _sha(clip)
    assert artifact["master_sha256"] == _sha(master)
    assert evidence.validate_evidence(artifact, row) == []
    first_frame = artifact["frames"][0][target + "_frame"]
    actual = _sha(type(clip)(artifact["directory"]) / first_frame)
    assert actual == _sha(reference_a)
    assert actual != _sha(reference_b)
    assert all(path != original for path in seen)
    assert all(not path.exists() for path in seen)


@pytest.mark.parametrize("with_evidence", [False, True])
def test_single_match_keeps_one_clip_snapshot(corpus, tmp_path, monkeypatch, with_evidence):
    clip = tmp_path / "clip.mp4"
    shutil.copy2(corpus.library / "source_00.mp4", clip)
    original = clip.read_bytes()
    before = hashlib.sha256(original).hexdigest()
    real_match = cli.match_clip_path
    seen = []

    def swap_during_match(path, *args, **kwargs):
        clip.write_bytes(corpus.replacement.read_bytes())
        seen.append((path, _sha(path)))
        try:
            return real_match(path, *args, **kwargs)
        finally:
            clip.write_bytes(original)

    monkeypatch.setattr(cli, "match_clip_path", swap_during_match)
    args = ["match", str(clip), "--index", str(corpus.index)]
    if with_evidence:
        args += ["--evidence-dir", str(tmp_path / "evidence")]
    result = CliRunner().invoke(cli.main, args)
    assert result.exit_code == 0, result.output
    row = json.loads(result.stdout)
    assert row["clip"] == str(clip)
    assert seen and all(digest == before for _, digest in seen)
    assert all(path != clip and not path.exists() for path, _ in seen)
    assert all(str(path) not in result.stdout for path, _ in seen)
    if with_evidence:
        assert row["evidence"]["clip_path"] == str(clip)
        assert row["evidence"]["clip_sha256"] == before
        assert evidence.validate_evidence(row["evidence"], row) == []


@pytest.mark.parametrize("with_evidence", [False, True])
def test_single_match_rejects_persistently_changed_input(corpus, tmp_path, monkeypatch, with_evidence):
    clip = tmp_path / "clip.mp4"
    shutil.copy2(corpus.library / "source_00.mp4", clip)
    real_match = cli.match_clip_path
    snapshots = []

    def replace_original(path, *args, **kwargs):
        snapshots.append(path)
        clip.write_bytes(corpus.replacement.read_bytes())
        return real_match(path, *args, **kwargs)

    monkeypatch.setattr(cli, "match_clip_path", replace_original)
    output = tmp_path / "result.json"
    args = ["match", str(clip), "--index", str(corpus.index), "--out", str(output)]
    if with_evidence:
        args += ["--evidence-dir", str(tmp_path / "evidence")]
    result = CliRunner().invoke(cli.main, args)
    assert result.exit_code == 2, result.output
    assert "changed" in result.stderr and "Traceback" not in result.stderr
    assert not output.exists()
    assert snapshots and all(path != clip and not path.exists() for path in snapshots)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFO required")
@pytest.mark.parametrize("target", ["clip", "manifest", "array"])
def test_nonregular_inputs_are_rejected_without_waiting_for_a_writer(corpus, tmp_path, target):
    env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    if target == "clip":
        path = tmp_path / "clip.mp4"
        os.mkfifo(path)
        args = ["match", str(path), "--index", str(corpus.index)]
    else:
        index = tmp_path / "index"
        shutil.copytree(corpus.index, index)
        path = index / "manifest.json"
        if target == "array":
            manifest = json.loads(path.read_text())
            entry = manifest["masters"][0]
            path = index / entry["shard_dir"] / entry["arrays"][0]["file"]
        path.unlink()
        os.mkfifo(path)
        args = ["match-dir", str(corpus.library), "--index", str(index),
                "--out", str(tmp_path / "out"), "--no-evidence"]
    result = subprocess.run([sys.executable, "-m", "clipprov.cli", *args],
                            capture_output=True, text=True, env=env, timeout=3, check=False)
    assert result.returncode == 2, result.stderr
    assert "regular" in result.stderr and "Traceback" not in result.stderr
