"""Regular-file admission must precede reads, including resume and error paths."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest
import test_phase5_integrity as phase5_tests

from clipprov.batch import match_directory
from clipprov.config import Config
from clipprov.index_batch import build_index_resumable

corpus = phase5_tests.corpus
pytestmark = pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFO required")


def _env(tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}}
    stage = tmp_path / "child-temporary"
    stage.mkdir(exist_ok=True)
    env["TMPDIR"] = str(stage)
    return env, stage


def _run(args, tmp_path):
    env, stage = _env(tmp_path)
    try:
        return subprocess.run([sys.executable, *args], capture_output=True, text=True,
                              env=env, timeout=3, check=False)
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _error2(result):
    assert result.returncode == 2, result.stderr
    assert "error:" in result.stderr and "Traceback" not in result.stderr


@pytest.mark.parametrize("reader", ["digest", "json_snapshot"])
def test_regular_reader_rejects_fifo_without_a_writer(tmp_path, reader):
    path = tmp_path / "input.json"
    os.mkfifo(path)
    code = """import sys
from clipprov import _io
try:
    getattr(_io, sys.argv[2])(sys.argv[1])
except OSError:
    print('REGULAR_FILE_REQUIRED')
    sys.exit(0)
sys.exit(4)
"""
    result = _run(["-c", code, str(path), reader], tmp_path)
    assert result.returncode == 0 and "REGULAR_FILE_REQUIRED" in result.stdout, result.stderr


def test_batch_discovery_rejects_fifo_only_directory(corpus, tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    os.mkfifo(clips / "blocked.mp4")
    _error2(_run(["-m", "clipprov.cli", "match-dir", str(clips), "--index", str(corpus.index),
                  "--out", str(tmp_path / "out"), "--no-evidence"], tmp_path))


def test_resume_checkpoint_array_fifo_is_rejected(corpus, tmp_path):
    index = tmp_path / "index"
    with pytest.raises(KeyboardInterrupt):
        build_index_resumable(corpus.library, index, Config(), interrupt_after=1)
    command = ["-m", "clipprov.cli", "index", str(corpus.library), "--out", str(index), "--resume"]
    journal = index / ".index-progress.json"
    state = json.loads(journal.read_text())
    control = _run(command, tmp_path)
    assert control.returncode == 0, control.stderr
    # Successful publication may remove its finished journal. Restore the same
    # valid checkpoint before corrupting its named shard for the resume probe.
    journal.write_text(json.dumps(state))
    entry = next(iter(state["entries"].values()))
    array = index / entry["shard_dir"] / entry["arrays"][0]["file"]
    array.unlink()
    os.mkfifo(array)
    _error2(_run(command, tmp_path))


@pytest.mark.parametrize("kind", ["calibration", "batch-journal", "index-journal"])
def test_nonregular_metadata_is_clean_cli_error(corpus, tmp_path, kind):
    out = tmp_path / "out"
    out.mkdir()
    if kind == "index-journal":
        os.mkfifo(out / ".index-progress.json")
        command = ["index", str(corpus.library), "--out", str(out), "--resume"]
    else:
        command = ["match-dir", str(corpus.library), "--index", str(corpus.index),
                   "--out", str(out), "--no-evidence"]
        if kind == "batch-journal":
            os.mkfifo(out / ".batch-progress.json")
        else:
            path = tmp_path / "calibration.json"
            os.mkfifo(path)
            command += ["--calibration", str(path)]
    _error2(_run(["-m", "clipprov.cli", *command], tmp_path))


def test_nonregular_report_pointer_is_domain_error(tmp_path):
    root = tmp_path / "report"
    root.mkdir()
    os.mkfifo(root / "current.json")
    code = """import sys
from clipprov.batch import BatchError,current_report_dir
try:
    current_report_dir(sys.argv[1])
except BatchError:
    print('REJECTED_POINTER')
    sys.exit(0)
sys.exit(4)
"""
    result = _run(["-c", code, str(root)], tmp_path)
    assert result.returncode == 0 and "REJECTED_POINTER" in result.stdout, result.stderr


def test_single_match_input_replaced_with_fifo_does_not_block(corpus, tmp_path):
    clip = tmp_path / "clip.mp4"
    shutil.copy2(corpus.library / "source_00.mp4", clip)
    code = """import os,sys
from pathlib import Path
import clipprov.cli as cli
clip=Path(sys.argv[1]); real=cli.match_clip_path
def replace(path,*args,**kwargs):
    clip.unlink();os.mkfifo(clip)
    return real(path,*args,**kwargs)
cli.match_clip_path=replace
cli.main(['match',str(clip),'--index',sys.argv[2]])
"""
    _error2(_run(["-c", code, str(clip), str(corpus.index)], tmp_path))


def test_json_nesting_errors_are_value_errors(tmp_path):
    from clipprov._io import json_snapshot

    path = tmp_path / "deep.json"
    path.write_text("[" * 2000 + "0" + "]" * 2000)
    with pytest.raises(ValueError):
        json_snapshot(path)

    # Brackets inside an escaped JSON string do not contribute to nesting.
    path.write_text(json.dumps({"text": "[" * 2000 + '\\"' + "]" * 2000}))
    assert json_snapshot(path)[0]["text"].startswith("[")


def test_deep_manifest_is_clean_cli_error(corpus, tmp_path):
    index = tmp_path / "index"
    index.mkdir()
    (index / "manifest.json").write_text("[" * 2000 + "0" + "]" * 2000)
    _error2(_run(["-m", "clipprov.cli", "match-dir", str(corpus.library), "--index", str(index),
                  "--out", str(tmp_path / "out"), "--no-evidence"], tmp_path))


@pytest.mark.parametrize("field", ["decision", "calibration-status", "timestamp-container"])
def test_malformed_resume_field_types_are_clean_errors(corpus, tmp_path, field):
    out = tmp_path / "out"
    report = match_directory(corpus.library, corpus.index, out, evidence=False)
    assert len(report["results"]) == 1 and not report["failures"]
    path = out / ".batch-progress.json"
    state = json.loads(path.read_text())
    row = next(iter(state["completed"].values()))
    if field == "decision":
        row["confidence"]["decision"] = []
    elif field == "calibration-status":
        row["confidence"]["calibration"]["status"] = []
    else:
        row["confidence"]["timing"]["aligned_timestamps"] = {}
    path.write_text(json.dumps(state))
    _error2(_run(["-m", "clipprov.cli", "match-dir", str(corpus.library), "--index", str(corpus.index),
                  "--out", str(out), "--no-evidence"], tmp_path))


def test_unhashable_index_status_is_clean_error(corpus, tmp_path):
    index = tmp_path / "index"
    shutil.copytree(corpus.index, index)
    path = index / "manifest.json"
    value = json.loads(path.read_text())
    value["status"] = []
    path.write_text(json.dumps(value))
    _error2(_run(["-m", "clipprov.cli", "match-dir", str(corpus.library), "--index", str(index),
                  "--out", str(tmp_path / "out"), "--no-evidence"], tmp_path))
