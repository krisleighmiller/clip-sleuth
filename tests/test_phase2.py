"""Phase 2 tests: index, match, CLI (portable)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from clipprov.cli import main as cli_main
from clipprov.config import Config, MatchConfig
from clipprov.index import IndexError as IdxError
from clipprov.index import build_index, load_index


def _make_video(path: Path, color: str = "black") -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c={color}:s=64x64:d=1",
         "-r", "8", str(path)],
        capture_output=True, check=False,
    )
    return r.returncode == 0 and path.exists()


@pytest.fixture
def corpus(tmp_path):
    masters = tmp_path / "masters"
    masters.mkdir()
    clips = tmp_path / "clips"
    clips.mkdir()
    if not _make_video(masters / "m0.mp4", "black"):
        pytest.skip("ffmpeg unavailable")
    _make_video(masters / "m1.mp4", "white")
    shutil.copy(masters / "m0.mp4", clips / "c0.mp4")
    return masters, clips / "c0.mp4"


def test_build_and_load(corpus, tmp_path):
    masters, _ = corpus
    out = tmp_path / "idx"
    build_index(masters, out, Config())
    manifest, masters_loaded = load_index(out)
    assert manifest["status"] == "complete"
    assert len(masters_loaded) == 2
    for m in manifest["masters"]:
        assert "source_relpath" in m
        assert "display_name" in m
        assert "original_root_hint" in m
        assert "content_digest" in m


def test_incremental_reuse(corpus, tmp_path):
    masters, _ = corpus
    out = tmp_path / "idx"
    build_index(masters, out, Config())
    build_index(masters, out, Config())
    _manifest, masters_loaded = load_index(out)
    assert len(masters_loaded) == 2


def test_config_mismatch_rebuilds(corpus, tmp_path):
    masters, _ = corpus
    out = tmp_path / "idx"
    build_index(masters, out, Config(match=MatchConfig(score_floor=0.99999)))
    build_index(masters, out, Config())
    manifest, _ = load_index(out)
    assert manifest["extractor_config"]["score_floor"] == 0.35


def test_corrupt_shard_rebuilds(corpus, tmp_path):
    masters, _ = corpus
    out = tmp_path / "idx"
    build_index(masters, out, Config())
    import glob
    for f in glob.glob(str(out / "masters" / "*" / "*" / "motion.npy")):
        Path(f).write_bytes(b"corrupt")
    build_index(masters, out, Config(), force=True)
    _manifest, masters_loaded = load_index(out)
    assert len(masters_loaded) == 2


def test_unsupported_schema(corpus, tmp_path):
    masters, _ = corpus
    out = tmp_path / "idx"
    build_index(masters, out, Config())
    manifest_path = out / "manifest.json"
    m = json.loads(manifest_path.read_text())
    m["schema_version"] = 999
    manifest_path.write_text(json.dumps(m))
    with pytest.raises(IdxError):
        load_index(out)


def test_duplicate_basename_rejected(tmp_path):
    d = tmp_path / "lib"
    (d / "a").mkdir(parents=True)
    (d / "b").mkdir(parents=True)
    if not _make_video(d / "a" / "dup.mp4", "black"):
        pytest.skip("ffmpeg unavailable")
    shutil.copy(d / "a" / "dup.mp4", d / "b" / "dup.mp4")
    with pytest.raises(IdxError):
        build_index(d, tmp_path / "idx", Config())


def test_cli_index_and_match(corpus, tmp_path):
    masters, clip = corpus
    out = tmp_path / "idx"
    runner = CliRunner()
    r = runner.invoke(cli_main, ["index", str(masters), "--out", str(out)])
    assert r.exit_code == 0, r.output
    r2 = runner.invoke(cli_main, ["match", str(clip), "--index", str(out)])
    assert r2.exit_code in (0, 1), r2.output
    data = json.loads(r2.output)
    assert "accepted" in data
    assert "source_record" in data
    if data["accepted"]:
        assert data["best_master"] == "m0"
        assert data["source_record"]["source_relpath"] == "m0.mp4"


def test_decision_threshold_only_reuse(corpus, tmp_path):
    masters, _ = corpus
    out = tmp_path / "idx"
    build_index(masters, out, Config(match=MatchConfig(score_floor=0.1, margin=0.01)))
    build_index(masters, out, Config(match=MatchConfig(score_floor=0.9, margin=0.5)))
    manifest, _ = load_index(out)
    assert manifest["extractor_config"]["score_floor"] == 0.9


def test_forced_rebuild_failure_preserves_active(corpus, tmp_path, monkeypatch):
    masters, _ = corpus
    out = tmp_path / "idx"
    build_index(masters, out, Config())
    # baseline loads
    load_index(out)
    original = (out / "manifest.json").read_text()

    import clipprov.index as idxmod

    def failing_build(*a, **k):
        raise RuntimeError("injected shard build failure")

    monkeypatch.setattr(idxmod, "_build_shard", failing_build)
    with pytest.raises(RuntimeError):
        build_index(masters, out, Config(), force=True)
    # old manifest still present
    assert (out / "manifest.json").read_text() == original
    # load still works
    load_index(out)


def test_cli_output_write_failure(corpus, tmp_path):
    masters, clip = corpus
    out = tmp_path / "idx"
    runner = CliRunner()
    runner.invoke(cli_main, ["index", str(masters), "--out", str(out)])
    bad_out = tmp_path  # existing dir
    r = runner.invoke(cli_main, ["match", str(clip), "--index", str(out), "--out", str(bad_out)])
    assert r.exit_code == 2
