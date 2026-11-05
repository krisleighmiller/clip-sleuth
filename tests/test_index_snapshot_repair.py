"""Repairs for index snapshot identity, lock cleanup, and resume location.

Scope:
- A: load_index exposes the digest of the exact manifest bytes it consumed via
  a read-only _ManifestSnapshot.manifest_sha256 attribute (never a mapping key).
- B: build_index_resumable releases its writer lock exactly once even when the
  resume journal fails validation before the publish phase.
- C: resumed journal entries must agree with the inventory on location fields
  (original_root_hint, source_path, display_name), not just digest/size/id.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from clipprov import index as index_module
from clipprov.config import Config
from clipprov.index import IndexError as ClipIndexError
from clipprov.index import load_index
from clipprov.index_batch import IndexBatchError, build_index_resumable

requires_ffmpeg = pytest.mark.skipif(
    not Path("/usr/bin/ffmpeg").exists(), reason="ffmpeg required"
)


def _video(path: Path, pattern="testsrc", seconds=2):
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
         f"{pattern}=size=96x64:rate=8", "-t", str(seconds), "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )


def _library(tmp_path: Path) -> Path:
    library = tmp_path / "library"
    library.mkdir()
    return library


def _interrupted_index(tmp_path: Path) -> tuple[Path, Path, dict]:
    """Two-master build interrupted after one item; returns library, index, journal."""
    library = _library(tmp_path)
    _video(library / "a.mp4", "testsrc")
    _video(library / "b.mp4", "testsrc2")
    index = tmp_path / "index"
    with pytest.raises(KeyboardInterrupt):
        build_index_resumable(library, index, Config(), interrupt_after=1)
    journal = json.loads((index / ".index-progress.json").read_text())
    assert set(journal["entries"]) == {"a"}
    return library, index, journal


def _lock_is_free(index_dir: Path) -> bool:
    """Non-blocking re-acquire of the writer lock with a fresh descriptor."""
    fd = os.open(index_dir / "lock", os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return True
    except OSError:
        return False
    finally:
        os.close(fd)


# --------------------------------------------------------------------------- A


@requires_ffmpeg
def test_manifest_snapshot_exposes_consumed_bytes_digest(tmp_path):
    library = _library(tmp_path)
    _video(library / "single.mp4", "testsrc")
    index = tmp_path / "index"
    build_index_resumable(library, index, Config())
    manifest_path = index / "manifest.json"
    manifest, masters = load_index(index)

    assert isinstance(manifest, index_module._ManifestSnapshot)
    assert isinstance(manifest, dict)
    on_disk = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert manifest.manifest_sha256 == on_disk
    assert len(manifest.manifest_sha256) == 64

    # The digest is an attribute, never a mapping key, and never serialized.
    assert "_manifest_sha256" not in manifest
    assert "manifest_sha256" not in manifest
    serialized = json.dumps(manifest, sort_keys=True)
    assert "manifest_sha256" not in serialized
    assert json.loads(serialized) == dict(manifest)

    # Read-only: assignment must not succeed.
    with pytest.raises(AttributeError):
        manifest.manifest_sha256 = "0" * 64

    # Config extraction sees exactly the persisted mapping (no pollution).
    config = index_module.config_from_manifest(manifest)
    assert config.media.max_frames == manifest["extractor_config"]["max_frames"]
    assert len(masters) == 1


@requires_ffmpeg
def test_manifest_replacement_at_read_boundary_keeps_snapshot_identity(tmp_path, monkeypatch):
    library = _library(tmp_path)
    _video(library / "single.mp4", "testsrc")
    index = tmp_path / "index"
    build_index_resumable(library, index, Config())
    manifest_path = index / "manifest.json"
    original_bytes = manifest_path.read_bytes()
    original_sha = hashlib.sha256(original_bytes).hexdigest()
    assert original_bytes == (index / "manifest.json").read_bytes()

    real_read = index_module._read_manifest
    calls = {"n": 0}

    def swapping_read(index_dir, directory_fd=None, manifest_name="manifest.json"):
        result = real_read(index_dir, directory_fd, manifest_name)
        calls["n"] += 1
        if calls["n"] == 1 and manifest_name == "manifest.json":
            mutated = json.loads(original_bytes)
            mutated["numpy_version"] = "999.0.0-swap"
            swap = index_dir / ".swap-manifest.tmp"
            swap.write_text(json.dumps(mutated, indent=2, sort_keys=True) + "\n")
            os.replace(swap, manifest_path)
        return result

    monkeypatch.setattr(index_module, "_read_manifest", swapping_read)

    # First load consumed the ORIGINAL bytes; the snapshot must keep that digest
    # even though the file was replaced before load_index continued.
    manifest, _ = load_index(index)
    assert calls["n"] == 1
    assert manifest.manifest_sha256 == original_sha
    assert manifest["numpy_version"] == json.loads(original_bytes)["numpy_version"]
    assert manifest.manifest_sha256 != hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()

    # A subsequent load strictly validates and hashes the NEW bytes.
    reloaded, _ = load_index(index)
    assert calls["n"] == 2
    assert reloaded["numpy_version"] == "999.0.0-swap"
    assert reloaded.manifest_sha256 == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    assert reloaded.manifest_sha256 != original_sha

    # Production reader stays strict: corrupt bytes are rejected, not digested.
    monkeypatch.undo()
    manifest_path.write_text("{ not json")
    with pytest.raises(ClipIndexError, match="corrupt manifest"):
        load_index(index)


# --------------------------------------------------------------------------- B


@requires_ffmpeg
def test_truncated_journal_shard_error_releases_writer_lock(tmp_path):
    library, index, journal = _interrupted_index(tmp_path)
    entry = journal["entries"]["a"]
    (index / entry["shard_dir"] / "motion.npy").write_bytes(b"")

    with pytest.raises(IndexBatchError, match="resumed shard failed validation"):
        build_index_resumable(library, index, Config())
    assert _lock_is_free(index), "writer lock leaked by journal validation failure"


@requires_ffmpeg
def test_malformed_journal_json_releases_writer_lock(tmp_path):
    library, index, _journal = _interrupted_index(tmp_path)
    (index / ".index-progress.json").write_text("{ truncated")

    with pytest.raises(IndexBatchError, match="corrupt index journal"):
        build_index_resumable(library, index, Config())
    assert _lock_is_free(index), "writer lock leaked by malformed journal"


# --------------------------------------------------------------------------- C


@requires_ffmpeg
@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        pytest.param("original_root_hint", "/mnt/elsewhere/library", id="root-hint"),
        pytest.param("source_path", None, id="source-path"),
        pytest.param("display_name", "renamed.mp4", id="display-name"),
    ],
)
def test_resume_rejects_drifted_location_identity(tmp_path, field, bad_value):
    library, index, journal = _interrupted_index(tmp_path)
    if field == "source_path":
        bad_value = str(library / "elsewhere" / "a.mp4")
    entry = journal["entries"]["a"]
    assert field in entry, f"journal entry lacks location field {field}"
    entry[field] = bad_value
    (index / ".index-progress.json").write_text(json.dumps(journal, indent=2))

    with pytest.raises(IndexBatchError, match="disagrees with inventory"):
        build_index_resumable(library, index, Config())
    assert not (index / "manifest.json").exists()
    assert _lock_is_free(index)


@requires_ffmpeg
def test_unmodified_journal_still_resumes_and_publishes(tmp_path):
    library, index, _journal = _interrupted_index(tmp_path)
    manifest = build_index_resumable(library, index, Config())
    assert manifest["status"] == "complete"
    assert manifest["build_receipt"]["resumed_masters"] == 1
    loaded, masters = load_index(index)
    assert len(loaded["masters"]) == len(masters) == 2
    assert loaded.manifest_sha256 == hashlib.sha256(
        (index / "manifest.json").read_bytes()
    ).hexdigest()
    assert _lock_is_free(index)
