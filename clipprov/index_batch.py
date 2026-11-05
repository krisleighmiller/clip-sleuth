"""Fault-contained resumable index construction for Phase 5."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from contextlib import ExitStack, closing
from pathlib import Path

import numpy as np

from clipprov._io import atomic_json, digest, json_snapshot
from clipprov.config import Config
from clipprov.index import (
    ALGORITHM_VERSION,
    SCHEMA_VERSION,
    IndexError,
    _discover_master_files,
    _fsync_dir,
    _open_index_directory,
    _open_shard_directory,
    _pin_array,
    _tool_version,
    _validate_source_path,
    acquire_writer_lock,
    artifact_config,
    build_index,
    extractor_config,
    load_index,
    release_writer_lock,
)
from clipprov.media import MediaError

INDEX_PROGRESS_VERSION = 1


class IndexBatchError(Exception):
    """Raised when an index journal cannot be resumed safely."""


def _implementation_receipt(root: Path | None = None) -> dict:
    from clipprov.harness import _git_receipt

    root = Path(root) if root is not None else Path(__file__).resolve().parent
    files = [
        {"path": f"clipprov/{name}", "sha256": digest(root / name)}
        for name in (
            "__init__.py", "_io.py", "appearance.py", "config.py", "index.py",
            "index_batch.py", "media.py", "motion.py",
        )
    ]
    payload = json.dumps(files, sort_keys=True, separators=(",", ":"))

    return {
        "files": files,
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "git": _git_receipt(root.parent),
    }


def _identity(library: Path, files: list[Path], config: Config, continue_on_error: bool):
    inventory = []
    for path in files:
        stat = path.stat()
        inventory.append({
            "relpath": path.relative_to(library).as_posix(),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": digest(path),
        })
    return {
        "version": INDEX_PROGRESS_VERSION,
        "library": str(library),
        "inventory": inventory,
        "config": config.model_dump(mode="json"),
        "algorithm_version": ALGORITHM_VERSION,
        "ffmpeg_version": _tool_version(config.media.ffmpeg),
        "numpy_version": np.__version__,
        "implementation_sha256": _implementation_receipt()["sha256"],
        "continue_on_error": continue_on_error,
    }


def _entry_valid(index_dir: Path, entry: dict) -> bool:
    try:
        token = entry["shard_dir"]
        if not isinstance(token, str) or token.startswith("/"):
            return False
        parts = tuple(token.split("/"))
        if any(part in {"", ".", ".."} or "\0" in part for part in parts):
            return False
        arrays = entry["arrays"]
        if (
            not isinstance(arrays, list) or len(arrays) != 3
            or any(not isinstance(row, dict) for row in arrays)
            or {row.get("name") for row in arrays} != {"motion", "shortlist_views", "views"}
        ):
            return False
        with ExitStack() as resources:
            directory = _open_index_directory(index_dir)
            resources.callback(directory.close)
            shard = _open_shard_directory(directory.fileno(), parts, index_dir / token)
            resources.callback(shard.close)
            for row in arrays:
                filename = row["file"]
                if (
                    not isinstance(filename, str) or filename in {"", ".", ".."}
                    or "/" in filename or "\0" in filename
                ):
                    return False
                with closing(_pin_array(shard.fileno(), filename, index_dir / token / filename)) as payload:
                    if payload.checksum() != row["checksum"]:
                        return False
                    array = payload.load()
                    if list(array.shape) != row["shape"] or str(array.dtype) != row["dtype"]:
                        return False
        return True
    except (IndexError, KeyError, OSError, EOFError, ValueError, TypeError, OverflowError):
        return False


def _assert_source_identity(source: Path, expected: dict) -> None:
    try:
        stat = source.stat()
    except OSError as exc:
        raise IndexBatchError(f"source changed during indexing: {expected['relpath']}") from exc
    if (
        stat.st_size != expected["size_bytes"]
        or stat.st_mtime_ns != expected["mtime_ns"]
        or digest(source) != expected["sha256"]
    ):
        raise IndexBatchError(f"source changed during indexing: {expected['relpath']}")


def _rewrite_entry(
    entry: dict, source: Path, library: Path, shard_dir: str, expected: dict
) -> dict:
    return {
        **entry,
        "id": source.stem,
        "source_relpath": source.relative_to(library).as_posix(),
        "display_name": source.name,
        "original_root_hint": str(library),
        "source_path": str(source),
        "size_bytes": expected["size_bytes"],
        "mtime_ns": expected["mtime_ns"],
        "content_digest": expected["sha256"],
        "shard_dir": shard_dir,
    }


def _build_one(
    source: Path, library: Path, index_dir: Path, config: Config, expected: dict
) -> dict:
    _assert_source_identity(source, expected)
    work = Path(tempfile.mkdtemp(prefix=".index-item-", dir=index_dir))
    try:
        one_library = work / "library"
        one_index = work / "index"
        one_library.mkdir()
        staged = one_library / source.name
        try:
            os.link(source, staged)
        except OSError:
            shutil.copy2(source, staged)
        build_index(one_library, one_index, config=config, force=True)
        _assert_source_identity(source, expected)
        if digest(staged) != expected["sha256"]:
            raise IndexBatchError(f"staged source changed during indexing: {expected['relpath']}")
        manifest, _ = load_index(one_index)
        entry = manifest["masters"][0]
        source_shard = one_index / entry["shard_dir"]
        destination_rel = f"masters/{source.stem}/{Path(entry['shard_dir']).name}"
        destination = index_dir / destination_rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            candidate = _rewrite_entry(entry, source, library, destination_rel, expected)
            if not _entry_valid(index_dir, candidate):
                destination = destination.with_name(destination.name + "_" + uuid.uuid4().hex[:8])
                destination_rel = destination.relative_to(index_dir).as_posix()
                os.replace(source_shard, destination)
        else:
            os.replace(source_shard, destination)
        _fsync_dir(destination)
        _fsync_dir(destination.parent)
        return _rewrite_entry(entry, source, library, destination_rel, expected)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _validate_progress_state(index_dir: Path, state: dict, identity: dict) -> tuple[dict, dict]:
    if not isinstance(state, dict) or set(state) != {"version", "identity", "entries", "failures"}:
        raise IndexBatchError("invalid index journal schema")
    if state["version"] != INDEX_PROGRESS_VERSION or state["identity"] != identity:
        raise IndexBatchError("index journal does not match inputs, implementation, or options")
    entries = state["entries"]
    failures = state["failures"]
    if not isinstance(entries, dict) or not isinstance(failures, dict):
        raise IndexBatchError("invalid index journal records")
    inventory = {Path(row["relpath"]).stem: row for row in identity["inventory"]}
    if not set(entries).issubset(inventory) or not set(failures).issubset(inventory):
        raise IndexBatchError("index journal contains a source outside the inventory")
    if set(entries) & set(failures):
        raise IndexBatchError("index journal marks one source both complete and failed")
    entry_ids = []
    library_hint = identity["library"]
    for mid, entry in entries.items():
        item = inventory[mid]
        if (
            not isinstance(entry, dict)
            or entry.get("id") != mid
            or entry.get("source_relpath") != item["relpath"]
            or entry.get("content_digest") != item["sha256"]
            or entry.get("size_bytes") != item["size_bytes"]
            or entry.get("mtime_ns") != item["mtime_ns"]
            or entry.get("original_root_hint") != library_hint
            or entry.get("source_path") != str(Path(library_hint) / item["relpath"])
            or entry.get("display_name") != Path(item["relpath"]).name
        ):
            raise IndexBatchError(f"index journal entry disagrees with inventory: {mid}")
        if not _entry_valid(index_dir, entry):
            raise IndexBatchError(f"resumed shard failed validation: {mid}")
        entry_ids.append(entry["id"])
    if len(entry_ids) != len(set(entry_ids)):
        raise IndexBatchError("index journal contains duplicate master ids")
    for mid, failure in failures.items():
        if (
            not isinstance(failure, dict)
            or set(failure) != {"source_relpath", "type", "message"}
            or failure.get("source_relpath") != inventory[mid]["relpath"]
            or any(not isinstance(failure.get(key), str) or not failure.get(key) for key in failure)
        ):
            raise IndexBatchError(f"invalid index journal failure: {mid}")
    return dict(entries), dict(failures)


def build_index_resumable(
    library: Path,
    index_dir: Path,
    config: Config | None = None,
    *,
    resume: bool = True,
    continue_on_error: bool = False,
    progress=None,
    interrupt_after: int | None = None,
) -> dict:
    """Build isolated master shards, checkpoint each, then atomically publish one manifest."""
    library = Path(library).resolve()
    index_dir = Path(index_dir).resolve()
    config = config or Config()
    if not library.is_dir():
        raise IndexBatchError(f"library not found: {library}")
    files = _discover_master_files(library)
    if not files:
        raise IndexBatchError(f"no *.mp4 masters in {library}")
    seen = set()
    for path in files:
        _validate_source_path(library, path)
        if path.stem in seen:
            raise IndexBatchError(f"duplicate master basename: {path.stem}")
        seen.add(path.stem)

    index_dir.mkdir(parents=True, exist_ok=True)
    identity = _identity(library, files, config, continue_on_error)
    inventory_by_id = {
        Path(item["relpath"]).stem: item for item in identity["inventory"]
    }
    digest_paths = {}
    for item in identity["inventory"]:
        previous = digest_paths.get(item["sha256"])
        if previous is not None:
            raise IndexBatchError(
                f"duplicate source content: {previous} and {item['relpath']}"
            )
        digest_paths[item["sha256"]] = item["relpath"]
    progress_path = index_dir / ".index-progress.json"
    fd = acquire_writer_lock(index_dir)
    # The writer lock is released exactly once in the finally below. Journal
    # reads, validation, per-item work, and publication all run inside it so
    # any raise (including BaseException) cannot leak a held flock.
    try:
        entries = {}
        failures = {}
        if progress_path.exists() and not resume:
            progress_path.unlink()
        if progress_path.exists() and resume:
            try:
                state, _ = json_snapshot(progress_path)
                entries, failures = _validate_progress_state(index_dir, state, identity)
            except (OSError, ValueError) as exc:
                raise IndexBatchError(f"corrupt index journal: {exc}") from exc
            except IndexBatchError:
                raise
        resumed = len(entries)
        processed = 0
        for position, source in enumerate(files, start=1):
            mid = source.stem
            if mid in entries:
                if progress:
                    progress({"event": "resumed", "master": source.relative_to(library).as_posix(), "position": position, "total": len(files)})
                continue
            if progress:
                progress({"event": "indexing", "master": source.relative_to(library).as_posix(), "position": position, "total": len(files)})
            try:
                entries[mid] = _build_one(
                    source, library, index_dir, config, inventory_by_id[mid]
                )
                failures.pop(mid, None)
            except (IndexError, MediaError, OSError, ValueError) as exc:
                failures[mid] = {
                    "source_relpath": source.relative_to(library).as_posix(),
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                if not continue_on_error:
                    raise
            processed += 1
            atomic_json(progress_path, {
                "version": INDEX_PROGRESS_VERSION,
                "identity": identity,
                "entries": entries,
                "failures": failures,
            })
            if interrupt_after is not None and processed >= interrupt_after:
                raise KeyboardInterrupt("injected index interruption")

        for source in files:
            _assert_source_identity(source, inventory_by_id[source.stem])
        if not entries:
            raise IndexBatchError("no valid masters were indexed")
        new_extractor = extractor_config(config.media, config.match)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "algorithm_version": ALGORITHM_VERSION,
            "extractor_config": new_extractor,
            "ffmpeg_version": _tool_version(config.media.ffmpeg),
            "numpy_version": np.__version__,
            "masters": [entries[mid] for mid in sorted(entries)],
            "status": "complete_with_errors" if failures else "complete",
            "failures": [failures[mid] for mid in sorted(failures)],
            "build_receipt": {
                "resume_schema_version": INDEX_PROGRESS_VERSION,
                "resumed_masters": resumed,
                "processed_masters": processed,
                "artifact_config": artifact_config(new_extractor),
                "implementation": _implementation_receipt(),
            },
        }
        _fsync_dir(index_dir / "masters")
        _fsync_dir(index_dir)
        candidate_path = index_dir / ".manifest-candidate.json"
        atomic_json(candidate_path, manifest)
        try:
            _, candidate_masters = load_index(
                index_dir, manifest_name=".manifest-candidate.json"
            )
            for candidate_master in candidate_masters.values():
                candidate_master["load_views"]()
        except (IndexError, OSError, ValueError):
            candidate_path.unlink(missing_ok=True)
            raise
        os.replace(candidate_path, index_dir / "manifest.json")
        _fsync_dir(index_dir)
        if not failures:
            progress_path.unlink(missing_ok=True)
        if progress:
            progress({"event": "complete", "masters": len(entries), "failures": len(failures)})
        return manifest
    finally:
        release_writer_lock(fd)
