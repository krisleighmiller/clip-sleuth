"""Persistent index: directory-bundle format with atomic writes."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path

import numpy as np

from clipprov._io import finite_number
from clipprov.appearance import view_of
from clipprov.config import Config, MediaConfig
from clipprov.media import MediaError
from clipprov.media import probe as media_probe
from clipprov.motion import (
    FPS,
    GRID,
    SHORTLIST_VIEWS,
    VIEWS,
    decode,
    motion,
)

# Backward compatibility alias for _finite_number
_finite_number = finite_number

SCHEMA_VERSION = 2
ALGORITHM_VERSION = "1.0.0"


class IndexError(Exception):
    """Raised when an index cannot be read or written."""


class CommitVisibilityUncertain(IndexError):
    """Manifest replacement succeeded, but its directory fsync failed."""


class _ManifestSnapshot(dict):
    """Manifest dict bound to the digest of the exact bytes that produced it.

    The binding digest is exposed only as the read-only ``manifest_sha256``
    attribute. It is deliberately not a mapping key, so it can never leak into
    serialized manifests or dict equality against persisted content.
    """

    __slots__ = ("_snapshot_sha256",)

    def __init__(self, mapping, snapshot_sha256):
        super().__init__(mapping)
        self._snapshot_sha256 = snapshot_sha256

    @property
    def manifest_sha256(self):
        return self._snapshot_sha256


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class _PinnedDescriptor:
    """Own a descriptor so validation and NumPy loading use one inode."""

    def __init__(self, fd: int, display_path: Path):
        self.fd = fd
        self.display_path = display_path

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def fileno(self) -> int:
        if self.fd is None:
            raise OSError("descriptor is closed")
        return self.fd

    def __del__(self):
        self.close()


class _PinnedArray(_PinnedDescriptor):
    def checksum(self) -> str:
        digest = hashlib.sha256()
        offset = 0
        try:
            while chunk := os.pread(self.fileno(), 1024 * 1024, offset):
                digest.update(chunk)
                offset += len(chunk)
        except OSError as exc:
            raise IndexError(f"cannot checksum array {self.display_path}: {exc}") from exc
        return digest.hexdigest()

    def load(self, *, deferred=False):
        try:
            return np.load(
                f"/proc/self/fd/{self.fd}", mmap_mode="r", allow_pickle=False
            )
        except (OSError, ValueError, EOFError, OverflowError, RecursionError) as exc:
            kind = "deferred views" if deferred else "array"
            raise IndexError(f"cannot load {kind} {self.display_path}: {exc}") from exc


def _open_index_directory(index_dir: Path) -> _PinnedDescriptor:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    fd = None
    try:
        fd = os.open(index_dir, flags)
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("not a directory")
    except OSError as exc:
        if fd is not None:
            os.close(fd)
        raise IndexError(f"cannot safely open index directory: {index_dir}") from exc
    return _PinnedDescriptor(fd, index_dir)


def _open_shard_directory(index_fd: int, parts: tuple[str, ...], display_path: Path):
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    current_fd = None
    try:
        current_fd = os.dup(index_fd)
        for part in parts:
            next_fd = os.open(part, flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        return _PinnedDescriptor(current_fd, display_path)
    except (OSError, ValueError) as exc:
        if current_fd is not None:
            os.close(current_fd)
        raise IndexError(f"cannot safely open shard directory: {display_path}") from exc


def _pin_array(shard_fd: int, filename: str, display_path: Path) -> _PinnedArray:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = None
    try:
        fd = os.open(filename, flags, dir_fd=shard_fd)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError("not a regular file")
    except OSError as exc:
        if fd is not None:
            os.close(fd)
        raise IndexError(f"cannot safely open array file: {display_path}") from exc
    return _PinnedArray(fd, display_path)


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


class _PinnedSource:
    def __init__(self, path, fd, source_stat, digest, directory_fds, directory_stats):
        self.path = path
        self.fd = fd
        self.source_stat = source_stat
        self.digest = digest
        self.directory_fds = directory_fds
        self.directory_stats = directory_stats

    def close(self):
        os.close(self.fd)
        for directory_fd in self.directory_fds:
            os.close(directory_fd)

    def unchanged(self):
        if _stat_identity(os.fstat(self.fd)) != _stat_identity(self.source_stat):
            return False
        return all(
            _stat_identity(os.fstat(directory_fd)) == _stat_identity(directory_stat)
            for directory_fd, directory_stat in zip(
                self.directory_fds, self.directory_stats, strict=True
            )
        )

    def stage(self, staging_dir):
        stage_path = staging_dir / self.path.name
        stage_fd = os.open(
            stage_path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        digest = hashlib.sha256()
        try:
            before = os.fstat(self.fd)
            os.lseek(self.fd, 0, os.SEEK_SET)
            while chunk := os.read(self.fd, 1024 * 1024):
                digest.update(chunk)
                remaining = memoryview(chunk)
                while remaining:
                    remaining = remaining[os.write(stage_fd, remaining) :]
            os.fsync(stage_fd)
            staged_stat = os.fstat(stage_fd)
            after = os.fstat(self.fd)
            if (
                _stat_identity(before) != _stat_identity(self.source_stat)
                or _stat_identity(after) != _stat_identity(self.source_stat)
                or digest.hexdigest() != self.digest
            ):
                raise IndexError(f"source changed while staging: {self.path}")
            if (
                not stat.S_ISREG(staged_stat.st_mode)
                or staged_stat.st_size != self.source_stat.st_size
            ):
                raise IndexError(f"invalid staged source: {self.path}")
            os.fchmod(stage_fd, 0o400)
            return stage_path
        except OSError as exc:
            stage_path.unlink(missing_ok=True)
            raise IndexError(f"cannot stage source: {self.path}") from exc
        except Exception:
            stage_path.unlink(missing_ok=True)
            raise
        finally:
            os.close(stage_fd)


def _open_source_without_symlinks(library: Path, path: Path):
    relative = path.relative_to(library)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(library, directory_flags | nofollow)
    directory_fds = [directory_fd]
    try:
        for part in relative.parts[:-1]:
            next_fd = os.open(part, directory_flags | nofollow, dir_fd=directory_fd)
            directory_fd = next_fd
            directory_fds.append(directory_fd)
        source_fd = os.open(
            relative.parts[-1],
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow | getattr(os, "O_NONBLOCK", 0),
            dir_fd=directory_fd,
        )
        return source_fd, directory_fds
    except Exception:
        for opened_fd in directory_fds:
            os.close(opened_fd)
        raise


def _pin_source(library: Path, path: Path) -> _PinnedSource:
    try:
        fd, directory_fds = _open_source_without_symlinks(library, path)
    except (OSError, ValueError) as exc:
        raise IndexError(f"cannot safely open source without symlinks: {path}") from exc
    try:
        directory_stats = [os.fstat(directory_fd) for directory_fd in directory_fds]
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise IndexError(f"source is not a regular file: {path}")
        digest = hashlib.sha256()
        while chunk := os.read(fd, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(fd)
    except OSError as exc:
        os.close(fd)
        for directory_fd in directory_fds:
            os.close(directory_fd)
        raise IndexError(f"cannot safely hash source: {path}") from exc
    except Exception:
        os.close(fd)
        for directory_fd in directory_fds:
            os.close(directory_fd)
        raise
    if _stat_identity(before) != _stat_identity(after):
        os.close(fd)
        for directory_fd in directory_fds:
            os.close(directory_fd)
        raise IndexError(f"source changed while hashing: {path}")
    return _PinnedSource(path, fd, after, digest.hexdigest(), directory_fds, directory_stats)


def _source_snapshot(library: Path, path: Path) -> tuple[os.stat_result, str]:
    pinned = _pin_source(library, path)
    try:
        return pinned.source_stat, pinned.digest
    finally:
        pinned.close()


def _staged_source_snapshot(path: Path):
    """Yield a private verified copy and its digest, closing every resource on exit."""
    from contextlib import contextmanager

    @contextmanager
    def _manager():
        source = Path(path).resolve()
        pinned = _pin_source(source.parent, source)
        try:
            with tempfile.TemporaryDirectory(prefix=".clipprov-source-") as directory:
                yield pinned.stage(Path(directory)), pinned.digest
        finally:
            pinned.close()

    return _manager()




def _validate_source_path(library: Path, path: Path) -> None:
    try:
        relative = path.relative_to(library)
    except ValueError as exc:
        raise IndexError(f"source is outside library root: {path}") from exc
    current = library
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise IndexError(f"library contains symlink: {current}")
    try:
        path.resolve(strict=True).relative_to(library.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise IndexError(f"source resolves outside library root: {path}") from exc


def _discover_master_files(library: Path) -> list[Path]:
    if library.is_symlink():
        raise IndexError(f"library contains symlink: {library}")
    entries = sorted(library.rglob("*"))
    for entry in entries:
        if entry.is_symlink():
            raise IndexError(f"library contains symlink: {entry}")
    masters = [entry for entry in entries if entry.suffix == ".mp4" and entry.is_file()]
    for master in masters:
        _validate_source_path(library, master)
    return masters


def _fsync_file(path: Path):
    with path.open("rb") as f:
        os.fsync(f.fileno())


def _fsync_dir(path: Path):
    fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _path_has_symlink(root: Path, relative: Path) -> bool:
    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            return True
    return False


def acquire_writer_lock(index_dir: Path):
    """Acquire an exclusive writer lock.

    Platform boundary: uses fcntl (POSIX/Linux only). On non-POSIX platforms
    this lock is unavailable; wrap or replace for cross-platform support.
    """
    index_dir.mkdir(parents=True, exist_ok=True)
    lock = index_dir / "lock"
    fd = os.open(str(lock), os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def release_writer_lock(fd):
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _canonical(obj):
    return json.loads(json.dumps(obj, sort_keys=True, default=str))


def extractor_config(media_config, match_config):
    return _canonical(
        {
            "sample_rate_fps": FPS,
            "max_frames": media_config.max_frames,
            "speed_min": match_config.speed_min,
            "speed_max": match_config.speed_max,
            "speed_steps": match_config.speed_steps,
            "score_floor": match_config.score_floor,
            "margin": match_config.margin,
            "offset_tolerance_s": match_config.offset_tolerance_s,
            "views": list(VIEWS),
            "shortlist_views": SHORTLIST_VIEWS,
        }
    )


def artifact_config(ec):
    return {
        key: value
        for key, value in ec.items()
        if key
        not in {
            "score_floor",
            "margin",
            "offset_tolerance_s",
            "speed_min",
            "speed_max",
            "speed_steps",
            "max_frames",
        }
    }


_EXTRACTOR_FIELDS = {
    "sample_rate_fps",
    "max_frames",
    "speed_min",
    "speed_max",
    "speed_steps",
    "score_floor",
    "margin",
    "offset_tolerance_s",
    "views",
    "shortlist_views",
}


def _validate_extractor_config(value):
    if not isinstance(value, dict) or not _EXTRACTOR_FIELDS.issubset(value):
        raise IndexError("extractor_config missing required fields")
    if not _finite_number(value["sample_rate_fps"]) or value["sample_rate_fps"] != FPS:
        raise IndexError("invalid extractor_config sample_rate_fps")
    if type(value["max_frames"]) is not int or value["max_frames"] < 4:
        raise IndexError("invalid extractor_config max_frames")
    for field in ("speed_min", "speed_max", "offset_tolerance_s", "score_floor", "margin"):
        if not _finite_number(value[field]):
            raise IndexError(f"invalid extractor_config {field}")
    if value["speed_min"] <= 0 or value["speed_min"] >= value["speed_max"]:
        raise IndexError("invalid extractor_config speed range")
    if type(value["speed_steps"]) is not int or value["speed_steps"] <= 0:
        raise IndexError("invalid extractor_config speed_steps")
    if value["offset_tolerance_s"] < 0:
        raise IndexError("invalid extractor_config offset_tolerance_s")
    if value["margin"] < 0:
        raise IndexError("invalid extractor_config margin")
    if value["views"] != _canonical(list(VIEWS)):
        raise IndexError("invalid extractor_config views")
    if type(value["shortlist_views"]) is not int or value["shortlist_views"] != SHORTLIST_VIEWS:
        raise IndexError("invalid extractor_config shortlist_views")
    return value


def config_from_manifest(manifest):
    from clipprov.config import Config, MatchConfig, MediaConfig

    if not isinstance(manifest, dict):
        raise IndexError("manifest must be an object")
    ec = _validate_extractor_config(manifest.get("extractor_config"))
    media_cfg = MediaConfig(max_frames=ec["max_frames"])
    match_cfg = MatchConfig(
        speed_min=ec["speed_min"],
        speed_max=ec["speed_max"],
        speed_steps=ec["speed_steps"],
        offset_tolerance_s=ec["offset_tolerance_s"],
        score_floor=ec["score_floor"],
        margin=ec["margin"],
    )
    return Config(media=media_cfg, match=match_cfg)


def _tool_version(tool: str) -> str:
    import subprocess

    try:
        r = subprocess.run([tool, "-version"], capture_output=True, text=True, check=False)
        first = (r.stdout or r.stderr or "").splitlines()
        return first[0] if first else "unknown"
    except OSError:
        return "unknown"


def _build_shard(master_path: Path, gen_dir: Path, media_config: MediaConfig):
    media_probe(master_path, media_config)
    frames = decode(master_path, "", ffmpeg=media_config.ffmpeg)
    if frames.shape[0] < 4:
        raise MediaError(
            f"master decodes to {frames.shape[0]} frames; at least 4 are required"
        )
    m = motion(frames)
    views = [view_of(frames, v) for v in VIEWS]
    views_arr = np.stack(views, axis=0).astype(np.float32)
    shortlist_arr = views_arr[:SHORTLIST_VIEWS]

    gen_dir.mkdir(parents=True, exist_ok=True)
    motion_path = gen_dir / "motion.npy"
    views_path = gen_dir / "views.npy"
    shortlist_path = gen_dir / "shortlist_views.npy"
    meta_path = gen_dir / "metadata.json"

    np.save(motion_path, m.astype(np.float32))
    _fsync_file(motion_path)
    np.save(views_path, views_arr)
    _fsync_file(views_path)
    np.save(shortlist_path, shortlist_arr)
    _fsync_file(shortlist_path)

    meta = {
        "frame_count": int(frames.shape[0]),
        "height": int(frames.shape[1]),
        "width": int(frames.shape[2]),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    _fsync_file(meta_path)
    _fsync_dir(gen_dir)


def build_index(library: Path, index_dir: Path, config: Config | None = None, force: bool = False):
    library = Path(library)
    index_dir = Path(index_dir)
    if config is None:
        config = Config()
    media_config = config.media
    match_config = config.match

    if not library.is_dir():
        raise IndexError(f"library not found: {library}")

    fd = acquire_writer_lock(index_dir)
    pinned_sources = {}
    staging_dir = None
    created_generations = set()
    manifest_replaced = False
    try:
        index_dir.mkdir(parents=True, exist_ok=True)
        masters_dir = index_dir / "masters"
        masters_dir.mkdir(parents=True, exist_ok=True)

        master_files = _discover_master_files(library)
        if not master_files:
            raise IndexError(f"no *.mp4 masters in {library}")

        # reject duplicate basenames
        seen = {}
        for mp in master_files:
            if mp.stem in seen:
                raise IndexError(f"duplicate master basename: {mp.stem} ({seen[mp.stem]} and {mp})")
            seen[mp.stem] = mp

        paths_by_digest = {}
        for mp in master_files:
            _validate_source_path(library, mp)
            pinned = _pin_source(library, mp)
            pinned_sources[mp] = pinned
            digest = pinned.digest
            source_relpath = str(mp.relative_to(library))
            previous_relpath = paths_by_digest.get(digest)
            if previous_relpath is not None:
                raise IndexError(
                    "duplicate source content is not allowed: "
                    f"{previous_relpath} and {source_relpath} share SHA-256 {digest}"
                )
            paths_by_digest[digest] = source_relpath

        manifest_path = index_dir / "manifest.json"
        existing = {}
        if manifest_path.exists():
            try:
                existing, loaded_existing = load_index(index_dir)
                # Opening an index deliberately defers the heavy view payload,
                # but artifact reuse requires every checksum to match.
                for loaded_master in loaded_existing.values():
                    loaded_master["load_views"]()
            except (IndexError, OSError, ValueError):
                existing = {}
        existing_by_id = (
            {m["id"]: m for m in existing.get("masters", []) if isinstance(m, dict) and "id" in m}
            if isinstance(existing.get("masters"), list)
            else {}
        )

        existing_algo = existing.get("algorithm_version")
        existing_ffmpeg = existing.get("ffmpeg_version")
        existing_extractor = existing.get("extractor_config")
        new_extractor = extractor_config(media_config, match_config)
        new_artifact = artifact_config(new_extractor)
        existing_artifact = (
            artifact_config(existing_extractor) if isinstance(existing_extractor, dict) else None
        )
        new_ffmpeg = _tool_version(media_config.ffmpeg)

        can_reuse_global = (
            (not force)
            and existing_algo == ALGORITHM_VERSION
            and existing_ffmpeg == new_ffmpeg
            and existing_artifact == new_artifact
        )

        referenced = set()
        master_entries = []

        for mp in master_files:
            mid = mp.stem
            pinned = pinned_sources[mp]
            source_stat, digest = pinned.source_stat, pinned.digest
            size = source_stat.st_size
            mtime_ns = source_stat.st_mtime_ns
            import hashlib as _hl
            import json as _json

            artifact_key = {
                "source_digest": digest,
                "schema_version": SCHEMA_VERSION,
                "algorithm_version": ALGORITHM_VERSION,
                "ffmpeg_version": new_ffmpeg,
                "numpy_version": np.__version__,
                "artifact_config": new_artifact,
                "decoder": media_config.ffmpeg,
                "dtype_contract": "float32",
            }
            gen_id = _hl.sha256(_json.dumps(artifact_key, sort_keys=True).encode()).hexdigest()[:16]
            gen_dir_rel = f"masters/{mid}/{gen_id}"
            gen_dir = index_dir / gen_dir_rel

            prev = existing_by_id.get(mid)
            unchanged = (
                can_reuse_global
                and prev is not None
                and prev.get("size_bytes") == size
                and prev.get("mtime_ns") == mtime_ns
                and prev.get("content_digest") == digest
                and prev.get("shard_dir") == gen_dir_rel
            )

            if unchanged:
                gen_dir_rel = prev["shard_dir"]
                gen_dir = index_dir / gen_dir_rel
            else:
                # Generations are immutable: construction always chooses a fresh path.
                if gen_dir.exists():
                    i = 2
                    while (index_dir / f"{gen_dir_rel}_{i}").exists():
                        i += 1
                    gen_dir_rel = f"{gen_dir_rel}_{i}"
                    gen_dir = index_dir / gen_dir_rel
                created_generations.add(gen_dir)
                if staging_dir is None:
                    staging_dir = Path(tempfile.mkdtemp(prefix=".staging-", dir=index_dir))
                staged_source = pinned.stage(staging_dir)
                try:
                    _build_shard(staged_source, gen_dir, media_config)
                finally:
                    staged_source.unlink(missing_ok=True)
                try:
                    _validate_source_path(library, mp)
                    current_stat, current_digest = _source_snapshot(library, mp)
                except IndexError as exc:
                    shutil.rmtree(gen_dir, ignore_errors=True)
                    raise IndexError(f"source changed during extraction: {mp}") from exc
                if (
                    not pinned.unchanged()
                    or _stat_identity(current_stat) != _stat_identity(source_stat)
                    or current_digest != digest
                ):
                    shutil.rmtree(gen_dir, ignore_errors=True)
                    raise IndexError(f"source changed during extraction: {mp}")
                motion_path = gen_dir / "motion.npy"
                views_path = gen_dir / "views.npy"
                shortlist_path = gen_dir / "shortlist_views.npy"
                for artifact in (
                    motion_path,
                    shortlist_path,
                    views_path,
                    gen_dir / "metadata.json",
                ):
                    if artifact.is_file():
                        _fsync_file(artifact)
                _fsync_dir(gen_dir)
                _fsync_dir(gen_dir.parent)
                if (
                    not motion_path.exists()
                    or not views_path.exists()
                    or not shortlist_path.exists()
                ):
                    raise IndexError(f"shard build failed for {mid}")

            referenced.add(gen_dir_rel)

            motion_path = gen_dir / "motion.npy"
            views_path = gen_dir / "views.npy"
            shortlist_path = gen_dir / "shortlist_views.npy"
            arrays = [
                {
                    "name": "motion",
                    "file": "motion.npy",
                    "shape": list(np.load(motion_path, mmap_mode="r").shape),
                    "dtype": "float32",
                    "checksum": _sha256_file(motion_path),
                },
                {
                    "name": "shortlist_views",
                    "file": "shortlist_views.npy",
                    "shape": list(np.load(shortlist_path, mmap_mode="r").shape),
                    "dtype": "float32",
                    "checksum": _sha256_file(shortlist_path),
                },
                {
                    "name": "views",
                    "file": "views.npy",
                    "shape": list(np.load(views_path, mmap_mode="r").shape),
                    "dtype": "float32",
                    "checksum": _sha256_file(views_path),
                },
            ]

            entry = {
                "id": mid,
                "source_relpath": str(mp.relative_to(library)),
                "display_name": mp.name,
                "original_root_hint": str(library),
                "source_path": str(mp),
                "size_bytes": size,
                "mtime_ns": mtime_ns,
                "content_digest": digest,
                "shard_dir": gen_dir_rel,
                "arrays": arrays,
            }
            master_entries.append(entry)

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "algorithm_version": ALGORITHM_VERSION,
            "extractor_config": extractor_config(media_config, match_config),
            "ffmpeg_version": _tool_version(media_config.ffmpeg),
            "numpy_version": np.__version__,
            "masters": master_entries,
            "status": "complete",
        }

        # Persist the hierarchy leading to every completed generation before
        # publishing a manifest that names those generations.
        _fsync_dir(masters_dir)
        _fsync_dir(index_dir)
        _fsync_dir(index_dir.parent)

        t_fd, tmp_name = tempfile.mkstemp(dir=str(index_dir), suffix=".manifest.tmp")
        tmp_manifest = Path(tmp_name)
        try:
            with os.fdopen(t_fd, "w") as f:
                f.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
                f.flush()
                os.fsync(f.fileno())
            _fsync_file(tmp_manifest)
            os.replace(tmp_manifest, manifest_path)
            manifest_replaced = True
            try:
                _fsync_dir(index_dir)
            except OSError as exc:
                raise CommitVisibilityUncertain(
                    "manifest may be visible but not durable; reload the index before retrying"
                ) from exc
        except Exception:
            try:
                tmp_manifest.unlink(missing_ok=True)
            except OSError:
                pass
            raise

        # Retain prior generations for reader safety; no immediate GC.
    finally:
        if not manifest_replaced:
            for generation in created_generations:
                shutil.rmtree(generation, ignore_errors=True)
        for pinned in pinned_sources.values():
            pinned.close()
        if staging_dir is not None:
            shutil.rmtree(staging_dir, ignore_errors=True)
        release_writer_lock(fd)


def _read_manifest(index_dir, directory_fd=None, manifest_name="manifest.json"):
    if manifest_name not in {"manifest.json", ".manifest-candidate.json"}:
        raise IndexError("unsafe manifest name")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    owns_directory_fd = directory_fd is None
    if owns_directory_fd:
        try:
            directory_fd = os.open(index_dir, directory_flags | nofollow)
        except OSError as exc:
            raise IndexError(f"cannot safely open index directory: {index_dir}") from exc
    try:
        try:
            manifest_fd = os.open(
                manifest_name,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow | getattr(os, "O_NONBLOCK", 0),
                dir_fd=directory_fd,
            )
        except FileNotFoundError as exc:
            raise IndexError(f"no manifest in {index_dir}") from exc
        except OSError as exc:
            raise IndexError(f"cannot safely open manifest in {index_dir}") from exc
        try:
            if not stat.S_ISREG(os.fstat(manifest_fd).st_mode):
                raise IndexError(f"manifest is not a regular file in {index_dir}")
            with os.fdopen(manifest_fd, "rb") as manifest_file:
                manifest_fd = None
                raw = manifest_file.read()
            decoded = json.loads(raw)
        except (ValueError, RecursionError) as exc:
            raise IndexError(f"corrupt manifest: {exc}") from exc
        except OSError as exc:
            raise IndexError(f"cannot read manifest in {index_dir}") from exc
        finally:
            if manifest_fd is not None:
                os.close(manifest_fd)
        if isinstance(decoded, dict):
            # Bind the parsed object to the digest of the exact bytes consumed.
            return _ManifestSnapshot(decoded, hashlib.sha256(raw).hexdigest())
        return decoded
    finally:
        if owns_directory_fd:
            os.close(directory_fd)


def load_index(index_dir: Path, *, manifest_name: str = "manifest.json"):
    index_dir = Path(index_dir)
    index_directory = _open_index_directory(index_dir)
    manifest = _read_manifest(index_dir, index_directory.fd, manifest_name)
    if not isinstance(manifest, dict):
        raise IndexError("manifest must be an object")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise IndexError(f"unsupported schema version: {manifest.get('schema_version')}")
    for key in ("algorithm_version", "ffmpeg_version", "numpy_version"):
        value = manifest.get(key)
        if not isinstance(value, str) or not value.strip():
            raise IndexError(f"invalid {key}")
    if manifest.get("algorithm_version") != ALGORITHM_VERSION:
        raise IndexError(f"unsupported algorithm version: {manifest.get('algorithm_version')}")
    status = manifest.get("status")
    if not isinstance(status, str) or status not in {"complete", "complete_with_errors"}:
        raise IndexError("index incomplete")
    failures = manifest.get("failures", [])
    if not isinstance(failures, list):
        raise IndexError("invalid index failures")
    if status == "complete_with_errors" and not failures:
        raise IndexError("complete_with_errors index must report failures")
    for failure in failures:
        if (
            not isinstance(failure, dict)
            or set(failure) != {"source_relpath", "type", "message"}
            or any(not isinstance(failure[key], str) or not failure[key] for key in failure)
        ):
            raise IndexError("invalid index failure record")
    if not isinstance(manifest.get("masters"), list) or not manifest["masters"]:
        raise IndexError("master list must be nonempty")
    if manifest.get("extractor_config") is None:
        raise IndexError("missing extractor_config")
    _validate_extractor_config(manifest.get("extractor_config"))
    masters = {}
    digests = {}
    for m in manifest.get("masters", []):
        if not isinstance(m, dict):
            raise IndexError("master entry not an object")
        mid = m.get("id")
        if not isinstance(mid, str) or not mid:
            raise IndexError("missing master id")
        if mid in masters:
            raise IndexError(f"duplicate master id: {mid}")
        for sf in (
            "source_relpath",
            "display_name",
            "original_root_hint",
            "content_digest",
            "size_bytes",
            "mtime_ns",
        ):
            if m.get(sf) is None:
                raise IndexError(f"missing {sf} for {mid}")
        if not isinstance(m.get("source_relpath"), str):
            raise IndexError(f"invalid source_relpath for {mid}")
        source_rel = Path(m["source_relpath"])
        if (
            not m["source_relpath"]
            or source_rel.is_absolute()
            or ".." in source_rel.parts
            or "\\" in m["source_relpath"]
        ):
            raise IndexError(f"unsafe source_relpath for {mid}")
        if not isinstance(m.get("display_name"), str) or m["display_name"] != source_rel.name:
            raise IndexError(f"invalid display_name for {mid}")
        if mid != source_rel.stem:
            raise IndexError(f"invalid source identity for {mid}")
        if (
            not isinstance(m.get("original_root_hint"), str)
            or not Path(m["original_root_hint"]).is_absolute()
        ):
            raise IndexError(f"invalid original_root_hint for {mid}")
        if not isinstance(m.get("content_digest"), str) or not re.fullmatch(
            r"[0-9a-f]{64}", m["content_digest"]
        ):
            raise IndexError(f"invalid content_digest for {mid}")
        previous_mid = digests.get(m["content_digest"])
        if previous_mid is not None:
            raise IndexError(
                f"duplicate content_digest for {previous_mid} and {mid}: {m['content_digest']}"
            )
        digests[m["content_digest"]] = mid
        if type(m.get("size_bytes")) is not int or m["size_bytes"] < 0:
            raise IndexError(f"invalid size_bytes for {mid}")
        if type(m.get("mtime_ns")) is not int or m["mtime_ns"] < 0:
            raise IndexError(f"invalid mtime_ns for {mid}")
        shard_rel = m.get("shard_dir")
        if not isinstance(shard_rel, str) or not shard_rel:
            raise IndexError(f"missing shard_dir for {mid}")
        shard_parts = Path(shard_rel).parts
        if (
            os.path.isabs(shard_rel)
            or "\\" in shard_rel
            or len(shard_parts) != 3
            or shard_parts[:2] != ("masters", mid)
        ):
            raise IndexError(f"unsafe shard path for {mid}: {shard_rel}")
        shard = index_dir / shard_rel
        shard_directory = _open_shard_directory(index_directory.fileno(), shard_parts, shard)
        arrays = m.get("arrays")
        if not isinstance(arrays, list):
            raise IndexError(f"missing arrays for {mid}")
        if any(not isinstance(a, dict) for a in arrays):
            raise IndexError(f"invalid array entry for {mid}")
        for arr in arrays:
            if not isinstance(arr.get("file"), str) or not arr["file"]:
                raise IndexError(f"invalid array file for {mid}")
        motion_entry = next((a for a in arrays if a.get("name") == "motion"), None)
        shortlist_entry = next((a for a in arrays if a.get("name") == "shortlist_views"), None)
        views_entry = next((a for a in arrays if a.get("name") == "views"), None)
        if motion_entry is None or shortlist_entry is None or views_entry is None:
            raise IndexError(f"missing required arrays for {mid}")
        views_path = shard / views_entry["file"]

        arrays = m.get("arrays")
        if not isinstance(arrays, list):
            raise IndexError(f"missing arrays for {mid}")
        names = [a.get("name") for a in arrays if isinstance(a, dict)]
        if set(names) != {"motion", "shortlist_views", "views"} or len(names) != 3:
            raise IndexError(f"missing required arrays for {mid}")
        seen_names = set()
        expected_files = {
            "motion": "motion.npy",
            "shortlist_views": "shortlist_views.npy",
            "views": "views.npy",
        }
        pinned_arrays = {
            name: _pin_array(shard_directory.fileno(), filename, shard / filename)
            for name, filename in expected_files.items()
        }
        shard_directory.close()
        for arr in arrays:
            if not isinstance(arr, dict):
                raise IndexError(f"invalid array entry for {mid}")
            aname = arr.get("name")
            if not isinstance(aname, str) or aname in seen_names:
                raise IndexError(f"duplicate/invalid array name for {mid}")
            seen_names.add(aname)
            fname = arr.get("file")
            if not isinstance(fname, str) or fname != expected_files.get(aname):
                raise IndexError(f"invalid array file for {mid}")
            if os.path.isabs(fname) or ".." in Path(fname).parts:
                raise IndexError(f"unsafe array path for {mid}: {fname}")
            for req in ("checksum", "shape", "dtype"):
                if arr.get(req) is None:
                    raise IndexError(f"missing {req} for {mid}/{aname}")
            if not isinstance(arr.get("checksum"), str):
                raise IndexError(f"invalid checksum for {mid}/{aname}")
            if not re.fullmatch(r"[0-9a-f]{64}", arr["checksum"]):
                raise IndexError(f"invalid checksum format for {mid}/{aname}")
            if not isinstance(arr.get("shape"), list):
                raise IndexError(f"invalid shape for {mid}/{aname}")
            if not isinstance(arr.get("dtype"), str):
                raise IndexError(f"invalid dtype for {mid}/{aname}")
            fpath = shard / fname
            if arr["dtype"] != "float32":
                raise IndexError(f"declared dtype must be float32 for {fpath}")
            shape = arr["shape"]
            if any(type(dimension) is not int for dimension in shape):
                raise IndexError(f"invalid shape for {mid}/{aname}")
            if aname == "motion" and (len(shape) != 1 or not 0 <= shape[0] <= 1000):
                raise IndexError(f"invalid motion dimensions: {fpath}")
            if aname == "shortlist_views" and (
                len(shape) != 3
                or shape[0] != SHORTLIST_VIEWS
                or not 1 <= shape[1] <= 1000
                or shape[2] != GRID * GRID
            ):
                raise IndexError(f"invalid shortlist dimensions: {fpath}")
            if aname == "views" and (
                len(shape) != 3
                or shape[0] != len(VIEWS)
                or not 1 <= shape[1] <= 1000
                or shape[2] != GRID * GRID
            ):
                raise IndexError(f"invalid views shape/dimensions: {fpath}")
            if aname != "views":
                if arr.get("checksum") and pinned_arrays[aname].checksum() != arr["checksum"]:
                    raise IndexError(f"checksum mismatch: {fpath}")
                arr_np = pinned_arrays[aname].load()
                if str(arr_np.dtype) != "float32":
                    raise IndexError(f"expected float32, got {arr_np.dtype} for {fpath}")

                if aname == "motion" and arr_np.ndim != 1:
                    raise IndexError(f"motion must be 1-D: {fpath}")
                if aname == "shortlist_views" and arr_np.ndim != 3:
                    raise IndexError(f"views must be 3-D: {fpath}")
                if aname == "shortlist_views" and (
                    arr_np.shape[0] != SHORTLIST_VIEWS
                    or not 1 <= arr_np.shape[1] <= 1000
                    or arr_np.shape[2] != GRID * GRID
                ):
                    raise IndexError(f"invalid shortlist dimensions: {fpath}")
                if aname == "motion" and arr_np.size > 1000:
                    raise IndexError(f"motion dimensions too large: {fpath}")
                if not np.isfinite(arr_np).all():
                    raise IndexError(f"nonfinite array: {fpath}")
                if list(arr_np.shape) != list(arr["shape"]):
                    raise IndexError(f"shape mismatch for {fpath}")
                if str(arr_np.dtype) != str(arr["dtype"]):
                    raise IndexError(f"dtype mismatch for {fpath}")
        motion_frames = motion_entry["shape"][0]
        shortlist_frames = shortlist_entry["shape"][1]
        views_frames = views_entry["shape"][1]
        if motion_frames + 1 != shortlist_frames or shortlist_frames != views_frames:
            raise IndexError(f"inconsistent array frame count for {mid}")

        motion_arr = pinned_arrays["motion"].load()
        shortlist_arr = pinned_arrays["shortlist_views"].load()
        pinned_arrays["motion"].close()
        pinned_arrays["shortlist_views"].close()
        views_checksum = None
        for arr in arrays:
            if arr.get("name") == "views":
                views_checksum = arr.get("checksum")

        _views_shape = views_entry.get("shape")
        _views_dtype = views_entry.get("dtype")

        def _load_views(
            pinned=pinned_arrays["views"],
            sp=views_path,
            cs=views_checksum,
            declared_shape=_views_shape,
            declared_dtype=_views_dtype,
        ):
            try:
                checksum = pinned.checksum() if cs is not None else None
            except OSError as exc:
                raise IndexError(f"cannot checksum deferred views {sp}: {exc}") from exc
            if cs is not None and checksum != cs:
                raise IndexError(f"checksum mismatch: {sp}")
            arr_np = pinned.load(deferred=True)
            if declared_shape is not None and list(arr_np.shape) != list(declared_shape):
                raise IndexError(f"shape mismatch: {sp}")
            if declared_dtype is not None and str(arr_np.dtype) != str(declared_dtype):
                raise IndexError(f"dtype mismatch: {sp}")
            if str(arr_np.dtype) != "float32":
                raise IndexError(f"expected float32: {sp}")
            if arr_np.ndim != 3:
                raise IndexError(f"views must be 3-D: {sp}")
            if (
                arr_np.shape[0] != len(VIEWS)
                or not 1 <= arr_np.shape[1] <= 1000
                or arr_np.shape[2] != GRID * GRID
            ):
                raise IndexError(f"invalid views dimensions: {sp}")
            if not np.isfinite(arr_np).all():
                raise IndexError(f"nonfinite array: {sp}")
            return arr_np

        masters[mid] = {
            "entry": m,
            "motion": motion_arr,
            "shortlist_views": shortlist_arr,
            "views_path": views_path,
            "views_checksum": views_checksum,
            "load_views": _load_views,
            "shard_dir": shard,
        }
    index_directory.close()
    return manifest, masters
