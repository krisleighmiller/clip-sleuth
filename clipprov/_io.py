"""Small shared helpers for content identity and atomic JSON publication."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path


def finite_number(value) -> bool:
    """Return True only if value is a finite JSON number (bool rejected)."""
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, ValueError, TypeError):
        return False


@contextmanager
def _regular_reader(path):
    """Admit a regular inode without waiting on pipes, then keep that inode open."""
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0))
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError(f"not a regular file: {path}")
        with os.fdopen(fd, "rb") as handle:
            fd = None
            yield handle
    finally:
        if fd is not None:
            os.close(fd)


def digest(path):
    h = hashlib.sha256()
    with _regular_reader(path) as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def atomic_text(path, text):
    path = Path(path)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
        dirfd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(dirfd)
        finally:
            os.close(dirfd)
    finally:
        Path(name).unlink(missing_ok=True)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _check_json_depth(raw, path):
    """Keep the input limit stable when the runtime JSON decoder has no depth limit."""
    depth = 0
    quoted = escaped = False
    for byte in raw:
        if quoted:
            if escaped:
                escaped = False
            elif byte == 92:  # backslash
                escaped = True
            elif byte == 34:  # quote
                quoted = False
        elif byte == 34:
            quoted = True
        elif byte in (91, 123):  # [ or {
            depth += 1
            if depth > 1000:
                raise ValueError(f"invalid JSON encoding or nesting: {path}")
        elif byte in (93, 125):  # ] or }
            depth -= 1


def json_snapshot(path):
    """Read JSON in one pass: return (data, exact_sha256) from same bytes."""
    with _regular_reader(path) as handle:
        raw = handle.read()
    _check_json_depth(raw, path)
    try:
        data = json.loads(raw)
    except (UnicodeError, RecursionError) as exc:
        raise ValueError(f"invalid JSON encoding or nesting: {path}") from exc
    return data, hashlib.sha256(raw).hexdigest()
