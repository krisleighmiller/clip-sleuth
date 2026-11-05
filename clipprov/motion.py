"""Motion traces, motion scoring, and shortlist generation for Clip Provenance."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from clipprov.media import MediaError

FPS = 8
SIDE = 64
GRID = 24
MAX_FRAMES = 20
FLAT_VAR = 12.0
BAR_LUMA = 14
BAR_STD = 3.0
SPEEDS = np.arange(0.968, 1.0321, 0.002)
SHORTLIST_APPEARANCE = 10
SHORTLIST_MOTION = 5
SHORTLIST_VIEWS = 3
VIEWS = [(1.0, 0.5, 0.5), (0.90, 0.5, 0.5), (0.81, 0.5, 0.5)]
VIEWS += [
    (c, px, py)
    for c in (0.90, 0.81)
    for px in (0.0, 0.5, 1.0)
    for py in (0.0, 0.5, 1.0)
    if (px, py) != (0.5, 0.5)
]
ACCEPT_BEST = 0.35
ACCEPT_MARGIN = 0.12


def decode(path: Path | str, vf_extra: str = "", *, ffmpeg: str = "ffmpeg") -> np.ndarray:
    """Decode a video into a (N, SIDE, SIDE) float32 frame array.

    Raises MediaError on missing tool, unreadable media, missing video stream,
    zero decoded frames, or invalid duration.
    """
    import shutil

    resolved = shutil.which(ffmpeg)
    if resolved is None:
        raise MediaError(f"required tool not found on PATH: {ffmpeg}")
    target = Path(path)
    if not target.is_file():
        raise MediaError(f"media file not found: {target}")
    chain = f"fps={FPS},{vf_extra}scale={SIDE}:{SIDE},format=gray"
    try:
        proc = subprocess.run(
            [resolved, "-v", "error", "-i", str(target), "-vf", chain,
             "-f", "rawvideo", "-pix_fmt", "gray", "-"],
            capture_output=True, check=False,
            )
    except OSError as exc:
        raise MediaError(f"failed to run ffmpeg: {exc}") from exc
    if proc.returncode != 0:
        raise MediaError(
            f"ffmpeg decode failed for {target}: {proc.stderr.decode(errors='replace').strip()}"
        )
    buf = np.frombuffer(proc.stdout, dtype=np.uint8)
    n = buf.size // (SIDE * SIDE)
    if n == 0:
        raise MediaError(f"zero decoded frames from {target}")
    return buf[: n * SIDE * SIDE].reshape(n, SIDE, SIDE).astype(np.float32)


def motion(frames: np.ndarray) -> np.ndarray:
    """Motion trace (z-scored, clipped) of mean abs frame differences."""
    if frames.shape[0] < 3:
        return np.zeros(0, dtype=np.float32)
    d = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))
    s = d.std()
    if s <= 1e-6:
        return np.zeros_like(d, dtype=np.float32)
    return np.clip((d - d.mean()) / s, -3.0, 3.0).astype(np.float32)


def motion_score(clip_m: np.ndarray, master_m: np.ndarray, speeds=None) -> float:
    """Peak motion correlation, used only to rank masters."""
    if speeds is None:
        speeds = SPEEDS
    n = clip_m.size
    if n < 4:
        return -1.0
    k = round(1.0 * FPS)
    core = clip_m[k : n - k] if n - 2 * k >= 16 else clip_m
    t = np.arange(core.size) / FPS
    best = -1.0
    for s in speeds[::4]:
        grid = np.arange(0.0, s * t[-1] + 1e-9, 1.0 / FPS)
        if grid.size < 4 or grid.size > master_m.size:
            continue
        from clipprov.appearance import ncc

        cc = ncc(np.interp(grid, s * t, core), master_m)
        if cc is not None and cc.size:
            best = max(best, float(cc.max()))
    return best


def speeds_from_config(match_config) -> np.ndarray:
    import numpy as np

    steps = max(1, int(match_config.speed_steps))
    if steps == 1:
        return np.array([float(match_config.speed_min)], dtype=float)
    return np.linspace(float(match_config.speed_min), float(match_config.speed_max), steps)
