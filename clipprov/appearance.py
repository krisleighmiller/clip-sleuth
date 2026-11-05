"""Appearance: bar detection/cropping, reframing views, grids, and comparison."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from clipprov.media import MediaError
from clipprov.motion import BAR_LUMA, BAR_STD, FLAT_VAR, FPS, GRID, MAX_FRAMES, SIDE, SPEEDS, VIEWS


def bars_crop(path: Path | str, *, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> str:
    """Return an ffmpeg crop filter fragment (or '' if no bars / failure)."""
    ffmpeg_bin = shutil.which(ffmpeg)
    ffprobe_bin = shutil.which(ffprobe)
    if ffmpeg_bin is None or ffprobe_bin is None:
        raise MediaError(f"required tool not found on PATH: {ffmpeg}/{ffprobe}")
    target = Path(path)
    if not target.is_file():
        raise MediaError(f"media file not found: {target}")

    try:
        wh_proc = subprocess.run(
            [ffprobe_bin, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(target)],
            capture_output=True, text=True, check=False,
        )
    except OSError as exc:
        raise MediaError(f"failed to run ffprobe: {exc}") from exc
    if wh_proc.returncode != 0:
        raise MediaError(f"ffprobe failed for {target}: {wh_proc.stderr.strip()}")
    wh = wh_proc.stdout.strip()
    try:
        w, h = (int(v) for v in wh.split("x"))
    except ValueError as exc:
        raise MediaError(f"invalid ffprobe dimensions for {target}: {wh!r}") from exc

    try:
        proc = subprocess.run(
            [ffmpeg_bin, "-v", "error", "-i", str(target), "-vf", "fps=2,format=gray",
             "-f", "rawvideo", "-pix_fmt", "gray", "-"],
            capture_output=True, check=False,
        )
    except OSError as exc:
        raise MediaError(f"failed to run ffmpeg: {exc}") from exc
    if proc.returncode != 0:
        raise MediaError(f"ffmpeg failed for {target}: {proc.stderr.decode(errors='replace').strip()}")
    n = len(proc.stdout) // (w * h)
    if n < 2:
        return ""
    f = np.frombuffer(proc.stdout[: n * w * h], dtype=np.uint8).reshape(n, h, w)

    rmed = np.median(f, axis=2)
    cmed = np.median(f, axis=1)
    row_bar = (rmed.max(axis=0) < BAR_LUMA) & (rmed.std(axis=0) < BAR_STD)
    col_bar = (cmed.max(axis=0) < BAR_LUMA) & (cmed.std(axis=0) < BAR_STD)

    def run(v):
        i = 0
        while i < v.size and v[i]:
            i += 1
        return i

    top = run(row_bar)
    bottom = run(row_bar[::-1])
    left = run(col_bar)
    right = run(col_bar[::-1])
    if top + bottom > 0.45 * h or left + right > 0.45 * w:
        return ""
    nw = (w - left - right) // 2 * 2
    nh = (h - top - bottom) // 2 * 2
    if nw < 16 or nh < 16 or (nw >= w and nh >= h):
        return ""
    return f"crop={nw}:{nh}:{left}:{top},"


def _norm(g: np.ndarray) -> np.ndarray:
    g = g.reshape(g.shape[0], -1)
    g = g - g.mean(axis=1, keepdims=True)
    n = np.sqrt((g * g).sum(axis=1, keepdims=True))
    n[n < 1e-6] = 1.0
    return (g / n).astype(np.float32)


def view_of(frames: np.ndarray, spec) -> np.ndarray:
    c, px, py = spec
    w = max(GRID, round(c * SIDE))
    x0 = round(px * (SIDE - w))
    y0 = round(py * (SIDE - w))
    sub = frames[:, y0 : y0 + w, x0 : x0 + w]
    take = (np.arange(GRID) * w // GRID).clip(0, w - 1)
    return _norm(sub[:, take][:, :, take])


def clip_grid(frames: np.ndarray, max_frames: int = MAX_FRAMES):
    """Return (grid vectors, kept frame indices)."""
    if frames.shape[0] == 0:
        return np.zeros((0, GRID * GRID), np.float32), np.zeros(0, int)
    keep = np.flatnonzero(frames.reshape(frames.shape[0], -1).var(axis=1) > FLAT_VAR)
    if keep.size < 4:
        keep = np.arange(frames.shape[0])
    if keep.size > max_frames:
        keep = keep[np.linspace(0, keep.size - 1, max_frames).astype(int)]
    return view_of(frames[keep], (1.0, 0.5, 0.5)), keep


def geometry_hypotheses(clip_g: np.ndarray, mode: str = "identity") -> list[tuple[str, np.ndarray]]:
    """Return explicitly enabled square-symmetry hypotheses for clip descriptors."""
    if mode not in {"identity", "d4"}:
        raise ValueError(f"unsupported geometry mode: {mode}")
    if clip_g.ndim != 2 or clip_g.shape[1] != GRID * GRID:
        raise ValueError(f"clip grids must have shape (frames, {GRID * GRID})")
    square = clip_g.reshape(clip_g.shape[0], GRID, GRID)
    rotations = (
        ("identity", square),
        ("rotate-90-ccw", np.rot90(square, 1, axes=(1, 2))),
        ("rotate-180", np.rot90(square, 2, axes=(1, 2))),
        ("rotate-270-ccw", np.rot90(square, 3, axes=(1, 2))),
    )
    if mode == "identity":
        return [("identity", clip_g)]
    mirrored = np.flip(square, axis=2)
    transforms = rotations + (
        ("mirror-horizontal", mirrored),
        ("mirror-horizontal-rotate-90-ccw", np.rot90(mirrored, 1, axes=(1, 2))),
        ("mirror-horizontal-rotate-180", np.rot90(mirrored, 2, axes=(1, 2))),
        ("mirror-horizontal-rotate-270-ccw", np.rot90(mirrored, 3, axes=(1, 2))),
    )
    return [(name, value.reshape(clip_g.shape)) for name, value in transforms]


def ncc(a: np.ndarray, b: np.ndarray):
    n = a.size
    if n < 4 or b.size < n:
        return None
    a = a - a.mean()
    da = np.sqrt((a * a).sum())
    if da < 1e-9:
        return None
    win = np.ones(n)
    s1 = np.convolve(b, win, "valid")
    var = np.convolve(b * b, win, "valid") - s1 * s1 / n
    var[var < 1e-9] = np.inf
    return np.convolve(b, a[::-1], "valid") / (da * np.sqrt(var))


def search(clip_g: np.ndarray, idx: np.ndarray, master_views: list, speeds=None):
    """Best (score, offset_seconds, speed) over reframings, offsets, speeds."""
    detail = search_diagnostics(clip_g, idx, master_views, speeds)
    return detail.score, detail.offset_sec, detail.speed


def search_geometry_diagnostics(
    clip_g: np.ndarray, idx: np.ndarray, master_views: list, speeds=None, *, mode="identity"
):
    """Return the best source alignment across an explicit geometry hypothesis set."""
    winner = None
    for order, (name, hypothesis) in enumerate(geometry_hypotheses(clip_g, mode)):
        detail = search_diagnostics(hypothesis, idx, master_views, speeds)
        candidate = (detail.score, -order)
        if winner is None or candidate > winner[0]:
            winner = (candidate, detail, name)
    assert winner is not None
    return winner[1], winner[2]


@dataclass(frozen=True)
class SearchDiagnostics:
    """Non-probabilistic diagnostics for one candidate alignment."""

    score: float
    offset_sec: float | None
    speed: float | None
    supporting_samples: int
    sample_count: int
    support_fraction: float | None
    temporal_residual_sec: float | None
    offset_profile_width_sec: float | None
    offset_ambiguity: float | None
    reframe_index: int | None
    reframe: tuple[float, float, float] | None
    aligned_timestamps: tuple[tuple[float, float], ...]


def search_diagnostics(clip_g: np.ndarray, idx: np.ndarray, master_views: list, speeds=None):
    """Return the best alignment plus auditable sample and timing diagnostics.

    Confidence values are descriptive measurements, not probabilities. The
    offset profile is measured on the winning reframe/speed. Its width is the
    contiguous region within 0.02 score of the peak; ambiguity is the gap to
    the best offset outside that region.
    """
    if speeds is None:
        speeds = SPEEDS
    f = clip_g.shape[0]
    empty = SearchDiagnostics(
        -1.0, None, None, 0, f, None, None, None, None, None, None, ()
    )
    if f < 4:
        return empty
    rows = np.arange(f)
    winner = None
    for view_index, mv in enumerate(master_views):
        m = mv.shape[0]
        if m < 4:
            continue
        similarities = clip_g @ mv.T
        for speed in speeds:
            shift = np.rint(float(speed) * idx).astype(int)
            lo = -int(shift.min())
            hi = m - 1 - int(shift.max())
            if hi < lo:
                continue
            offsets = np.arange(lo, hi + 1)
            per_offset = similarities[
                rows[None, :], offsets[:, None] + shift[None, :]
            ]
            profile = per_offset.mean(axis=1)
            peak = int(np.argmax(profile))
            candidate = (float(profile[peak]), -view_index, -float(speed))
            if winner is None or candidate > winner[0]:
                winner = (candidate, view_index, float(speed), shift, offsets, profile, per_offset, peak)
    if winner is None:
        return empty
    _, view_index, speed, shift, offsets, profile, per_offset, peak = winner
    score = float(profile[peak])
    sample_scores = per_offset[peak]
    supporting = int(np.count_nonzero(sample_scores >= 0.35))
    support_fraction = supporting / f

    near = profile >= score - 0.02
    left = peak
    right = peak
    while left > 0 and near[left - 1]:
        left -= 1
    while right + 1 < near.size and near[right + 1]:
        right += 1
    width = float(offsets[right] - offsets[left]) / FPS
    outside = np.concatenate((profile[:left], profile[right + 1 :]))
    ambiguity = score - float(outside.max()) if outside.size else score + 1.0
    weights = np.maximum(profile[left : right + 1] - (score - 0.02), 0.0)
    positions = offsets[left : right + 1].astype(float)
    if weights.sum() > 0:
        residual = float(np.sqrt(np.average((positions - offsets[peak]) ** 2, weights=weights))) / FPS
    else:
        residual = 0.0
    chosen_offset = int(offsets[peak])
    timestamps = tuple(
        (float(frame_index) / FPS, float(chosen_offset + shifted) / FPS)
        for frame_index, shifted in zip(idx.tolist(), shift.tolist(), strict=True)
    )
    if view_index < len(VIEWS):
        crop, position_x, position_y = VIEWS[view_index]
        reframe = (float(crop), float(position_x), float(position_y))
    else:
        reframe = None
    return SearchDiagnostics(
        score=score,
        offset_sec=float(chosen_offset) / FPS,
        speed=speed,
        supporting_samples=supporting,
        sample_count=f,
        support_fraction=support_fraction,
        temporal_residual_sec=residual,
        offset_profile_width_sec=width,
        offset_ambiguity=ambiguity,
        reframe_index=view_index,
        reframe=reframe,
        aligned_timestamps=timestamps,
    )
