"""Probed, validated ffmpeg controls adapted from Kimi's synth_gen.py.

Durations are source spans, before speed adjustment. Failures raise; a completed
output replaces the destination only after probing and a full decode audit.
"""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

from clipprov._io import finite_number
from clipprov.config import MediaConfig
from clipprov.media import MediaError, probe

RECIPE_FIELDS = {
    "crop",
    "resize",
    "crf",
    "watermark",
    "brightness",
    "saturation",
    "padding",
    "speed",
    "audio",
    "audio_frequency",
    "grayscale",
}


def number(value, name, low, high):
    if type(value) not in (int, float) or not finite_number(value) or not low <= value <= high:
        raise ValueError(f"{name} must be finite and in [{low}, {high}]")
    return float(value)


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def run_ffmpeg(args: list[str], *, ffmpeg="ffmpeg", progress=False):
    try:
        r = subprocess.run(
            [ffmpeg, "-v", "error", "-nostdin", "-y", *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise MediaError(f"failed to run ffmpeg: {exc}") from exc
    if r.returncode:
        raise MediaError(f"ffmpeg failed: {r.stderr.strip()}")
    return r.stdout if progress else True


def audit_decode(path, config=None):
    """Check every video frame, including failures beyond a valid header."""
    cfg = config or MediaConfig()
    output = run_ffmpeg(
        [
            "-xerror",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-progress",
            "pipe:1",
            "-nostats",
            "-f",
            "null",
            "-",
        ],
        ffmpeg=cfg.ffmpeg,
        progress=True,
    )
    frames = [
        int(line.split("=", 1)[1]) for line in output.splitlines() if line.startswith("frame=")
    ]
    if not frames or max(frames) <= 0:
        raise MediaError(f"zero decoded frames from {path}")
    return max(frames)


def validate_recipe(recipe, info):
    if not isinstance(recipe, dict) or recipe.keys() - RECIPE_FIELDS:
        raise ValueError("unknown or malformed transform recipe")
    w, h = info.width, info.height
    if (
        type(w) is not int
        or type(h) is not int
        or w < 2
        or h < 2
        or not math.isfinite(info.fps)
        or info.fps <= 0
    ):
        raise ValueError("source dimensions and frame rate must be positive")
    filters = []
    if "crop" in recipe:
        crop = recipe["crop"]
        if not isinstance(crop, list) or len(crop) not in (3, 4):
            raise ValueError("crop must be [scale, x-position, y-position] or [x,y,w,h]")
        if len(crop) == 3:
            scale = number(crop[0], "crop scale", 0.1, 1)
            px, py = (number(v, "crop position", 0, 1) for v in crop[1:])
            cw, ch = max(2, int(w * scale) // 2 * 2), max(2, int(h * scale) // 2 * 2)
            x, y = int((w - cw) * px) // 2 * 2, int((h - ch) * py) // 2 * 2
        else:
            x, y, cw, ch = [_integer(v, "crop", 0, 16384) for v in crop]
            if cw < 2 or ch < 2 or x + cw > w or y + ch > h:
                raise ValueError("crop exceeds source dimensions")
            if any(v % 2 for v in crop):
                raise ValueError("crop coordinates and dimensions must be even")
        filters.append(f"crop={cw}:{ch}:{x}:{y}")
        w, h = cw, ch
    if "resize" in recipe:
        size = recipe["resize"]
        if not isinstance(size, list) or len(size) != 2:
            raise ValueError("resize must contain width and height")
        w, h = [_integer(v, "resize", 2, 4096) for v in size]
        if w % 2 or h % 2:
            raise ValueError("resize dimensions must be even")
        filters.append(f"scale={w}:{h}")
    for name, bounds, default in [("brightness", (-1, 1), 0), ("saturation", (0, 3), 1)]:
        number(recipe.get(name, default), name, *bounds)
    if "brightness" in recipe or "saturation" in recipe:
        filters.append(
            f"eq=brightness={recipe.get('brightness', 0)}:saturation={recipe.get('saturation', 1)}"
        )
    if "grayscale" in recipe:
        if type(recipe["grayscale"]) is not bool:
            raise ValueError("grayscale must be boolean")
        if recipe["grayscale"]:
            filters.append("hue=s=0")
    if "padding" in recipe:
        padding = recipe["padding"]
        if not isinstance(padding, list) or len(padding) != 2:
            raise ValueError("padding must contain horizontal and vertical border sizes")
        x, y = [_integer(v, "padding", 0, 1024) for v in padding]
        filters.append(f"pad={w + 2 * x}:{h + 2 * y}:{x}:{y}:black")
    if "watermark" in recipe:
        watermark = recipe["watermark"]
        if not isinstance(watermark, list) or len(watermark) != 3:
            raise ValueError("watermark must contain relative x, y, and opacity")
        x, y, opacity = [number(v, "watermark", 0, 1) for v in watermark]
        # Font-independent opaque graphic, matching Astra's validation design.
        filters.append(f"drawbox=x=iw*{x}:y=ih*{y}:w=iw*0.2:h=ih*0.1:color=white@{opacity}:t=fill")
    speed = number(recipe.get("speed", 1), "speed", 0.968, 1.032)
    crf = _integer(recipe.get("crf", 23), "crf", 0, 51)
    audio = recipe.get("audio", "keep")
    if audio not in ("keep", "remove", "replace"):
        raise ValueError("audio must be keep, remove or replace")
    frequency = number(recipe.get("audio_frequency", 440), "audio frequency", 20, 20000)
    return filters, speed, crf, audio, frequency


def transform(src, dst, *, start=0, duration=None, recipe=None, config=None):
    """Atomically create one control, returning actual probed output metadata."""
    cfg = config or MediaConfig()
    src, dst = Path(src), Path(dst)
    if src.resolve() == dst.resolve():
        raise ValueError("transform destination must differ from source")
    info = probe(src, cfg)
    filters, rate, crf, audio, frequency = validate_recipe({} if recipe is None else recipe, info)
    start = number(start, "start", 0, info.duration_s)
    duration = number(
        info.duration_s - start if duration is None else duration,
        "duration",
        1 / info.fps,
        info.duration_s,
    )
    if start + duration > info.duration_s + 1e-9:
        raise ValueError("source-end boundary exceeded")
    first, end = round(start * info.fps), round((start + duration) * info.fps)
    if end <= first:
        raise ValueError("source span must contain a frame")
    actual_start, span = first / info.fps, (end - first) / info.fps
    if end / info.fps > info.duration_s + 1e-9:
        raise ValueError("rounded source-end boundary exceeded")
    filters = [
        f"trim=start_frame={first}:end_frame={end}",
        f"setpts=(PTS-STARTPTS)/{rate}",
        *filters,
        f"fps={info.fps}",
    ]
    args = ["-threads", "1", "-i", str(src)]
    if audio == "replace":
        args += [
            "-f",
            "lavfi",
            "-i",
            (f"sine=frequency={frequency}:sample_rate=44100:duration={span / rate}"),
        ]
    args += ["-map", "0:v:0", "-vf", ",".join(filters), "-filter_threads", "1"]
    if audio == "remove" or (audio == "keep" and not info.has_audio):
        args += ["-an"]
    elif audio == "replace":
        args += ["-map", "1:a:0", "-c:a", "aac"]
    else:
        args += [
            "-map",
            "0:a:0",
            "-af",
            (f"atrim=start={actual_start}:duration={span},asetpts=PTS-STARTPTS,atempo={rate}"),
            "-c:a",
            "aac",
        ]
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=dst.parent, suffix=".tmp.mp4")
    os.close(fd)
    tmp = Path(name)
    try:
        args += [
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-crf",
            str(crf),
            "-preset",
            "fast",
            "-pix_fmt",
            "yuv420p",
            "-map_metadata",
            "-1",
            "-fflags",
            "+bitexact",
            "-flags:v",
            "+bitexact",
            str(tmp),
        ]
        run_ffmpeg(args, ffmpeg=cfg.ffmpeg)
        result = probe(tmp, cfg)
        audit_decode(tmp, cfg)
        with tmp.open("rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, dst)
        return replace(result, path=dst)
    finally:
        tmp.unlink(missing_ok=True)


def crop_scale(src, dst, x, y, w, h, scale=1.0):
    scale = number(scale, "scale", 0.1, 10)
    transform(
        src,
        dst,
        recipe={
            "crop": [x, y, w, h],
            "resize": [round(w * scale / 2) * 2, round(h * scale / 2) * 2],
        },
    )
    return True


def resize(src, dst, w, h):
    transform(src, dst, recipe={"resize": [w, h]})
    return True


def speed(src, dst, factor):
    transform(src, dst, recipe={"speed": factor})
    return True


def grayscale(src, dst):
    transform(src, dst, recipe={"grayscale": True})
    return True


def pad(src, dst, w, h):
    info = probe(Path(src))
    if w < info.width or h < info.height or (w - info.width) % 2 or (h - info.height) % 2:
        raise ValueError("padding target must enclose the source with symmetric borders")
    transform(src, dst, recipe={"padding": [(w - info.width) // 2, (h - info.height) // 2]})
    return True
