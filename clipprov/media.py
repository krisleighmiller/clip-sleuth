"""ffmpeg/ffprobe media helpers for Clip Provenance."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from clipprov.config import MediaConfig


class MediaError(Exception):
    """Raised when media cannot be read or decoded."""


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    duration_s: float
    width: int
    height: int
    fps: float
    frame_count: int | None
    has_video: bool
    has_audio: bool


def _resolve(tool: str, config: MediaConfig | None) -> str:
    name = tool
    if config is not None:
        name = getattr(config, tool, tool)
    resolved = shutil.which(name)
    if resolved is None:
        raise MediaError(f"required tool not found on PATH: {name}")
    return resolved


def probe(path: Path, config: MediaConfig | None = None) -> VideoInfo:
    """Probe a media file with ffprobe and return structured metadata."""
    ffprobe = _resolve("ffprobe", config)
    target = Path(path)
    if not target.is_file():
        raise MediaError(f"media file not found: {target}")
    cmd = [
        ffprobe,
        "-v", "error",
        "-show_entries", "format=duration:stream=index,codec_type,width,height,r_frame_rate,nb_frames,duration",
        "-of", "json",
        str(target),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise MediaError(f"failed to run ffprobe: {exc}") from exc
    if proc.returncode != 0:
        raise MediaError(f"ffprobe failed for {target}: {proc.stderr.strip()}")
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise MediaError(f"invalid ffprobe output for {target}: {exc}") from exc

    streams = data.get("streams") or []
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    has_video = video_stream is not None
    has_audio = audio_stream is not None

    if not has_video:
        raise MediaError(f"no video stream in {target}")

    width = int(video_stream.get("width", 0)) if video_stream else 0
    height = int(video_stream.get("height", 0)) if video_stream else 0

    fps = 0.0
    if video_stream:
        rate = video_stream.get("r_frame_rate", "0/1")
        try:
            num, den = rate.split("/")
            fps = float(num) / float(den) if float(den) != 0 else 0.0
        except (ValueError, ZeroDivisionError):
            fps = 0.0

    frame_count = None
    if video_stream and video_stream.get("nb_frames") not in (None, "N/A"):
        try:
            frame_count = int(video_stream["nb_frames"])
        except (ValueError, TypeError):
            frame_count = None

    duration = 0.0
    try:
        duration = float((data.get("format") or {}).get("duration", 0.0))
    except (TypeError, ValueError):
        duration = 0.0
    if duration <= 0 and video_stream:
        # fall back to stream duration if present
        try:
            duration = float(video_stream.get("duration", 0.0))
        except (TypeError, ValueError):
            duration = 0.0

    if not math.isfinite(duration) or duration <= 0:
        raise MediaError(f"invalid or zero duration for {target}")

    return VideoInfo(
        path=target,
        duration_s=duration,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        has_video=has_video,
        has_audio=has_audio,
    )


def extract_frames(
    path: Path,
    output_dir: Path,
    *,
    fps: float = 1.0,
    config: MediaConfig | None = None,
) -> list[Path]:
    """Extract frames into versioned generation dirs; return current frames."""
    import fcntl
    import json
    import os
    import uuid

    ffmpeg = _resolve("ffmpeg", config)
    target = Path(path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    generations = out / "generations"
    generations.mkdir(exist_ok=True)
    lock_path = out / ".extract.lock"
    lock_fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        gen_id = uuid.uuid4().hex
        gen_dir = generations / gen_id
        gen_dir.mkdir()
        pattern = gen_dir / "frame_%06d.png"
        cmd = [ffmpeg, "-y", "-i", str(target), "-vf", f"fps={fps}", str(pattern)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except OSError as exc:
            import shutil as _sh
            _sh.rmtree(gen_dir, ignore_errors=True)
            raise MediaError(f"failed to run ffmpeg: {exc}") from exc
        if proc.returncode != 0:
            import shutil as _sh
            _sh.rmtree(gen_dir, ignore_errors=True)
            raise MediaError(f"ffmpeg frame extraction failed for {target}: {proc.stderr.strip()}")
        frames = sorted(gen_dir.glob("frame_*.png"))
        if not frames:
            import shutil as _sh
            _sh.rmtree(gen_dir, ignore_errors=True)
            raise MediaError(f"no frames extracted from {target}")

        prev_gen = None
        pointer = out / "current.json"
        if pointer.exists():
            try:
                prev_gen = json.loads(pointer.read_text()).get("generation")
            except (json.JSONDecodeError, OSError):
                prev_gen = None

        pointer_tmp = out / ".current.tmp"
        pointer_tmp.write_text(json.dumps({"generation": gen_id, "frames": [p.name for p in frames]}) + "\n")
        os.replace(pointer_tmp, pointer)

        if prev_gen and prev_gen != gen_id:
            import shutil as _sh
            _sh.rmtree(generations / prev_gen, ignore_errors=True)

        return [gen_dir / p.name for p in frames]
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(lock_fd)
