"""Configuration model for Clip Provenance."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class MediaConfig(BaseModel):
    """ffmpeg/ffprobe and media extraction settings."""

    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    max_frames: int = Field(default=20, ge=4)


class MatchConfig(BaseModel):
    """Matching thresholds and search parameters (baseline defaults)."""

    speed_min: float = 0.968
    speed_max: float = 1.032
    speed_steps: int = 33
    offset_tolerance_s: float = 1.5
    score_floor: float = 0.35
    margin: float = 0.12
    geometry_mode: Literal["identity", "d4"] = "identity"
    audio_enabled: bool = False

    @field_validator("speed_min", "speed_max", "offset_tolerance_s", "score_floor", "margin")
    @classmethod
    def _finite(cls, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("must be a finite number")
        if not math.isfinite(value):
            raise ValueError("must be finite (no NaN/inf)")
        return float(value)

    @field_validator("speed_steps")
    @classmethod
    def _positive_steps(cls, value):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("must be an integer")
        if value <= 0:
            raise ValueError("speed_steps must be positive")
        return value

    @model_validator(mode="after")
    def _ranges(self):
        if self.speed_min >= self.speed_max:
            raise ValueError("speed_min must be < speed_max")
        if self.offset_tolerance_s < 0:
            raise ValueError("offset_tolerance_s must be >= 0")
        if self.margin < 0:
            raise ValueError("margin must be >= 0")
        return self


class Config(BaseModel):
    """Top-level Clip Provenance configuration."""

    media: MediaConfig = Field(default_factory=MediaConfig)
    match: MatchConfig = Field(default_factory=MatchConfig)
    index_dir: Path | None = None

    @field_validator("index_dir", mode="before")
    @classmethod
    def _coerce_path(cls, value):
        if value is None or isinstance(value, Path):
            return value
        return Path(value)


SPEED_PROFILE_SCHEMA_VERSION = 1
WIDER_SPEED_PROFILE = "wider-speed-v1"


def resolve_speed_profile(name: str | None, base: Config) -> tuple[Config, dict]:
    """Apply a named search policy without changing the index-derived base policy."""
    selected = name or "v1-default"
    if selected == "v1-default":
        match = base.match
        measured_range = None
    elif selected == WIDER_SPEED_PROFILE:
        match = base.match.model_copy(update={
            "speed_min": 0.5,
            "speed_max": 2.0,
            "speed_steps": 401,
        })
        measured_range = {"minimum": 0.5, "maximum": 2.0}
    else:
        raise ValueError(f"unsupported speed profile: {selected}")
    receipt = {
        "schema_version": SPEED_PROFILE_SCHEMA_VERSION,
        "name": selected,
        "speed_min": match.speed_min,
        "speed_max": match.speed_max,
        "speed_steps": match.speed_steps,
        "measured_range": measured_range,
    }
    return base.model_copy(update={"match": match}), receipt
