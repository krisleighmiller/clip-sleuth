"""Smoke tests for the product skeleton."""

from __future__ import annotations

from pathlib import Path

import pytest

from clipprov import __version__
from clipprov.config import Config
from clipprov.media import MediaError, probe


def test_version():
    assert __version__ == "0.1.0"


def test_config_defaults():
    c = Config()
    assert c.media.ffmpeg == "ffmpeg"
    assert c.match.speed_min < c.match.speed_max


def test_config_matches_search_defaults():
    from clipprov.config import MatchConfig
    from clipprov.motion import ACCEPT_BEST, ACCEPT_MARGIN, SPEEDS

    c = MatchConfig()
    assert c.score_floor == ACCEPT_BEST
    assert c.margin == ACCEPT_MARGIN
    assert abs(c.speed_min - float(SPEEDS.min())) < 1e-9
    assert abs(c.speed_max - float(SPEEDS.max())) < 1e-9


def test_probe_missing_file():
    with pytest.raises(MediaError):
        probe(Path("/no/such/file.mp4"))
