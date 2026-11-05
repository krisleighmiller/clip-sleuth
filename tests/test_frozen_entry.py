"""The downloadable launcher must dispatch GUI and CLI worker separately."""

import runpy
import sys
from pathlib import Path

import pytest

from clipprov import _gui_cli, gui

ENTRY = Path(__file__).resolve().parents[1] / "packaging" / "clip_sleuth.py"


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [([], "gui"), (["--gui-worker", "index", "library"], "worker")],
)
def test_download_launcher_dispatches_without_opening_worker_gui(monkeypatch, arguments, expected):
    calls = []
    monkeypatch.setattr(sys, "argv", ["clip-sleuth", *arguments])
    monkeypatch.setattr(gui, "main", lambda: calls.append("gui"))
    monkeypatch.setattr(_gui_cli, "main", lambda: calls.append(("worker", sys.argv[1:])))
    runpy.run_path(str(ENTRY), run_name="__main__")
    assert calls == (["gui"] if expected == "gui" else [("worker", ["index", "library"])])
