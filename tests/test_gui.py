"""Real-Tk GUI tests: constructor smoke plus full proof against the real CLI.

The integration test drives actual widgets (invoke on Run/Open Report, polling
the Tk event loop) through the actual clipprov CLI via the checkout-pinned
child launcher, over a real ffmpeg demo corpus, in paths containing spaces.
It only runs when a display exists; Hermes CLI sessions have none, so proof
runs export DISPLAY/XAUTHORITY/WAYLAND_DISPLAY/XDG_RUNTIME_DIR from
``systemctl --user show-environment`` without ever logging credential values.
"""

from __future__ import annotations

import json
import os
import threading
import time
import tkinter as tk
from pathlib import Path

import pytest

DISPLAYABLE = bool(os.environ.get("DISPLAY")) or bool(os.environ.get("WAYLAND_DISPLAY"))
requires_display = pytest.mark.skipif(
    not DISPLAYABLE, reason="no DISPLAY/WAYLAND_DISPLAY in environment"
)


def _make_app():
    """Bootstrap exactly like main(): hidden root, withdraw, app, update pass."""
    root = tk.Tk()
    root.withdraw()
    from clipprov.gui import ClipProvenanceGUI

    app = ClipProvenanceGUI(root)
    root.update()
    return root, app


def _pump(root, predicate, deadline_sec=300.0):
    """Process Tk events on the main thread until predicate() or deadline."""
    end = time.monotonic() + deadline_sec
    while time.monotonic() < end:
        root.update()
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _live_job_threads():
    return [
        t
        for t in threading.enumerate()
        if t.name.startswith(("clipprov-cli-job", "clipprov-cli-stream"))
    ]


@requires_display
def test_constructor_real_tk():
    """RED-then-GREEN gate. The old code failed here on a real interpreter with
    AttributeError: module 'tkinter.ttk' has no attribute 'StringVar'."""
    root, app = _make_app()
    try:
        assert app.workflow_var.get() == "index"
        assert app.library_path_var.get() == ""
        assert app.resume_var.get() is True
        assert app.continue_on_error_var.get() is True
        assert app.speed_profile_var.get() == "v1-default"
        assert app.geometry_mode_var.get() == "identity"
        assert tuple(app.speed_profile_combo["values"]) == (
            "v1-default",
            "wider-speed-v1",
        )
        assert tuple(app.geometry_mode_combo["values"]) == ("identity", "d4")
        assert app.log_text.get("1.0", "end-1c") == ""
        assert str(app.run_button) != ""
        assert str(app.open_report_button) != ""
        assert str(app.log_text) != ""  # tk.Text, not ttk (ttk has no Text)
    finally:
        root.destroy()


@requires_display
def test_real_widgets_forward_v1_speed_and_identity_geometry_defaults(tmp_path, monkeypatch):
    clips = tmp_path / "clips"
    index = tmp_path / "index"
    clips.mkdir()
    index.mkdir()
    _root, app = _make_app()
    captured = {}
    try:
        app.workflow_var.set("match_dir")
        app._on_workflow_change()
        app.clips_path_var.set(str(clips))
        app.index_path_var.set(str(index))
        app.match_out_path_var.set(str(tmp_path / "out"))
        monkeypatch.setattr(
            app.controller,
            "start_match_dir",
            lambda *args, **kwargs: captured.update(args=args, kwargs=kwargs),
        )

        app.run_button.invoke()

        assert captured["kwargs"] == {
            "speed_profile": None,
            "geometry_mode": "identity",
            "audio_enabled": False,
        }
    finally:
        app.close()


@requires_display
def test_real_widgets_forward_combined_speed_and_geometry_opt_ins(tmp_path, monkeypatch):
    clips = tmp_path / "clips"
    index = tmp_path / "index"
    clips.mkdir()
    index.mkdir()
    _root, app = _make_app()
    captured = {}
    try:
        app.workflow_var.set("match_dir")
        app._on_workflow_change()
        app.clips_path_var.set(str(clips))
        app.index_path_var.set(str(index))
        app.match_out_path_var.set(str(tmp_path / "out"))
        app.speed_profile_combo.set("wider-speed-v1")
        app.geometry_mode_combo.set("d4")
        app.audio_enabled_var.set(True)
        monkeypatch.setattr(
            app.controller,
            "start_match_dir",
            lambda *args, **kwargs: captured.update(args=args, kwargs=kwargs),
        )

        app.run_button.invoke()

        assert captured["kwargs"] == {
            "speed_profile": "wider-speed-v1",
            "geometry_mode": "d4",
            "audio_enabled": True,
        }
    finally:
        app.close()


@requires_display
def test_open_report_rejects_unverified_directory(tmp_path, monkeypatch):
    """current_report_dir(out) is the authoritative integrity resolver: a bogus
    or incomplete output dir must never open a browser. The opener is mocked
    ONLY to stop a real popup; the CLI and resolver are genuine."""
    import clipprov.gui as gui_module

    opened: list[str] = []
    monkeypatch.setattr(gui_module.webbrowser, "open", lambda uri: opened.append(uri))
    root, app = _make_app()
    try:
        app.workflow_var.set("match_dir")
        app._on_workflow_change()
        app.match_out_path_var.set(str(tmp_path / "no such report dir"))
        app.open_report_button.configure(state="normal")
        app.open_report_button.invoke()
        assert _pump(root, lambda: app._report_error is not None, 15.0)
        assert opened == []
    finally:
        app.close()
        assert _live_job_threads() == []


@requires_display
def test_full_workflow_via_real_widgets(tmp_path, monkeypatch):
    """Index a library in a spaced path, match a partial clip dir, verify real
    index manifest, current_report_dir resolution, report HTML, evidence
    images, and clean teardown (no streams, no Popen, no after callbacks)."""
    from clipprov.batch import current_report_dir
    from clipprov.harness import generate_demo_corpus

    demo = tmp_path / "demo corpus"
    generate_demo_corpus(demo)  # 3 spaced-path sources; first two in "indexed sources"
    library = demo / "indexed sources"
    clips = demo / "sources"
    index_out = tmp_path / "index out"
    match_out = tmp_path / "match out"

    root, app = _make_app()
    closed = False
    try:
        # ---- Index Library (real CLI subprocess over spaced-path dirs) ----
        app.library_path_var.set(str(library))
        app.index_out_path_var.set(str(index_out))
        app.run_button.invoke()
        assert _pump(root, lambda: app.finished and not app.controller.busy)
        summary = app.controller.summary
        assert summary is not None and summary["status"] == "complete"
        assert summary["masters"] == 2
        assert "indexing" in app._visible_log()  # live dispatch, not only post-exit
        manifest_path = index_out / "manifest.json"
        assert manifest_path.is_file()
        manifest = json.loads(manifest_path.read_text())
        assert len(manifest["masters"]) == 2
        # Index has no offline HTML report; the button must stay disabled.
        assert str(app.open_report_button["state"]) == "disabled"

        # ---- Match Clips (partial directory: one source withheld from index) ----
        app.workflow_var.set("match_dir")
        app._on_workflow_change()
        root.update()
        app.clips_path_var.set(str(clips))
        app.index_path_var.set(str(index_out))
        app.match_out_path_var.set(str(match_out))
        app.run_button.invoke()
        # Progress must reach the log while the child is still running.
        assert _pump(root, lambda: "matching" in app._visible_log(), 120.0)
        assert app.controller.busy
        assert _pump(root, lambda: app.finished and not app.controller.busy)
        summary = app.controller.summary
        assert summary is not None
        assert summary["results"] + summary["failures"] == 3
        report_dir = current_report_dir(match_out)  # raises unless generation is intact
        assert (report_dir / "report.html").is_file()
        assert b"data:image/jpeg" in (report_dir / "report.html").read_bytes()
        evidence_files = [p for p in (match_out / "evidence").rglob("*") if p.is_file()]
        assert len(evidence_files) >= 2
        assert Path(summary["report"]).parent == report_dir
        assert str(app.open_report_button["state"]) == "normal"

        # ---- Open Report through the validated resolver ----
        import clipprov.gui as gui_module

        opened: list[str] = []
        monkeypatch.setattr(gui_module.webbrowser, "open", lambda uri: opened.append(uri))
        app.open_report_button.invoke()
        assert _pump(root, lambda: bool(opened), 30.0)
        assert opened == [(report_dir / "report.html").as_uri()]
        assert "report_dir" not in json.dumps(app.controller.summary)  # summary stays minimal
    finally:
        if not closed:
            app.close()
            closed = True

    # ---- Teardown hygiene ----
    assert _live_job_threads() == []
    assert app.controller.process is None
    with pytest.raises(tk.TclError):
        root.winfo_exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
