"""Controller tests use explicitly synthetic backends; real CLI tests live in test_gui.py."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import clipprov.gui_controller as module
from clipprov.gui_controller import CLIController, parse_progress_event, parse_summary


@pytest.mark.parametrize("value", ["", "garbage", "[]", "null", None])
def test_invalid_progress_is_not_a_summary(value):
    assert parse_progress_event(value) is None
    assert parse_summary(value) is None


def test_pretty_summary_and_progress_are_parsed():
    assert parse_summary('{\n"results": 2,\n"failures": 1\n}') == {"results": 2, "failures": 1}
    assert parse_progress_event('{"event":"matching","clip":"a b.mp4"}')["clip"] == "a b.mp4"


def backend(monkeypatch, code, capture=None):
    real = subprocess.Popen

    def launch(args, **kwargs):
        if capture is not None:
            capture.update(args=args, kwargs=kwargs)
        return real([sys.executable, "-u", "-c", code], **kwargs)

    monkeypatch.setattr(module.subprocess, "Popen", launch)


def test_authoritative_summary_is_not_truncated(monkeypatch):
    data = {"status": "complete_with_errors", "failures": ["x" * 100 for _ in range(200)]}
    backend(monkeypatch, f"import json;print(json.dumps({data!r},indent=2))")
    controller = CLIController()
    controller.start(["index", "input", "--out", "output"])
    assert controller.wait(5)
    assert controller.returncode == 0 and controller.summary == data
    assert not controller.busy and controller.process is None


def test_progress_is_live_and_cancel_reaps_owned_job(monkeypatch):
    backend(
        monkeypatch,
        'import sys,time;print(\'{"event":"matching","position":1,"total":2}\',file=sys.stderr,flush=True);time.sleep(20)',
    )
    controller = CLIController()
    controller.start(["match-dir", "clips", "--index", "index", "--out", "out"])
    end = time.monotonic() + 5
    while time.monotonic() < end and not controller.progress_events:
        time.sleep(0.01)
    assert controller.progress_events and controller.busy and not controller.done.is_set()
    controller.cancel()
    assert controller.wait(7)
    assert controller.cancelled and controller.process is None and not controller.busy


def test_double_start_is_rejected_before_process_finishes(monkeypatch):
    backend(monkeypatch, "import time;time.sleep(20)")
    controller = CLIController()
    controller.start(["index", "lib", "--out", "out"])
    try:
        with pytest.raises(RuntimeError, match="already running"):
            controller.start(["index", "other", "--out", "out"])
    finally:
        assert controller.close(7)


def test_cancel_during_process_creation_is_remembered(monkeypatch):
    real = subprocess.Popen
    entered, release = threading.Event(), threading.Event()

    def delayed(args, **kwargs):
        entered.set()
        assert release.wait(5)
        return real([sys.executable, "-u", "-c", "import time;time.sleep(20)"], **kwargs)

    monkeypatch.setattr(module.subprocess, "Popen", delayed)
    controller = CLIController()
    controller.start(["index", "lib", "--out", "out"])
    assert entered.wait(5)
    controller.cancel()
    release.set()
    assert controller.wait(7)
    assert controller.cancelled and not controller.busy


def test_launch_failure_is_reported_and_not_busy(monkeypatch):
    def fail(*args, **kwargs):
        raise FileNotFoundError("synthetic launch failure")

    monkeypatch.setattr(module.subprocess, "Popen", fail)
    controller = CLIController()
    controller.start(["index", "lib", "--out", "out"])
    assert controller.wait(5)
    assert controller.returncode == 2 and "launch failure" in controller.error
    assert controller.process is None and not controller.busy


def test_closed_controller_cannot_launch():
    controller = CLIController()
    assert controller.close()
    with pytest.raises(RuntimeError, match="closed"):
        controller.start(["index", "lib", "--out", "out"])


def test_invalid_stdout_cannot_become_success(monkeypatch):
    backend(monkeypatch, 'print("not a JSON summary")')
    controller = CLIController()
    controller.start(["index", "lib", "--out", "out"])
    assert controller.wait(5)
    assert controller.returncode == 2 and controller.error


def test_new_job_does_not_reuse_cancel_or_summary(monkeypatch):
    backend(monkeypatch, 'print(\'{"results":1,"failures":0}\')')
    controller = CLIController()
    controller.start(["match-dir", "clips", "--index", "index", "--out", "out"])
    assert controller.wait(5)
    controller.cancel()
    controller.start(["match-dir", "clips", "--index", "index", "--out", "out"])
    assert controller.wait(5)
    assert not controller.cancelled and controller.summary == {"results": 1, "failures": 0}


def test_child_environment_and_launcher_are_pinned(monkeypatch):
    captured = {}
    backend(monkeypatch, 'print(\'{"status":"complete","masters":1}\')', captured)
    monkeypatch.setenv("PYTHONPATH", "/untrusted")
    monkeypatch.setenv("PYTHONHOME", "/untrusted")
    controller = CLIController()
    controller.start(["index", "a folder", "--out", "an output"])
    assert controller.wait(5)
    assert captured["args"][1].endswith("clipprov/_gui_cli.py")
    assert captured["args"][-3:] == ["a folder", "--out", "an output"]
    assert "PYTHONPATH" not in captured["kwargs"]["env"]
    assert "PYTHONHOME" not in captured["kwargs"]["env"]
    assert captured["kwargs"]["start_new_session"] is True


def test_match_dir_argv_exposes_combined_speed_and_geometry_opt_ins(monkeypatch, tmp_path):
    controller = CLIController()
    captured = {}
    monkeypatch.setattr(
        controller,
        "start",
        lambda args, **kwargs: captured.update(args=args, kwargs=kwargs),
    )

    controller.start_match_dir(
        "clips",
        "index",
        tmp_path / "out",
        speed_profile="wider-speed-v1",
        geometry_mode="d4",
    )

    assert captured["args"][-5:] == [
        "--speed-profile",
        "wider-speed-v1",
        "--geometry",
        "d4",
        "--no-audio",
    ]


def test_match_dir_argv_keeps_v1_speed_and_identity_geometry_defaults(monkeypatch, tmp_path):
    controller = CLIController()
    captured = {}
    monkeypatch.setattr(
        controller,
        "start",
        lambda args, **kwargs: captured.update(args=args, kwargs=kwargs),
    )

    controller.start_match_dir("clips", "index", tmp_path / "out")

    assert "--speed-profile" not in captured["args"]
    assert captured["args"][-3:] == ["--geometry", "identity", "--no-audio"]


def test_utf8_split_across_pipe_reads_is_preserved(monkeypatch):
    code = "import os,time;raw='café'.encode();os.write(2,raw[:-1]);time.sleep(.05);os.write(2,raw[-1:]+b'\\n');print('{}')"
    backend(monkeypatch, code)
    controller = CLIController()
    controller.start(["index", "lib", "--out", "out"])
    assert controller.wait(5)
    logs = []
    while not controller.events.empty():
        event = controller.events.get_nowait()
        if event["kind"] == "log":
            logs.append(event["text"])
    assert "café" in logs


def test_child_that_holds_pipe_open_is_reaped_with_process_group(monkeypatch, tmp_path):
    pid_file = tmp_path / "child.pid"
    code = (
        "import pathlib,subprocess,sys;"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)']);"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid));"
        "print('{}')"
    )
    backend(monkeypatch, code)
    controller = CLIController()
    controller.start(["index", "lib", "--out", "out"])
    assert controller.wait(7)
    child_pid = int(pid_file.read_text())
    try:
        state_path = Path(f"/proc/{child_pid}/stat")
        end = time.monotonic() + 2
        while state_path.exists():
            try:
                if state_path.read_text().split()[2] == "Z":
                    break
            except FileNotFoundError:
                break
            if time.monotonic() >= end:
                pytest.fail("owned descendant is still running")
            time.sleep(0.02)
    finally:
        try:
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
