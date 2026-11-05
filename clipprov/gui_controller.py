"""One owned CLI process group, non-blocking pipe reads, and UI-neutral events."""

from __future__ import annotations

import codecs
import json
import os
import selectors
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from queue import Queue


def parse_progress_event(text: str) -> dict | None:
    try:
        value = json.loads(text)
    except (ValueError, TypeError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def parse_summary(text: str) -> dict | None:
    return parse_progress_event(text)


class CLIController:
    """Run the installed/checkout CLI without ever calling Tk from a worker."""

    def __init__(self):
        self.process = None
        self.summary = None
        self.returncode = None
        self.error = None
        self.cancelled = False
        self.events = Queue()
        self.done = threading.Event()
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self._starting = False
        self._closed = False
        self._progress = deque(maxlen=2000)

    @property
    def busy(self):
        with self._lock:
            return self._starting or bool(self._thread and self._thread.is_alive())

    @property
    def progress_events(self):
        with self._lock:
            return list(self._progress)

    def _build_cli_env(self):
        env = os.environ.copy()
        env.pop("PYTHONHOME", None)
        env.pop("PYTHONPATH", None)
        return env

    def _get_cloned_clipprov_path(self):
        return Path(__file__).resolve().parent.parent

    def start(self, args, *, log_root=None):
        with self._lock:
            if self._closed:
                raise RuntimeError("controller is closed")
            if self._starting or (self._thread and self._thread.is_alive()):
                raise RuntimeError("a job is already running")
            self._starting = True
            self.summary = self.returncode = self.error = None
            self.cancelled = False
            self._cancel.clear()
            self.done.clear()
            self._progress.clear()
            self.events = Queue()
            self._thread = threading.Thread(
                target=self._worker,
                args=(tuple(args), log_root),
                name="clipprov-cli-job",
                daemon=False,
            )
            self._thread.start()

    def _line(self, stream, line):
        if stream == "stderr":
            event = parse_progress_event(line)
            if event is not None and isinstance(event.get("event"), str):
                with self._lock:
                    self._progress.append(event)
                self.events.put({"kind": "progress", "payload": event})
        self.events.put({"kind": "log", "stream": stream, "text": line})

    def _worker(self, args, log_root):
        process = None
        streams = {}
        logs = {}
        selector = selectors.DefaultSelector()
        stdout_parts = []
        buffers = {"stdout": "", "stderr": ""}
        decoders = {name: codecs.getincrementaldecoder("utf-8")("replace") for name in buffers}
        cancel_stage = 0
        deadline = 0.0
        exited_at = None
        force_group_cleanup = False
        code = 2
        try:
            if self._cancel.is_set():
                self.cancelled = True
                code = 130
                return
            if log_root is not None:
                import uuid

                log_root = Path(log_root)
                log_root.mkdir(parents=True, exist_ok=True)
                job = uuid.uuid4().hex
                for name in buffers:
                    logs[name] = (log_root / f"{job}.{name}.log").open("w", encoding="utf-8")
            launcher = Path(__file__).with_name("_gui_cli.py")
            process = subprocess.Popen(
                [sys.executable, str(launcher), *args],
                cwd=self._get_cloned_clipprov_path(),
                env=self._build_cli_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                bufsize=0,
            )
            with self._lock:
                self.process = process
                self._starting = False
            for name in buffers:
                stream = getattr(process, name)
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, name)
                streams[name] = stream
            while selector.get_map() or process.poll() is None:
                alive = process.poll() is None
                now = time.monotonic()
                if self._cancel.is_set():
                    self.cancelled = True
                    if alive and (cancel_stage == 0 or now >= deadline):
                        sig = (signal.SIGINT, signal.SIGTERM, signal.SIGKILL)[min(cancel_stage, 2)]
                        try:
                            os.killpg(process.pid, sig)
                        except ProcessLookupError:
                            pass
                        cancel_stage += 1
                        deadline = now + (3.0 if cancel_stage == 1 else 2.0)
                for key, _ in selector.select(0.05):
                    name = key.data
                    try:
                        raw = os.read(key.fd, 65536)
                    except BlockingIOError:
                        continue
                    text = decoders[name].decode(raw, final=not raw)
                    if name == "stdout":
                        stdout_parts.append(text)
                    if name in logs:
                        logs[name].write(text)
                        logs[name].flush()
                    buffers[name] += text
                    while "\n" in buffers[name]:
                        line, buffers[name] = buffers[name].split("\n", 1)
                        self._line(name, line)
                    if not raw:
                        if buffers[name]:
                            self._line(name, buffers[name])
                            buffers[name] = ""
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                if not alive:
                    exited_at = exited_at or now
                    if selector.get_map() and now - exited_at > 2:
                        raise RuntimeError("CLI exited but left an output pipe open")
            code = process.wait()
            self.summary = parse_summary("".join(stdout_parts))
            if code in (0, 1) and self.summary is None and not self.cancelled:
                raise ValueError("CLI completed without a valid JSON summary")
        except Exception as exc:  # noqa: BLE001 - job boundary reports errors to the UI
            force_group_cleanup = True
            self.error = str(exc)
            self.events.put({"kind": "log", "stream": "stderr", "text": self.error})
            code = 2
        finally:
            if process is not None and (
                force_group_cleanup or self.cancelled or process.poll() is None
            ):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
            selector.close()
            for stream in streams.values():
                stream.close()
            for handle in logs.values():
                handle.close()
            with self._lock:
                self.process = None
                self._starting = False
                self.returncode = code
            self.events.put(
                {
                    "kind": "done",
                    "returncode": code,
                    "summary": self.summary,
                    "cancelled": self.cancelled,
                    "error": self.error,
                }
            )
            self.done.set()

    def cancel(self):
        self._cancel.set()

    def shutdown(self):
        with self._lock:
            self._closed = True
        self.cancel()

    def wait(self, timeout=15):
        started = time.monotonic()
        if not self.done.wait(timeout):
            return False
        thread = self._thread
        if thread:
            thread.join(max(0, timeout - (time.monotonic() - started)))
        return not self.busy

    def close(self, timeout=10):
        self.shutdown()
        if not self._thread:
            return True
        return self.wait(timeout)

    def start_index(self, library, out, resume=True, continue_on_error=True):
        args = [
            "index",
            str(library),
            "--out",
            str(out),
            "--resume" if resume else "--no-resume",
            "--continue-on-error" if continue_on_error else "--fail-fast",
        ]
        self.start(args, log_root=Path(out) / ".gui" / "logs")

    def start_match_dir(
        self,
        clips,
        index,
        out,
        resume=True,
        continue_on_error=True,
        evidence=True,
        calibration=None,
        speed_profile=None,
        geometry_mode="identity",
        audio_enabled=False,
    ):
        args = [
            "match-dir",
            str(clips),
            "--index",
            str(index),
            "--out",
            str(out),
            "--resume" if resume else "--no-resume",
            "--continue-on-error" if continue_on_error else "--fail-fast",
            "--evidence" if evidence else "--no-evidence",
        ]
        if calibration:
            args += ["--calibration", str(calibration)]
        if speed_profile:
            args += ["--speed-profile", str(speed_profile)]
        args += ["--geometry", str(geometry_mode)]
        args += ["--audio" if audio_enabled else "--no-audio"]
        self.start(args, log_root=Path(out) / ".gui" / "logs")

    def start_report(self, out):
        self.start(["--gui-resolve-report", str(out)])
