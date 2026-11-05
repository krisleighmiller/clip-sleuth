"""Native desktop adapter for the authoritative Clip Provenance CLI."""

from __future__ import annotations

import tkinter as tk
import webbrowser
from pathlib import Path
from queue import Empty
from tkinter import filedialog, ttk

from clipprov.gui_controller import CLIController


class ClipProvenanceGUI:
    """All widget access stays on the Tk thread; backend work is process-owned."""

    def __init__(self, root):
        self.root = root
        root.title("Clip Sleuth")
        root.minsize(850, 650)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.controller = CLIController()
        self.current_workflow = "index"
        self.running = self.finished = self.closed = self._closing = False
        self._pending_done = None
        self._after_id = None
        self._report_error = None
        self._job_kind = None
        self.workflow_var = tk.StringVar(value="index")
        for name in [
            "library_path",
            "index_out_path",
            "clips_path",
            "index_path",
            "match_out_path",
            "calibration_path",
        ]:
            setattr(self, name + "_var", tk.StringVar())
        self.resume_var = tk.BooleanVar(value=True)
        self.continue_on_error_var = tk.BooleanVar(value=True)
        self.evidence_var = tk.BooleanVar(value=True)
        self.audio_enabled_var = tk.BooleanVar(value=False)
        self.speed_profile_var = tk.StringVar(value="v1-default")
        self.geometry_mode_var = tk.StringVar(value="identity")
        self.status_var = tk.StringVar(value="Choose a workflow and its folders.")
        self.progress_var = tk.DoubleVar(value=0)
        self._inputs = []
        self._build()
        root.bind("<Escape>", lambda _event: self._on_cancel())
        self._after_id = root.after(25, self._poll)

    def _build(self):
        outer = ttk.Frame(self.root, padding=18)
        outer.grid(sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(6, weight=1)
        ttk.Label(outer, text="Clip Sleuth", font="TkHeadingFont").grid(sticky="w")
        ttk.Label(outer, text="Find the source and timestamp. Keep the evidence.").grid(
            sticky="w", pady=(4, 12)
        )
        selector = ttk.Frame(outer)
        selector.grid(sticky="ew", pady=(0, 12))
        for label, value in [
            ("Index a source library", "index"),
            ("Match a clip folder", "match_dir"),
        ]:
            button = ttk.Radiobutton(
                selector,
                text=label,
                variable=self.workflow_var,
                value=value,
                command=self._on_workflow_change,
            )
            button.pack(side="left", padx=(0, 20))
            self._inputs.append(button)
        self._index_frame = ttk.LabelFrame(outer, text="Source library", padding=12)
        self._index_frame.grid(row=3, sticky="ew")
        self._match_frame = ttk.LabelFrame(outer, text="Clip matching", padding=12)
        self._match_frame.grid(row=3, sticky="ew")
        self._path_row(self._index_frame, 0, "Source video folder", "library_path")
        self._path_row(self._index_frame, 1, "Output index folder", "index_out_path")
        self._path_row(self._match_frame, 0, "Clip folder", "clips_path")
        self._path_row(self._match_frame, 1, "Existing index", "index_path")
        self._path_row(self._match_frame, 2, "Output report folder", "match_out_path")
        self._path_row(
            self._match_frame, 3, "Calibration JSON (optional)", "calibration_path", file=True
        )
        options = ttk.Frame(outer)
        options.grid(row=4, sticky="ew", pady=12)
        for text, variable in [
            ("Resume verified work", self.resume_var),
            ("Continue after isolated errors", self.continue_on_error_var),
            ("Paired visual evidence", self.evidence_var),
            ("Audio corroboration", self.audio_enabled_var),
        ]:
            button = ttk.Checkbutton(options, text=text, variable=variable)
            button.pack(side="left", padx=(0, 14))
            self._inputs.append(button)
        match_options = ttk.Frame(outer)
        match_options.grid(row=5, sticky="ew", pady=(0, 12))
        ttk.Label(match_options, text="Speed search").pack(side="left", padx=(0, 6))
        self.speed_profile_combo = ttk.Combobox(
            match_options,
            textvariable=self.speed_profile_var,
            values=("v1-default", "wider-speed-v1"),
            state="readonly",
            width=18,
        )
        self.speed_profile_combo.pack(side="left", padx=(0, 18))
        ttk.Label(match_options, text="Geometry").pack(side="left", padx=(0, 6))
        self.geometry_mode_combo = ttk.Combobox(
            match_options,
            textvariable=self.geometry_mode_var,
            values=("identity", "d4"),
            state="readonly",
            width=12,
        )
        self.geometry_mode_combo.pack(side="left")
        self._inputs.extend((self.speed_profile_combo, self.geometry_mode_combo))
        self.log_text = tk.Text(outer, height=13, wrap="word", state="disabled", font="TkFixedFont")
        self.log_text.grid(row=6, sticky="nsew")
        self.progress_bar = ttk.Progressbar(outer, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=7, sticky="ew", pady=(12, 4))
        ttk.Label(outer, textvariable=self.status_var, wraplength=780).grid(
            row=8, sticky="w", pady=4
        )
        actions = ttk.Frame(outer)
        actions.grid(row=9, sticky="ew", pady=(10, 0))
        self.run_button = ttk.Button(actions, text="Run", command=self._on_run, padding=(16, 8))
        self.run_button.pack(side="left")
        self.cancel_button = ttk.Button(
            actions, text="Cancel", command=self._on_cancel, state="disabled", padding=(16, 8)
        )
        self.cancel_button.pack(side="left", padx=8)
        self.open_report_button = ttk.Button(
            actions,
            text="Open verified report",
            command=self._on_open_report,
            state="disabled",
            padding=(16, 8),
        )
        self.open_report_button.pack(side="right")
        self._on_workflow_change()

    def _path_row(self, parent, row, label, name, *, file=False):
        parent.columnconfigure(1, weight=1)
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=5)
        variable = getattr(self, name + "_var")
        entry = ttk.Entry(parent, textvariable=variable, width=55)
        entry.grid(row=row, column=1, sticky="ew", pady=5)
        setattr(self, name + "_entry", entry)

        def browse():
            value = (
                filedialog.askopenfilename(parent=self.root, title=label)
                if file
                else filedialog.askdirectory(parent=self.root, title=label)
            )
            if value:
                variable.set(value)

        button = ttk.Button(parent, text="Browse…", command=browse)
        button.grid(row=row, column=2, padx=(8, 0), pady=5)
        self._inputs.extend((entry, button))

    def _on_workflow_change(self):
        if self.running:
            self.workflow_var.set(self.current_workflow)
            return
        self.current_workflow = self.workflow_var.get()
        if self.current_workflow == "index":
            self._match_frame.grid_remove()
            self._index_frame.grid()
        else:
            self._index_frame.grid_remove()
            self._match_frame.grid()
        self.open_report_button.configure(state="disabled")

    def _log(self, text):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", str(text) + "\n")
        count = int(self.log_text.index("end-1c").split(".")[0])
        if count > 2000:
            self.log_text.delete("1.0", f"{count - 2000}.0")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _visible_log(self):
        return self.log_text.get("1.0", "end-1c")

    def _set_running(self, kind):
        self.running, self.finished = True, False
        self._job_kind = kind
        self._pending_done = None
        self.progress_var.set(0)
        for widget in self._inputs:
            widget.configure(state="disabled")
        self.run_button.configure(state="disabled")
        self.open_report_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.status_var.set("Running…")

    def _on_run(self):
        if self.running or self._closing:
            return
        try:
            if self.current_workflow == "index":
                library, out = self.library_path_var.get(), self.index_out_path_var.get()
                if not Path(library).is_dir() or not out:
                    raise ValueError("Choose an existing source folder and an output index folder.")
                self.controller.start_index(
                    library, out, self.resume_var.get(), self.continue_on_error_var.get()
                )
            else:
                clips, index, out = (
                    self.clips_path_var.get(),
                    self.index_path_var.get(),
                    self.match_out_path_var.get(),
                )
                if not Path(clips).is_dir() or not Path(index).is_dir() or not out:
                    raise ValueError(
                        "Choose existing clips/index folders and an output report folder."
                    )
                self.controller.start_match_dir(
                    clips,
                    index,
                    out,
                    self.resume_var.get(),
                    self.continue_on_error_var.get(),
                    self.evidence_var.get(),
                    self.calibration_path_var.get() or None,
                    speed_profile=(
                        None
                        if self.speed_profile_var.get() == "v1-default"
                        else self.speed_profile_var.get()
                    ),
                    geometry_mode=self.geometry_mode_var.get(),
                    audio_enabled=self.audio_enabled_var.get(),
                )
            self._set_running(self.current_workflow)
        except (OSError, ValueError, RuntimeError) as exc:
            self.status_var.set("Cannot start: " + str(exc))
            self._log(self.status_var.get())

    def _on_cancel(self):
        if self.running:
            self.controller.cancel()
            self.status_var.set("Cancelling, preserving completed checkpoints…")

    def _on_open_report(self):
        if self.running or self._closing:
            return
        self._report_error = None
        try:
            self.controller.start_report(self.match_out_path_var.get())
            self._set_running("report")
        except (OSError, ValueError, RuntimeError) as exc:
            self._report_error = str(exc)
            self.status_var.set("Cannot open report: " + str(exc))

    def _complete(self, event):
        self.running, self.finished = False, True
        for widget in self._inputs:
            widget.configure(state="normal")
        self.speed_profile_combo.configure(state="readonly")
        self.geometry_mode_combo.configure(state="readonly")
        self.run_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")
        summary = event.get("summary") or {}
        if event.get("cancelled"):
            self.status_var.set("Cancelled. Completed checkpoints are available for resume.")
        elif event.get("error") or event["returncode"] not in (0, 1):
            self.status_var.set("Failed. See the diagnostic log.")
            if self._job_kind == "report":
                self._report_error = event.get("error") or "Report integrity validation failed"
        elif self._job_kind == "report":
            uri = summary.get("uri")
            if isinstance(uri, str) and uri.startswith("file:"):
                webbrowser.open(uri)
                self.status_var.set("Opened integrity-verified local report.")
            else:
                self._report_error = "Resolver did not return a local report URI"
                self.status_var.set(self._report_error)
        else:
            failures = summary.get("failures", 0)
            failures = len(failures) if isinstance(failures, list) else failures
            partial = (
                summary.get("status") == "complete_with_errors"
                or bool(failures)
                or event["returncode"] == 1
            )
            self.status_var.set(
                "Completed with isolated errors. Review the report/log."
                if partial
                else "Completed successfully."
            )
            if self._job_kind == "match_dir" and isinstance(summary.get("report"), str):
                self.open_report_button.configure(state="normal")
        self._pending_done = None

    def _poll(self):
        self._after_id = None
        try:
            while True:
                event = self.controller.events.get_nowait()
                if event["kind"] == "log":
                    self._log(event["text"])
                elif event["kind"] == "progress":
                    data = event["payload"]
                    self.status_var.set(
                        str(data.get("event", "Running"))
                        + ": "
                        + str(data.get("clip", data.get("source", "")))
                    )
                    total = data.get("total")
                    position = data.get("position")
                    if isinstance(total, int) and total > 0 and isinstance(position, int):
                        self.progress_var.set(position / total * 100)
                elif event["kind"] == "done":
                    self._pending_done = event
        except Empty:
            pass
        if self._pending_done and not self.controller.busy:
            self._complete(self._pending_done)
        if self._closing and not self.controller.busy:
            self._destroy()
            return
        if not self.closed:
            self._after_id = self.root.after(25, self._poll)

    def close(self):
        if self.closed:
            return
        self._closing = True
        self.controller.shutdown()
        self.run_button.configure(state="disabled")
        if not self.controller.busy:
            self._destroy()
        else:
            self.status_var.set("Stopping owned job before closing…")

    def _destroy(self):
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None
        self.closed = True
        self.root.destroy()


def main():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise SystemExit(
            "A desktop display is required for the GUI; use clip-prov for headless runs."
        ) from exc
    ClipProvenanceGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
