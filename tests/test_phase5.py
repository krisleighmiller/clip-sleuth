from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from clipprov.batch import (
    BatchError,
    _load_calibration,
    current_report_dir,
    discover_clips,
    match_directory,
    render_reports,
)
from clipprov.batch import (
    _implementation_receipt as batch_implementation_receipt,
)
from clipprov.config import Config
from clipprov.core import MatchDiagnostics
from clipprov.index import IndexError as ClipIndexError
from clipprov.index import _ManifestSnapshot, build_index, load_index
from clipprov.index_batch import IndexBatchError, build_index_resumable
from clipprov.index_batch import (
    _implementation_receipt as index_implementation_receipt,
)
from clipprov.media import MediaError


def _video(path: Path, pattern="testsrc", seconds=2):
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
         f"{pattern}=size=96x64:rate=8", "-t", str(seconds), "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )


def _diag(name="master_00"):
    return MatchDiagnostics(
        best_score=0.9, second_score=0.4, margin=0.5, best_master=name,
        raw_alignment_offset_sec=0.0, source_offset_sec=0.0, speed=1.0,
        accepted=True, accepted_master=name, accepted_offset_sec=0.0,
        accepted_speed=1.0, reason="matched",
        ranked=[(0.9, name, 0.0, 1.0)], shortlist=[name],
        acceptance_floor=0.35, acceptance_margin=0.12,
        supporting_samples=3, sample_count=3, support_fraction=1.0,
        temporal_residual_sec=0.0, offset_profile_width_sec=0.0,
        offset_ambiguity=0.5, reframe_index=0, reframe=(1.0, 0.5, 0.5),
        aligned_timestamps=((0.0, 0.0), (0.5, 0.5), (1.0, 1.0)),
        decision="matched", source_confidence="uncalibrated",
        timing_confidence="uncalibrated",
    )


def _manifest(master: Path):
    return {
        "masters": [{
            "id": "master_00", "source_relpath": master.name,
            "display_name": master.name, "original_root_hint": str(master.parent),
            "content_digest": hashlib.sha256(master.read_bytes()).hexdigest(),
        }]
    }


def _snapshot_manifest(value: dict, source_path: Path) -> _ManifestSnapshot:
    """Mirror production load_index: bind the parsed manifest to its file bytes."""
    return _ManifestSnapshot(value, hashlib.sha256(Path(source_path).read_bytes()).hexdigest())


def test_discovery_is_recursive_deterministic_and_skips_symlinks(tmp_path):
    root = tmp_path / "clips"
    (root / "z").mkdir(parents=True)
    (root / "a").mkdir()
    for relative in ("z/b.MP4", "a/c.mkv", "a/a.mp4"):
        path = root / relative
        path.write_bytes(relative.encode())
    (root / "link.mp4").symlink_to(root / "a/a.mp4")
    (root / "linked-dir").symlink_to(root / "a", target_is_directory=True)
    assert [row["relpath"] for row in discover_clips(root)] == [
        "a/a.mp4", "a/c.mkv", "z/b.MP4",
    ]


def test_batch_interrupt_resume_failures_and_reproducible_exports(tmp_path, monkeypatch):
    clips = tmp_path / "clips"
    clips.mkdir()
    for name in ("b.mp4", "<evil>.mp4", "corrupt.mp4"):
        (clips / name).write_bytes(name.encode())
    index = tmp_path / "index"
    index.mkdir()
    (index / "manifest.json").write_text('{"index":"fixture"}\n')
    master = tmp_path / "master_00.mp4"
    master.write_bytes(b"master")
    manifest = _manifest(master)
    calls = []

    monkeypatch.setattr("clipprov.batch.load_index", lambda _path: (_snapshot_manifest(manifest, index / "manifest.json"), {}))

    def fake_match(path, *_args, **_kwargs):
        calls.append(path.name)
        if path.name == "corrupt.mp4":
            raise ValueError("decode <failed>")
        return _diag(), manifest

    monkeypatch.setattr("clipprov.batch.match_clip_path", fake_match)
    out = tmp_path / "out"
    with pytest.raises(KeyboardInterrupt):
        match_directory(clips, index, out, evidence=False, interrupt_after=1)
    assert (out / ".batch-progress.json").is_file()
    report = match_directory(clips, index, out, evidence=False)
    assert [row["clip"] for row in report["results"]] == ["<evil>.mp4", "b.mp4"]
    assert report["failures"] == [{
        "clip": "corrupt.mp4", "type": "ValueError", "message": "decode <failed>",
    }]
    assert report["run_receipt"]["resumed_count"] == 1
    assert calls.count("<evil>.mp4") == 1
    assert calls.count("corrupt.mp4") == 1
    assert calls.count("b.mp4") == 1
    report_dir = current_report_dir(out)
    assert (report_dir / "report.csv").is_file()
    assert (report_dir / "source-summary.json").is_file()
    assert (report_dir / "queues" / "decision-matched.json").is_file()
    report_json = (report_dir / "report.json").read_bytes()
    second_report_dir = render_reports(report, out)
    assert (second_report_dir / "report.json").read_bytes() == report_json
    page = (second_report_dir / "report.html").read_text()
    assert "&lt;evil&gt;.mp4" in page and "decode &lt;failed&gt;" in page
    assert "https://" not in page and "http://" not in page
    assert report["performance"]["one_loaded_index_reused"] is True
    assert report["performance"]["peak_rss_bytes"] < report["performance"]["memory_ceiling_bytes"]
    assert report["performance"]["clips_per_sec"] >= report["performance"]["throughput_target_clips_per_sec"]


def test_batch_resume_rejects_changed_input(tmp_path, monkeypatch):
    clips = tmp_path / "clips"
    clips.mkdir()
    clip = clips / "a.mp4"
    clip.write_bytes(b"a")
    index = tmp_path / "index"
    index.mkdir()
    (index / "manifest.json").write_text("{}")
    master = tmp_path / "master_00.mp4"
    master.write_bytes(b"master")
    manifest = _manifest(master)
    monkeypatch.setattr("clipprov.batch.load_index", lambda _path: (_snapshot_manifest(manifest, index / "manifest.json"), {}))
    monkeypatch.setattr("clipprov.batch.match_clip_path", lambda *_args, **_kwargs: (_diag(), manifest))
    match_directory(clips, index, tmp_path / "out", evidence=False)
    clip.write_bytes(b"changed")
    with pytest.raises(BatchError, match="does not match"):
        match_directory(clips, index, tmp_path / "out", evidence=False)


@pytest.mark.skipif(not Path("/usr/bin/ffmpeg").exists(), reason="ffmpeg required")
def test_index_interruption_resumes_verified_shards(tmp_path):
    library = tmp_path / "library"
    library.mkdir()
    _video(library / "a.mp4", "testsrc")
    _video(library / "b.mp4", "testsrc2")
    index = tmp_path / "index"
    with pytest.raises(KeyboardInterrupt):
        build_index_resumable(library, index, Config(), interrupt_after=1)
    assert (index / ".index-progress.json").is_file()
    journal = json.loads((index / ".index-progress.json").read_text())
    assert len(journal["identity"]["implementation_sha256"]) == 64
    assert not (index / "manifest.json").exists()
    manifest = build_index_resumable(library, index, Config())
    assert manifest["status"] == "complete"
    assert manifest["build_receipt"]["resumed_masters"] == 1
    loaded, masters = load_index(index)
    assert len(loaded["masters"]) == len(masters) == 2


@pytest.mark.skipif(not Path("/usr/bin/ffmpeg").exists(), reason="ffmpeg required")
def test_index_continues_after_bad_master_and_preserves_last_valid_manifest(tmp_path):
    library = tmp_path / "library"
    library.mkdir()
    _video(library / "a.mp4")
    (library / "z.mp4").write_bytes(b"not video")
    index = tmp_path / "index"
    manifest = build_index_resumable(library, index, Config(), continue_on_error=True)
    assert manifest["status"] == "complete_with_errors"
    assert [row["source_relpath"] for row in manifest["failures"]] == ["z.mp4"]
    load_index(index)
    before = (index / "manifest.json").read_bytes()
    (library / "y.mp4").write_bytes(b"also bad")
    with pytest.raises(MediaError):
        build_index_resumable(library, index, Config(), resume=False, continue_on_error=False)
    assert (index / "manifest.json").read_bytes() == before
    load_index(index)


def test_html_uses_phase4_evidence_paths_without_reconstruction(tmp_path):
    frame_dir = tmp_path / "evidence" / "generations" / "abc"
    frame_dir.mkdir(parents=True)
    for name in ("00-clip.jpg", "00-master.jpg"):
        (frame_dir / name).write_bytes(b"fixture")
    result = {
        "best_master": "master_00", "best_score": 0.9,
        "confidence": {"decision": "uncertain", "source": {"tier": "review"}, "timing": {"tier": "review"}},
        "evidence": {"directory": str(frame_dir), "frames": [{
            "clip_sec": 1.25, "master_sec": 9.5,
            "clip_frame": "00-clip.jpg", "master_frame": "00-master.jpg",
        }]},
    }
    report = {
        "results": [{"clip": "clip.mp4", "result": result}], "failures": [],
        "run_receipt": {}, "inventory": [], "performance": {}, "schema_version": 1,
    }
    report_dir = render_reports(report, tmp_path)
    page = (report_dir / "report.html").read_text()
    assert page.count("data:image/jpeg;base64,") == 2
    assert "clip 1.250s · master 9.500s" in page
    assert "ffmpeg" not in page.lower()


@pytest.mark.skipif(not Path("/usr/bin/ffmpeg").exists(), reason="ffmpeg required")
def test_index_journal_extra_success_cannot_replace_last_valid_manifest(tmp_path):
    library = tmp_path / "library"
    library.mkdir()
    _video(library / "a.mp4", "testsrc")
    index = tmp_path / "index"
    build_index_resumable(library, index, Config())
    previous = (index / "manifest.json").read_bytes()
    _video(library / "b.mp4", "testsrc2")
    with pytest.raises(KeyboardInterrupt):
        build_index_resumable(library, index, Config(), interrupt_after=1)
    journal_path = index / ".index-progress.json"
    journal = json.loads(journal_path.read_text())
    first = next(iter(journal["entries"].values()))
    journal["entries"]["outside"] = copy.deepcopy(first)
    journal_path.write_text(json.dumps(journal))
    with pytest.raises(IndexBatchError, match="outside the inventory"):
        build_index_resumable(library, index, Config())
    assert (index / "manifest.json").read_bytes() == previous
    manifest, masters = load_index(index)
    assert [row["id"] for row in manifest["masters"]] == ["a"]
    assert set(masters) == {"a"}


def test_batch_journal_extra_completion_is_rejected(tmp_path, monkeypatch):
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "a.mp4").write_bytes(b"a")
    index = tmp_path / "index"
    index.mkdir()
    (index / "manifest.json").write_text("{}")
    master = tmp_path / "master_00.mp4"
    master.write_bytes(b"master")
    manifest = _manifest(master)
    monkeypatch.setattr("clipprov.batch.load_index", lambda _path: (_snapshot_manifest(manifest, index / "manifest.json"), {}))
    monkeypatch.setattr("clipprov.batch.match_clip_path", lambda *_args, **_kwargs: (_diag(), manifest))
    out = tmp_path / "out"
    match_directory(clips, index, out, evidence=False)
    journal_path = out / ".batch-progress.json"
    journal = json.loads(journal_path.read_text())
    journal["completed"]["outside.mp4"] = copy.deepcopy(journal["completed"]["a.mp4"])
    journal_path.write_text(json.dumps(journal))
    with pytest.raises(BatchError, match="outside the input inventory"):
        match_directory(clips, index, out, evidence=False)


def test_journals_bind_implementation_identity(tmp_path, monkeypatch):
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "a.mp4").write_bytes(b"a")
    index = tmp_path / "index"
    index.mkdir()
    (index / "manifest.json").write_text("{}")
    master = tmp_path / "master_00.mp4"
    master.write_bytes(b"master")
    manifest = _manifest(master)
    monkeypatch.setattr("clipprov.batch.load_index", lambda _path: (_snapshot_manifest(manifest, index / "manifest.json"), {}))
    monkeypatch.setattr("clipprov.batch.match_clip_path", lambda *_args, **_kwargs: (_diag(), manifest))
    out = tmp_path / "out"
    match_directory(clips, index, out, evidence=False)
    journal_path = out / ".batch-progress.json"
    journal = json.loads(journal_path.read_text())
    assert len(journal["identity"]["implementation_sha256"]) == 64
    journal["identity"]["implementation_sha256"] = "0" * 64
    journal_path.write_text(json.dumps(journal))
    with pytest.raises(BatchError, match="implementation"):
        match_directory(clips, index, out, evidence=False)


def test_report_generation_failure_never_mixes_published_files(tmp_path, monkeypatch):
    report = {
        "results": [], "failures": [], "run_receipt": {}, "inventory": [],
        "performance": {}, "schema_version": 1,
    }
    first = render_reports(report, tmp_path)
    first_pointer = (tmp_path / "current.json").read_bytes()
    first_csv = (first / "report.csv").read_bytes()
    real_atomic_text = __import__("clipprov.batch", fromlist=["atomic_text"]).atomic_text

    def fail_csv(path, text):
        if Path(path).name == "report.csv":
            raise OSError("injected csv publication failure")
        return real_atomic_text(path, text)

    monkeypatch.setattr("clipprov.batch.atomic_text", fail_csv)
    changed = copy.deepcopy(report)
    changed["failures"] = [{"clip": "x.mp4", "type": "Injected", "message": "new"}]
    with pytest.raises(OSError, match="injected"):
        render_reports(changed, tmp_path)
    assert (tmp_path / "current.json").read_bytes() == first_pointer
    current = current_report_dir(tmp_path)
    assert current == first
    assert (current / "report.csv").read_bytes() == first_csv
    assert json.loads((current / "report.json").read_text()) == report


def test_calibration_validation_rejects_semantically_invalid_signed_receipt(tmp_path):
    value = {
        "method": "development_medians_v1",
        "semantics": "diagnostic_not_probability",
        "development_rows": 2,
        "eligible_correct_alignments": 2,
        "thresholds": {
            "strong_score": 0.8,
            "strong_margin": 0.2,
            "focused_support_fraction": 2.0,
            "focused_profile_width_sec": 0.5,
        },
    }
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    value["calibration_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(value))
    with pytest.raises(BatchError, match="focused_support_fraction"):
        _load_calibration(path)


@pytest.mark.skipif(not Path("/usr/bin/ffmpeg").exists(), reason="ffmpeg required")
def test_two_frame_master_is_rejected_by_native_and_resumable_index(tmp_path):
    library = tmp_path / "library"
    library.mkdir()
    short = library / "short.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
         "testsrc=size=96x64:rate=8", "-frames:v", "2", "-pix_fmt", "yuv420p", str(short)],
        check=True,
    )
    with pytest.raises(MediaError, match="at least 4"):
        build_index(library, tmp_path / "native", Config())
    with pytest.raises(MediaError, match="at least 4"):
        build_index_resumable(library, tmp_path / "resumable", Config())


@pytest.mark.skipif(not Path("/usr/bin/ffmpeg").exists(), reason="ffmpeg required")
def test_candidate_preflight_failure_preserves_previous_manifest(tmp_path, monkeypatch):
    library = tmp_path / "library"
    library.mkdir()
    _video(library / "a.mp4")
    index = tmp_path / "index"
    build_index_resumable(library, index, Config())
    previous = (index / "manifest.json").read_bytes()
    real_load = __import__("clipprov.index_batch", fromlist=["load_index"]).load_index

    def fail_candidate(path, **kwargs):
        if kwargs.get("manifest_name") == ".manifest-candidate.json":
            raise ClipIndexError("injected candidate rejection")
        return real_load(path, **kwargs)

    monkeypatch.setattr("clipprov.index_batch.load_index", fail_candidate)
    with pytest.raises(ClipIndexError, match="candidate rejection"):
        build_index_resumable(library, index, Config(), resume=False)
    assert (index / "manifest.json").read_bytes() == previous
    load_index(index)


@pytest.mark.skipif(not Path("/usr/bin/ffmpeg").exists(), reason="ffmpeg required")
def test_source_replacement_after_extraction_cannot_change_manifest_identity(tmp_path, monkeypatch):
    library = tmp_path / "library"
    library.mkdir()
    source = library / "a.mp4"
    replacement = tmp_path / "replacement.mp4"
    _video(source, "testsrc")
    _video(replacement, "testsrc2")
    index = tmp_path / "index"
    build_index_resumable(library, index, Config())
    previous = (index / "manifest.json").read_bytes()
    real_build = __import__("clipprov.index_batch", fromlist=["build_index"]).build_index

    def replace_after_extraction(*args, **kwargs):
        result = real_build(*args, **kwargs)
        os.replace(replacement, source)
        return result

    monkeypatch.setattr("clipprov.index_batch.build_index", replace_after_extraction)
    with pytest.raises(IndexBatchError, match="source changed during indexing"):
        build_index_resumable(library, index, Config(), resume=False)
    assert (index / "manifest.json").read_bytes() == previous
    manifest, _ = load_index(index)
    assert manifest["masters"][0]["content_digest"] != hashlib.sha256(source.read_bytes()).hexdigest()


def test_current_report_rejects_every_damaged_or_missing_listed_artifact(tmp_path):
    report = {
        "results": [], "failures": [], "run_receipt": {}, "inventory": [],
        "performance": {}, "schema_version": 1,
    }
    generation = render_reports(report, tmp_path)
    (generation / "report.html").write_text("damaged")
    with pytest.raises(BatchError, match="integrity"):
        current_report_dir(tmp_path)
    shutil.rmtree(tmp_path)
    generation = render_reports(report, tmp_path)
    (generation / "report.csv").unlink()
    with pytest.raises(BatchError, match="missing"):
        current_report_dir(tmp_path)


def test_implementation_fingerprints_cover_appearance_and_core(tmp_path):
    source_root = Path(__import__("clipprov").__file__).resolve().parent
    copy_root = tmp_path / "clipprov"
    copy_root.mkdir()
    names = {
        Path(row["path"]).name
        for row in batch_implementation_receipt()["files"]
    } | {
        Path(row["path"]).name
        for row in index_implementation_receipt()["files"]
    }
    for name in names:
        shutil.copy2(source_root / name, copy_root / name)
    index_before = index_implementation_receipt(copy_root)["sha256"]
    batch_before = batch_implementation_receipt(copy_root)["sha256"]
    (copy_root / "appearance.py").write_text((copy_root / "appearance.py").read_text() + "\n# semantic change\n")
    assert index_implementation_receipt(copy_root)["sha256"] != index_before
    assert batch_implementation_receipt(copy_root)["sha256"] != batch_before
    batch_after_appearance = batch_implementation_receipt(copy_root)["sha256"]
    (copy_root / "core.py").write_text((copy_root / "core.py").read_text() + "\n# decision change\n")
    assert batch_implementation_receipt(copy_root)["sha256"] != batch_after_appearance
    batch_after_core = batch_implementation_receipt(copy_root)["sha256"]
    assert "audio.py" in names
    (copy_root / "audio.py").write_text((copy_root / "audio.py").read_text() + "\n# audio change\n")
    assert batch_implementation_receipt(copy_root)["sha256"] != batch_after_core


def test_report_generation_is_synced_before_pointer_publication(tmp_path, monkeypatch):
    report = {
        "results": [], "failures": [], "run_receipt": {}, "inventory": [],
        "performance": {}, "schema_version": 1,
    }
    events = []
    real_sync = __import__("clipprov.batch", fromlist=["_sync_directory"])._sync_directory
    real_atomic = __import__("clipprov.batch", fromlist=["atomic_json"]).atomic_json

    def record_sync(path):
        events.append(("sync", Path(path).name))
        return real_sync(path)

    def record_atomic(path, value):
        if Path(path).name == "current.json":
            events.append(("pointer", "current.json"))
        return real_atomic(path, value)

    monkeypatch.setattr("clipprov.batch._sync_directory", record_sync)
    monkeypatch.setattr("clipprov.batch.atomic_json", record_atomic)
    render_reports(report, tmp_path)
    pointer_index = events.index(("pointer", "current.json"))
    assert any(event == ("sync", "generations") for event in events[:pointer_index])
    assert any(event[0] == "sync" and event[1].endswith(".tmp") for event in events[:pointer_index])


def test_csv_exports_neutralize_spreadsheet_formulas(tmp_path):
    result = {
        "best_master": "-master", "best_score": 0.9,
        "confidence": {
            "decision": "uncertain", "source": {"tier": "review"},
            "timing": {"tier": "review"},
        },
        "evidence": None,
    }
    report = {
        "results": [{"clip": "=2+2.mp4", "result": result}],
        "failures": [{"clip": "+bad.mp4", "type": "@Error", "message": "-formula"}],
        "run_receipt": {}, "inventory": [], "performance": {}, "schema_version": 1,
    }
    generation = render_reports(report, tmp_path)
    rows = list(csv.reader((generation / "report.csv").open()))
    assert rows[1][0] == "'+bad.mp4"
    assert rows[1][7] == "'@Error"
    assert rows[1][8] == "'-formula"
    assert rows[2][0] == "'=2+2.mp4"
    assert rows[2][3] == "'-master"
    source_rows = list(csv.reader((generation / "source-summary.csv").open()))
    assert source_rows[1][0] == "'-master"
    assert source_rows[1][5] == "'=2+2.mp4"
