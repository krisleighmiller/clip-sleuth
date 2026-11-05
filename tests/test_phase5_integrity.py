"""Phase 5 batch integrity: staged clip identity, snapshot receipts,
journal source binding, and calibration tolerance receipts.

The index manifest reader, writer locks, and truncated-shard recovery are
covered by tests/test_index_snapshot_repair.py and are deliberately not
duplicated here.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import clipprov.match as match_module
from clipprov._io import digest
from clipprov.batch import BatchError, _load_calibration, match_directory
from clipprov.confidence import derive_tiers
from clipprov.config import Config
from clipprov.core import MatchDiagnostics
from clipprov.index import _ManifestSnapshot, build_index, load_index

requires_ffmpeg = pytest.mark.skipif(
    not Path("/usr/bin/ffmpeg").exists(), reason="ffmpeg required"
)


def _video(path: Path, pattern: str, seconds: int = 2) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
         f"{pattern}=size=96x64:rate=8", "-t", str(seconds),
         "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )


def _snapshot(value: dict, source_path: Path) -> _ManifestSnapshot:
    return _ManifestSnapshot(
        value, hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
    )


def _matched_diag(master_id: str = "source_00") -> MatchDiagnostics:
    return MatchDiagnostics(
        best_score=0.9, second_score=0.4, margin=0.5, best_master=master_id,
        raw_alignment_offset_sec=0.0, source_offset_sec=0.0, speed=1.0,
        accepted=True, accepted_master=master_id, accepted_offset_sec=0.0,
        accepted_speed=1.0, reason="matched",
        ranked=[(0.9, master_id, 0.0, 1.0)], shortlist=[master_id],
        acceptance_floor=0.35, acceptance_margin=0.12,
        supporting_samples=3, sample_count=3, support_fraction=1.0,
        temporal_residual_sec=0.0, offset_profile_width_sec=0.0,
        offset_ambiguity=0.5, reframe_index=0, reframe=(1.0, 0.5, 0.5),
        aligned_timestamps=((0.0, 0.0), (0.5, 0.5), (1.0, 1.0)),
        decision="matched",
    )


def _unmatched_diag() -> MatchDiagnostics:
    return MatchDiagnostics(
        best_score=0.2, second_score=0.1, margin=0.1, best_master=None,
        raw_alignment_offset_sec=None, source_offset_sec=None, speed=None,
        accepted=False, accepted_master=None, accepted_offset_sec=None,
        accepted_speed=None, reason="unmatched",
        ranked=[], shortlist=[],
        acceptance_floor=0.35, acceptance_margin=0.12,
        supporting_samples=0, sample_count=0, support_fraction=None,
        temporal_residual_sec=None, offset_profile_width_sec=None,
        offset_ambiguity=None, reframe_index=None, reframe=None,
        aligned_timestamps=(), decision="unmatched",
    )


def _fake_manifest(master_root: Path) -> dict:
    return {
        "masters": [{
            "id": "source_00",
            "source_relpath": "source_00.mp4",
            "display_name": "source_00.mp4",
            "original_root_hint": str(master_root),
            "content_digest": hashlib.sha256(b"master").hexdigest(),
        }],
    }


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("phase5-corpus")
    library = root / "library"
    library.mkdir()
    _video(library / "source_00.mp4", "testsrc")
    index = root / "index"
    build_index(library, index, Config())
    outside = root / "outside"
    outside.mkdir()
    replacement = outside / "B.mp4"
    _video(replacement, "testsrc2")
    return SimpleNamespace(library=library, index=index, replacement=replacement)


@requires_ffmpeg
def test_replace_restore_race_consumes_staged_bytes(corpus, tmp_path, monkeypatch):
    clips = tmp_path / "clips"
    clips.mkdir()
    original = clips / "clip-differentlynamed.mp4"
    shutil.copy2(corpus.library / "source_00.mp4", original)
    a_bytes = original.read_bytes()
    frozen = digest(original)
    staged_paths = []

    def racing_match(clip_path, index_dir, *args, **kwargs):
        clip_path = Path(clip_path)
        staged_paths.append(clip_path)
        assert clip_path != original
        assert digest(clip_path) == frozen
        original.write_bytes(corpus.replacement.read_bytes())
        try:
            return match_module.match_clip_path(clip_path, index_dir, *args, **kwargs)
        finally:
            original.write_bytes(a_bytes)

    monkeypatch.setattr("clipprov.batch.match_clip_path", racing_match)
    out = tmp_path / "out"
    report = match_directory(clips, corpus.index, out, evidence=True)
    assert report["failures"] == []
    result = report["results"][0]["result"]
    assert result["clip"] == str(original)
    assert result["best_master"] == "source_00"
    assert result["evidence"] is not None
    assert result["evidence"]["clip_path"] == str(original)
    assert result["evidence"]["clip_sha256"] == frozen
    persisted = (out / ".batch-progress.json").read_text() + json.dumps(report)
    assert ".clip-" not in persisted
    assert str(staged_paths[0].parent) not in persisted
    assert [p for p in out.iterdir() if p.name.startswith(".clip-")] == []
    monkeypatch.setattr(
        "clipprov.batch.match_clip_path",
        lambda *_a, **_k: pytest.fail("completed clip was recomputed"),
    )
    resumed = match_directory(clips, corpus.index, out, evidence=True)
    assert resumed["run_receipt"]["resumed_count"] == 1


@requires_ffmpeg
def test_swap_before_staging_is_a_contained_failure(corpus, tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    for name in ("a-good.mp4", "b-swap.mp4", "c-restore.mp4"):
        shutil.copy2(corpus.library / "source_00.mp4", clips / name)
    a_bytes = (clips / "b-swap.mp4").read_bytes()

    def progress(event):
        if event["event"] != "matching":
            return
        if event["clip"] == "b-swap.mp4":
            (clips / "b-swap.mp4").write_bytes(corpus.replacement.read_bytes())
        if event["clip"] == "c-restore.mp4":
            (clips / "b-swap.mp4").write_bytes(a_bytes)

    out = tmp_path / "out"
    report = match_directory(
        clips, corpus.index, out, evidence=False, progress=progress
    )
    assert [row["clip"] for row in report["results"]] == ["a-good.mp4", "c-restore.mp4"]
    assert [row["clip"] for row in report["failures"]] == ["b-swap.mp4"]
    assert report["failures"][0]["type"] == "BatchError"
    assert "changed since inventory" in report["failures"][0]["message"]
    journal = json.loads((out / ".batch-progress.json").read_text())
    assert "b-swap.mp4" not in journal["completed"]
    assert [p for p in out.iterdir() if p.name.startswith(".clip-")] == []


def _fake_env(tmp_path, monkeypatch, *, diag=None):
    clips = tmp_path / "clips"
    clips.mkdir()
    clip = clips / "differentlynamed.mp4"
    clip.write_bytes(b"clip-bytes")
    index = tmp_path / "index"
    index.mkdir()
    (index / "manifest.json").write_text('{"generation": 1}')
    snapshot = _snapshot(_fake_manifest(tmp_path / "master-root"), index / "manifest.json")
    match = diag if diag is not None else _matched_diag()
    monkeypatch.setattr("clipprov.batch.load_index", lambda *_a, **_k: (snapshot, {}))
    monkeypatch.setattr(
        "clipprov.batch.match_clip_path", lambda *_a, **_k: (match, snapshot)
    )
    return clip, clips, index, tmp_path / "out"


def _journal(out: Path) -> dict:
    return json.loads((out / ".batch-progress.json").read_text())


def test_resume_binds_source_by_master_id_not_clip_name(tmp_path, monkeypatch):
    clip, clips, index, out = _fake_env(tmp_path, monkeypatch)
    match_directory(clips, index, out, evidence=False)
    monkeypatch.setattr(
        "clipprov.batch.match_clip_path",
        lambda *_a, **_k: pytest.fail("completed clip was recomputed"),
    )
    report = match_directory(clips, index, out, evidence=False)
    assert report["failures"] == []
    assert report["run_receipt"]["resumed_count"] == 1
    assert _journal(out)["completed"]["differentlynamed.mp4"]["clip"] == str(clip)


def test_resume_accepts_valid_nonmatch_without_source_record(tmp_path, monkeypatch):
    _clip, clips, index, out = _fake_env(tmp_path, monkeypatch, diag=_unmatched_diag())
    match_directory(clips, index, out, evidence=False)
    report = match_directory(clips, index, out, evidence=False)
    assert report["failures"] == []
    assert report["run_receipt"]["resumed_count"] == 1


def test_resume_rejects_persistently_swapped_clip(tmp_path, monkeypatch):
    clip, clips, index, out = _fake_env(tmp_path, monkeypatch)
    match_directory(clips, index, out, evidence=False)
    clip.write_bytes(b"swapped-to-B")
    with pytest.raises(BatchError, match="does not match"):
        match_directory(clips, index, out, evidence=False)


def test_resume_accepts_replaced_then_restored_clip(tmp_path, monkeypatch):
    clip, clips, index, out = _fake_env(tmp_path, monkeypatch)
    match_directory(clips, index, out, evidence=False)
    clip.write_bytes(b"swapped-to-B")
    clip.write_bytes(b"clip-bytes")  # restored; digest, not mtime, binds
    report = match_directory(clips, index, out, evidence=False)
    assert report["failures"] == []
    assert report["run_receipt"]["resumed_count"] == 1


@pytest.mark.parametrize(
    ("field", "replacement", "pattern"),
    [
        ("content_digest", "f" * 64, "content_digest changed"),
        ("original_root_hint", "/mutated-root", "original_root_hint changed"),
        ("source_relpath", "other.mp4", "source_relpath changed"),
        ("id", "other_source", "id changed"),
    ],
)
def test_journal_rejects_mutated_source_binding(
    tmp_path, monkeypatch, field, replacement, pattern
):
    _clip, clips, index, out = _fake_env(tmp_path, monkeypatch)
    match_directory(clips, index, out, evidence=False)
    journal = _journal(out)
    journal["completed"]["differentlynamed.mp4"]["source_record"][field] = replacement
    (out / ".batch-progress.json").write_text(json.dumps(journal))
    with pytest.raises(BatchError, match=pattern):
        match_directory(clips, index, out, evidence=False)


def test_journal_rejects_source_absent_from_manifest(tmp_path, monkeypatch):
    _clip, clips, index, out = _fake_env(tmp_path, monkeypatch)
    match_directory(clips, index, out, evidence=False)
    journal = _journal(out)
    record = journal["completed"]["differentlynamed.mp4"]
    record["best_master"] = "ghost"
    record["source_record"]["id"] = "ghost"
    (out / ".batch-progress.json").write_text(json.dumps(journal))
    with pytest.raises(BatchError, match="unknown source"):
        match_directory(clips, index, out, evidence=False)


def test_plain_manifest_without_snapshot_attribute_is_rejected(tmp_path, monkeypatch):
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "a.mp4").write_bytes(b"a")
    index = tmp_path / "index"
    index.mkdir()
    (index / "manifest.json").write_text("{}")
    monkeypatch.setattr(
        "clipprov.batch.load_index", lambda *_a, **_k: (_fake_manifest(tmp_path), {})
    )
    with pytest.raises(BatchError, match="snapshot digest"):
        match_directory(clips, index, tmp_path / "out", evidence=False)


@requires_ffmpeg
def test_identity_uses_loaded_snapshot_not_current_manifest(
    corpus, tmp_path, monkeypatch
):
    index_copy = tmp_path / "index"
    shutil.copytree(corpus.index, index_copy)
    manifest_file = index_copy / "manifest.json"
    frozen_manifest, masters = load_index(index_copy)
    snapshot_sha = frozen_manifest.manifest_sha256
    generation_two = json.loads(manifest_file.read_text())
    generation_two["generation_marker"] = "2"
    manifest_file.write_text(json.dumps(generation_two))
    clips = tmp_path / "clips"
    clips.mkdir()
    shutil.copy2(corpus.library / "source_00.mp4", clips / "clip.mp4")
    out = tmp_path / "out"
    monkeypatch.setattr(
        "clipprov.batch.load_index", lambda *_a, **_k: (frozen_manifest, masters)
    )
    report = match_directory(clips, index_copy, out, evidence=False)
    assert report["failures"] == []
    assert report["run_receipt"]["index_manifest_sha256"] == snapshot_sha
    assert snapshot_sha != digest(manifest_file)
    assert _journal(out)["identity"]["index_manifest_sha256"] == snapshot_sha
    monkeypatch.setattr("clipprov.batch.load_index", load_index)
    with pytest.raises(BatchError, match="does not match"):
        match_directory(clips, index_copy, out, evidence=False)


LEGACY_RECEIPT = {
    "method": "development_medians_v1",
    "semantics": "diagnostic_not_probability",
    "development_rows": 2,
    "eligible_correct_alignments": 2,
    "thresholds": {
        "strong_score": 0.8,
        "strong_margin": 0.2,
        "focused_support_fraction": 0.6,
        "focused_profile_width_sec": 0.5,
    },
}


def _receipt_with_sha(value: dict) -> dict:
    body = {key: item for key, item in value.items() if key != "calibration_sha256"}
    payload = json.dumps(body, sort_keys=True, separators=(",", ":"))
    receipt = dict(value)
    receipt["calibration_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    return receipt


def _receipt_path(tmp_path, value: dict) -> Path:
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(value))
    return path


def test_legacy_receipt_without_tolerance_still_loads(tmp_path):
    value, _file_digest = _load_calibration(
        _receipt_path(tmp_path, _receipt_with_sha(LEGACY_RECEIPT))
    )
    assert "tolerance_seconds" not in value
    assert value["development_rows"] == 2


def test_positive_tolerance_receipt_loads(tmp_path):
    value, _ = _load_calibration(_receipt_path(
        tmp_path, _receipt_with_sha({**LEGACY_RECEIPT, "tolerance_seconds": 2.0})
    ))
    assert value["tolerance_seconds"] == 2.0


def test_derived_receipt_round_trips_through_loader(tmp_path):
    row = {
        "clip": "c.mp4", "best_master": "m", "source_offset_sec": 1.0,
        "margin": 0.5, "best_score": 0.9,
        "confidence": {"timing": {
            "support_fraction": 1.0, "offset_profile_width_sec": 0.0,
        }},
    }
    truth = {"c.mp4": {"master": "m.mp4", "offset_sec": 1.0}}
    receipt = derive_tiers([row], truth, tolerance=0.75)
    value, _ = _load_calibration(_receipt_path(tmp_path, receipt))
    assert value["tolerance_seconds"] == 0.75


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.5, True, False, "0.5"])
def test_invalid_tolerance_rejected_after_sha_recompute(tmp_path, bad):
    value = _receipt_with_sha({**LEGACY_RECEIPT, "tolerance_seconds": bad})
    with pytest.raises(BatchError, match="tolerance_seconds"):
        _load_calibration(_receipt_path(tmp_path, value))


def test_extra_receipt_field_rejected_after_sha_recompute(tmp_path):
    value = _receipt_with_sha(
        {**LEGACY_RECEIPT, "tolerance_seconds": 1.0, "notes": "hand-tuned"}
    )
    with pytest.raises(BatchError, match="invalid calibration receipt"):
        _load_calibration(_receipt_path(tmp_path, value))
