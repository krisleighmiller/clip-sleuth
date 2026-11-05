"""Portable generated-media controls, audit, shortlist and CLI contracts."""

import copy
import hashlib
import json
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from clipprov import harness, match, transforms
from clipprov.cli import main
from clipprov.config import Config
from clipprov.index import IndexError as ClipIndexError
from clipprov.index import build_index
from clipprov.media import MediaError, probe
from clipprov.motion import decode


@pytest.fixture(scope="module")
def media_corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("phase3 media with spaces")
    library = root / "library"
    library.mkdir()
    for i, (w, h, fps, duration) in enumerate(
        [(96, 64, 12, 2), (80, 72, 16, 2.5), (64, 96, 10, 3)]
    ):
        p = library / f"nested {i}" / f"source {i}.mp4"
        p.parent.mkdir()
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"testsrc2=s={w}x{h}:r={fps}:d={duration}",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency={300 + i * 100}:duration={duration}",
                "-vf",
                f"hue=h={i * 75}",
                "-c:v",
                "libx264",
                "-threads",
                "1",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-shortest",
                str(p),
            ],
            check=True,
        )
    return root, library


@pytest.mark.parametrize(
    "recipe",
    [
        {"crop": [0, 0, 0]},
        {"crop": [0.8, -0.1, 0]},
        {"crop": [1.1, 0, 0]},
        {"resize": [0, 10]},
        {"resize": [11, 10]},
        {"resize": [True, 10]},
        {"speed": 0},
        {"speed": 1.1},
        {"speed": float("nan")},
        {"brightness": float("inf")},
        {"crf": -1},
        {"crf": True},
        {"padding": [-2, 2]},
        {"audio": "unknown"},
        {"unknown": 1},
    ],
)
def test_invalid_transforms(media_corpus, tmp_path, recipe):
    _, lib = media_corpus
    src = next(lib.rglob("*.mp4"))
    dst = tmp_path / "out.mp4"
    with pytest.raises(ValueError):
        transforms.transform(src, dst, recipe=recipe)
    assert not dst.exists()


def test_transform_errors_atomic_and_boundary(media_corpus, tmp_path, monkeypatch):
    _, lib = media_corpus
    src = min(lib.rglob("*.mp4"))
    dst = tmp_path / "out.mp4"
    dst.write_bytes(b"previous output")
    info = probe(src)
    for start, duration in [
        (-1, 1),
        (0, 0),
        (info.duration_s, 1),
        (info.duration_s - 0.1, 0.2),
        (float("nan"), 1),
    ]:
        with pytest.raises(ValueError):
            transforms.transform(src, dst, start=start, duration=duration)
    with pytest.raises(MediaError):
        transforms.transform(
            src, dst, config=Config().media.model_copy(update={"ffmpeg": "no-such-ffmpeg"})
        )
    assert dst.read_bytes() == b"previous output"
    assert not list(tmp_path.glob("*.tmp*"))
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"corrupt")
    with pytest.raises(MediaError):
        transforms.transform(bad, dst)


def test_all_families_compound_probed_audio_and_speed(media_corpus, tmp_path):
    _, lib = media_corpus
    src = min(lib.rglob("*.mp4"))
    for audio in ("remove", "replace", "keep"):
        dst = tmp_path / f"{audio}.mp4"
        result = transforms.transform(
            src,
            dst,
            start=0.25,
            duration=1.5,
            recipe={
                "crop": [0.8, 1, 0],
                "resize": [64, 48],
                "crf": 31,
                "watermark": [0.1, 0.2, 0.3],
                "brightness": 0.03,
                "saturation": 0.7,
                "padding": [4, 6],
                "speed": 1.02,
                "audio": audio,
                "audio_frequency": 640,
            },
        )
        actual = probe(dst)
        assert result.width == actual.width == 72
        assert actual.height == 60
        assert actual.fps == probe(src).fps
        assert actual.has_audio == (audio != "remove")
        assert abs(actual.duration_s - 1.5 / 1.02) <= 2 / actual.fps
        assert decode(dst).shape[0] > 0


def test_frozen_manifest_and_no_stale_clips(media_corpus, tmp_path):
    _, lib = media_corpus
    output = tmp_path / "controls"
    output.mkdir()
    (output / "stale.mp4").write_bytes(b"ignore me")
    manifest = harness.generate_controls(lib, output)
    again = harness.generate_controls(lib, output)
    assert manifest == again
    assert json.loads((output / "manifest.json").read_text()) == manifest
    assert {r["group"] for r in manifest["cases"]} == {"development", "holdout", "source_withheld"}
    dev = {r["source_sha256"] for r in manifest["cases"] if r["group"] == "development"}
    hold = {r["source_sha256"] for r in manifest["cases"] if r["group"] == "holdout"}
    assert dev.isdisjoint(hold)
    values = {
        g: [r["recipe"] for r in manifest["cases"] if r["group"] == g]
        for g in ("development", "holdout")
    }
    for key in (
        "crop",
        "resize",
        "crf",
        "watermark",
        "brightness",
        "saturation",
        "padding",
        "speed",
        "audio_frequency",
    ):
        assert {json.dumps(r[key]) for r in values["development"] if key in r}.isdisjoint(
            {json.dumps(r[key]) for r in values["holdout"] if key in r}
        )
    assert not any("stale" in c["clip"] for c in manifest["cases"])
    for case in manifest["cases"]:
        assert case["source_sha256"]
        assert case["clip_sha256"]
        assert case["audits"]["decode_ok"]
        assert case["audits"]["source_end_ok"]
        assert case["source_metadata"]["fps"] == probe(lib / case["source_relpath"]).fps
        assert case["source_end_sec"] <= case["source_metadata"]["duration_s"]
        assert case["offset_sec"] * case["source_metadata"]["fps"] == pytest.approx(
            round(case["offset_sec"] * case["source_metadata"]["fps"])
        )


def controls_index(lib, manifest, root):
    selected = root / "selected library"
    selected.mkdir()
    for src in manifest["sources"]:
        if src["group"] != "source_withheld":
            dst = selected / src["source_relpath"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(lib / src["source_relpath"], dst)
    index = root / "index"
    build_index(selected, index)
    return selected, index


def test_suite_api_benchmark_groups_tampering_membership(media_corpus, tmp_path, monkeypatch):
    _, lib = media_corpus
    output = tmp_path / "generated"
    manifest = harness.generate_controls(lib, output)
    _selected, idx = controls_index(lib, manifest, tmp_path)
    # Console script invocation is forbidden; matching must use the library.
    real_run = subprocess.run

    def guarded(args, *a, **kw):
        assert Path(args[0]).name != "clip-prov"
        return real_run(args, *a, **kw)

    monkeypatch.setattr(subprocess, "run", guarded)
    from test_phase3_eval import product

    br, bt = tmp_path / "benchmark.json", tmp_path / "benchmark-truth.json"
    br.write_text(json.dumps([product()]))
    bt.write_text(json.dumps([{"clip": "clip", "master": "m0", "offset_sec": 1}]))
    report = harness.evaluate_controls(
        output / "manifest.json",
        idx,
        tmp_path / "report.json",
        benchmark_results=br,
        benchmark_truth=bt,
        shortlist_samples=2,
    )
    assert report["benchmark"]["metrics"]["rows"] == 1
    assert set(report["groups"]) == {"development", "holdout", "source_withheld"}
    assert report["shortlist"]["samples"] == 2
    assert 0 <= report["shortlist"]["exhaustive_winner_recall"] <= 1
    assert report["audits"]["source_membership_ok"]
    assert report["input_digests"]["manifest"]
    assert report["groups"]["source_withheld"]["metrics"]["foreign_clips"] > 0
    for case in report["results"]:
        assert case["schema_version"] == 1
    bad = copy.deepcopy(manifest)
    bad["cases"][0]["source_end_sec"] += 100
    (output / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="boundary|source.end"):
        harness.evaluate_controls(output / "bad.json", idx, tmp_path / "bad-report.json")
    # Membership must use nested relative paths and content, never top-level glob/stem alone.
    index_json = idx / "manifest.json"
    old = index_json.read_text()
    altered = json.loads(old)
    altered["masters"][0]["source_relpath"] = "wrong/source.mp4"
    index_json.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="membership|source"):
        harness.evaluate_controls(output / "manifest.json", idx, tmp_path / "bad-report.json")
    index_json.write_text(old)
    first = output / manifest["cases"][0]["clip"]
    first.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash|digest"):
        harness.evaluate_controls(output / "manifest.json", idx, tmp_path / "bad-report.json")
    assert not (tmp_path / "bad-report.json").exists()


def test_negative_must_really_be_withheld(media_corpus, tmp_path):
    _, lib = media_corpus
    out = tmp_path / "controls"
    harness.generate_controls(lib, out)
    idx = tmp_path / "all-index"
    build_index(lib, idx)
    with pytest.raises(ValueError, match="withheld|membership"):
        harness.evaluate_controls(out / "manifest.json", idx, tmp_path / "report.json")


def test_cli_generate_evaluate_and_limits(media_corpus, tmp_path):
    _, lib = media_corpus
    out = tmp_path / "controls"
    runner = CliRunner()
    result = runner.invoke(main, ["generate", str(lib), "--out", str(out)])
    assert result.exit_code == 0, result.output
    manifest = json.loads((out / "manifest.json").read_text())
    _, idx = controls_index(lib, manifest, tmp_path)
    result = runner.invoke(
        main,
        [
            "evaluate",
            str(out / "manifest.json"),
            "--index",
            str(idx),
            "--out",
            str(tmp_path / "report.json"),
            "--shortlist-samples",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["benchmark"] is None
    for n in (-1, 65):
        with pytest.raises(ValueError):
            harness.evaluate_controls(
                out / "manifest.json", idx, tmp_path / "report.json", shortlist_samples=n
            )


def test_shortlist_exhaustive_can_find_omitted_winner(monkeypatch):
    # 18 masters exceed the real 10+5 shortlist. Exhaustive must score every full view.
    from clipprov import core

    masters = {
        f"m{i:02}": {
            "motion": np.zeros(8),
            "shortlist_views": np.zeros((3, 8, 576)),
            "load_views": lambda i=i: np.full((19, 8, 576), i),
        }
        for i in range(18)
    }
    monkeypatch.setattr(
        match, "probe_clip", lambda *a, **k: (np.zeros((4, 576)), np.arange(4), np.zeros(4), "")
    )
    monkeypatch.setattr(match, "rank_candidates", lambda *a, **k: [(0.5, "m00", 0, 1)])
    seen = []

    def search(cg, idx, views, speeds):
        i = int(views[0][0, 0])
        seen.append(i)
        return i / 20, 0, 1

    monkeypatch.setattr(match, "search", search)
    monkeypatch.setattr(match, "load_index", lambda p: ({}, masters))
    diag, _, audit = match.audit_shortlist("clip", "index", config=Config())
    assert set(seen) == set(range(18))
    assert audit["exhaustive_best_master"] == "m17"
    assert audit["winner_in_shortlist"] is False
    assert diag.shortlist == ["m00"]
    assert core is not None


def test_generation_rejects_duplicate_source_content_like_index(media_corpus, tmp_path):
    _, lib = media_corpus
    copied = tmp_path / "library"
    shutil.copytree(lib, copied)
    shutil.copy2(next(copied.rglob("*.mp4")), copied / "duplicate.mp4")
    with pytest.raises(ValueError, match="duplicate source content is not allowed"):
        harness.generate_controls(copied, tmp_path / "out")


def test_short_static_repeated_duplicate_and_near_duplicate(tmp_path):
    library = tmp_path / "library"
    library.mkdir()
    # Exact duplicate masters are rejected; near duplicates remain match diagnostics.
    src = library / "a.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=64x64:r=8:d=1",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            str(src),
        ],
        check=True,
    )
    shutil.copy2(src, library / "b.mp4")
    transforms.transform(src, library / "near.mp4", recipe={"crf": 25})
    idx = tmp_path / "index"
    with pytest.raises(ClipIndexError, match="duplicate source content"):
        build_index(library, idx)
    (library / "b.mp4").unlink()
    build_index(library, idx)
    diag, _ = match.match_clip_path(src, idx)
    assert not diag.accepted
    assert diag.margin < Config().match.margin
    for kind, vf, duration in [
        ("short", "testsrc2", "0.125"),
        ("static", "color=c=gray", "1"),
        ("repeated", "testsrc2", "2"),
    ]:
        clip = tmp_path / f"{kind}.mp4"
        args = [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"{vf}:s=64x64:r=8:d={duration}"
            if vf != "testsrc2"
            else f"{vf}=s=64x64:r=8:d={duration}",
        ]
        if kind == "repeated":
            args += ["-vf", "loop=loop=1:size=8:start=0", "-t", "2"]
        subprocess.run([*args, "-c:v", "libx264", "-threads", "1", str(clip)], check=True)
        diag, _ = match.match_clip_path(clip, idx)
        assert not diag.accepted


def test_generation_failure_preserves_pointer(media_corpus, tmp_path, monkeypatch):
    _, lib = media_corpus
    out = tmp_path / "out"
    harness.generate_controls(lib, out)
    before = (out / "manifest.json").read_bytes()

    def fail(*a, **k):
        raise MediaError("injected transform error")

    monkeypatch.setattr(harness, "transform", fail)
    with pytest.raises(MediaError):
        harness.generate_controls(lib, out)
    assert (out / "manifest.json").read_bytes() == before
    assert not list(out.glob(".generate-*"))


def test_repeated_generation_audits_every_frozen_negative(media_corpus, tmp_path):
    _, lib = media_corpus
    out = tmp_path / "out"
    manifest = harness.generate_controls(lib, out)
    before = (out / "manifest.json").read_bytes()
    negatives = manifest["negatives"]

    deleted = out / negatives[0]["clip"]
    deleted_bytes = deleted.read_bytes()
    deleted.unlink()
    with pytest.raises(ValueError, match="generation content hash mismatch"):
        harness.generate_controls(lib, out)
    assert (out / "manifest.json").read_bytes() == before

    deleted.write_bytes(deleted_bytes)
    corrupted = out / negatives[1]["clip"]
    corrupted.write_bytes(b"corrupt frozen negative")
    with pytest.raises(ValueError, match="generation content hash mismatch"):
        harness.generate_controls(lib, out)
    assert (out / "manifest.json").read_bytes() == before


def test_negative_controls_are_unique_and_honestly_related(media_corpus, tmp_path):
    _, lib = media_corpus
    manifest = harness.generate_controls(lib, tmp_path / "out")
    negatives = manifest["negatives"]
    assert {n["category"] for n in negatives} == {
        "unrelated_visual", "synthetic_noise", "static_color", "repeated_pattern",
        "independent_test_pattern_variant",
    }
    assert len({n["clip"] for n in negatives}) == len(negatives)
    assert len({n["sha256"] for n in negatives}) == len(negatives)
    assert not {n["clip"] for n in negatives} & {c["clip"] for c in manifest["cases"]}
    assert all(isinstance(n["relation"], dict) for n in negatives)
    variant = next(n for n in negatives if n["category"] == "independent_test_pattern_variant")
    assert variant["relation"] == {
        "kind": "procedural_variant_of",
        "category": "unrelated_visual",
    }
    positives = {s["visual_signature"] for s in manifest["sources"] if s["group"] != "source_withheld"}
    assert positives
    for negative in negatives:
        separation = negative["visual_separation"]
        assert separation["method"] == manifest["visual_audit"]["method"]
        assert separation["minimum_distance"] > separation["threshold"]
        assert separation["nearest_signature"] in positives


@pytest.mark.parametrize(
    "change", ["overlap", "path", "nonfinite", "duplicate", "version", "negative_duplicate",
               "negative_relation", "receipt"]
)
def test_manifest_validation(media_corpus, tmp_path, change):
    _, lib = media_corpus
    out = tmp_path / "out"
    manifest = harness.generate_controls(lib, out)
    _, idx = controls_index(lib, manifest, tmp_path)
    bad = copy.deepcopy(manifest)
    if change == "overlap":
        hold = next(c for c in bad["cases"] if c["group"] == "holdout")
        hold["source_sha256"] = bad["cases"][0]["source_sha256"]
    elif change == "path":
        bad["cases"][0]["clip"] = "../escape.mp4"
    elif change == "nonfinite":
        bad["cases"][0]["offset_sec"] = float("nan")
    elif change == "duplicate":
        bad["cases"].append(copy.deepcopy(bad["cases"][0]))
    elif change == "negative_duplicate":
        bad["negatives"][1]["clip"] = bad["negatives"][0]["clip"]
        bad["negatives"][1]["sha256"] = bad["negatives"][0]["sha256"]
    elif change == "negative_relation":
        bad["negatives"][-1]["relation"]["category"] = "missing_category"
    elif change == "receipt":
        bad["generation_receipt"]["generator_implementation"]["files"][0]["sha256"] = "0" * 64
    else:
        bad["schema_version"] = True
    (out / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        harness.evaluate_controls(out / "bad.json", idx, tmp_path / "report.json")


def test_not_enough_distinct_sources_and_corrupt_input(tmp_path):
    lib = tmp_path / "library"
    lib.mkdir()
    (lib / "bad.mp4").write_bytes(b"corrupt")
    with pytest.raises((ValueError, MediaError)):
        harness.generate_controls(lib, tmp_path / "out")


@pytest.mark.parametrize("recipe", [[], False, "", 0])
def test_falsey_malformed_recipes(media_corpus, tmp_path, recipe):
    _, lib = media_corpus
    with pytest.raises(ValueError):
        transforms.transform(min(lib.rglob("*.mp4")), tmp_path / "bad.mp4", recipe=recipe)


def test_source_hash_tamper_and_relocation(media_corpus, tmp_path):
    _, lib = media_corpus
    out = tmp_path / "controls"
    manifest = harness.generate_controls(lib, out)
    _, idx = controls_index(lib, manifest, tmp_path)
    moved = tmp_path / "moved library"
    shutil.copytree(lib, moved)
    report = harness.evaluate_controls(
        out / "manifest.json", idx, tmp_path / "report.json", library=moved, shortlist_samples=0
    )
    assert report["shortlist"]["samples"] == 0
    p = moved / manifest["sources"][0]["source_relpath"]
    p.write_bytes(b"changed source")
    with pytest.raises((ValueError, MediaError)):
        harness.evaluate_controls(out / "manifest.json", idx, tmp_path / "bad.json", library=moved)


def test_shortlist_bound_before_decoding(monkeypatch):
    monkeypatch.setattr(match, "load_index", lambda p: ({}, {str(i): {} for i in range(65)}))
    with pytest.raises(ValueError, match="bounded"):
        match.audit_shortlist("missing.mp4", "index")


def test_manifest_timing_must_match_frozen_recipe(media_corpus, tmp_path):
    _, lib = media_corpus
    out = tmp_path / "out"
    manifest = harness.generate_controls(lib, out)
    _, idx = controls_index(lib, manifest, tmp_path)
    # A self-consistent offset within the source is still invalid if it changes frozen truth.
    case = manifest["cases"][0]
    step = 1 / case["source_metadata"]["fps"]
    case["offset_sec"] += step
    case["source_end_sec"] += step
    (out / "bad.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="frozen|timing"):
        harness.evaluate_controls(out / "bad.json", idx, tmp_path / "bad-report.json")


def test_generator_publishes_explicit_ground_truth(media_corpus, tmp_path):
    _, lib = media_corpus
    manifest = harness.generate_controls(lib, tmp_path / "out")
    assert len(manifest["truth"]) == len(manifest["cases"])
    for case, truth in zip(manifest["cases"], manifest["truth"], strict=True):
        assert truth["clip"] == Path(case["clip"]).name
        assert truth["master"] == (
            None if case["group"] == "source_withheld" else Path(case["source_relpath"]).name
        )


def test_successful_process_with_zero_decoded_frames_is_not_an_audit(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="frame=0\nprogress=end\n", stderr=""),
    )
    with pytest.raises(MediaError, match="zero|frames"):
        transforms.audit_decode("valid-header-no-frames.mp4")


def test_frozen_case_order_cannot_change_shortlist_sample(media_corpus, tmp_path):
    _, lib = media_corpus
    out = tmp_path / "out"
    manifest = harness.generate_controls(lib, out)
    _, idx = controls_index(lib, manifest, tmp_path)
    manifest["cases"].reverse()
    manifest["truth"].reverse()
    (out / "bad.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="frozen|order"):
        harness.evaluate_controls(out / "bad.json", idx, tmp_path / "report.json")


def test_candidate_universe_is_exact_and_reported(media_corpus, tmp_path):
    _, lib = media_corpus
    out = tmp_path / "out"
    manifest = harness.generate_controls(lib, out)
    selected, idx = controls_index(lib, manifest, tmp_path)
    report = harness.evaluate_controls(
        out / "manifest.json", idx, tmp_path / "report.json", shortlist_samples=0
    )
    expected = sorted(
        (s["source_relpath"], s["sha256"])
        for s in manifest["sources"]
        if s["group"] in {"development", "holdout"}
    )
    assert report["candidate_universe"]["identities"] == [
        {"source_relpath": path, "content_digest": digest} for path, digest in expected
    ]
    frozen = json.dumps(report["candidate_universe"]["identities"], sort_keys=True, separators=(",", ":"))
    assert report["candidate_universe"]["sha256"] == hashlib.sha256(frozen.encode()).hexdigest()

    extra = selected / "extra.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
         "color=c=green:s=96x64:r=12:d=2", "-c:v", "libx264", "-threads", "1", str(extra)],
        check=True,
    )
    build_index(selected, tmp_path / "extra-index")
    with pytest.raises(ValueError, match="candidate universe.*extra"):
        harness.evaluate_controls(
            out / "manifest.json", tmp_path / "extra-index", tmp_path / "bad.json",
            shortlist_samples=0,
        )


def test_diagnostic_matrix_and_generation_receipt(media_corpus, tmp_path):
    _, lib = media_corpus
    manifest = harness.generate_controls(lib, tmp_path / "out")
    labels = {case["diagnostic"] for case in manifest["cases"] if case["group"] == "development"}
    assert labels == {
        "baseline", "reencode", "crop", "speed", "watermark", "audio_keep",
        "grayscale", "resize", "brightness", "saturation", "padding",
        "audio_remove", "audio_replace", "crop_speed", "compound_stress",
    }
    single_factor = {
        "resize": {"resize"}, "brightness": {"brightness"},
        "saturation": {"saturation"}, "padding": {"padding"},
        "audio_remove": {"audio"}, "audio_replace": {"audio", "audio_frequency"},
    }
    dev = {c["diagnostic"]: c["recipe"] for c in manifest["cases"] if c["group"] == "development"}
    for label, changed in single_factor.items():
        assert set(dev[label]) - {"audio"} == changed - {"audio"}
    receipt = manifest["generation_receipt"]
    assert receipt["package_version"]
    assert receipt["python_version"]
    assert receipt["numpy_version"]
    assert receipt["ffmpeg_version"]
    assert len(receipt["generator_sha256"]) == 64
    implementation = receipt["generator_implementation"]
    expected_paths = [
        "clipprov/__init__.py", "clipprov/_io.py", "clipprov/cli.py", "clipprov/config.py",
        "clipprov/harness.py", "clipprov/media.py", "clipprov/transforms.py",
    ]
    assert [row["path"] for row in implementation["files"]] == expected_paths
    assert all(len(row["sha256"]) == 64 for row in implementation["files"])
    frozen = json.dumps(implementation["files"], sort_keys=True, separators=(",", ":"))
    assert implementation["sha256"] == hashlib.sha256(frozen.encode()).hexdigest()
    assert receipt["generator_sha256"] == implementation["sha256"]
    assert receipt["recipe_version"] == manifest["recipe_version"]
    assert set(receipt["git"]) == {"commit", "dirty"}


def test_generator_identity_changes_when_a_behavior_module_changes(tmp_path):
    package = Path(harness.__file__).resolve().parent
    copied = tmp_path / "clipprov"
    copied.mkdir()
    for relative in harness.GENERATOR_IMPLEMENTATION_FILES:
        source = package.parent / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    before = harness._generator_implementation(tmp_path)
    transforms_copy = copied / "transforms.py"
    transforms_copy.write_text(transforms_copy.read_text() + "\n# behavior change\n")
    after = harness._generator_implementation(tmp_path)
    assert before["sha256"] != after["sha256"]
    changed = [
        new["path"]
        for old, new in zip(before["files"], after["files"], strict=True)
        if old["sha256"] != new["sha256"]
    ]
    assert changed == ["clipprov/transforms.py"]


def test_git_receipt_ignores_unrelated_parent_repository(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    (tmp_path / "parent.txt").write_text("unrelated\n")
    subprocess.run(["git", "add", "parent.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "unrelated parent"], cwd=tmp_path, check=True)

    installed_root = tmp_path / "venv" / "site-packages"
    installed_root.mkdir(parents=True)
    assert harness._git_receipt(installed_root) == {"commit": "unavailable", "dirty": None}


def test_evaluation_receipt_binds_complete_evaluation_implementation(media_corpus, tmp_path):
    _, lib = media_corpus
    out = tmp_path / "out"
    manifest = harness.generate_controls(lib, out)
    _, idx = controls_index(lib, manifest, tmp_path)
    report = harness.evaluate_controls(
        out / "manifest.json", idx, tmp_path / "report.json", shortlist_samples=0
    )

    assert report["run_receipt"]["generation"] == manifest["generation_receipt"]
    evaluation = report["run_receipt"]["evaluation"]
    implementation = evaluation["evaluation_implementation"]
    expected_paths = [
        "clipprov/__init__.py", "clipprov/_io.py", "clipprov/appearance.py",
        "clipprov/cli.py", "clipprov/config.py", "clipprov/confidence.py",
        "clipprov/core.py", "clipprov/eval.py", "clipprov/evidence.py",
        "clipprov/harness.py", "clipprov/index.py", "clipprov/match.py",
        "clipprov/media.py", "clipprov/motion.py", "clipprov/transforms.py",
    ]
    assert [row["path"] for row in implementation["files"]] == expected_paths
    frozen = json.dumps(implementation["files"], sort_keys=True, separators=(",", ":"))
    assert implementation["sha256"] == hashlib.sha256(frozen.encode()).hexdigest()
    assert evaluation["evaluation_sha256"] == implementation["sha256"]
    assert "generator_implementation" not in evaluation
    assert set(evaluation["git"]) == {"commit", "dirty"}


def test_evaluation_identity_changes_when_an_evaluation_module_changes(tmp_path):
    for relative in harness.EVALUATION_IMPLEMENTATION_FILES:
        source = Path(harness.__file__).resolve().parent.parent / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    before = harness._evaluation_implementation(tmp_path)
    match_copy = tmp_path / "clipprov/match.py"
    match_copy.write_text(match_copy.read_text() + "\n# evaluation behavior change\n")
    after = harness._evaluation_implementation(tmp_path)
    assert before["sha256"] != after["sha256"]
    changed = [
        new["path"]
        for old, new in zip(before["files"], after["files"], strict=True)
        if old["sha256"] != new["sha256"]
    ]
    assert changed == ["clipprov/match.py"]


def test_frozen_negative_categories_are_reported(media_corpus, tmp_path):
    _, lib = media_corpus
    out = tmp_path / "out"
    manifest = harness.generate_controls(lib, out)
    _, idx = controls_index(lib, manifest, tmp_path)
    report = harness.evaluate_controls(out / "manifest.json", idx, tmp_path / "report.json")
    assert set(report["negative_categories"]) == {
        "source_withheld", "unrelated_visual", "synthetic_noise", "static_color",
        "repeated_pattern", "independent_test_pattern_variant",
    }
    assert all(metrics["metrics"]["rows"] for metrics in report["negative_categories"].values())


def test_source_family_audit_clusters_near_visual_variants(tmp_path):
    library = tmp_path / "library"
    library.mkdir()
    sources = []
    for i, source in enumerate((
        "testsrc2=s=96x64:r=12:d=2",
        "color=c=green:s=96x64:r=12:d=2",
        "color=c=purple:s=96x64:r=12:d=2",
    )):
        path = library / f"source-{i}.mp4"
        subprocess.run([
            "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", source,
            "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p", str(path),
        ], check=True)
        sources.append(path)
    transforms.transform(sources[0], library / "source-near.mp4", recipe={"crf": 31})
    manifest = harness.generate_controls(library, tmp_path / "controls")
    audit = manifest["visual_audit"]
    assert audit["method"] == "rgb32_normalized_timeline_mae_v1"
    assert audit["threshold"] > 0
    assert audit["pairs"]
    by_path = {s["source_relpath"]: s for s in manifest["sources"]}
    assert by_path["source-0.mp4"]["source_family_id"] == by_path["source-near.mp4"]["source_family_id"]
    assert by_path["source-0.mp4"]["group"] == by_path["source-near.mp4"]["group"]
    assert audit["family_count"] == 3


def test_shortlist_sampling_is_stratified_and_reports_composition(media_corpus, tmp_path):
    _, lib = media_corpus
    out = tmp_path / "controls"
    manifest = harness.generate_controls(lib, out)
    _, idx = controls_index(lib, manifest, tmp_path)
    report = harness.evaluate_controls(
        out / "manifest.json", idx, tmp_path / "report.json", shortlist_samples=8
    )
    composition = report["shortlist"]["composition"]
    assert set(composition["groups"]) == {
        "development", "holdout", "source_withheld", "foreign_procedural",
    }
    assert set(composition["classes"]) == {"positive", "negative"}
    assert len(composition["source_ids"]) >= 4
    assert len(composition["recipe_families"]) >= 4
    assert sum(composition["groups"].values()) == report["shortlist"]["samples"] == 8
    assert all("sample" in row for row in report["shortlist"]["rows"])


def test_demo_corpus_is_installed_cli_functionality(tmp_path):
    out = tmp_path / "demo with spaces"
    result = CliRunner().invoke(main, ["demo-corpus", str(out)])
    assert result.exit_code == 0, result.output
    assert len(list((out / "sources").rglob("*.mp4"))) == 3
    assert len(list((out / "indexed sources").rglob("*.mp4"))) == 2


def test_built_artifacts_ship_docs_and_installed_cli_runs_workflow(tmp_path):
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required for the packaging integration test")
    root = Path(__file__).resolve().parents[1]
    dist = tmp_path / "dist"
    subprocess.run([uv, "build", "--out-dir", str(dist)], cwd=root, check=True, capture_output=True)
    wheel = next(dist.glob("*.whl"))
    sdist = next(dist.glob("*.tar.gz"))
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert all(
            any(name.endswith(f"share/doc/clip-provenance/{doc}") for name in names)
            for doc in (
                "evaluation.md", "confidence-evidence.md", "batch-workflow.md",
                "v1-input-contract.md", "geometry.md", "audio-qualification.md",
            )
        )
        metadata = archive.read(next(name for name in names if name.endswith("METADATA"))).decode()
        assert all(f"Requires-Dist: {name}" in metadata for name in ("pydantic", "click", "numpy"))
    with tarfile.open(sdist) as archive:
        names = archive.getnames()
        assert all(
            any(name.endswith(f"/docs/{doc}") for name in names)
            for doc in (
                "evaluation.md", "confidence-evidence.md", "batch-workflow.md",
                "v1-input-contract.md", "geometry.md", "audio-qualification.md",
            )
        )

    venv = tmp_path / "venv"
    subprocess.run([uv, "venv", str(venv)], check=True, capture_output=True)
    python = venv / "bin" / "python"
    subprocess.run(
        [uv, "pip", "install", "--python", str(python), str(wheel)],
        check=True, capture_output=True,
    )
    cli = venv / "bin" / "clip-prov"
    demo = tmp_path / "installed demo"
    controls, index, report = demo / "controls", demo / "index", demo / "report.json"
    commands = [
        [str(cli), "demo-corpus", str(demo)],
        [str(cli), "generate", str(demo / "sources"), "--out", str(controls)],
        [str(cli), "index", str(demo / "indexed sources"), "--out", str(index)],
        [str(cli), "evaluate", str(controls / "manifest.json"), "--index", str(index),
         "--out", str(report), "--shortlist-samples", "0"],
    ]
    for command in commands:
        subprocess.run(command, check=True, capture_output=True)
    installed_report = json.loads(report.read_text())
    assert len(installed_report["candidate_universe"]["identities"]) == 2
    assert len(installed_report["negative_categories"]) == 6
    receipt = json.loads((controls / "manifest.json").read_text())["generation_receipt"]
    assert [row["path"] for row in receipt["generator_implementation"]["files"]] == list(
        harness.GENERATOR_IMPLEMENTATION_FILES
    )
    assert [
        row["path"]
        for row in installed_report["run_receipt"]["evaluation"]["evaluation_implementation"]["files"]
    ] == list(harness.EVALUATION_IMPLEMENTATION_FILES)


def test_evaluate_controls_tolerance_propagates_to_calibration_and_groups(media_corpus, tmp_path):
    _, lib = media_corpus
    out = tmp_path / "controls"
    manifest = harness.generate_controls(lib, out)
    _, idx = controls_index(lib, manifest, tmp_path)

    report_default = harness.evaluate_controls(
        out / "manifest.json", idx, tmp_path / "report-default.json", shortlist_samples=0
    )
    report_tight = harness.evaluate_controls(
        out / "manifest.json", idx, tmp_path / "report-tight.json",
        shortlist_samples=0, tolerance=0.01,
    )

    assert report_tight["confidence"]["calibration"]["tolerance_seconds"] == 0.01
    assert report_tight["confidence"]["heldout_behavior"]["tolerance_seconds"] == 0.01

    for group in ("development", "holdout"):
        default_metrics = report_default["groups"][group]["metrics"]
        tight_metrics = report_tight["groups"][group]["metrics"]
        # Tighter tolerance can only reduce or equal joint correctness counts.
        assert tight_metrics["accepted_joint_recall"] <= default_metrics["accepted_joint_recall"]
        assert tight_metrics["all_rows_joint_correct"] <= default_metrics["all_rows_joint_correct"]