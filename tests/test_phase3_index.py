"""Regressions exposed by the evaluation audit, independent of source media."""

import json
import os

import numpy as np
import pytest

from clipprov import index
from clipprov.config import Config, MatchConfig, MediaConfig
from clipprov.motion import GRID, SHORTLIST_VIEWS, VIEWS


@pytest.fixture
def tiny_index(tmp_path, monkeypatch):
    lib = tmp_path / "nested library"
    (lib / "sub directory").mkdir(parents=True)
    for name in ("a", "b"):
        (lib / "sub directory" / f"{name}.mp4").write_bytes(name.encode())

    def build(src, dst, cfg):
        dst.mkdir(parents=True)
        np.save(dst / "motion.npy", np.zeros(7, np.float32))
        np.save(dst / "shortlist_views.npy", np.zeros((SHORTLIST_VIEWS, 8, GRID**2), np.float32))
        np.save(dst / "views.npy", np.zeros((len(VIEWS), 8, GRID**2), np.float32))

    monkeypatch.setattr(index, "_build_shard", build)
    out = tmp_path / "index"
    index.build_index(lib, out)
    return lib, out, build


@pytest.mark.parametrize("entry", [None, False, 3, [], "bad"])
def test_array_entries_validated_before_get(tiny_index, entry):
    _, out, _ = tiny_index
    path = out / "manifest.json"
    m = json.loads(path.read_text())
    m["masters"][0]["arrays"].insert(0, entry)
    path.write_text(json.dumps(m))
    with pytest.raises(index.IndexError, match="array"):
        index.load_index(out)


@pytest.mark.parametrize(
    "shape", [(4, 8, 576), (3, 8, 575), (3, 1001, 576), (0, 8, 576), (3, 0, 576)]
)
def test_bounded_shortlist_shape(tiny_index, shape):
    _, out, _ = tiny_index
    path = out / "manifest.json"
    m = json.loads(path.read_text())
    entry = m["masters"][0]
    arr = next(a for a in entry["arrays"] if a["name"] == "shortlist_views")
    p = out / entry["shard_dir"] / arr["file"]
    np.save(p, np.zeros(shape, np.float32))
    arr.update(shape=list(shape), checksum=index._sha256_file(p))
    path.write_text(json.dumps(m))
    with pytest.raises(index.IndexError, match="dimensions|shape"):
        index.load_index(out)


def test_force_rebuilds_create_immutable_generations_and_preserve_old_loaders(
    tiny_index, monkeypatch
):
    lib, out, _ = tiny_index
    _, old_loaded = index.load_index(out)
    old_loaders = [entry["load_views"] for entry in old_loaded.values()]
    generations = set(out.glob("masters/*/*"))

    index.build_index(lib, out, Config(), force=True)
    after_first = set(out.glob("masters/*/*"))
    index.build_index(lib, out, Config(), force=True)
    after_second = set(out.glob("masters/*/*"))

    assert generations < after_first < after_second
    assert all(loader().shape == (len(VIEWS), 8, GRID**2) for loader in old_loaders)


def test_failed_rebuild_removes_only_generations_created_by_attempt(tiny_index, monkeypatch):
    lib, out, build = tiny_index
    before = set(out.glob("masters/*/*"))
    orphan = out / "masters" / "a" / "preexisting-orphan"
    orphan.mkdir()
    before.add(orphan)
    active = (out / "manifest.json").read_bytes()

    def fail(src, dst, cfg):
        build(src, dst, cfg)
        if src.stem == "b":
            raise RuntimeError("injected after second write")

    monkeypatch.setattr(index, "_build_shard", fail)
    with pytest.raises(RuntimeError):
        index.build_index(lib, out, Config(), force=True)
    assert set(out.glob("masters/*/*")) == before
    assert (out / "manifest.json").read_bytes() == active
    index.load_index(out)


def test_post_replace_directory_fsync_failure_has_uncertain_visibility(tiny_index, monkeypatch):
    lib, out, _ = tiny_index
    before = set(out.glob("masters/*/*"))
    replaced = False
    real_replace = os.replace
    real_fsync_dir = index._fsync_dir

    def track_replace(source, target):
        nonlocal replaced
        real_replace(source, target)
        replaced = True

    def fail_after_replace(path):
        if replaced and path == out:
            raise OSError("injected directory fsync failure")
        real_fsync_dir(path)

    monkeypatch.setattr(os, "replace", track_replace)
    monkeypatch.setattr(index, "_fsync_dir", fail_after_replace)
    with pytest.raises(index.CommitVisibilityUncertain, match="reload"):
        index.build_index(lib, out, Config(), force=True)
    manifest, _ = index.load_index(out)
    assert manifest["status"] == "complete"
    assert set(out.glob("masters/*/*")) > before


def test_manifest_switch_failure_removes_new_generations(tiny_index, monkeypatch):
    lib, out, _ = tiny_index
    before = set(out.glob("masters/*/*"))
    import os

    def fail(*a):
        raise OSError("manifest switch failed")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError):
        index.build_index(lib, out, force=True)
    assert set(out.glob("masters/*/*")) == before
    index.load_index(out)


@pytest.mark.parametrize("shape", [(20, 8, 576), (19, 8, 575), (19, 0, 576)])
def test_bounded_full_view_dimensions(tiny_index, shape):
    _, out, _ = tiny_index
    path = out / "manifest.json"
    m = json.loads(path.read_text())
    entry = m["masters"][0]
    arr = next(a for a in entry["arrays"] if a["name"] == "views")
    p = out / entry["shard_dir"] / arr["file"]
    np.save(p, np.zeros(shape, np.float32))
    arr.update(shape=list(shape), checksum=index._sha256_file(p))
    path.write_text(json.dumps(m))
    with pytest.raises(index.IndexError, match="dimensions|shape"):
        index.load_index(out)


def test_query_only_settings_do_not_change_generation_identity(tiny_index):
    lib, out, _ = tiny_index
    before_manifest, _ = index.load_index(out)
    before_shards = {entry["id"]: entry["shard_dir"] for entry in before_manifest["masters"]}
    before_generations = set(out.glob("masters/*/*"))
    config = Config(
        media=MediaConfig(max_frames=99),
        match=MatchConfig(
            speed_min=0.8,
            speed_max=1.2,
            speed_steps=17,
            offset_tolerance_s=9.0,
        ),
    )

    index.build_index(lib, out, config)

    after_manifest, _ = index.load_index(out)
    assert {entry["id"]: entry["shard_dir"] for entry in after_manifest["masters"]} == before_shards
    assert set(out.glob("masters/*/*")) == before_generations
    assert after_manifest["extractor_config"]["max_frames"] == 99
    assert after_manifest["extractor_config"]["speed_steps"] == 17


def test_heavy_view_declaration_rejected_at_open_without_reading_bytes(tiny_index, monkeypatch):
    _, out, _ = tiny_index
    path = out / "manifest.json"
    manifest = json.loads(path.read_text())
    entry = manifest["masters"][0]
    views = next(array for array in entry["arrays"] if array["name"] == "views")
    views["shape"] = [len(VIEWS) - 1, 8, GRID**2]
    path.write_text(json.dumps(manifest))
    views_path = (out / entry["shard_dir"] / views["file"]).resolve()
    real_hash = index._sha256_file
    real_load = np.load

    def guarded_hash(candidate):
        if candidate.resolve() == views_path:
            raise AssertionError("heavy view bytes hashed during index open")
        return real_hash(candidate)

    def guarded_load(candidate, *args, **kwargs):
        if hasattr(candidate, "resolve") and candidate.resolve() == views_path:
            raise AssertionError("heavy view bytes loaded during index open")
        return real_load(candidate, *args, **kwargs)

    monkeypatch.setattr(index, "_sha256_file", guarded_hash)
    monkeypatch.setattr(np, "load", guarded_load)
    with pytest.raises(index.IndexError, match="views.*shape|dimensions"):
        index.load_index(out)


def test_deferred_malformed_views_raise_typed_index_error(tiny_index):
    _, out, _ = tiny_index
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["masters"][0]
    views = next(array for array in entry["arrays"] if array["name"] == "views")
    views_path = out / entry["shard_dir"] / views["file"]
    views_path.write_bytes(b"not a numpy array")
    views["checksum"] = index._sha256_file(views_path)
    manifest_path.write_text(json.dumps(manifest))

    _, masters = index.load_index(out)
    with pytest.raises(index.IndexError, match="cannot load deferred views"):
        masters[entry["id"]]["load_views"]()


def test_deferred_views_checksum_oserror_is_typed(tiny_index, monkeypatch):
    _, out, _ = tiny_index
    _, masters = index.load_index(out)
    loader = next(iter(masters.values()))["load_views"]

    def fail_hash(pinned):
        raise OSError("injected read failure")

    monkeypatch.setattr(index._PinnedArray, "checksum", fail_hash)
    with pytest.raises(index.IndexError, match="cannot checksum deferred views"):
        loader()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sample_rate_fps", True),
        ("sample_rate_fps", 0),
        ("max_frames", True),
        ("max_frames", 3),
        ("speed_min", float("nan")),
        ("speed_steps", True),
        ("speed_steps", 0),
        ("offset_tolerance_s", -1),
        ("score_floor", float("inf")),
        ("margin", -1),
        ("views", list(reversed(VIEWS))),
        ("shortlist_views", SHORTLIST_VIEWS + 1),
    ],
)
def test_extractor_config_schema_is_strict(tiny_index, field, value):
    _, out, _ = tiny_index
    path = out / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["extractor_config"][field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(index.IndexError, match="extractor_config"):
        index.load_index(out)


@pytest.mark.parametrize("field", ["algorithm_version", "ffmpeg_version", "numpy_version"])
@pytest.mark.parametrize("value", [True, {}, ""])
def test_tool_metadata_requires_nonempty_strings(tiny_index, field, value):
    _, out, _ = tiny_index
    path = out / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest))

    with pytest.raises(index.IndexError, match=f"invalid {field}"):
        index.load_index(out)


@pytest.mark.parametrize("field", ["score_floor", "margin"])
def test_config_from_manifest_fields_are_required(tiny_index, field):
    _, out, _ = tiny_index
    path = out / "manifest.json"
    manifest = json.loads(path.read_text())
    del manifest["extractor_config"][field]
    path.write_text(json.dumps(manifest))
    with pytest.raises(index.IndexError, match="extractor_config"):
        index.load_index(out)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("content_digest", "A" * 64),
        ("content_digest", "0" * 63),
        ("size_bytes", True),
        ("size_bytes", -1),
        ("mtime_ns", True),
        ("mtime_ns", -1),
        ("display_name", "wrong.mp4"),
        ("original_root_hint", "relative/root"),
    ],
)
def test_master_provenance_schema_is_strict(tiny_index, field, value):
    _, out, _ = tiny_index
    path = out / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["masters"][0][field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(index.IndexError, match=field):
        index.load_index(out)


def test_array_frame_counts_must_agree(tiny_index):
    _, out, _ = tiny_index
    path = out / "manifest.json"
    manifest = json.loads(path.read_text())
    entry = manifest["masters"][0]
    motion = next(array for array in entry["arrays"] if array["name"] == "motion")
    motion_path = out / entry["shard_dir"] / motion["file"]
    np.save(motion_path, np.zeros(8, np.float32))
    motion.update(shape=[8], checksum=index._sha256_file(motion_path))
    path.write_text(json.dumps(manifest))
    with pytest.raises(index.IndexError, match="frame count"):
        index.load_index(out)


def test_reuse_requires_deferred_view_checksum_match(tiny_index):
    lib, out, _ = tiny_index
    before, _ = index.load_index(out)
    entry = before["masters"][0]
    old_shard = entry["shard_dir"]
    views = next(array for array in entry["arrays"] if array["name"] == "views")
    np.save(out / old_shard / views["file"], np.ones((len(VIEWS), 8, GRID**2), np.float32))

    index.build_index(lib, out, Config())

    after, loaded = index.load_index(out)
    rebuilt = next(master for master in after["masters"] if master["id"] == entry["id"])
    assert rebuilt["shard_dir"] != old_shard
    loaded[entry["id"]]["load_views"]()


def test_rebuild_rejects_unsafe_candidate_before_external_access(tiny_index, monkeypatch, tmp_path):
    lib, out, _ = tiny_index
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    external = tmp_path / "external-generation"
    external.mkdir()
    active_shard = out / manifest["masters"][0]["shard_dir"]
    for artifact in ("motion.npy", "shortlist_views.npy", "views.npy"):
        (external / artifact).write_bytes((active_shard / artifact).read_bytes())
    manifest["masters"][0]["shard_dir"] = str(external)
    manifest_path.write_text(json.dumps(manifest))

    original_open = type(external).open

    def guarded_open(path, *args, **kwargs):
        if path == external or external in path.parents:
            raise AssertionError("unsafe candidate path was opened")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(type(external), "open", guarded_open)
    index.build_index(lib, out, Config())
    index.load_index(out)


def test_index_rejects_symlinked_generation_even_within_root(tiny_index):
    _, out, _ = tiny_index
    manifest = json.loads((out / "manifest.json").read_text())
    shard = out / manifest["masters"][0]["shard_dir"]
    target = shard.with_name(f"{shard.name}-target")
    shard.rename(target)
    shard.symlink_to(target.name, target_is_directory=True)
    with pytest.raises(index.IndexError, match="symlink"):
        index.load_index(out)


@pytest.mark.parametrize("rel", ["../escape.mp4", "/absolute.mp4", "nested/../../escape.mp4"])
def test_index_rejects_unsafe_source_relative_paths(tiny_index, rel):
    _, out, _ = tiny_index
    path = out / "manifest.json"
    m = json.loads(path.read_text())
    m["masters"][0]["source_relpath"] = rel
    path.write_text(json.dumps(m))
    with pytest.raises(index.IndexError, match="source"):
        index.load_index(out)


def test_source_mutation_during_extraction_aborts_and_removes_generation(tiny_index, monkeypatch):
    lib, out, build = tiny_index
    source = lib / "sub directory" / "a.mp4"
    before_generations = set(out.glob("masters/*/*"))
    before_manifest = (out / "manifest.json").read_bytes()

    mutated = False

    def mutate_after_extract(src, dst, cfg):
        nonlocal mutated
        build(src, dst, cfg)
        if not mutated:
            mutated = True
            original = source.stat()
            source.write_bytes(b"z")
            os.utime(source, ns=(original.st_atime_ns, original.st_mtime_ns))

    monkeypatch.setattr(index, "_build_shard", mutate_after_extract)

    with pytest.raises(index.IndexError, match="changed during extraction"):
        index.build_index(lib, out, Config(), force=True)

    assert set(out.glob("masters/*/*")) == before_generations
    assert (out / "manifest.json").read_bytes() == before_manifest
    assert not list(out.glob(".staging-*"))


def test_parent_swap_during_extraction_cannot_read_external_bytes_or_publish(tmp_path, monkeypatch):
    library = tmp_path / "library"
    source_dir = library / "nested"
    source_dir.mkdir(parents=True)
    source = source_dir / "master.mp4"
    source.write_bytes(b"validated source bytes")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "master.mp4").write_bytes(b"attacker bytes")
    extracted_bytes = []

    def swap_while_extracting(src, dst, cfg):
        parked = library / "parked"
        source_dir.rename(parked)
        source_dir.symlink_to(outside, target_is_directory=True)
        try:
            extracted_bytes.append(src.read_bytes())
            dst.mkdir(parents=True)
            np.save(dst / "motion.npy", np.zeros(7, np.float32))
            np.save(
                dst / "shortlist_views.npy",
                np.zeros((SHORTLIST_VIEWS, 8, GRID**2), np.float32),
            )
            np.save(
                dst / "views.npy",
                np.zeros((len(VIEWS), 8, GRID**2), np.float32),
            )
        finally:
            source_dir.unlink()
            parked.rename(source_dir)

    monkeypatch.setattr(index, "_build_shard", swap_while_extracting)
    out = tmp_path / "index"

    with pytest.raises(index.IndexError, match="changed during extraction"):
        index.build_index(library, out)

    assert extracted_bytes == [b"validated source bytes"]
    assert not (out / "manifest.json").exists()
    assert not list(out.glob(".staging-*"))


@pytest.mark.parametrize("symlink_component", [False, True])
def test_library_source_symlinks_are_rejected(tmp_path, monkeypatch, symlink_component):
    library = tmp_path / "library"
    outside = tmp_path / "outside"
    library.mkdir()
    outside.mkdir()
    (outside / "source.mp4").write_bytes(b"source")
    if symlink_component:
        (library / "linked directory").symlink_to(outside, target_is_directory=True)
    else:
        (library / "source.mp4").symlink_to(outside / "source.mp4")

    def unexpected_build(*args):
        pytest.fail("a symlinked source must not be extracted")

    monkeypatch.setattr(index, "_build_shard", unexpected_build)

    with pytest.raises(index.IndexError) as exc_info:
        index.build_index(library, tmp_path / "index")

    assert str(exc_info.value).startswith("library contains symlink:")
    assert not (tmp_path / "index" / "manifest.json").exists()


def test_manifest_symlink_to_external_file_is_rejected(tiny_index):
    _, out, _ = tiny_index
    manifest_path = out / "manifest.json"
    external_manifest = out.parent / "external-manifest.json"
    manifest_path.rename(external_manifest)
    manifest_path.symlink_to(external_manifest)

    with pytest.raises(index.IndexError, match="manifest"):
        index.load_index(out)


def test_duplicate_content_source_aliases_are_rejected_before_extraction(tmp_path, monkeypatch):
    library = tmp_path / "library"
    (library / "first").mkdir(parents=True)
    (library / "second").mkdir()
    source_bytes = b"identical source bytes"
    (library / "first" / "alpha.mp4").write_bytes(source_bytes)
    (library / "second" / "beta.mp4").write_bytes(source_bytes)
    digest = index._sha256_file(library / "first" / "alpha.mp4")

    def unexpected_build(*args):
        pytest.fail("duplicate-content sources must be rejected before extraction")

    monkeypatch.setattr(index, "_build_shard", unexpected_build)
    out = tmp_path / "index"

    with pytest.raises(index.IndexError) as exc_info:
        index.build_index(library, out)

    message = str(exc_info.value)
    assert "first/alpha.mp4" in message
    assert "second/beta.mp4" in message
    assert digest in message
    assert not (out / "manifest.json").exists()


def test_load_rejects_duplicate_content_digests(tiny_index):
    _, out, _ = tiny_index
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["masters"][1]["content_digest"] = manifest["masters"][0]["content_digest"]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(index.IndexError, match="duplicate.*content_digest"):
        index.load_index(out)


@pytest.mark.parametrize("array_name", ["motion", "shortlist_views"])
def test_eager_array_swap_after_validation_never_accepts_external_bytes(
    tiny_index, monkeypatch, tmp_path, array_name
):
    _, out, _ = tiny_index
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["masters"][0]
    array = next(item for item in entry["arrays"] if item["name"] == array_name)
    target = out / entry["shard_dir"] / array["file"]
    external = tmp_path / f"external-{array_name}.npy"
    np.save(external, np.full(tuple(array["shape"]), 7, np.float32))
    array["checksum"] = index._sha256_file(external)
    manifest_path.write_text(json.dumps(manifest))

    real_sha256_file = index._sha256_file
    real_open = os.open
    swapped = False

    def swap():
        nonlocal swapped
        if not swapped:
            swapped = True
            target.unlink()
            target.symlink_to(external)

    def racing_sha256_file(path):
        if path == target:
            swap()
        return real_sha256_file(path)

    def racing_open(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        if path == array["file"] and kwargs.get("dir_fd") is not None:
            swap()
        return fd

    monkeypatch.setattr(index, "_sha256_file", racing_sha256_file)
    monkeypatch.setattr(os, "open", racing_open)

    with pytest.raises(index.IndexError, match="checksum mismatch"):
        index.load_index(out)


def test_deferred_array_swap_after_validation_never_accepts_external_bytes(
    tiny_index, monkeypatch, tmp_path
):
    _, out, _ = tiny_index
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["masters"][0]
    array = next(item for item in entry["arrays"] if item["name"] == "views")
    target = out / entry["shard_dir"] / array["file"]
    external = tmp_path / "external-views.npy"
    np.save(external, np.full(tuple(array["shape"]), 7, np.float32))
    array["checksum"] = index._sha256_file(external)
    manifest_path.write_text(json.dumps(manifest))

    real_sha256_file = index._sha256_file
    real_open = os.open
    swapped = False

    def swap():
        nonlocal swapped
        if not swapped:
            swapped = True
            target.unlink()
            target.symlink_to(external)

    def racing_sha256_file(path):
        if path == target:
            swap()
        return real_sha256_file(path)

    def racing_open(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        if path == array["file"] and kwargs.get("dir_fd") is not None:
            swap()
        return fd

    monkeypatch.setattr(index, "_sha256_file", racing_sha256_file)
    monkeypatch.setattr(os, "open", racing_open)
    _, masters = index.load_index(out)

    with pytest.raises(index.IndexError, match="checksum mismatch"):
        masters[entry["id"]]["load_views"]()
