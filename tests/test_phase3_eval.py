"""Product-schema evaluation contract and scorer regression tests."""

import copy
import json

import pytest
from click.testing import CliRunner

from clipprov.cli import main
from clipprov.eval import adapt_product, load_product_results, load_truth, score_product


def product(clip="clip.mp4", accepted=True, master="m0", offset=1.0):
    return {
        "schema_version": 1,
        "clip": clip,
        "accepted": accepted,
        "best_master": master,
        "source_record": (
            {
                "id": master,
                "source_relpath": f"nested/{master}.mp4",
                "display_name": f"{master}.mp4",
                "original_root_hint": "/library",
                "content_digest": "a" * 64,
            }
            if master
            else None
        ),
        "raw_alignment_offset_sec": offset,
        "source_offset_sec": offset,
        "speed": 1.0 if offset is not None else None,
        "best_score": 0.8,
        "second_score": 0.1,
        "margin": 0.7,
        "reason": "matched" if accepted else "below_score_floor",
        "accepted_master": master if accepted else None,
        "accepted_offset_sec": offset if accepted else None,
        "accepted_speed": 1.0 if accepted else None,
    }


def test_windows_names_and_arbitrary_master_ids(tmp_path):
    row = product(r"C:\clips with spaces\clip.mp4")
    row["source_record"]["source_relpath"] = r"nested\m0.mp4"
    assert adapt_product(row)["clip"] == "clip.mp4"
    p = tmp_path / "truth.json"
    p.write_text(json.dumps([{"clip": r"C:\clips\clip.mp4", "master": "m0", "offset_sec": 1}]))
    assert load_truth(p)["clip.mp4"]["master"] == "m0.mp4"


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("accepted", 1),
        ("source_record", False),
        ("source_record", []),
        ("source_record", ""),
        ("best_score", float("nan")),
        ("second_score", float("inf")),
        ("margin", float("-inf")),
        ("raw_alignment_offset_sec", float("nan")),
        ("speed", float("inf")),
        ("speed", 0),
        ("accepted_speed", 2),
        ("accepted_master", "other"),
        ("accepted_offset_sec", 2),
        ("accepted_offset_sec", None),
        ("source_offset_sec", -0.1),
        ("best_master", None),
        ("reason", "below_score_floor"),
        ("schema_version", None),
        ("unexpected", 3),
    ],
)
def test_strict_schema(field, value):
    row = product()
    row[field] = value
    with pytest.raises(ValueError):
        adapt_product(row)


@pytest.mark.parametrize("field", list(product()))
def test_missing_schema_fields(field):
    row = product()
    del row[field]
    with pytest.raises(ValueError):
        adapt_product(row)


@pytest.mark.parametrize(
    "field,value", [("id", "other"), ("display_name", "other.mp4"), ("source_relpath", "other.mp4")]
)
def test_source_identity_consistency(field, value):
    row = product()
    row["source_record"][field] = value
    with pytest.raises(ValueError):
        adapt_product(row)


def test_rejected_accepted_fields_must_be_null():
    for field, value in [
        ("accepted_master", "m0"),
        ("accepted_offset_sec", 1),
        ("accepted_speed", 1),
    ]:
        row = product(accepted=False)
        row[field] = value
        with pytest.raises(ValueError):
            adapt_product(row)


def test_three_tiers_and_proposed_accuracy():
    rows = [
        adapt_product(product("a", True)),
        adapt_product(product("b", False)),
        adapt_product(product("c", False)),
        adapt_product(product("d", False, None, None)),
        adapt_product(product("e", False, None, None)),
    ]
    assert [r["tier"] for r in rows] == [
        "accepted",
        "uncertain",
        "uncertain",
        "unidentified",
        "unidentified",
    ]
    truth = {
        f"{c}.mp4": {
            "master": "m0.mp4" if c in "abe" else None,
            "offset_sec": 1 if c in "abe" else None,
        }
        for c in "abcde"
    }
    m = score_product(rows, truth)["metrics"]
    assert m["proposed_matches"] == 3
    assert m["proposed_source_accuracy"] == m["proposed_joint_accuracy"] == 0.666667
    assert m["accepted_joint_recall"] == 0.333333
    assert m["accepted_joint_precision"] == 1
    assert m["uncertain_genuine"] == m["uncertain_foreign"] == 1
    # Any accepted=false outcome explicitly rejects a foreign clip; an
    # uncertain diagnostic candidate is still distinct from unidentified.
    assert m["explicit_foreign_rejections"] == 2
    assert m["genuine_unidentified"] == 1
    assert m["foreign_rejection_recall"] == 1
    assert score_product(rows, truth)["accepted_tiers"] == ["accepted"]


def test_candidate_without_timing_is_still_uncertain():
    row = adapt_product(product(accepted=False, offset=None))
    assert row["tier"] == "uncertain"
    assert row["candidate_master"] == "m0.mp4"
    assert row["candidate_offset"] is None


@pytest.mark.parametrize("tol", [-1, True, float("nan"), float("inf")])
def test_invalid_tolerance(tol):
    with pytest.raises(ValueError):
        score_product([], {}, tol)


def test_coverage_duplicates_unknown_and_missing(tmp_path):
    row = adapt_product(product())
    for rows, truth in [([row, row], {"clip.mp4": {}}), ([row], {}), ([], {"clip.mp4": {}})]:
        with pytest.raises(ValueError):
            score_product(rows, truth)
    p = tmp_path / "bad.json"
    for raw in [False, [None], [product(), 0]]:
        p.write_text(json.dumps(raw))
        with pytest.raises(ValueError):
            load_product_results(p)
    for raw in [
        [{"clip": "a", "master": True, "offset_sec": 1}],
        [{"clip": "a", "master": "m0", "offset_sec": float("nan")}],
        [{"clip": "a", "master": None, "offset_sec": 1}],
        [{"clip": "a", "master": "m0", "offset_sec": -1}],
    ]:
        p.write_text(json.dumps(raw))
        with pytest.raises(ValueError):
            load_truth(p)


def test_tolerance_boundary_and_empty_denominators():
    row = adapt_product(product(offset=2.5))
    truth = {"clip.mp4": {"master": "m0.mp4", "offset_sec": 1}}
    assert score_product([row], truth)["metrics"]["accepted_joint_precision"] == 1
    assert score_product([row], truth, 1.499)["metrics"]["accepted_joint_precision"] == 0
    assert score_product([], {})["metrics"]["proposed_source_accuracy"] is None


def test_eval_cli_atomic_and_error(tmp_path, monkeypatch):
    results, truth, out = [tmp_path / n for n in ("results.json", "truth.json", "out.json")]
    results.write_text(json.dumps([product()]))
    truth.write_text(json.dumps([{"clip": "clip", "master": "m0", "offset_sec": 1}]))
    runner = CliRunner()
    args = ["eval", "--results", str(results), "--truth", str(truth), "--out", str(out)]
    r = runner.invoke(main, args)
    assert r.exit_code == 0, r.output
    saved = out.read_bytes()
    import os

    def fail(*args):
        raise OSError("injected replace failure")

    monkeypatch.setattr(os, "replace", fail)
    r = runner.invoke(main, args)
    assert r.exit_code == 2
    assert out.read_bytes() == saved
    assert not list(tmp_path.glob("*.tmp"))
    results.write_text(json.dumps(copy.deepcopy(product()) | {"speed": float("nan")}))
    assert runner.invoke(main, args).exit_code == 2


@pytest.mark.parametrize("accepted", [True, False])
def test_falsey_accepted_master_is_never_an_identifier(accepted):
    row = product(accepted=accepted)
    row["accepted_master"] = False
    with pytest.raises(ValueError):
        adapt_product(row)


@pytest.mark.parametrize(
    "field,value", [("margin", 0.5), ("speed", False), ("accepted_speed", True)]
)
def test_numeric_and_score_consistency(field, value):
    row = product()
    row[field] = value
    with pytest.raises(ValueError):
        adapt_product(row)


def test_truth_requires_all_three_fields(tmp_path):
    path = tmp_path / "truth.json"
    path.write_text(json.dumps([{"clip": "c"}]))
    with pytest.raises(ValueError):
        load_truth(path)


def test_score_api_rejects_malformed_normalized_rows():
    row = adapt_product(product())
    truth = {"clip.mp4": {"master": "m0.mp4", "offset_sec": 1}}
    for bad in [
        row | {"tier": "bogus"},
        row | {"candidate_offset": float("nan")},
        row | {"candidate_master": None},
    ]:
        with pytest.raises(ValueError):
            score_product([bad], truth)
