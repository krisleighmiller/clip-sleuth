"""Development-only confidence tier calibration and held-out measurement."""
from __future__ import annotations

import copy
import hashlib
import json
import statistics
from pathlib import Path

from clipprov._io import finite_number


def _expected(truth_row):
    master = truth_row.get("master_id", truth_row.get("master"))
    return (Path(master).stem if master else None), truth_row.get("offset_sec")


def _candidate(row):
    return row["best_master"], row["source_offset_sec"]


def _validate_tolerance(tolerance, name="tolerance"):
    if not finite_number(tolerance):
        raise ValueError(f"{name} must be a finite number")
    tol = float(tolerance)
    if tol < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return tol


def derive_tiers(development_rows: list[dict], truth: dict[str, dict], tolerance=1.5) -> dict:
    """Freeze descriptive thresholds from correct development alignments only."""
    tolerance = _validate_tolerance(tolerance)
    eligible = []
    for row in development_rows:
        expected = truth[row["clip"]]
        expected_master, expected_offset = _expected(expected)
        master, offset = _candidate(row)
        confidence = row["confidence"]
        timing = confidence["timing"]
        if (
            master == expected_master
            and offset is not None
            and expected_offset is not None
            and abs(offset - expected_offset) <= tolerance
            and timing["support_fraction"] is not None
            and timing["offset_profile_width_sec"] is not None
        ):
            eligible.append(row)
    if not eligible:
        thresholds = {
            "strong_score": 1.0,
            "strong_margin": 1.0,
            "focused_support_fraction": 1.0,
            "focused_profile_width_sec": 0.0,
        }
    else:
        thresholds = {
            "strong_score": statistics.median(r["best_score"] for r in eligible),
            "strong_margin": statistics.median(r["margin"] for r in eligible),
            "focused_support_fraction": statistics.median(
                r["confidence"]["timing"]["support_fraction"] for r in eligible
            ),
            "focused_profile_width_sec": statistics.median(
                r["confidence"]["timing"]["offset_profile_width_sec"] for r in eligible
            ),
        }
    receipt = {
        "method": "development_medians_v1",
        "semantics": "diagnostic_not_probability",
        "tolerance_seconds": tolerance,
        "development_rows": len(development_rows),
        "eligible_correct_alignments": len(eligible),
        "thresholds": thresholds,
    }
    receipt["calibration_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return receipt


def apply_calibration(row: dict, calibration: dict) -> dict:
    """Return a result whose published tiers use one frozen calibration receipt."""
    calibrated = copy.deepcopy(row)
    confidence = calibrated["confidence"]
    thresholds = calibration["thresholds"]
    confidence["calibration"] = {
        "status": "applied",
        "sha256": calibration["calibration_sha256"],
    }
    if confidence["decision"] in {"unmatched", "ambiguous"}:
        confidence["source"]["tier"] = "not_applicable"
        confidence["timing"]["tier"] = "not_applicable"
        return calibrated
    confidence["source"]["tier"] = (
        "strong"
        if calibrated["best_score"] >= thresholds["strong_score"]
        and calibrated["margin"] >= thresholds["strong_margin"]
        else "review"
    )
    timing = confidence["timing"]
    confidence["timing"]["tier"] = (
        "strong"
        if timing["support_fraction"] is not None
        and timing["support_fraction"] >= thresholds["focused_support_fraction"]
        and timing["offset_profile_width_sec"] is not None
        and timing["offset_profile_width_sec"] <= thresholds["focused_profile_width_sec"]
        else "review"
    )
    return calibrated


def tier_for(row: dict, calibration: dict) -> str | None:
    confidence = apply_calibration(row, calibration)["confidence"]
    if confidence["decision"] in {"unmatched", "ambiguous"}:
        return None
    if confidence["source"]["tier"] == "strong" and confidence["timing"]["tier"] == "strong":
        return "strong"
    return "review"


def measure_heldout(rows: list[dict], truth: dict[str, dict], calibration: dict, tolerance=1.5) -> dict:
    """Measure frozen tiers on held-out rows without retuning or hiding empty tiers."""
    tolerance = _validate_tolerance(tolerance)
    published = ("strong", "review")
    buckets = {
        tier: {"count": 0, "source_correct": 0, "timing_correct": 0, "joint_correct": 0}
        for tier in published
    }
    decisions = {name: 0 for name in ("matched", "uncertain", "ambiguous", "unmatched")}
    for row in rows:
        decision = row["confidence"]["decision"]
        decisions[decision] += 1
        tier = tier_for(row, calibration)
        if tier is None:
            continue
        bucket = buckets[tier]
        bucket["count"] += 1
        expected = truth[row["clip"]]
        expected_master, expected_offset = _expected(expected)
        source_ok = row["best_master"] == expected_master
        timing_ok = (
            row["source_offset_sec"] is not None
            and expected_offset is not None
            and abs(row["source_offset_sec"] - expected_offset) <= tolerance
        )
        bucket["source_correct"] += int(source_ok)
        bucket["timing_correct"] += int(timing_ok)
        bucket["joint_correct"] += int(source_ok and timing_ok)
    for bucket in buckets.values():
        count = bucket["count"]
        bucket["source_accuracy"] = bucket["source_correct"] / count if count else None
        bucket["timing_accuracy"] = bucket["timing_correct"] / count if count else None
        bucket["joint_accuracy"] = bucket["joint_correct"] / count if count else None
    return {
        "split": "holdout_only",
        "semantics": "observed_frequency_not_probability",
        "tolerance_seconds": tolerance,
        "calibration_sha256": calibration["calibration_sha256"],
        "published_tiers": list(published),
        "all_published_tiers_observed": all(buckets[tier]["count"] > 0 for tier in published),
        "decision_counts": decisions,
        "tiers": buckets,
    }
