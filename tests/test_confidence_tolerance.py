"""Tolerance propagation regressions for confidence calibration and heldout.

Tolerance must flow from the harness scorer into derive_tiers eligibility,
measure_heldout correctness, and the calibration/measurement receipts.
Backwards-compatible default is 1.5.
"""
from __future__ import annotations

import pytest

from clipprov.confidence import derive_tiers, measure_heldout


def _row(name, master="master_00", offset=1.0, support=0.9, width=0.25):
    return {
        "clip": name,
        "best_master": master,
        "source_offset_sec": offset,
        "best_score": 0.8,
        "margin": 0.3,
        "confidence": {
            "calibration": {"status": "uncalibrated"},
            "decision": "matched",
            "source": {
                "tier": "uncalibrated",
                "best_score": 0.8,
                "competing_score": 0.5,
                "cross_master_margin": 0.3,
            },
            "timing": {
                "tier": "uncalibrated",
                "support_fraction": support,
                "offset_profile_width_sec": width,
            },
        },
    }


# Backwards-compatible default: 1.5 still works.
def test_default_tolerance_backwards_compatible():
    truth = {
        "a.mp4": {"master": "master_00.mp4", "offset_sec": 1.0},
    }
    # error = 1.4, within default 1.5
    row = _row("a.mp4", offset=2.4)
    cal = derive_tiers([row], truth)
    assert cal["eligible_correct_alignments"] == 1
    measured = measure_heldout([row], truth, cal)
    assert measured["tiers"]["strong"]["timing_correct"] == 1


# Custom tolerance excludes rows whose offset error falls between the
# custom tolerance and the default 1.5. Ordinary scorer and confidence
# eligibility/tier correctness must agree.
def test_custom_tolerance_propagates_to_eligibility_and_heldout():
    truth = {
        "a.mp4": {"master": "master_00.mp4", "offset_sec": 1.0},
    }
    # error = 1.2, between 0.01 and 1.5
    row = _row("a.mp4", offset=2.2)
    cal = derive_tiers([row], truth, tolerance=0.01)
    assert cal["eligible_correct_alignments"] == 0
    measured = measure_heldout([row], truth, cal, tolerance=0.01)
    assert measured["tiers"]["review"]["timing_correct"] == 0


# Tolerance is recorded explicitly in the calibration receipt.
def test_tolerance_recorded_in_calibration_receipt():
    truth = {
        "a.mp4": {"master": "master_00.mp4", "offset_sec": 1.0},
    }
    cal = derive_tiers([_row("a.mp4")], truth, tolerance=0.01)
    assert cal["tolerance_seconds"] == 0.01


# Calibration SHA must change when tolerance changes, so the receipt binds it.
def test_calibration_sha_binds_tolerance():
    truth = {
        "a.mp4": {"master": "master_00.mp4", "offset_sec": 1.0},
    }
    cal_01 = derive_tiers([_row("a.mp4")], truth, tolerance=0.01)
    cal_15 = derive_tiers([_row("a.mp4")], truth, tolerance=1.5)
    assert cal_01["calibration_sha256"] != cal_15["calibration_sha256"]


# Zero tolerance is permitted (ordinary scorer allows it).
def test_zero_tolerance_permitted():
    truth = {
        "a.mp4": {"master": "master_00.mp4", "offset_sec": 1.0},
    }
    row = _row("a.mp4", offset=1.0)
    cal = derive_tiers([row], truth, tolerance=0.0)
    assert cal["eligible_correct_alignments"] == 1
    measured = measure_heldout([row], truth, cal, tolerance=0.0)
    assert measured["tiers"]["strong"]["timing_correct"] == 1


# Negative tolerance is rejected at the library boundary.
def test_negative_tolerance_rejected():
    truth = {
        "a.mp4": {"master": "master_00.mp4", "offset_sec": 1.0},
    }
    with pytest.raises(ValueError, match="tolerance"):
        derive_tiers([_row("a.mp4")], truth, tolerance=-0.1)


# NaN tolerance is rejected.
def test_nan_tolerance_rejected():
    truth = {
        "a.mp4": {"master": "master_00.mp4", "offset_sec": 1.0},
    }
    with pytest.raises(ValueError, match="tolerance"):
        derive_tiers([_row("a.mp4")], truth, tolerance=float("nan"))


# Infinite tolerance is rejected.
def test_inf_tolerance_rejected():
    truth = {
        "a.mp4": {"master": "master_00.mp4", "offset_sec": 1.0},
    }
    with pytest.raises(ValueError, match="tolerance"):
        derive_tiers([_row("a.mp4")], truth, tolerance=float("inf"))


# Boolean tolerance is rejected.
def test_bool_tolerance_rejected():
    truth = {
        "a.mp4": {"master": "master_00.mp4", "offset_sec": 1.0},
    }
    with pytest.raises(ValueError, match="tolerance"):
        derive_tiers([_row("a.mp4")], truth, tolerance=True)


# measure_heldout also records tolerance in its receipt.
def test_measurement_receipt_records_tolerance():
    truth = {
        "a.mp4": {"master": "master_00.mp4", "offset_sec": 1.0},
    }
    row = _row("a.mp4")
    cal = derive_tiers([row], truth, tolerance=0.01)
    measured = measure_heldout([row], truth, cal, tolerance=0.01)
    assert measured["tolerance_seconds"] == 0.01
