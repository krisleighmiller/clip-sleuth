"""Evaluation adapter and scoring for the Clip Provenance product."""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from clipprov._io import finite_number

TOLERANCE_SECONDS = 1.5


def _basename(value):
    return value.replace("\\", "/").rsplit("/", 1)[-1]


def _canonical_clip(value: Any) -> str:
    if not isinstance(value, str) or not value or not _basename(value):
        raise ValueError("clip name must be a non-empty string")
    name = _basename(value)
    return name if name.endswith(".mp4") else f"{name}.mp4"


def _canonical_master(value: Any) -> str | None:
    if value is None:
        return None
    if type(value) in (int, float):
        if not finite_number(value) or value < 0 or int(value) != value:
            raise ValueError("master id must be a non-negative integer")
        return f"master_{int(value):02d}.mp4"
    if not isinstance(value, str) or not value or not _basename(value):
        raise ValueError("master must be a non-empty name or null")
    return _canonical_clip(value)


def _number_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if not finite_number(value):
        raise ValueError("number must be a finite JSON number or null")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError("number too large to represent as float") from exc
    return number


PRODUCT_FIELDS = {
    "schema_version",
    "clip",
    "accepted",
    "best_master",
    "source_record",
    "raw_alignment_offset_sec",
    "source_offset_sec",
    "speed",
    "best_score",
    "second_score",
    "margin",
    "reason",
    "accepted_master",
    "accepted_offset_sec",
    "accepted_speed",
}


PRODUCT_V2_FIELDS = PRODUCT_FIELDS | {"confidence", "evidence"}
PRODUCT_V3_FIELDS = PRODUCT_V2_FIELDS | {"matching_receipt"}


def _confidence_result(diag, *, geometry_enabled=False, audio_enabled=False):
    competitor = diag.ranked[1] if len(diag.ranked) > 1 else None
    result = {
        "semantics": "diagnostic_not_probability",
        "calibration": {"status": "uncalibrated"},
        "decision": diag.decision,
        "source": {
            "tier": diag.source_confidence,
            "best_score": diag.best_score,
            "competing_master": competitor[1] if competitor else None,
            "competing_score": competitor[0] if competitor else diag.second_score,
            "cross_master_margin": diag.margin,
            "acceptance_floor": diag.acceptance_floor,
            "acceptance_margin": diag.acceptance_margin,
        },
        "timing": {
            "tier": diag.timing_confidence,
            "supporting_samples": diag.supporting_samples,
            "sample_count": diag.sample_count,
            "support_fraction": diag.support_fraction,
            "temporal_residual_sec": diag.temporal_residual_sec,
            "offset_profile_width_sec": diag.offset_profile_width_sec,
            "offset_ambiguity": diag.offset_ambiguity,
            "selected_reframe": {
                "index": diag.reframe_index,
                "crop": list(diag.reframe) if diag.reframe is not None else None,
            },
            "speed_ratio": diag.speed,
            "aligned_timestamps": [
                {"clip_sec": clip_sec, "master_sec": master_sec}
                for clip_sec, master_sec in diag.aligned_timestamps
            ],
        },
    }
    if audio_enabled:
        result["audio"] = diag.audio
    if geometry_enabled:
        result["timing"]["selected_geometry_hypothesis"] = diag.geometry_hypothesis
    return result


GRID_EPSILON_SEC = 1e-6


def alignment_timestamp_problems(row: dict[str, Any]) -> list[str]:
    """Check published timestamps against the frame-grid alignment rule.

    ``appearance.search_diagnostics`` samples masters on an integer frame grid
    at ``clipprov.motion.FPS``. With raw alignment offset ``O`` (seconds, may
    be negative before the non-negative clamp applied to ``source_offset_sec``)
    and speed ``s``, a clip frame at grid index ``k`` satisfies exactly:

        clip_sec  = k / FPS
        master_sec = O + round(s * k) / FPS

    where ``round`` is ties-to-even, matching ``np.rint``. Every reported pair
    must satisfy this within ``GRID_EPSILON_SEC``. Returns problem strings;
    an empty list means consistent. Malformed types are reported before any
    arithmetic so callers never see a raw TypeError escape validation.
    """
    pairs = None
    if isinstance(row, dict):
        confidence = row.get("confidence")
        if isinstance(confidence, dict):
            timing = confidence.get("timing")
            if isinstance(timing, dict):
                pairs = timing.get("aligned_timestamps")
    if pairs is None:
        return []
    if not isinstance(pairs, list):
        return ["aligned_timestamps must be a list"]
    if not pairs:
        return []

    def _finite(value):
        try:
            return _number_or_none(value), None
        except ValueError as exc:
            return None, str(exc)

    raw, problem = _finite(row.get("raw_alignment_offset_sec"))
    if problem:
        return [f"invalid raw alignment offset: {problem}"]
    speed, problem = _finite(row.get("speed"))
    if problem:
        return [f"invalid speed: {problem}"]
    if raw is None or speed is None:
        return ["aligned timestamps require a raw offset and speed"]
    if speed <= 0:
        return ["speed must be positive"]

    from clipprov.motion import FPS  # deferred: keeps eval import-light

    problems = []
    for pair in pairs:
        if not isinstance(pair, dict) or pair.keys() != {"clip_sec", "master_sec"}:
            problems.append("invalid aligned timestamp")
            continue
        clip_sec, clip_problem = _finite(pair["clip_sec"])
        master_sec, master_problem = _finite(pair["master_sec"])
        if clip_problem or master_problem:
            problems.extend(p for p in (clip_problem, master_problem) if p)
            continue
        if clip_sec is None or master_sec is None or clip_sec < 0 or master_sec < 0:
            problems.append("aligned timestamps must be non-negative numbers")
            continue
        try:
            frame_index = round(clip_sec * FPS)
        except (OverflowError, ValueError):
            problems.append("clip timestamp overflow during frame index calculation")
            continue
        if abs(clip_sec - frame_index / FPS) > GRID_EPSILON_SEC:
            problems.append("clip timestamp off the sample grid")
            continue
        try:
            expected_master = raw + round(speed * frame_index) / FPS
        except (OverflowError, ValueError):
            problems.append("master timestamp overflow during expected calculation")
            continue
        if abs(master_sec - expected_master) > GRID_EPSILON_SEC:
            problems.append("master timestamp disagrees with raw offset and speed")
    return problems


def product_result(
    clip, diag, manifest, *, evidence=None, schema_version=2, matching_receipt=None
):
    """Serialize a match; schema v3 binds the search policy and immutable inputs."""
    source = next((m for m in manifest["masters"] if m["id"] == diag.best_master), None)
    record_fields = ("id", "source_relpath", "display_name", "original_root_hint", "content_digest")
    row = {
        key: getattr(diag, key)
        for key in sorted(PRODUCT_FIELDS - {"schema_version", "clip", "source_record"})
    }
    base = {
        **row,
        "schema_version": 1,
        "clip": str(clip),
        "source_record": {k: source[k] for k in record_fields} if source else None,
    }
    if schema_version == 1:
        return base
    if matching_receipt is not None:
        schema_version = 3
    if schema_version not in {2, 3}:
        raise ValueError("unsupported product schema version")
    receipt_version = matching_receipt.get("schema_version") if matching_receipt else None
    geometry_enabled = bool(
        matching_receipt
        and (
            (receipt_version == 2 and matching_receipt.get("geometry_mode") == "d4")
            or (receipt_version == 1 and diag.geometry_mode != "identity")
        )
    )
    audio_enabled = bool(
        matching_receipt
        and (
            (receipt_version == 2 and matching_receipt.get("audio_enabled") is True)
            or (receipt_version == 1 and diag.audio is not None)
        )
    )
    result = {
        **base,
        "schema_version": schema_version,
        "confidence": _confidence_result(
            diag, geometry_enabled=geometry_enabled, audio_enabled=audio_enabled
        ),
        "evidence": evidence,
    }
    if schema_version == 3:
        if matching_receipt is None:
            raise ValueError("product schema v3 requires a matching receipt")
        result["matching_receipt"] = matching_receipt
    return result


def _validate_matching_receipt(value):
    if not isinstance(value, dict):
        raise TypeError("invalid matching receipt")
    version = value.get("schema_version")
    v1_fields = {"schema_version", "speed_profile", "input_receipt"}
    v2_fields = v1_fields | {"geometry_mode", "audio_enabled"}
    if (version == 1 and value.keys() != v1_fields) or (
        version == 2 and value.keys() != v2_fields
    ):
        raise ValueError("invalid matching receipt")
    if version not in {1, 2}:
        raise ValueError("unsupported matching receipt schema version")
    if version == 2:
        if value["geometry_mode"] not in {"identity", "d4"}:
            raise ValueError("invalid matching receipt geometry mode")
        if type(value["audio_enabled"]) is not bool:
            raise ValueError("invalid matching receipt audio flag")
    profile = value["speed_profile"]
    fields = {
        "schema_version", "name", "speed_min", "speed_max", "speed_steps",
        "measured_range",
    }
    if not isinstance(profile, dict) or profile.keys() != fields or profile["schema_version"] != 1:
        raise ValueError("invalid speed profile receipt")
    name = profile["name"]
    if name not in {"v1-default", "wider-speed-v1"}:
        raise ValueError("unsupported speed profile")
    speed_min = _number_or_none(profile["speed_min"])
    speed_max = _number_or_none(profile["speed_max"])
    steps = profile["speed_steps"]
    if speed_min is None or speed_max is None or speed_min <= 0 or speed_min >= speed_max:
        raise ValueError("invalid speed profile range")
    if type(steps) is not int or steps <= 0:
        raise ValueError("invalid speed profile steps")
    if name == "wider-speed-v1" and (
        speed_min != 0.5 or speed_max != 2.0 or steps != 401
        or profile["measured_range"] != {"minimum": 0.5, "maximum": 2.0}
    ):
        raise ValueError("wider speed profile options disagree with schema")
    if name == "v1-default" and profile["measured_range"] is not None:
        raise ValueError("default speed profile cannot claim a measured range")
    inputs = value["input_receipt"]
    if not isinstance(inputs, dict) or inputs.keys() != {
        "clip_sha256", "index_manifest_sha256",
    }:
        raise ValueError("invalid matching input receipt")
    for digest_value in inputs.values():
        if (
            not isinstance(digest_value, str) or len(digest_value) != 64
            or any(char not in "0123456789abcdef" for char in digest_value)
        ):
            raise ValueError("invalid matching input digest")


def _same_number(left, right):
    left = _number_or_none(left)
    right = _number_or_none(right)
    if left is None or right is None:
        return left is right
    return math.isclose(left, right, rel_tol=0, abs_tol=1e-9)


def _validate_confidence(value, row, *, allow_phase6=False):
    required = {"semantics", "calibration", "decision", "source", "timing"}
    if not isinstance(value, dict) or set(value) not in {frozenset(required), frozenset(required | {"audio"})}:
        raise ValueError("invalid confidence object")
    if not allow_phase6 and set(value) != required:
        raise ValueError("schema v2 confidence must use the exact Phase 5 fields")
    if value["semantics"] != "diagnostic_not_probability":
        raise ValueError("confidence semantics must be non-probabilistic")
    decision = value["decision"]
    if decision not in {"matched", "uncertain", "ambiguous", "unmatched"}:
        raise ValueError("invalid decision")
    if row["reason"] != decision:
        raise ValueError("confidence decision disagrees with result reason")
    if row["accepted"] != (decision == "matched"):
        raise ValueError("confidence decision disagrees with accepted")

    if "audio" in value:
        audio = value["audio"]
        if audio is not None:
            fields = {
                "status", "reason", "score", "threshold", "algorithm", "sample_rate_hz",
                "compared_samples", "lag_samples", "clip_sha256", "source_sha256",
                "source_content_digest", "source_offset_sec",
            }
            if not isinstance(audio, dict) or audio.keys() != fields:
                raise ValueError("invalid audio corroboration receipt")
            if audio["status"] not in {"corroborated", "not_corroborated", "unavailable"}:
                raise ValueError("invalid audio corroboration status")
            if audio["algorithm"] != "pcm_ncc_v1":
                raise ValueError("invalid audio corroboration algorithm")
            if audio["sample_rate_hz"] != 16000:
                raise ValueError("invalid audio sample rate")
            if type(audio["compared_samples"]) is not int or audio["compared_samples"] < 0:
                raise ValueError("invalid audio sample count")
            if type(audio["lag_samples"]) is not int:
                raise ValueError("invalid audio lag")
            if abs(audio["lag_samples"]) > audio["sample_rate_hz"] // 2:
                raise ValueError("audio lag exceeds bounded search")
            if not _same_number(audio["source_offset_sec"], row["source_offset_sec"]):
                raise ValueError("audio offset disagrees with visual alignment")
            if not isinstance(audio["reason"], str) or not audio["reason"]:
                raise ValueError("invalid audio reason")
            if not _same_number(audio["threshold"], 0.45):
                raise ValueError("invalid audio threshold")

            def _digest_or_none(candidate):
                if candidate is None:
                    return None
                if (
                    not isinstance(candidate, str)
                    or len(candidate) != 64
                    or any(character not in "0123456789abcdef" for character in candidate)
                ):
                    raise ValueError("invalid audio snapshot digest")
                return candidate

            clip_digest = _digest_or_none(audio["clip_sha256"])
            source_digest = _digest_or_none(audio["source_sha256"])
            indexed_digest = _digest_or_none(audio["source_content_digest"])
            source_record = row.get("source_record")
            if not isinstance(source_record, dict) or indexed_digest != source_record.get("content_digest"):
                raise ValueError("audio digest disagrees with source record")
            if audio["status"] == "unavailable" and audio["score"] is not None:
                raise ValueError("unavailable audio cannot publish a score")
            if audio["status"] != "unavailable":
                score = _number_or_none(audio["score"])
                if score is None or not 0 <= score <= 1:
                    raise ValueError("available audio requires a bounded score")
                if clip_digest is None or source_digest != indexed_digest:
                    raise ValueError("available audio requires verified snapshots")
                if audio["compared_samples"] < 2 * audio["sample_rate_hz"]:
                    raise ValueError("available audio has insufficient samples")
                corroborates = score >= audio["threshold"]
                if corroborates != (audio["status"] == "corroborated"):
                    raise ValueError("audio status disagrees with score")

    source = value["source"]
    source_fields = {
        "tier", "best_score", "competing_master", "competing_score",
        "cross_master_margin", "acceptance_floor", "acceptance_margin",
    }
    if not isinstance(source, dict) or source.keys() != source_fields:
        raise ValueError("invalid source confidence")
    if not _same_number(source["best_score"], row["best_score"]):
        raise ValueError("confidence best score disagrees with result")
    if not _same_number(source["competing_score"], row["second_score"]):
        raise ValueError("confidence competing score disagrees with result")
    if not _same_number(source["cross_master_margin"], row["margin"]):
        raise ValueError("confidence margin disagrees with result")
    for key in ("acceptance_floor", "acceptance_margin"):
        value_number = _number_or_none(source[key])
        if value_number is None or value_number < 0:
            raise ValueError(f"{key} must be a non-negative number")
    competitor = source["competing_master"]
    if competitor is not None and (
        not isinstance(competitor, str) or not competitor or _basename(competitor) != competitor
    ):
        raise ValueError("invalid competing master")

    timing = value["timing"]
    timing_fields = {
        "tier", "supporting_samples", "sample_count", "support_fraction",
        "temporal_residual_sec", "offset_profile_width_sec", "offset_ambiguity",
        "selected_reframe", "speed_ratio", "aligned_timestamps",
    }
    geometry_field = "selected_geometry_hypothesis"
    if not isinstance(timing, dict) or set(timing) not in {
        frozenset(timing_fields), frozenset(timing_fields | {geometry_field})
    }:
        raise ValueError("invalid timing confidence")
    if not allow_phase6 and set(timing) != timing_fields:
        raise ValueError("schema v2 timing must use the exact Phase 5 fields")
    if geometry_field in timing and timing[geometry_field] not in {
        "identity", "rotate-90-ccw", "rotate-180", "rotate-270-ccw",
        "mirror-horizontal", "mirror-horizontal-rotate-90-ccw",
        "mirror-horizontal-rotate-180", "mirror-horizontal-rotate-270-ccw",
    }:
        raise ValueError("invalid selected geometry hypothesis")
    for key in ("supporting_samples", "sample_count"):
        if type(timing[key]) is not int or timing[key] < 0:
            raise ValueError(f"{key} must be a non-negative integer")
    if timing["supporting_samples"] > timing["sample_count"]:
        raise ValueError("supporting_samples exceeds sample_count")
    support = _number_or_none(timing["support_fraction"])
    if support is not None and not 0 <= support <= 1:
        raise ValueError("support_fraction must be in [0, 1]")
    if timing["sample_count"]:
        expected_support = timing["supporting_samples"] / timing["sample_count"]
        if support is None or not math.isclose(support, expected_support, abs_tol=1e-9):
            raise ValueError("support_fraction disagrees with sample counts")
    elif support is not None:
        raise ValueError("support_fraction requires samples")
    for key in ("temporal_residual_sec", "offset_profile_width_sec", "offset_ambiguity"):
        number = _number_or_none(timing[key])
        if number is not None and number < 0:
            raise ValueError(f"{key} must be non-negative")
    if not _same_number(timing["speed_ratio"], row["speed"]):
        raise ValueError("confidence speed disagrees with result")
    reframe = timing["selected_reframe"]
    if not isinstance(reframe, dict) or reframe.keys() != {"index", "crop"}:
        raise ValueError("invalid selected reframe")
    if reframe["index"] is not None and (type(reframe["index"]) is not int or reframe["index"] < 0):
        raise ValueError("invalid selected reframe index")
    crop = reframe["crop"]
    if crop is not None and (
        not isinstance(crop, list)
        or len(crop) != 3
        or any(_number_or_none(item) is None for item in crop)
    ):
        raise ValueError("invalid selected reframe crop")
    if not isinstance(timing["aligned_timestamps"], list):
        raise TypeError("aligned_timestamps must be a list")
    for pair in timing["aligned_timestamps"]:
        if not isinstance(pair, dict) or pair.keys() != {"clip_sec", "master_sec"}:
            raise ValueError("invalid aligned timestamp")
        clip_sec = _number_or_none(pair["clip_sec"])
        master_sec = _number_or_none(pair["master_sec"])
        if clip_sec is None or master_sec is None or clip_sec < 0 or master_sec < 0:
            raise ValueError("aligned timestamps must be non-negative numbers")

    calibration = value["calibration"]
    if not isinstance(calibration, dict) or calibration.get("status") not in {"uncalibrated", "applied"}:
        raise ValueError("invalid confidence calibration")
    if calibration["status"] == "uncalibrated":
        if calibration != {"status": "uncalibrated"}:
            raise ValueError("invalid uncalibrated confidence receipt")
        if source["tier"] != "uncalibrated" or timing["tier"] != "uncalibrated":
            raise ValueError("uncalibrated confidence cannot publish fixed tiers")
    else:
        import re

        if calibration.keys() != {"status", "sha256"} or not re.fullmatch(
            r"[0-9a-f]{64}", str(calibration.get("sha256", ""))
        ):
            raise ValueError("invalid applied calibration receipt")
        allowed = {"strong", "review", "not_applicable"}
        if source["tier"] not in allowed or timing["tier"] not in allowed:
            raise ValueError("invalid calibrated tier")
        if decision in {"unmatched", "ambiguous"} and (
            source["tier"] != "not_applicable" or timing["tier"] != "not_applicable"
        ):
            raise ValueError("non-match decisions cannot publish calibrated tiers")


def _validate_v3_bindings(row):
    receipt = row["matching_receipt"]
    profile = receipt["speed_profile"]
    lower = profile["speed_min"]
    upper = profile["speed_max"]
    for field in ("speed", "accepted_speed"):
        value = row[field]
        if value is not None and not lower <= value <= upper:
            raise ValueError(f"{field} is outside matching receipt speed profile bounds")
    audio = row["confidence"].get("audio")
    if receipt["schema_version"] == 2:
        timing = row["confidence"]["timing"]
        geometry_present = "selected_geometry_hypothesis" in timing
        if geometry_present != (receipt["geometry_mode"] == "d4"):
            raise ValueError("geometry diagnostic disagrees with matching receipt")
        if ("audio" in row["confidence"]) != receipt["audio_enabled"]:
            raise ValueError("audio diagnostic disagrees with matching receipt")
    if (
        isinstance(audio, dict)
        and audio.get("clip_sha256") is not None
        and audio["clip_sha256"] != receipt["input_receipt"]["clip_sha256"]
    ):
        raise ValueError("audio clip digest disagrees with matching receipt")


def adapt_product(row: dict[str, Any], *, _allow_phase6=False) -> dict[str, Any]:
    """Validate product schema v1-v3 without discarding diagnostic proposals."""
    if isinstance(row, dict) and row.get("schema_version") == 3:
        if row.keys() != PRODUCT_V3_FIELDS:
            raise ValueError("result must contain exactly the product schema v3 fields")
        _validate_matching_receipt(row["matching_receipt"])
        base = {key: row[key] for key in PRODUCT_V2_FIELDS}
        base["schema_version"] = 2
        adapted = adapt_product(base, _allow_phase6=True)
        _validate_v3_bindings(row)
        return adapted
    if isinstance(row, dict) and row.get("schema_version") == 2:
        if row.keys() != PRODUCT_V2_FIELDS:
            raise ValueError("result must contain exactly the product schema v2 fields")
        if row["evidence"] is not None and not isinstance(row["evidence"], dict):
            raise ValueError("evidence must be an object or null")
        base = {key: row[key] for key in PRODUCT_FIELDS}
        base["schema_version"] = 1
        adapted = adapt_product(base)
        _validate_confidence(
            row["confidence"], row, allow_phase6=_allow_phase6
        )
        timestamp_problems = alignment_timestamp_problems(row)
        if timestamp_problems:
            raise ValueError(
                "aligned timestamps disagree with alignment: " + "; ".join(timestamp_problems)
            )
        if row["evidence"] is not None:
            from clipprov.evidence import validate_evidence

            problems = validate_evidence(row["evidence"], row)
            if problems:
                raise ValueError("evidence disagrees with result: " + "; ".join(problems))
        adapted["native_tier"] = row["confidence"]["decision"]
        return adapted
    if not isinstance(row, dict) or row.keys() != PRODUCT_FIELDS:
        raise ValueError("result must contain exactly the product schema v1 fields")
    if type(row["schema_version"]) is not int or row["schema_version"] != 1:
        raise ValueError("unsupported schema version")
    if type(row["accepted"]) is not bool:
        raise ValueError("accepted must be a boolean")
    if not isinstance(row["reason"], str) or not row["reason"]:
        raise ValueError("reason must be a non-empty string")
    accepted = row["accepted"]
    if (row["reason"] == "matched") != accepted:
        raise ValueError("reason disagrees with accepted")
    best = row["best_master"]
    if best is not None and (not isinstance(best, str) or not best or _basename(best) != best):
        raise ValueError("best_master must be a source identifier or null")
    source = row["source_record"]
    if source is not None and not isinstance(source, dict):
        raise ValueError("source_record must be an object or null")
    if (best is None) != (source is None):
        raise ValueError("source_record must identify the best candidate")
    master = _canonical_master(best)
    if source is not None:
        fields = {"id", "source_relpath", "display_name", "original_root_hint", "content_digest"}
        if source.keys() != fields or any(
            not isinstance(source[k], str) or not source[k] for k in fields
        ):
            raise ValueError("invalid source_record fields")
        if (
            source["id"] != best
            or _canonical_master(source["source_relpath"]) != master
            or source["display_name"] != _basename(source["source_relpath"])
        ):
            raise ValueError("inconsistent source_record identity")
        import re

        if not re.fullmatch(r"[0-9a-f]{64}", source["content_digest"]):
            raise ValueError("invalid source content digest")
    numbers = {
        k: _number_or_none(row[k])
        for k in (
            "raw_alignment_offset_sec",
            "source_offset_sec",
            "speed",
            "best_score",
            "second_score",
            "margin",
            "accepted_offset_sec",
            "accepted_speed",
        )
    }
    if any(numbers[k] is None for k in ("best_score", "second_score", "margin")):
        raise ValueError("scores and margin must be numbers")
    offset, raw, speed = (
        numbers[k] for k in ("source_offset_sec", "raw_alignment_offset_sec", "speed")
    )
    if (offset is None) != (raw is None) or (offset is None) != (speed is None):
        raise ValueError("candidate alignment fields must be set together")
    if offset is not None and (offset != max(0, raw) or speed <= 0):
        raise ValueError("invalid candidate alignment")
    if best is None and offset is not None:
        raise ValueError("alignment without candidate")
    if best is not None and not math.isclose(
        numbers["margin"], numbers["best_score"] - numbers["second_score"], abs_tol=1e-9
    ):
        raise ValueError("margin disagrees with best and second scores")
    if accepted:
        if best is None or offset is None:
            raise ValueError("accepted row requires a candidate and offset")
        if (
            row["accepted_master"] != best
            or numbers["accepted_offset_sec"] != offset
            or numbers["accepted_speed"] != speed
        ):
            raise ValueError("accepted fields disagree with candidate")
    elif any(
        row[k] is not None for k in ("accepted_master", "accepted_offset_sec", "accepted_speed")
    ):
        raise ValueError("rejected rows must have null accepted fields")
    return {
        "clip": _canonical_clip(row["clip"]),
        "native_tier": "accepted" if accepted else row["reason"],
        "tier": "accepted" if accepted else ("uncertain" if best is not None else "unidentified"),
        "candidate_master": master,
        "candidate_offset": offset,
    }


def load_truth(path: Path, *, with_digest: bool = False) -> dict[str, dict[str, Any]] | tuple[dict[str, dict[str, Any]], str]:
    from clipprov._io import json_snapshot

    raw, sha = json_snapshot(path)
    if not isinstance(raw, list):
        raise ValueError("answer key must be a JSON array")  # noqa: TRY004
    truth = {}
    for index, row in enumerate(raw):
        if not isinstance(row, dict):
            raise ValueError(f"answer-key row {index} is not an object")  # noqa: TRY004
        if not {"clip", "master", "offset_sec"}.issubset(row):
            raise ValueError("answer-key row requires clip, master and offset_sec")
        clip = _canonical_clip(row.get("clip"))
        if clip in truth:
            raise ValueError(f"answer key contains duplicate clip {clip}")
        master = _canonical_master(row.get("master"))
        offset = _number_or_none(row.get("offset_sec"))
        if (master is None) != (offset is None) or (offset is not None and offset < 0):
            raise ValueError(f"answer key {clip}: master and offset must both be set or null")
        truth[clip] = {"clip": clip, "master": master, "offset_sec": offset}
    return (truth, sha) if with_digest else truth


def load_product_results(path: Path, *, with_digest: bool = False) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], str]:
    from clipprov._io import json_snapshot

    raw, sha = json_snapshot(path)
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        raise ValueError("product results must be a JSON array or object")  # noqa: TRY004
    rows = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"row {index} is not an object")  # noqa: TRY004
        rows.append(adapt_product(item))
    return (rows, sha) if with_digest else rows


def validate_coverage(rows, truth):
    names = [r["clip"] for r in rows]
    counts = Counter(names)
    duplicates = sorted(n for n, c in counts.items() if c > 1)
    missing = sorted(set(truth) - set(names))
    unknown = sorted(set(names) - set(truth))
    problems = []
    if duplicates:
        problems.append(f"duplicate clips: {duplicates[:10]}")
    if missing:
        problems.append(f"missing clips: {missing[:10]}")
    if unknown:
        problems.append(f"unknown clips: {unknown[:10]}")
    if problems:
        raise ValueError("; ".join(problems))


def score_product(
    results,
    truth,
    tolerance_seconds=TOLERANCE_SECONDS,
    results_digest=None,
    truth_digest=None,
    scorer_version="1.0.0",
):
    if not isinstance(tolerance_seconds, (int, float)) or isinstance(tolerance_seconds, bool):
        raise ValueError("tolerance must be a finite number")  # noqa: TRY004
    if not finite_number(tolerance_seconds) or tolerance_seconds < 0:
        raise ValueError("tolerance must be finite and non-negative")
    tol = float(tolerance_seconds)
    for row in results:
        if not isinstance(row, dict) or not {
            "clip",
            "tier",
            "candidate_master",
            "candidate_offset",
        }.issubset(row):
            raise ValueError("invalid normalized result")
        tier, master = row["tier"], row["candidate_master"]
        offset = _number_or_none(row["candidate_offset"])
        if (
            tier not in {"accepted", "uncertain", "unidentified"}
            or (master is not None and _canonical_master(master) != master)
            or (offset is not None and (offset < 0 or master is None))
            or (tier == "accepted" and (master is None or offset is None))
            or (tier == "uncertain" and master is None)
            or (tier == "unidentified" and (master is not None or offset is not None))
        ):
            raise ValueError("inconsistent normalized candidate or tier")
    for expected in truth.values():
        if not isinstance(expected, dict) or not {"master", "offset_sec"}.issubset(expected):
            raise ValueError("invalid normalized truth")
        master = expected["master"]
        offset = _number_or_none(expected["offset_sec"])
        if (
            (master is None) != (offset is None)
            or (offset is not None and offset < 0)
            or (master is not None and _canonical_master(master) != master)
        ):
            raise ValueError("invalid normalized truth master or offset")
    validate_coverage(results, truth)
    scored_rows = []
    for row in results:
        expected = truth[row["clip"]]
        truth_kind = "foreign" if expected["master"] is None else "genuine"
        source_correct = row["candidate_master"] == expected["master"]
        offset_error = None
        if truth_kind == "genuine" and source_correct and row["candidate_offset"] is not None:
            offset_error = abs(row["candidate_offset"] - expected["offset_sec"])
        if truth_kind == "foreign":
            joint_correct = row["candidate_master"] is None and row["candidate_offset"] is None
        else:
            joint_correct = source_correct and offset_error is not None and offset_error <= tol
        scored_rows.append(
            {
                **row,
                "truth_kind": truth_kind,
                "truth_master": expected["master"],
                "truth_offset": expected["offset_sec"],
                "source_correct": source_correct,
                "offset_error_seconds": offset_error,
                "joint_correct": joint_correct,
                "accepted_match": row["tier"] == "accepted",
            }
        )
    genuine_total = sum(r["truth_kind"] == "genuine" for r in scored_rows)
    foreign_total = len(scored_rows) - genuine_total
    accepted = [r for r in scored_rows if r["accepted_match"]]
    proposed = [r for r in scored_rows if r["tier"] != "unidentified"]
    uncertain = [r for r in scored_rows if r["tier"] == "uncertain"]
    unidentified = [r for r in scored_rows if r["tier"] == "unidentified"]
    accepted_source_correct = sum(
        r["truth_kind"] == "genuine" and r["source_correct"] for r in accepted
    )
    accepted_joint_correct = sum(
        r["truth_kind"] == "genuine" and r["joint_correct"] for r in accepted
    )
    proposed_source_correct = sum(
        r["truth_kind"] == "genuine" and r["source_correct"] for r in proposed
    )
    proposed_joint_correct = sum(
        r["truth_kind"] == "genuine" and r["joint_correct"] for r in proposed
    )
    explicit_foreign_rejections = sum(
        r["truth_kind"] == "foreign" and not r["accepted_match"] for r in scored_rows
    )

    def ratio(a, b):
        return round(a / b, 6) if b else None

    grouped = defaultdict(list)
    for r in scored_rows:
        grouped[r["tier"]].append(r)

    def tier_metrics(rows):
        errors = [
            r["offset_error_seconds"]
            for r in rows
            if r["truth_kind"] == "genuine"
            and r["source_correct"]
            and r["offset_error_seconds"] is not None
        ]
        return {
            "count": len(rows),
            "genuine": sum(r["truth_kind"] == "genuine" for r in rows),
            "foreign": sum(r["truth_kind"] == "foreign" for r in rows),
            "correct_genuine_sources": sum(
                r["truth_kind"] == "genuine" and r["source_correct"] for r in rows
            ),
            "joint_correct": sum(r["joint_correct"] for r in rows),
            "median_offset_error_seconds": statistics.median(errors) if errors else None,
            "max_offset_error_seconds": max(errors) if errors else None,
        }

    return {
        "evaluation_schema_version": 1,
        "tolerance_seconds": tol,
        "scorer_version": scorer_version,
        "results_digest": results_digest,
        "truth_digest": truth_digest,
        "adapter": "product",
        "display_name": "Clip Provenance product",
        "accepted_tiers": ["accepted"],
        "metrics": {
            "rows": len(scored_rows),
            "genuine_clips": genuine_total,
            "foreign_clips": foreign_total,
            "accepted_matches": len(accepted),
            "accepted_correct_sources": accepted_source_correct,
            "accepted_correct_source_and_time": accepted_joint_correct,
            "accepted_source_precision": ratio(accepted_source_correct, len(accepted)),
            "accepted_joint_precision": ratio(accepted_joint_correct, len(accepted)),
            "accepted_source_recall": ratio(accepted_source_correct, genuine_total),
            "accepted_joint_recall": ratio(accepted_joint_correct, genuine_total),
            "proposed_matches": len(proposed),
            "proposed_source_accuracy": ratio(proposed_source_correct, len(proposed)),
            "proposed_joint_accuracy": ratio(proposed_joint_correct, len(proposed)),
            "proposed_correct_genuine_sources": proposed_source_correct,
            "proposed_correct_genuine_source_and_time": proposed_joint_correct,
            "explicit_foreign_rejections": explicit_foreign_rejections,
            "foreign_rejection_recall": ratio(explicit_foreign_rejections, foreign_total),
            "genuine_unidentified": sum(r["truth_kind"] == "genuine" for r in unidentified),
            "uncertain_rows": len(uncertain),
            "uncertain_genuine": sum(r["truth_kind"] == "genuine" for r in uncertain),
            "uncertain_foreign": sum(r["truth_kind"] == "foreign" for r in uncertain),
            "all_rows_joint_correct": sum(r["joint_correct"] for r in scored_rows),
        },
        "tiers": {tier: tier_metrics(rows) for tier, rows in sorted(grouped.items())},
        "rows": sorted(scored_rows, key=lambda r: r["clip"]),
    }
