"""Match a single clip against a loaded index."""

from __future__ import annotations

import numpy as np

from clipprov.appearance import search
from clipprov.core import (
    MatchDiagnostics,
    confidence_labels,
    decide,
    probe_clip,
    rank_candidates,
    search_geometry_diagnostics,
    speeds_from_config,
)
from clipprov.index import config_from_manifest, load_index


def attach_audio_corroboration(diag, clip_path, manifest):
    """Attach positive-only audio evidence without changing any visual field."""
    if not diag.accepted or diag.accepted_master is None:
        return diag
    records = {
        record.get("id"): record
        for record in manifest.get("masters", [])
        if isinstance(record, dict)
    }
    record = records.get(diag.accepted_master)
    if record is None or diag.source_offset_sec is None or diag.speed is None:
        return diag
    root = record.get("original_root_hint")
    relative = record.get("source_relpath")
    source_digest = record.get("content_digest")
    if not all(isinstance(value, str) and value for value in (root, relative, source_digest)):
        return diag
    assert isinstance(root, str)
    assert isinstance(relative, str)
    assert isinstance(source_digest, str)

    from pathlib import Path

    from clipprov.audio import corroborate_audio

    diag.audio = corroborate_audio(
        Path(clip_path),
        Path(root) / relative,
        source_digest,
        diag.source_offset_sec,
        diag.speed,
    )
    return diag


def match_clip_path(
    clip_path, index_dir, config=None, *, loaded_index=None, geometry_mode=None
):
    """Match via the library API, optionally reusing a validated loaded index."""
    manifest, masters = loaded_index if loaded_index is not None else load_index(index_dir)
    diag, _ = _match_loaded(
        clip_path, manifest, masters, config, geometry_mode=geometry_mode
    )
    return diag, manifest


def _match_loaded(
    clip_path, manifest, masters, config=None, *, exhaustive=False, geometry_mode=None
):
    if config is None:
        config = config_from_manifest(manifest)
    if geometry_mode is not None:
        config = config.model_copy(update={
            "match": config.match.model_copy(update={"geometry_mode": geometry_mode})
        })
    mc = config.match if config is not None else None
    cg, idx, cm, _filt = probe_clip(clip_path, config=config)

    mm = {}
    mshort = {}
    for mid, data in masters.items():
        mm[mid] = np.asarray(data["motion"]).ravel()
        mshort[mid] = [np.asarray(v) for v in data["shortlist_views"]]

    speeds = speeds_from_config(mc) if mc is not None else None
    score_floor = mc.score_floor if mc is not None else 0.35
    margin = mc.margin if mc is not None else 0.12
    geometry_mode = mc.geometry_mode if mc is not None else "identity"

    scored_short = rank_candidates(cg, idx, cm, mm, mshort, speeds=speeds, match_config=mc)
    cand_names = [n for (_, n, _, _) in scored_short]

    # load full views only for shortlisted candidates
    mfull = {}
    for name in cand_names:
        mfull[name] = [np.asarray(v) for v in masters[name]["load_views"]()]

    scored = []
    details = {}
    for name in sorted(cand_names):
        detail, hypothesis = search_geometry_diagnostics(
            cg, idx, mfull[name], speeds, mode=geometry_mode
        )
        details[name] = detail
        details[(name, "geometry")] = hypothesis
        scored.append((detail.score, name, detail.offset_sec, detail.speed))
    scored.sort(key=lambda row: (-row[0], row[1]))

    source_accepted, name, raw_off, sp, best, marg = decide(
        scored, accept_best=score_floor, accept_margin=margin
    )
    second = scored[1][0] if len(scored) > 1 else 0.0
    source_off = max(0.0, raw_off) if raw_off is not None else None
    detail = details.get(name)
    decision, source_confidence, timing_confidence = confidence_labels(
        source_accepted, best, marg, score_floor, margin, detail
    )
    accepted = decision == "matched"
    reason = decision
    diag = MatchDiagnostics(
        best_score=best,
        second_score=float(second),
        margin=float(marg),
        best_master=name,
        raw_alignment_offset_sec=raw_off,
        source_offset_sec=source_off,
        speed=sp,
        accepted=accepted,
        accepted_master=name if accepted else None,
        accepted_offset_sec=source_off if accepted else None,
        accepted_speed=sp if accepted else None,
        reason=reason,
        ranked=[(s, n, o, p) for (s, n, o, p) in scored],
        shortlist=[n for (_, n, _, _) in scored],
        acceptance_floor=float(score_floor),
        acceptance_margin=float(margin),
        supporting_samples=detail.supporting_samples if detail else 0,
        sample_count=detail.sample_count if detail else 0,
        support_fraction=detail.support_fraction if detail else None,
        temporal_residual_sec=detail.temporal_residual_sec if detail else None,
        offset_profile_width_sec=detail.offset_profile_width_sec if detail else None,
        offset_ambiguity=detail.offset_ambiguity if detail else None,
        reframe_index=detail.reframe_index if detail else None,
        reframe=detail.reframe if detail else None,
        aligned_timestamps=detail.aligned_timestamps if detail else (),
        decision=decision,
        source_confidence=source_confidence,
        timing_confidence=timing_confidence,
        geometry_mode=geometry_mode,
        geometry_hypothesis=details.get((name, "geometry"), "identity"),
    )
    if mc is not None and mc.audio_enabled:
        attach_audio_corroboration(diag, clip_path, manifest)
    audit = None
    if exhaustive:
        all_scored = []
        for mid in sorted(masters):
            views = mfull.get(mid)
            if views is None:
                views = [np.asarray(v) for v in masters[mid]["load_views"]()]
            if geometry_mode == "identity":
                sc, off, sp = search(cg, idx, views, speeds)
            else:
                exhaustive_detail, _hypothesis = search_geometry_diagnostics(
                    cg, idx, views, speeds, mode=geometry_mode
                )
                sc, off, sp = (
                    exhaustive_detail.score,
                    exhaustive_detail.offset_sec,
                    exhaustive_detail.speed,
                )
            all_scored.append((sc, mid, off, sp))
        all_scored.sort(key=lambda row: (-row[0], row[1]))
        winner = all_scored[0][1] if all_scored else None
        # A score of -1 with no offset is no alignment, not a successful recall hit.
        valid = bool(all_scored and all_scored[0][2] is not None)
        audit = {
            "clip": str(clip_path),
            "exhaustive_best_master": winner if valid else None,
            "exhaustive_best_score": all_scored[0][0] if valid else None,
            "winner_in_shortlist": winner in cand_names if valid else None,
            "masters_scored": len(all_scored),
            "shortlist": cand_names,
        }
    return diag, audit


def audit_shortlist(
    clip_path, index_dir, config=None, *, loaded_index=None, max_masters=64,
    geometry_mode=None,
):
    """Compare production shortlisting with every full-view score on a bounded index."""
    if type(max_masters) is not int or not 1 <= max_masters <= 64:
        raise ValueError("max_masters must be in [1, 64]")
    manifest, masters = loaded_index if loaded_index is not None else load_index(index_dir)
    if len(masters) > max_masters:
        raise ValueError(f"exhaustive audit is bounded to {max_masters} masters")
    diag, audit = _match_loaded(
        clip_path, manifest, masters, config, exhaustive=True, geometry_mode=geometry_mode
    )
    return diag, manifest, audit
