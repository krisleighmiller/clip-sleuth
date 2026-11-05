"""Matching core: candidate ranking, foreign rejection, diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from clipprov.appearance import bars_crop, clip_grid, search_geometry_diagnostics
from clipprov.media import probe as media_probe
from clipprov.motion import (
    ACCEPT_BEST,
    ACCEPT_MARGIN,
    SHORTLIST_APPEARANCE,
    SHORTLIST_MOTION,
    SHORTLIST_VIEWS,
    SPEEDS,
    motion,
    motion_score,
    speeds_from_config,
)


@dataclass
class MatchDiagnostics:
    best_score: float
    second_score: float
    margin: float
    best_master: str | None
    raw_alignment_offset_sec: float | None
    source_offset_sec: float | None
    speed: float | None
    accepted: bool
    accepted_master: str | None
    accepted_offset_sec: float | None
    accepted_speed: float | None
    reason: str
    ranked: list = field(default_factory=list)
    shortlist: list = field(default_factory=list)
    acceptance_floor: float | None = None
    acceptance_margin: float | None = None
    supporting_samples: int = 0
    sample_count: int = 0
    support_fraction: float | None = None
    temporal_residual_sec: float | None = None
    offset_profile_width_sec: float | None = None
    offset_ambiguity: float | None = None
    reframe_index: int | None = None
    reframe: tuple[float, float, float] | None = None
    aligned_timestamps: tuple[tuple[float, float], ...] = ()
    decision: str = "unmatched"
    source_confidence: str = "uncalibrated"
    timing_confidence: str = "uncalibrated"
    geometry_mode: str = "identity"
    geometry_hypothesis: str = "identity"
    audio: dict | None = None

    @property
    def offset_sec(self):
        return self.source_offset_sec


def _resolve_configs(config=None, media_config=None, match_config=None):
    if config is not None:
        if media_config is None:
            media_config = getattr(config, "media", None)
        if match_config is None:
            match_config = getattr(config, "match", None)
    return media_config, match_config


def rank_candidates(cg, idx, cm, mm, mviews, speeds=None, config=None, match_config=None, media_config=None):
    _, mc = _resolve_configs(config=config, match_config=match_config)
    if speeds is None:
        if mc is not None:
            speeds = speeds_from_config(mc)
        else:
            speeds = SPEEDS
    ms_speeds = speeds

    geometry_mode = mc.geometry_mode if mc is not None else "identity"

    def app_search(cg, idx, mv):
        detail, _hypothesis = search_geometry_diagnostics(
            cg, idx, mv, speeds, mode=geometry_mode
        )
        return detail.score, detail.offset_sec, detail.speed

    plain = {n: app_search(cg, idx, mviews[n][:SHORTLIST_VIEWS])[0] for n in mviews}
    appearance_cands = sorted(plain, key=lambda n: (-plain[n], n))[:SHORTLIST_APPEARANCE]
    motion_cands = sorted(mm, key=lambda n: (-motion_score(cm, mm[n], speeds=ms_speeds), n))[:SHORTLIST_MOTION]

    cand = set(appearance_cands) | set(motion_cands)

    scored = []
    for name in sorted(cand):
        sc, off, sp = app_search(cg, idx, mviews[name])
        scored.append((sc, name, off, sp))
    scored.sort(key=lambda row: (-row[0], row[1]))
    return scored


def decide(scored, *, accept_best=ACCEPT_BEST, accept_margin=ACCEPT_MARGIN):
    if not scored:
        return False, None, None, None, -1.0, 0.0
    sc0, name0, off0, sp0 = scored[0]
    second = scored[1][0] if len(scored) > 1 else 0.0
    margin = sc0 - second
    accepted = (sc0 >= accept_best) and (margin >= accept_margin)
    raw_off = float(off0) if off0 is not None else None
    return accepted, name0, raw_off, (float(sp0) if sp0 is not None else None), float(sc0), float(margin)


def probe_clip(path, config=None, media_config=None, match_config=None, speeds=None, max_frames=None):
    media_cfg, _match_cfg = _resolve_configs(config=config, media_config=media_config, match_config=match_config)
    target = Path(path)
    if media_cfg is not None:
        media_probe(target, media_cfg)
    else:
        media_probe(target)
    filt = bars_crop(
        target,
        ffmpeg=(media_cfg.ffmpeg if media_cfg else "ffmpeg"),
        ffprobe=(media_cfg.ffprobe if media_cfg else "ffprobe"),
    )
    from clipprov.motion import decode

    frames = decode(target, filt, ffmpeg=(media_cfg.ffmpeg if media_cfg else "ffmpeg"))
    if max_frames is None:
        max_frames = (media_cfg.max_frames if media_cfg else 20)
    cg, idx = clip_grid(frames, max_frames=max_frames)
    cm = motion(frames)
    return cg, idx, cm, filt


def confidence_labels(source_accepted, best, margin, score_floor, margin_floor, detail):
    """Return a final decision plus explicitly uncalibrated tier placeholders."""
    if detail is None or detail.offset_sec is None or best < score_floor:
        decision = "unmatched"
    elif margin < margin_floor:
        decision = "ambiguous"
    else:
        timing_ok = (
            detail.support_fraction is not None
            and detail.support_fraction >= 0.60
            and detail.offset_profile_width_sec is not None
            and detail.offset_profile_width_sec <= 1.0
        )
        decision = "matched" if source_accepted and timing_ok else "uncertain"
    return decision, "uncalibrated", "uncalibrated"


def match_clip(
    cg,
    idx,
    cm,
    mm,
    mviews,
    config=None,
    media_config=None,
    match_config=None,
    speeds=None,
    score_floor=None,
    margin=None,
):
    _, mc = _resolve_configs(config=config, match_config=match_config)
    if speeds is None and mc is not None:
        speeds = speeds_from_config(mc)
    if score_floor is None:
        score_floor = (mc.score_floor if mc is not None else ACCEPT_BEST)
    if margin is None:
        margin = (mc.margin if mc is not None else ACCEPT_MARGIN)
    scored = rank_candidates(cg, idx, cm, mm, mviews, speeds=speeds, match_config=mc)
    source_accepted, name, raw_off, sp, best, marg = decide(
        scored, accept_best=score_floor, accept_margin=margin
    )
    second = scored[1][0] if len(scored) > 1 else 0.0
    source_off = (max(0.0, raw_off) if raw_off is not None else None)
    if name in mviews:
        detail, geometry_hypothesis = search_geometry_diagnostics(
            cg, idx, mviews[name], speeds, mode=(mc.geometry_mode if mc else "identity")
        )
    else:
        detail, geometry_hypothesis = None, "identity"
    decision, source_confidence, timing_confidence = confidence_labels(
        source_accepted, best, marg, score_floor, margin, detail
    )
    accepted = decision == "matched"
    reason = decision
    return MatchDiagnostics(
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
        geometry_mode=mc.geometry_mode if mc else "identity",
        geometry_hypothesis=geometry_hypothesis,
    )
