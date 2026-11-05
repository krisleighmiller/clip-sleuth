"""Deterministic, positive-only audio corroboration diagnostics."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from clipprov.index import IndexError as IndexReadError
from clipprov.index import _staged_source_snapshot
from clipprov.media import MediaError, _resolve

AUDIO_SAMPLE_RATE = 16000
AUDIO_SPAN_SEC = 8.0
MIN_AUDIO_SEC = 2.0
AUDIO_THRESHOLD = 0.45
AUDIO_ALGORITHM = "pcm_ncc_v1"


def _base_receipt(expected_digest: str, source_offset_sec: float) -> dict:
    return {
        "status": "unavailable",
        "reason": None,
        "score": None,
        "threshold": AUDIO_THRESHOLD,
        "algorithm": AUDIO_ALGORITHM,
        "sample_rate_hz": AUDIO_SAMPLE_RATE,
        "compared_samples": 0,
        "lag_samples": 0,
        "clip_sha256": None,
        "source_sha256": None,
        "source_content_digest": expected_digest,
        "source_offset_sec": float(source_offset_sec),
    }


def _decode_pcm(path: Path, start_sec: float) -> np.ndarray | None:
    ffmpeg = _resolve("ffmpeg", None)
    command = [
        ffmpeg,
        "-v",
        "error",
        "-nostdin",
        "-ss",
        f"{start_sec:.9f}",
        "-i",
        str(path),
        "-t",
        f"{AUDIO_SPAN_SEC:.9f}",
        "-map",
        "0:a:0",
        "-ar",
        str(AUDIO_SAMPLE_RATE),
        "-ac",
        "1",
        "-c:a",
        "pcm_f32le",
        "-f",
        "f32le",
        "-",
    ]
    process = subprocess.run(command, capture_output=True, check=False, timeout=30)
    if process.returncode or not process.stdout or len(process.stdout) % 4:
        return None
    return np.frombuffer(process.stdout, dtype="<f4").astype(np.float64)


def _normalized_correlation(left: np.ndarray, right: np.ndarray) -> tuple[float, int] | None:
    count = min(left.size, right.size)
    if count < int(MIN_AUDIO_SEC * AUDIO_SAMPLE_RATE):
        return None
    left = left[:count]
    right = right[:count]
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        left = left - left.mean()
        right = right - right.mean()
        left_norm = float(np.linalg.norm(left))
        right_norm = float(np.linalg.norm(right))
    if not np.isfinite(left_norm) or not np.isfinite(right_norm):
        return float("nan"), 0
    if left_norm <= 1e-6 or right_norm <= 1e-6:
        return None
    fft_length = 1 << (left.size + right.size - 2).bit_length()
    circular = np.fft.irfft(
        np.fft.rfft(left, fft_length) * np.conj(np.fft.rfft(right, fft_length)),
        fft_length,
    )
    correlation = np.concatenate((circular[-(right.size - 1) :], circular[: left.size]))
    correlation /= left_norm * right_norm
    lags = np.arange(-(right.size - 1), left.size)
    selected = np.flatnonzero(np.abs(lags) <= AUDIO_SAMPLE_RATE // 2)
    bounded = np.abs(correlation[selected])
    if not np.isfinite(bounded).all():
        return float("nan"), 0
    peak = selected[np.argmax(bounded)]
    score = float(np.clip(abs(correlation[peak]), 0.0, 1.0))
    if not np.isfinite(score):
        return float("nan"), 0
    return score, int(lags[peak])


def corroborate_audio(
    clip_path: Path,
    source_path: Path,
    source_content_digest: str,
    source_offset_sec: float,
    speed: float,
) -> dict:
    """Return a provenance-bound receipt that can only corroborate a visual match."""
    receipt = _base_receipt(source_content_digest, source_offset_sec)
    if abs(float(speed) - 1.0) > 1e-9:
        receipt["reason"] = "unsupported_speed"
        return receipt
    if source_offset_sec < 0:
        receipt["reason"] = "unsupported_offset"
        return receipt
    if not Path(clip_path).is_file():
        receipt["reason"] = "clip_snapshot_unavailable"
        return receipt
    if not Path(source_path).is_file():
        receipt["reason"] = "source_snapshot_unavailable"
        return receipt

    try:
        with (
            _staged_source_snapshot(Path(clip_path)) as (clip_snapshot, clip_digest),
            _staged_source_snapshot(Path(source_path)) as (source_snapshot, source_digest),
        ):
            receipt["clip_sha256"] = clip_digest
            receipt["source_sha256"] = source_digest
            if source_digest != source_content_digest:
                receipt["reason"] = "source_digest_mismatch"
                return receipt
            clip_pcm = _decode_pcm(clip_snapshot, 0.0)
            source_pcm = _decode_pcm(source_snapshot, float(source_offset_sec))
    except (IndexReadError, MediaError, OSError, RuntimeError, ValueError, subprocess.SubprocessError):
        # Audio is deliberately optional: filesystem races, decoder failures,
        # and missing native tooling must remain diagnostics, never decisions.
        receipt["reason"] = "snapshot_or_decode_failed"
        return receipt

    if clip_pcm is None or source_pcm is None:
        receipt["reason"] = "audio_stream_unavailable"
        return receipt
    if not np.isfinite(clip_pcm).all() or not np.isfinite(source_pcm).all():
        receipt["reason"] = "nonfinite_audio"
        return receipt
    count = min(clip_pcm.size, source_pcm.size)
    receipt["compared_samples"] = int(count)
    if count < int(MIN_AUDIO_SEC * AUDIO_SAMPLE_RATE):
        receipt["reason"] = "insufficient_audio"
        return receipt
    correlated = _normalized_correlation(clip_pcm, source_pcm)
    if correlated is None:
        receipt["reason"] = "silent_audio"
        return receipt
    score, lag = correlated
    if not np.isfinite(score):
        receipt["reason"] = "nonfinite_audio"
        return receipt

    receipt["score"] = score
    receipt["lag_samples"] = lag
    if score >= AUDIO_THRESHOLD:
        receipt["status"] = "corroborated"
        receipt["reason"] = "score_at_or_above_threshold"
    else:
        receipt["status"] = "not_corroborated"
        receipt["reason"] = "score_below_threshold"
    return receipt
