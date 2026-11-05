# Optional audio qualification

## Decision

`pcm_ncc_v1` is an explicit `--audio` opt-in qualified only as **positive corroboration of an already accepted visual match**. It is disabled by default and does not alter ranking, acceptance, source identity, timing, or confidence tiers. Missing, silent, short, replaced, disagreeing, unsupported-speed, unverifiable, or undecodable audio produces `not_corroborated` or `unavailable`; neither result can reject or downgrade a visual match. Opt-in diagnostics are emitted only in product schema v3, whose matching receipt binds the audio flag alongside speed, geometry, and immutable input digests.

The implementation decodes the first eight clip seconds and the visually aligned source interval as mono 16 kHz float PCM. It removes DC, computes energy-normalized FFT correlation over a bounded ±0.5 second lag, and reports corroboration at a frozen threshold of `0.45`. Both inputs are private immutable snapshots; the source snapshot must match the index content digest. The receipt records both snapshot SHA-256 values, the indexed source digest, source offset, sample count, lag, algorithm, sample rate, score, and threshold.

## Repaired qualification

The earlier Chromaprint comparison was discarded: exact fingerprint equality
was not a valid similarity measure for aligned excerpts. A small internal
qualification of the production PCM method included intact, removed, and
replaced audio. Its media hashes and per-case receipts are not part of this
public source distribution. The public tests exercise the positive-only
decision boundary without requiring the private media.

## Limits

The qualification was small, with excluded cases where media was unavailable
and little headroom for one positive case. It supports optional positive
corroboration, not audio-based rejection, standalone identification, calibrated
probability, or a claim of broad codec/content robustness. Future independent
media should be evaluated without changing the frozen threshold.
