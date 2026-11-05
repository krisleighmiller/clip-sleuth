# Supported V1 input contract

Clip Provenance V1 matches transformed video clips against a library of source
("master") videos using visual evidence only.

## Supported inputs

- Container: MP4 (H.264 video) with an optional audio track.
- Master library: a directory of `*.mp4` files with unique basenames.
- Clips: a directory of `*.mp4` files.
- Video alone is matchable. Audio may be absent, replaced, or unrelated.
- Transformations within the tested hypothesis set: crop/reframe (crop
  position and scale), resize, compression, watermark, color/brightness
  changes, padding (letterbox/pillarbox), audio removal/replacement, and
  bounded speed changes (0.968x-1.032x).
- Duration: masters must decode to at least four frames, the minimum alignment
  window. Clips that decode to fewer than four usable frames are rejected as
  `unmatched` with diagnostics rather than indexed or accepted silently.

## Resource envelope

There is no qualified maximum duration or file-size guarantee. Indexing and
clip probing decode the **whole** video at 8 sampled frames per second into
memory before selecting the smaller matching grid; indexing also constructs
multiple in-memory reframe views. Long or high-volume media can exhaust memory,
and the decoder subprocess has no timeout. The batch 1 GiB smoke threshold is
for its small test corpus, not an arbitrary-input limit. Start with short clips
and modest masters on available hardware, monitor memory, and split large
libraries into separate indexing runs if necessary (this does not reduce the
memory required for one long master). Do not feed untrusted or arbitrarily long
videos expecting a bounded-resource service.

## Declared V1 boundaries (not supported, not hidden failures)

- cuts, compilations, or composited sources;
- perspective-transformed footage;
- mirrored or quarter-turn-rotated footage unless matching explicitly enables the
  experimental `d4` geometry mode; the default identity mode preserves V1 behavior;
- variable-speed or nonlinear time warps;
- duplicate or near-duplicate masters without reporting ambiguity;
- repeated or static scenes with a uniquely determined timestamp;
- reframing outside the tested hypothesis set.

Any expansion requires a measured spike against held-out data.

## Tooling requirements

- `ffmpeg` and `ffprobe` must be available on PATH.
- Python >= 3.11, NumPy, pydantic, click.
