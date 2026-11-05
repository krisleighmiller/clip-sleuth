# Clip Sleuth

Find where a clip came from in your own video library.

Find a likely source and timestamp for a continuous clip in a local video
library. Matching is visual-first; optional audio only corroborates an accepted
visual result. This is a local Python CLI and Tk desktop app, not a hosted
service or a claim of universal identification.

## Install from this source checkout

There is no published package or supported downloadable binary yet. On Linux
with Python 3.11 or newer, install `ffmpeg` and `ffprobe` on PATH and ensure
Python's Tkinter package is installed if you want the GUI (often a separate
`python3-tk` OS package). A graphical display is also required for the GUI.
From this checkout, in a fresh virtual environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/clip-prov --help
```

Python dependencies (Click, Pydantic, NumPy) are installed by pip; the two
FFmpeg executables and Tkinter are system prerequisites, not pip dependencies.
Linux is the declared platform; other operating systems are not qualified.

## First match

Place source MP4/H.264 videos with unique basenames in `masters/` and a
continuous MP4 clip in `clips/`. Use your own absolute paths in place of these
example folders; keep the original master library available at the indexed
path when using evidence or audio.

```bash
.venv/bin/clip-prov index masters --out index
.venv/bin/clip-prov match clips/example.mp4 --index index --out result.json
# For multiple clips and a searchable offline report:
.venv/bin/clip-prov match-dir clips --index index --out run
```

`index` prints a JSON summary; progress events go to stderr. `match` prints
the result JSON even when it cannot accept a match, including `reason`, best
candidate, offset, speed, and `confidence.decision` with separate source/timing
numerical diagnostics. Exit 0 means an accepted match, 1 means no accepted
match (or batch failures for `match-dir`), and 2 means a command error. A best
candidate in an uncertain, ambiguous, or unmatched result is **not** a
confirmed identification. `match-dir` defaults to paired frame evidence,
continues after individual errors, and prints the verified `report.json` path;
the same generation includes `report.html`, CSV and review queues. See
[batch output and recovery](docs/batch-workflow.md).

To use the desktop app after installation, run `.venv/bin/clip-prov-gui`,
select **Index a source library**, choose the master and index folders, and
run it. Then select **Match a clip folder**, choose the clip, index and report
folders, run it, and use **Open verified report**. This interface processes a
folder of clips; use `clip-prov match` for one clip. See the desktop section
below for cancellation and log behavior.

## Scope, confidence, and resources

Default V1 expects one continuous single-source clip or an outside-library
clip. Tested transforms include bounded crop/reframe, padding, resizing,
compression, watermark/color changes and small constant speed shifts
(0.968x–1.032x). It does not promise cuts, nonlinear speed, arbitrary
perspective, or unique timing in static/repeated scenes. Confidence tiers are
descriptive diagnostics, **not probabilities**; measured results are specific
to the evaluation datasets. Wider speed (0.5x–2x) and mirror/quarter-turn D4
are opt-in, separately bounded extensions; audio is positive-only and off by
default. See [input limits and resource caveats](docs/v1-input-contract.md),
[geometry](docs/geometry.md) and [audio](docs/audio-qualification.md).

The index is not fully relocatable: visual matching uses indexed features,
but paired evidence and audio need the original master paths. If the library
moves, reindex it at its new location and start a new match run; do not edit
the manifest or resume an old run against a changed index. See
[index format and relocation](docs/index-schema.md).

## Local data and sharing

The index and JSON/CSV reports contain source and clip paths (often absolute)
and SHA-256 content hashes. Evidence and HTML reports contain extracted
clip/master images. GUI job logs under the output folder's `.gui/logs` may
contain paths, progress and errors. Keep these folders private; inspect and
redact paths, hashes, frames and logs before sharing a report or archiving a
run. Delete an index, run output, evidence folder and GUI logs when no longer
needed. Removing a report does not remove the original media. The documented
matching workflow runs locally and its HTML report uses no external assets;
this is not a security audit or a blanket claim that every installation,
dependency or browser action is network-free.

## Trial provenance

The synthetic-control approach in `clipprov/transforms.py` was adapted from
`synth_gen.py` produced during Kris's clip-matching comparison. The Kimi trial
ran in OpenCode 2.0.29 with model `moonshotai/kimi-k3` via provider
`openrouter`, not the consumer Kimi chat application. This route is confirmed
by the original OpenCode session and assistant-message model records for the
trial; it is not an assertion that OpenRouter itself grants output rights.

The project is licensed under the [MIT License](LICENSE), copyright 2026
Kristen Leigh Miller. This is the public source repository; the package is not
published to a Python registry.

## Development

```bash
uv venv --python 3.11 .venv
uv pip install -e ".[dev]"
.venv/bin/pytest -q tests/
.venv/bin/ruff check clipprov tests
# The private benchmark task, answer key and oracle tests are not part of this public tree.
```

Requires `ffmpeg` and `ffprobe` on PATH.

## Desktop interface

```bash
clip-prov-gui
# From a source checkout:
python -m clipprov.gui
```

The desktop interface additionally requires Tkinter and a graphical display. It
runs the same CLI for indexing and directory matching, including resume,
continue-on-error, evidence and calibration settings. Cancel stops the owned job
and keeps completed checkpoints. Progress and diagnostics remain visible, with
full per-job logs under the selected output folder's `.gui/logs` directory.
`Open verified report` resolves the integrity-checked local HTML generation before
opening it. Headless use remains available through `clip-prov`.

## Evaluation and calibration

Generate frozen development, unseen holdout, source-withheld, and independent
synthetic controls, then match and score them through the library API:

```bash
clip-prov demo-corpus '/tmp/clip provenance mini'
clip-prov generate '/tmp/clip provenance mini/sources' --out '/tmp/clip provenance mini/controls'
clip-prov index '/tmp/clip provenance mini/indexed sources' --out '/tmp/clip provenance mini/index'
clip-prov evaluate '/tmp/clip provenance mini/controls/manifest.json' \
  --index '/tmp/clip provenance mini/index' --out '/tmp/clip provenance mini/report.json'
```

Add `--benchmark-results product-results.json --benchmark-truth ground-truth.json`
to emit benchmark and holdout metrics together. Score saved product results alone
with `clip-prov eval --results product-results.json --truth ground-truth.json`.
See [the evaluation contract and Python API](docs/evaluation.md) for source-family
near-duplicate leakage audits, visually separated procedural negatives, stratified
shortlist checks, single-factor diagnostics, distinct generation/evaluation
implementation receipts, metric definitions, and bounds. No source media or invented
benchmark measurements are bundled.

## Confidence and visual evidence

`clip-prov match` emits separate source and timing diagnostics. Add
`--evidence-dir PATH` to produce validated aligned clip/master frame pairs whose
timestamps match the machine-readable result. These tiers are descriptive, not
probabilities. See [the confidence and evidence contract](docs/confidence-evidence.md)
for field semantics, development-only tier calibration, held-out measurements, and
integrity checks.

Mirror and quarter-turn descriptor hypotheses are experimental and explicit:
`clip-prov match clip.mp4 --index index --geometry d4`. Identity-only matching remains
the default. See [the geometry evidence and boundary](docs/geometry.md).

Audio corroboration is also explicit: add `--audio` to `match` or `match-dir`.
It can only annotate an accepted visual match; it never changes ranking, acceptance,
source identity, timing, or confidence tiers. The default remains visual-only.

## Resumable directory workflow

```bash
clip-prov index masters --out index --resume --continue-on-error
clip-prov match-dir clips --index index --out run --resume --continue-on-error
# Opt-in mirror and quarter-turn hypotheses:
clip-prov match-dir clips --index index --out run --geometry d4
```

The batch workflow checkpoints verified work, contains isolated failures, and emits
canonical JSON plus CSV, review queues, source summaries, artifact checksums, and a
self-contained searchable HTML report backed by Phase 4 evidence frames. See
[the batch workflow contract](docs/batch-workflow.md) for resume invalidation,
failure semantics, performance ceilings, and the evidence-based decision not to add
a Docker image.

### Opt-in wider-speed profile

The default V1 search remains unchanged. For clips expected to be retimed within the
measured **0.5x–2.0x** range, opt in explicitly:

```bash
clip-prov match clip.mp4 --index index --speed-profile wider-speed-v1
clip-prov match-dir clips --index index --out run --speed-profile wider-speed-v1
```

`wider-speed-v1` uses 401 speed samples from 0.5 through 2.0. This is the measured
production envelope, not a claim of support outside that range. Default results retain
the exact Phase 5 schema-v2 confidence and timing shape. Selecting wider speed, D4
geometry, or audio emits schema v3 with a receipt binding all three policy choices and
the clip/index digests. Batch identity additionally binds the complete effective media
and match configurations, so any acceptance-affecting change invalidates resume state.
