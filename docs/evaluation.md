# Phase 3 evaluation contract

The scorer adapts the comparison project's `verification/score_trials.py`.
Control generation adapts Kimi's `trials/kimi/pipeline/synth_gen.py`, using
`clipprov.media.probe` instead of fixed source dimensions, frame rate or duration.
The disjoint controls, frame-aligned ground truth, and source-withheld audits
adapt Astra's `trials/astra/pipeline/validate.py`. Neither donor matcher nor its
OpenCV/SciPy dependencies are used. Production retains Phase 2 shortlisting,
alignment search, and source thresholds. Phase 4 also gates final acceptance on
timing-support and offset-width diagnostics. Its separate descriptive tier calibration
uses frozen development data and is measured on held-out rows.

## Frozen controls

`clip-prov generate LIBRARY --out CONTROLS` requires at least three visually
distinct MP4 source families, unique basenames, and four frames per source. It
recursively sorts relative source paths, clusters near visual variants, and assigns
each family to development, holdout, then source_withheld, repeating in that order.
Every member of one family stays in one group. Byte-identical sources are rejected.
The manifest freezes every input hash,
relative path, probed metadata, recipe, frame-aligned source offset/span/end,
output hash, output metadata, explicit ground truth, and decode/boundary audits.
For reproducibility, keep the manifest and original source bytes together.

Version 3 freezes the same diagnostic matrix for development and holdout:
no-transform baseline, reencode-only, crop-only, speed-only, watermark-only,
audio-preserving, grayscale-only, resize-only, brightness-only, saturation-only,
padding-only, audio-removal-only, audio-replacement-only, crop+speed interaction,
and compound stress.
Explicit numeric transform values are disjoint between development and holdout.
Holdout values must not be tuned using holdout scores; a recipe change requires
a new version and frozen run.

The manifest also freezes negative controls by category: source-withheld,
unrelated visual, synthetic noise, static color, repeated pattern, and an
independently encoded procedural test-pattern variant. The last category is not
called a near-duplicate: without a domain generator there is no honest way to
guarantee that a visually similar artifact is independent of an arbitrary
indexed source. Every synthetic negative uses procedural content distinct from the indexed source
generators, has distinct bytes and a distinct path, and must exceed the frozen
visual-distance threshold from every indexed positive and every other procedural
negative. Its nearest-positive distance and signature are frozen. Source-withheld
controls remain a separate negative class. Each negative's declared relation
(`independent_synthetic` or `procedural_variant_of`) is
frozen and audited. Reports score each category separately rather than pooling
unlike failure modes.

Generation uses temporary files and directories, validates actual outputs, and
publishes `manifest.json` last by atomic replacement. Content-addressed generation
paths and single-threaded encoding are intended to make repeated runs identical
with the same inputs and ffmpeg build, but this is verified by tests only for the
active environment, not promised across systems. Before republishing an existing
generation, the generator rehashes every case and every frozen negative. A
missing or changed artifact fails generation and leaves the active manifest
unchanged. Bit identity across different encoder builds is not promised. Unlisted
files and older generations are never evaluated. Failed runs preserve the active
manifest. Keep controls outside the source library.

`generation_receipt` records package, Python, NumPy and ffmpeg versions, recipe
version, Git commit plus dirty state (or an explicit unavailable value), and
`generator_implementation`. That generation identity contains an ordered list
and SHA-256 for only the installed modules that can affect control generation,
plus a SHA-256 over the canonical list. Exact duplicate source bytes are rejected,
matching the index contract; rename aliases are not silently assigned to a split.

The frozen source-family audit decodes RGB frames at 2 fps, area-resizes to 32×32,
samples eight evenly spaced normalized-timeline positions, and computes mean
absolute RGB distance on the [0, 1] scale (`rgb32_normalized_timeline_mae_v1`).
Pairs at or below 0.06 are joined transitively into a family. The manifest records
the method, threshold, comparison count, matched pairs, signatures, and family
membership. Evaluation recomputes all of it and fails if a family spans groups.
This is a deterministic near-duplicate screen, not semantic scene recognition: it
can miss brief shared segments, heavy crops, temporal reordering, or variants above
the threshold. Dataset curation is still required for those cases.

Evaluation reports preserve that manifest receipt under `run_receipt.generation`
and record a separate `run_receipt.evaluation`. The evaluation receipt hashes the
ordered installed implementation closure for report production: CLI, harness,
evaluation adapter/scorer, index loader, matcher, appearance/motion/core logic,
media, transforms, configuration, I/O, and package version modules. Its
`evaluation_sha256` is the SHA-256 over the canonical per-file identity list.
Both receipts work from a wheel: Git commit is `unavailable` and dirty state is
null when repository metadata is absent. Evaluation validates the generation
receipt schema and aggregate. A dirty-tree receipt is reproducible input
provenance, not release calibration evidence. Final benchmark evidence remains
pending until a clean committed checkout is run with the real source corpus.

## Index and audits

Build the evaluation index from every development/holdout source and **exclude
all source_withheld sources**, preserving nested relative paths. For an existing
library, read `manifest['sources']`, copy entries whose `group` is not
`source_withheld` into an index library using `source_relpath`, then run
`clip-prov index`. The installed `clip-prov demo-corpus OUT` command does this for
its known three-source demo. It is a smoke test, not an accuracy dataset.

`clip-prov evaluate CONTROLS/manifest.json --index INDEX --out REPORT.json`
recomputes source hashes and frozen membership, re-probes metadata, fully decodes
sources and listed controls, verifies output hashes, frame alignment and frozen
source-end boundaries, and checks indexed relative paths and content hashes.
The index candidate identities must equal the manifest development+holdout
identities exactly: missing and extra masters are both errors. The report freezes
the sorted identities and their SHA-256 digest. Withheld bytes, aliases with the
same digest, or withheld IDs in the index cause an error. Exact and detected near
visual variants cannot leak across groups; variants outside the documented audit's
scope still require dataset curation. Tests exercise ambiguous duplicate and
near-duplicate masters, repeated scenes, static footage, short/corrupt media,
malformed schemas, nonfinite values, and source-end cases without asserting
unique timestamps for ambiguous media.

Use `--library MOVED_LIBRARY` after relocation. The index's absolute original-root
hint is informational; membership uses nested relative paths and hashes.
Malformed data, missing artifacts, decode failures, membership violations, and
write failures exit 2 with stderr diagnostics. Evaluation writes a report only
after all audits and scoring succeed. Matching retains exit 0 for accepted,
1 for rejected, and 2 for errors.

## Result schema and metrics

`eval` accepts a schema v1 product object or array. Every field emitted by
`clip-prov match` is required; extra fields and conflicting accepted/candidate
fields are errors. Scores, margins, speeds and offsets must be finite JSON
numbers, never booleans. A source record must agree with the candidate ID and
basename. Accepted master/offset/speed must equal the candidate fields; rejected
accepted fields must be null. Raw alignment may be negative; the source offset
must equal `max(0, raw_alignment_offset_sec)`. Truth contains `clip`, `master`,
`offset_sec`; foreign truth sets the last two to null. POSIX and Windows paths
are compared by basename, including arbitrary source IDs. Duplicate, missing,
and unknown clip results are errors.

Three reporting tiers preserve the existing matcher diagnostics:

- `accepted`: matcher-accepted source and timestamp (not calibrated confidence);
- `uncertain`: rejected diagnostic candidate, including a candidate without timing;
- `unidentified`: no candidate.

Accepted source/joint precision divides correct accepted genuine proposals by
all accepted proposals. Genuine source/joint recall divides those counts by all
genuine clips. Joint correctness requires the right source and absolute time
error at most the tolerance (default 1.5 seconds, inclusive).
`proposed_source_accuracy` and `proposed_joint_accuracy` include both accepted
and uncertain proposals, dividing correct genuine proposals by all proposals,
including foreign proposals. An uncertain candidate without timing contributes
to source accuracy but cannot be jointly correct.

Explicit foreign rejection means any foreign clip with `accepted: false`.
Rejected foreign candidates remain in the uncertainty queue and are counted
separately from unidentified rejections. Reports
also include genuine unidentified counts, uncertainty-queue genuine/foreign
composition, tier counts and timing-error summaries. Empty denominators are
JSON null. These are diagnostics, not calibrated confidence probabilities.

## Shortlist checks and combined reports

`--shortlist-samples N` (default 8, range 0–64) deterministically balances samples
across development, holdout, source-withheld, foreign procedural negatives, source
IDs, recipe families, and positive/negative classes where applicable. The report
records the selected row metadata and composition counts. Each sampled clip scores
**all full views of every indexed master**, capped at 64 masters, with the same
appearance search, speed grid and
tie-breaking as production. Larger indexes require `N=0` or a separate bounded
validation index. Reports include the exact samples and shortlist IDs,
`exhaustive_winner_recall` among samples with a valid exhaustive alignment, and
`genuine_truth_recall` among sampled genuine controls. No valid alignments means
null winner recall; no samples means null recalls. Exhaustive scores do not
change product decisions. A regression with 18 candidates proves that an
omitted exhaustive winner produces a recall miss.

Supply both `--benchmark-results PRODUCT_RESULTS --benchmark-truth ANSWER_KEY`
to include an independently scored `benchmark` object alongside `groups` for
development, holdout and source_withheld. The combined report includes raw
product results, shortlist audits, and SHA-256 identities for the manifest,
index manifest and benchmark artifacts. Without benchmark artifacts,
`benchmark` is null. The Phase 3 gate is the ability to emit these measurements
from supplied artifacts; no dataset or accuracy numbers are fabricated.

## Python API

`load_product_results(path)` and `load_truth(path)` keep their list/dict return
contracts. Passing `with_digest=True` returns `(data, sha256)` from one exact-byte
JSON snapshot. Saved JSON inputs must be regular files, not streaming pipes.
The built-in `eval` and `evaluate` commands use those snapshots and
the loaded index's manifest identity for receipts, not later reads of mutable paths.

```python
from pathlib import Path
from clipprov.harness import generate_controls, evaluate_controls
from clipprov.eval import load_product_results, load_truth, score_product

manifest = generate_controls(Path('sources'), Path('controls'))
# Build an index from only manifest sources in development/holdout, as above.
report = evaluate_controls(
    'controls/manifest.json', 'index', 'report.json',
    shortlist_samples=8,
    benchmark_results='product-results.json',
    benchmark_truth='ground-truth.json',
)
metrics = score_product(
    load_product_results(Path('product-results.json')),
    load_truth(Path('ground-truth.json')),
)
```

`clipprov.transforms.transform(src, dst, start=0, duration=None, recipe=None)`
returns probed `VideoInfo`; duration is a **source span before speed adjustment**.
It rounds trim positions to source frame boundaries, validates crop/resize/range
constraints, handles absent or replaced audio, fully decodes the temporary
output, and replaces the destination atomically. Invalid arguments raise
`ValueError`; ffmpeg/probe/decode failures raise `MediaError` with diagnostics.
Existing convenience wrappers raise on failure and return True on success.
All matching uses `clipprov.match.match_clip_path`, reusing a loaded index;
`audit_shortlist` adds bounded exhaustive measurements without console-script
subprocesses.
