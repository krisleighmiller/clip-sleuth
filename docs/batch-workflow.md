# Batch workflow, resume, reports, and performance

## Commands

```bash
clip-prov index MASTERS --out INDEX --resume --continue-on-error
clip-prov match-dir CLIPS --index INDEX --out RUN --resume --continue-on-error
```

Progress events are JSON objects on stderr. Final command summaries are written to
stdout, so progress never corrupts machine-readable output.

Supported batch clip extensions are `.avi`, `.m4v`, `.mkv`, `.mov`, `.mp4`, and
`.webm`, case-insensitively. Discovery is recursive, does not follow symlinked files
or directories, and orders regular-file inputs by normalized relative POSIX path.
Non-regular entries, including FIFOs, are not inventoried as clips. Explicit media
and saved JSON/metadata inputs must be regular files; readers reject pipes and other
special entries without waiting for a writer.

## Index recovery

The resumable indexer isolates each master build. After a shard is checksummed and
synced, `.index-progress.json` atomically records it. The journal is bound to the
canonical library path, sorted source inventory and SHA-256 digests, complete config,
algorithm, NumPy, ffmpeg version, failure policy, and indexing implementation digest.
Resume refuses mismatched, malformed, extra, duplicate, or stale-code state and
revalidates every entry against its inventory identity plus every array path, shape,
dtype, and checksum before reuse. The implementation digest covers configuration,
decoding, motion, appearance, indexing, I/O, and resumable-index code. Each source is
rechecked against its inventoried size, mtime, and SHA-256 after extraction and again
before manifest publication; the manifest always records the inventoried bytes rather
than rereading a potentially replaced path.

`--continue-on-error` records an isolated source path, exception type, and message,
then continues. If at least one master succeeds, the published manifest status is
`complete_with_errors`; `load_index` validates and accepts that explicit state.
`--fail-fast` preserves the previous manifest and the progress journal. A candidate
manifest is fully loaded and all deferred arrays are checked before one atomic rename
publishes it, so malformed resume state cannot overwrite the last usable index.
Masters that decode to fewer than the four-frame matching window are explicit input
errors, consistent with the V1 contract.

## Batch recovery

`.batch-progress.json` is bound to the canonical clips root, each input digest, the
digest of the index manifest bytes actually loaded (the parsed `_ManifestSnapshot`
carries `manifest_sha256` as an attribute; batch identity never hashes whatever
`manifest.json` sits at the path now, so resume rejects under a later manifest
generation while the loaded snapshot stays correct), calibration file/receipt,
evidence options, and the matching implementation digest, which covers every module
affecting decoding, appearance, motion, decisions, matching, schemas, calibration,
evidence, indexing, and batch rendering. Before any clip is staged or matched, its
descriptor is pinned and bound to the inventoried digest and size; a clip already
replaced on disk fails containment before staging. Matching and evidence both consume
a private read-only snapshot staged from the pinned descriptor inside the output
directory, so a replaced-then-restored original can never poison frames or scores with
foreign bytes; persisted results and evidence carry only the original absolute
inventory path. The original's digest and size are rechecked after decoding and
evidence but before checkpoint, and every inventory row is rechecked again before
report publication, so stale-inventory reports are impossible; only content is bound,
never mtime. Resume validates each checkpointed `source_record` against the exact
manifest master selected by `best_master`/`source_record.id`, never the clip relpath,
so a clip named differently from its source resumes while a mutated root, digest,
name, or source association is rejected. Every successful result is validated and
checkpointed after its clip. Resume rejects malformed records and any completion
outside the exact inventory; it skips only validated completed results. Failed clips
are retried, while successful clips are not recomputed. Input, index, option,
evidence, calibration, or implementation mismatch is a hard resume error rather than
a silent new run.

Per-clip decode, match, and evidence errors are contained and recorded. The batch
finishes every possible clip, writes all reports, then the CLI exits nonzero when
failures remain.

## Outputs

`report.json` is the canonical finite schema. Every presentation artifact is first
written into one immutable `generations/<id>` directory. Every artifact and nested
queue directory is synced, the generation is renamed and its parent synced, and only
then does one atomic `current.json` pointer publish that generation. Resolution of
`current.json` verifies the report digest, artifact-manifest digest, every listed
artifact's path, size, and SHA-256, and exact coverage of the generation. A failed or
subsequently damaged CSV, HTML, queue, summary, or checksum is therefore rejected
rather than mixed with or mistaken for a valid generation. The CLI prints the
resolved current report path. Presentation artifacts are reproduced from canonical
report data with deterministic ordering:

- `report.csv`
- `source-summary.json` and `source-summary.csv`
- `queues/decision-*.json`, `queues/source-tier-*.json`, and
  `queues/timing-tier-*.json`
- `report.html`, a self-contained offline searchable/sortable report
- `artifact-manifest.json`, with relative paths, sizes, and SHA-256 digests

CSV exports neutralize untrusted cells whose first meaningful character is `=`, `+`,
`-`, or `@` (and leading tab/newline variants) by prefixing an apostrophe. JSON keeps
the original text. This prevents spreadsheet formula execution without mutating the
canonical machine report.

Without `--calibration`, tier queues explicitly say `uncalibrated`. Supplying a Phase
4 calibration report applies its development-derived receipt; the batch does not
invent thresholds. The receipt loader accepts both the exact legacy six-field signed
receipt (whose tolerance is the implicit 1.5 seconds at every derive site) and the
newer receipt carrying `tolerance_seconds`, which must be finite, non-negative, and
non-boolean; malformed tolerances are rejected semantically before any digest
recomputation, and unlisted fields are rejected even with a correctly recomputed
SHA-256.

HTML evidence panels consume `result.evidence.frames` directly. They embed the
validated clip/master JPEG bytes as local data URIs with the reported timestamps; the
report layer never decodes video or reconstructs alignment. All filenames,
identifiers, and errors are HTML escaped. There are no CDNs, external scripts, remote
assets, or server requirement.

## Performance contract

The batch opens and validates one index and reuses it for every clip. Full views stay
lazy and are loaded only for shortlisted masters by the existing matcher. Decoded
frame arrays are scoped to one clip and are not retained across the run. Product
results and the journal grow linearly because durable resume requires per-clip state.

Each report records elapsed seconds, clips per second, peak RSS, and these deliberately
conservative smoke thresholds:

- peak RSS below 1 GiB;
- throughput at least 0.05 clips/second on the portable mini corpus;
- one loaded index reused for the entire batch.

These are regression alarms, not performance promises for arbitrary media or hardware.
The exhaustive shortlist audit remains separately bounded to 64 masters.

## Docker decision

No Docker image is added for Phase 5. The project already produces a wheel and sdist,
requires only system `ffmpeg`/`ffprobe`, records tool versions in durable receipts,
and passes installed-artifact workflows. A container would duplicate a working
boundary without fixing a measured reproducibility gap. Add one later only if actual
distribution evidence changes that conclusion. Docker theater remains theater.
