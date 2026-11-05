# Versioned index schema (v2)

The index is a directory bundle, not a single file.

```
<index_dir>/
  manifest.json
  masters/
    <stable-master-id>/
      <generation-id>/
        motion.npy
        shortlist_views.npy
        views.npy
        metadata.json
  lock                      # single-writer lock (transient)
```

## Manifest (manifest.json)

- `schema_version`: integer, current `2`.
- `algorithm_version`: nonempty string identifying the matcher version.
- `extractor_config`: normalized extraction and query configuration. Schema v2
  requires `sample_rate_fps`, `max_frames`, `speed_min`, `speed_max`,
  `speed_steps`, `score_floor`, `margin`, `offset_tolerance_s`, `views`, and
  `shortlist_views`. Numeric values must be finite JSON numbers (not booleans),
  integer fields must be integers (not booleans), ranges must be valid, and the
  view declarations must exactly match the implementation.
- `ffmpeg_version`, `numpy_version`: nonempty string tool/library versions used
  to build. Booleans, objects, and empty strings are invalid for all three
  version fields.
- `masters`: list of objects with:
  - `id`: stable master identifier (basename without extension).
  - `source_relpath`: safe nested path relative to the library root; unique basenames.
  - `display_name`, `original_root_hint`, `source_path`: source metadata.
    `display_name` equals the final component of `source_relpath`.
    `original_root_hint` is the absolute library location at indexing time.
    Matching against the stored visual descriptors can work after moving the
    library, but evidence generation and opt-in audio read masters at this
    original path. The batch command enables evidence by default. Thus a moved
    library is **not** fully portable without rebuilding the index.
  - `size_bytes` and `mtime_ns`: nonnegative integers (booleans are invalid).
  - `content_digest`: lowercase 64-character SHA-256 digest of the source bytes.
  - `shard_dir`: relative path to the per-master directory.
  - `arrays`: list of `{name, file, shape, dtype, checksum}` (SHA-256 of file bytes).
- `status`: `"complete"` or `"incomplete"`.

Readers open `manifest.json` descriptor-relative to a no-follow-opened index
directory and require it to be a regular file. A symlinked manifest (including
one targeting an external path) is invalid.

Readers also traverse each shard descriptor-relative beneath that pinned index
directory and open every array with `O_NOFOLLOW`. Each array must be a regular
file. Eager checksum validation and NumPy loading use the same pinned array inode;
the `views.npy` descriptor remains pinned until its deferred loader is released,
so a pathname replacement cannot redirect an existing reader outside the index.

To relocate a library, build a new index from the new library path, then start a
new match run against it. Do not edit `manifest.json` by hand: its contents are
bound to validated source identity, checksums, and batch-resume receipts. An old
report or journal is not automatically relocated by rebuilding.

## Provenance identity policy

Source content has exactly one provenance identity within an index. Two distinct
`source_relpath` values must not have the same `content_digest`, even when their
basenames differ. Such duplicate-content aliases would put indistinguishable
candidates in the matcher and force a deterministic zero-margin ambiguity, so
index construction rejects them with both relative paths and the shared SHA-256
digest in the diagnostic. Rebuilding the same source path is not an alias and may
reuse its unchanged generation.

## Per-master files

- `motion.npy`: motion trace, float32, 1-D, at most 1000 entries. Its length is
  one less than the feature-array frame count.
- `shortlist_views.npy`: float32, shape `(3, num_frames, 576)`, with 1–1000 frames.
- `views.npy`: reframing views, float32, shape
  `(19, num_frames, 576)`, with 1–1000 frames (memory-mappable; no pickle/object arrays).
  The complete `views` declaration (required fields, safe path, checksum format,
  dtype, and exact dimensions) is validated when the index opens without
  hashing or loading `views.npy`. Its checksum, NumPy header/payload shape and
  finiteness validation remain deferred until `load_views()` is called. All
  array metadata entries must be objects before field lookup. Deferred checksum
  and NumPy loading failures are reported as `clipprov.index.IndexError`, never
  as raw `OSError` or `ValueError` exceptions.
- `metadata.json`: master-specific metadata (frame count, duration, etc.).

Each `.npy` stores one homogeneous array. If shortlist representations become
materially different later, give them their own `.npy` files in the same
directory. Do not use object arrays or a monolithic archive.

## Write protocol

- Source libraries must contain only real directories and regular source files:
  the library root, source path, and every source-path component beneath the root
  must not be symlinks. Source paths are resolved and required to remain beneath
  the resolved library root before extraction. Descriptor-relative, no-follow
  opens are used while hashing to reduce path-component race exposure.
- Source identity is pinned before extraction by descriptor-relative,
  `O_NOFOLLOW` traversal. The source and traversed directory descriptors remain
  open while its generation is built. Identity is snapshotted from stable stat
  fields (including device, inode, mode, size, mtime, and ctime) and a SHA-256
  digest read from the pinned source descriptor.
  All source digests are checked for the provenance-identity uniqueness rule
  before extraction or manifest publication.
- A source needing extraction is copied from that same pinned descriptor into a
  private, immutable regular staging file inside index storage. `ffmpeg` and
  `ffprobe` receive only this staged path and never reopen the mutable library
  path. The staged file and staging directory are removed on success and on all
  failure paths. Unchanged reusable generations do not require staging.
- After extraction, the pinned source and directory chain are re-statted, and
  the library path is reopened without following symlinks, re-statted, and
  re-hashed. Any source or traversal-directory identity change (including a
  transient parent swap) aborts with `IndexError`, removes that build's new
  generation, and leaves the active manifest unpublished and unchanged.
- Write each completed master into a unique versioned generation directory.
  Generation directories are immutable: construction never deletes, overwrites,
  or reuses a conflicting path. A currently referenced generation is reused only
  after the active manifest passes the same strict reader validation and its
  source identity, artifact identity, and checksums match.
- Query/decision-only settings (`max_frames`, speed bounds/steps,
  `offset_tolerance_s`, `score_floor`, and `margin`) are retained in the manifest
  for query semantics but excluded from generation artifact identity.
- Fsync completed generation files and directories and their parent hierarchy
  before publishing the manifest.
- Switch the manifest pointer last (atomic rename of manifest).
- Fsync the manifest directory after replacement. If that fsync fails,
  `CommitVisibilityUncertain` means the replacement may be visible but not
  durable; reload the index before deciding whether to retry.
- Never replace a non-empty directory in place (os.replace fails on non-empty dirs on Linux).
- Single-writer lock prevents concurrent writers.
- Before manifest replacement, a failed build removes only generation directories
  created by that build attempt. It preserves the active manifest, every generation
  referenced by it, and all pre-existing unreferenced generations for old readers.
  Once manifest replacement succeeds, generations from that attempt are retained;
  this includes `CommitVisibilityUncertain`, because publication may be visible.
  Broader reclamation remains a separate garbage-collection step.
- Readers reject incomplete, corrupt, unsupported, or stale indexes with
  actionable diagnostics.

## Invalidation

Cache invalidation accounts for source bytes, algorithm version, configuration,
and decoder/extractor changes.
