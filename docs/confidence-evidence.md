# Confidence and visual evidence contract

Product result schema v2 extends the v1 match fields with `confidence` and `evidence`.
The evaluator continues to accept strict v1 rows for existing frozen artifacts. New
match results use v2.

## Diagnostic confidence

Confidence is **diagnostic, not probabilistic**. A tier does not mean “X% likely.”
The fields expose the measurements behind a decision:

- `source` keeps candidate identity evidence separate: best and competing scores,
  competing master, cross-master margin, and the configured score/margin floors.
- `timing` reports supporting sample count/fraction, temporal residual, offset-profile
  width, offset ambiguity, selected crop/reframe, estimated speed ratio, and aligned
  clip/master timestamps.
- `decision` is one of `matched`, `uncertain`, `ambiguous`, or `unmatched`.
  `ambiguous` means the cross-master margin missed its floor. `uncertain` means a
  source candidate cleared source checks but its timing support remained broad.

The matcher emits `uncalibrated` for both tier fields unless a frozen calibration
receipt has been applied. It never substitutes hard-coded matcher cutoffs for a
measured tier. The evaluation harness derives the published `strong` and `review`
tiers from correct frozen **development** controls only, records the calibration
receipt and digest, then applies those exact thresholds to **holdout** controls.
The report preserves every calibrated development and holdout result, decision counts,
and per-tier source, timing, and joint accuracy. Empty published tiers remain explicit
with a zero count and null accuracy; a release gate may therefore require
`all_published_tiers_observed: true`. These are observed held-out frequencies, not
probabilities. Source-withheld controls do not tune the tiers.

## Evidence frames

Request visual evidence for one match with:

```bash
clip-prov match clip.mp4 --index INDEX --evidence-dir evidence --out result.json
```

The command matches and extracts evidence from verified private snapshots. Clip and
master frame extraction cannot switch to replacement bytes if an original pathname
changes during processing. Records retain the original absolute paths and content
digests, never temporary paths. Snapshots are removed on completion or interruption;
they require temporary disk space, whose location follows `TMPDIR`.

The command writes first, middle, and last representative aligned JPEG pairs into a
new immutable generation, then atomically promotes a `current.json` pointer. A failed
replacement before pointer promotion leaves the previous pointer intact; a failure
after pointer promotion never deletes the generation named by that pointer, even if
the final directory durability sync reports an error. Before the result is emitted,
validation checks:

- the result carries the original absolute clip identity whose verified bytes generated
  the evidence; basename-only identity cannot distinguish same-named inputs;
- clip and master files still exist and match their recorded digests;
- the master identity and digest agree with the index source record;
- every timestamp is present in the numerical alignment and lies inside both media
  timelines;
- every aligned pair satisfies the frame-grid rule (below);
- every evidence frame exists, matches its recorded digest, and decodes as an image.

## Frame-grid alignment rule

The search samples masters on an integer frame grid at `clipprov.motion.FPS`
(8 Hz). With raw alignment offset `O` seconds (before the non-negative clamp to
`source_offset_sec`; the raw value may be negative) and speed `s`, a published
pair for clip frame index `k` must satisfy exactly

```text
clip_sec   = k / FPS
master_sec = O + round(s * k) / FPS
```

where `round` is ties-to-even, matching `np.rint` in the search. `clip_sec`
must lie on the sample grid and both pair timestamps must be non-negative;
the relation is checked within a 1e-6 second numerical epsilon, never with
slack like the 1.5 s scoring tolerance. `clipprov.eval.alignment_timestamp_problems`
implements this single rule and is shared by schema-v2 evaluation and the
evidence validator, so the two cannot drift apart.

Schema-v2 evaluation repeats these cross-field and artifact checks. Confidence scores,
margin, speed, decision, reason, acceptance, sample fraction, calibration receipt, and
tiers must agree with the base match result; an arbitrary evidence object is rejected.

Evidence generation requires a candidate with aligned timestamps. A no-candidate
result remains auditable numerically but cannot truthfully manufacture pictures of
an alignment that does not exist. Refresh evidence by rerunning the match if either
source file changes.
