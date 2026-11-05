# Experimental mirror and quarter-turn matching

Geometry support is opt-in. `--geometry d4` evaluates the identity descriptor plus
the seven other square symmetries: three quarter-turn rotations, a horizontal mirror,
and the mirror composed with each rotation. The default is `identity`, which retains
the established V1 behavior and cost.

The matcher chooses the best geometry hypothesis *within each source* before applying
the acceptance margin across sources. Geometry hypotheses therefore do not compete
with one another as if they were independent sources. Results produced with `d4`
include `confidence.timing.selected_geometry_hypothesis`; batch receipts and resume
identity include `geometry_mode`.

## Adoption evidence and boundary

An isolated internal spike froze its case list before scoring. Its media hashes
and per-case artifacts are not included in this public source distribution.

On one development source family, one held-out source family, and one source-withheld
family, the D4 descriptor method produced:

| Group | Correct | Accepted |
|---|---:|---:|
| development | 8/8 | 8/8 |
| holdout | 7/8 | 7/8 |
| source withheld | 8/8 | 0/8 |

The sole held-out failure was the aggressive `crop_055` case. Mirror and rotations by
90, 180, and 270 degrees matched both included source families, while all eight
source-withheld transformations remained rejected. A production-path rerun reproduced
those outcomes and selected the corresponding mirror or rotation hypothesis for all
eight mirror/rotation positives.

This is narrow initial evidence, not broad geometry qualification. It supports only
an experimental, explicit mirror/quarter-turn option. It does **not** justify claims
for perspective correction, arbitrary-angle rotation, or aggressive cropping. The
existing V1 reframe hypotheses remain responsible for crop tolerance.
