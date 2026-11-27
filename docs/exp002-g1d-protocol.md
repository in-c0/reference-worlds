# EXP-002 G1-D: learned-geometry failure decomposition

Status: diagnostic-only, predeclared after the sealed G1 rank-3 result and before inspecting any hybrid-ablation scores.

G1 failed under reduced-resolution VGGT geometry despite an oracle depth-scale scalar and oracle anchor frame placement. The next step is not to tune rank-3 outputs; it is to decompose the already-opened rank-3 failure into depth-shape vs source-intrinsics contributions.

## Fixed data

- BlendedMVS frozen scenes 2–10.
- First `pair.txt` reference record.
- Already-opened source rank 3 targets from G1.
- Reuse the exact G1 392×392 VGGT source-geometry artifacts.
- Reuse the exact published anchor depth/camera and target camera already materialized for G1.
- No fresh target is consumed in G1-D.
- Target depth is never read.

## Four geometry conditions

1. `oracle`: published anchor depth + published anchor intrinsics + published anchor extrinsics.
2. `vggt_both`: G1 baseline: VGGT depth shape, one frozen oracle scalar, VGGT intrinsics, published anchor extrinsics.
3. `vggt_depth_oracle_K`: same scaled VGGT depth as G1, but published anchor intrinsics.
4. `oracle_depth_vggt_K`: published anchor depth, but VGGT intrinsics; published anchor extrinsics.

No pose refinement, focal fitting, offset, spatial depth correction, target-dependent tuning, or alternative scale estimator is allowed.

## Primary diagnostic metric

For each scene, score every condition against the already-opened rank-3 target RGB on the intersection of OBSERVED support shared by **all four** geometry conditions. Report PSNR and deltas from `oracle`.

Also report full-frame PSNR and each condition's own OBSERVED fraction as secondary diagnostics.

## Frozen aggregate decision bands

Use the median per-scene common-OBSERVED PSNR delta from `oracle` across scenes 2–10.

- `near oracle`: median delta >= -1.0 dB.
- `substantially below oracle`: median delta <= -3.0 dB.
- values strictly between -3.0 and -1.0 dB are intermediate and do not support a clean attribution.

These bands are fixed before any G1-D hybrid score is inspected.

## Interpretation rule

- Depth-shape dominant: `vggt_depth_oracle_K` is substantially below oracle and `oracle_depth_vggt_K` is near oracle.
- Intrinsics dominant: `oracle_depth_vggt_K` is substantially below oracle and `vggt_depth_oracle_K` is near oracle.
- Both-components failure: both hybrids are substantially below oracle.
- Coupling/registration failure: both hybrids are near oracle but `vggt_both` is substantially below oracle.
- Otherwise: mixed/inconclusive; do not choose a repair from G1-D alone.

G1-D is explanatory only. Any repair selected from these opened rank-3 diagnostics must be frozen before evaluation on a fresh rank-4 target set.
