# EXP-002 G1-R0: VGGT 392 vs 518 resolution control

Status: diagnostic-only, frozen after G1-D and before any 518 score exists.

G1/G1-D used the RTX-2080-fit 392×392 VGGT configuration, which is explicitly smoke-only. Before replacing VGGT, test whether reduced resolution materially explains the geometry failure.

## Fixed data

- BlendedMVS frozen scene order 2 only (first frozen diagnostic scene; no outcome-based scene selection).
- First `pair.txt` reference record, already-opened source rank 3 target (view 27).
- Reuse the already-opened target RGB and published anchor depth/cameras from G1. No fresh target is consumed.
- Target depth remains unread.
- Pinned VGGT commit and checkpoint remain unchanged.

## Memory-control inference

Run the same pinned VGGT twice with model weights explicitly converted to FP16 before inference:

1. `392-lowmem`
2. `518-lowmem`

The existing G1 `392-original` output remains the reference for an equivalence guard. All three use the same image, preprocessing family, seed, one-scalar oracle scale estimator, published anchor extrinsics for benchmark-frame placement, and published target camera.

No pose refinement, focal correction, depth offset/spatial correction, target-dependent fitting, or repaint is allowed.

## Primary metric

Score `392-original`, `392-lowmem`, `518-lowmem`, and oracle warp against the already-opened target on OBSERVED support common to all four conditions.

## Frozen decisions

### Low-memory equivalence guard

The memory-saving path is considered an acceptable numerical surrogate only if `392-lowmem` differs from `392-original` by **<= 0.25 dB** on the common-support target PSNR.

If this guard fails, do not interpret 518 as a resolution-only comparison.

### Resolution rescue

If the equivalence guard passes:

- **strong rescue:** `518-lowmem` improves by **>= +3.0 dB** over `392-lowmem` and is **better than -3.0 dB from oracle** on the same common support;
- **partial rescue:** improvement >= +1.0 dB but the 518 result remains <= -3.0 dB from oracle, or improvement is between +1 and +3 dB;
- **no meaningful rescue:** improvement < +1.0 dB.

Only a strong rescue justifies extending 518 across the remaining opened rank-3 scenes before selecting a repair. Partial/no rescue means resolution alone is insufficient and the next lane may evaluate alternative geometry/calibration models on opened rank-3 data.

G1-R0 is explanatory only. Rank 4 remains sealed.