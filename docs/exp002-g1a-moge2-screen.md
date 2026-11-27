# EXP-002 G1-A: alternative geometry screen — MoGe-2 first

Status: **development-only protocol frozen before any MoGe target score is inspected**.

G1 failed with pinned single-image VGGT geometry. G1-D attributed the failure to both learned depth shape and learned intrinsics, and G1-R0 found no meaningful 392→518 resolution rescue on the frozen control scene. Rank 3 is therefore a development set; rank 4 remains sealed.

## Question

Can a stable, permissively licensed single-image geometry model that jointly predicts depth and camera intrinsics materially close the calibrated warp gap before we spend a fresh target set?

## Fixed development data

- BlendedMVS frozen scene orders 2–10.
- First published `cams/pair.txt` record in each scene.
- Already-opened source rank 3 targets from G1 only.
- No fresh target RGB is materialized.
- Target depth is never read.
- Published anchor depth is used only to fit one positive global depth scalar.
- Published anchor extrinsics are used only for benchmark-frame placement.
- Published target camera is used for evaluation.

## Equalized scale bridge

The original G1 gave VGGT a top-50%-confidence scalar fit. MoGe-2 exposes a validity mask rather than a directly comparable raw confidence score, so G1-A equalizes the oracle advantage across models:

- fit `scale = median(anchor_depth / predicted_depth)` over **all finite positive overlapping anchor pixels**;
- implement this with the existing deterministic scale primitive at `top_fraction=1.0`;
- no confidence-based selection;
- no offset, spatial depth correction, focal fitting, pose refinement, per-scene tuning, or target-dependent fitting.

The equalized VGGT condition is therefore regenerated from the already-frozen 392px G1 source geometry using this all-valid scalar. This is a development baseline, not a replacement for the sealed G1 result.

## Candidate 1 — MoGe-2 ViT-B normal

Pinned stack:

- upstream code: `microsoft/MoGe` commit `925b8ed835a7a9cdb7578ba15c658a0afc969030`;
- model: `Ruicheng/moge-2-vitb-normal`;
- Hugging Face revision: `54ad3a693e61907ea4633d13dec6ee682fa09419`;
- `model.pt` SHA-256: `16b8110e86d5dc5a849db120ca96ef3a223fd30b0c9146d1d81db504073da5f6`;
- FP16 inference;
- MoGe-predicted normalized intrinsics are mapped to original image pixels;
- MoGe depth is mapped/output at original source resolution;
- MoGe native metric scale is **not** credited in this screen: the same one-scalar oracle bridge is used as for equalized VGGT.

Hardware-only resolution fallback, fixed before target scoring:

1. `resolution_level=9`;
2. if and only if CUDA OOM occurs, retry `resolution_level=7`;
3. if and only if CUDA OOM occurs again, retry `resolution_level=5`;
4. the first level that fits scene order 2 is frozen for all nine scenes.

A non-OOM failure is not a reason to change resolution or model configuration.

## Primary comparison

For every scene, score:

1. oracle source geometry;
2. equalized VGGT (392 source geometry + all-valid scalar);
3. MoGe-2 ViT-B (all-valid scalar).

Primary support is the intersection of pixels marked `OBSERVED` by **all three** conditions. Report per-scene PSNR and:

- `MoGe - oracle`;
- `VGGT_equalized - oracle`;
- `MoGe - VGGT_equalized`.

Aggregate medians are computed across the nine frozen scenes. Full-frame PSNR and each condition's own OBSERVED fraction are secondary diagnostics.

## Frozen decision rule

MoGe-2 **passes the geometry screen** only if all are true:

1. median `MoGe - oracle` on all-three common OBSERVED support is **greater than -3.0 dB**;
2. median `MoGe - VGGT_equalized` is **at least +3.0 dB**;
3. MoGe beats equalized VGGT on the same common support in **at least 7 of 9 scenes**.

If MoGe-2 passes, freeze this exact geometry configuration before any fresh rank-4 confirmation. Do not tune it on rank 3 further.

If MoGe-2 fails, **do not spend rank 4**. The next predeclared candidate is `depth-anything/DA3-BASE` under the same all-valid scalar/frame bridge. DA3 is not scored unless MoGe-2 fails this frozen screen.

## Claim boundary

G1-A is not end-to-end single-image RefWorld-0. It deliberately retains one oracle anchor-depth scale scalar, oracle anchor-frame placement, and the published target camera. Its only purpose is model selection for depth-shape + intrinsics adequacy on already-opened development targets.
