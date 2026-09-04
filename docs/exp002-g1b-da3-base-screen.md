# EXP-002 G1-B: DA3-BASE geometry screen

Status: development-only, frozen after G1-A MoGe-2 failed and before any DA3 rank-3 target score exists.

## Evidence scope

- Frozen BlendedMVS scene orders 2–10 only.
- Reuse only the already-opened rank-3 target RGBs from G1/G1-A.
- Rank 4 remains sealed throughout G1-B.
- Target depth is never read.

## Candidate pin

- code: `ByteDance-Seed/Depth-Anything-3`
- commit: `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`
- checkpoint: `depth-anything/DA3-BASE`
- HF revision: `ee22d50d2aeb9a58c06b2079d2d27bc220e801aa`
- `model.safetensors` SHA-256: `e01067dc1659613083d9145a9a2547ccdbe6ccbbf83c4fe7b3e8a4e2bdae78b5`
- model size: `541518028` bytes
- license: Apache-2.0

DA3's native code defines depth in the model camera space and predicts pixel-space camera intrinsics. G1-B uses the predicted depth shape and intrinsics only; predicted pose and native scale do not earn the gate.

## Frozen bridge

For DA3 and the equalized VGGT reference:

1. infer source geometry from anchor RGB;
2. map predicted depth/intrinsics to original anchor pixels;
3. fit exactly one positive multiplicative depth scalar from published anchor depth over all finite positive candidate-valid pixels;
4. use published anchor extrinsics only for benchmark-frame placement;
5. use the published target camera;
6. no offset, spatial correction, focal correction, pose refinement, target-dependent fit or repaint.

DA3 sky pixels are candidate-invalid because upstream replaces predicted sky depth with a synthetic maximum-depth value. They cannot become hard OBSERVED support in this screen.

## Hardware rule

First frozen scene probes `process_res` in the fixed order `504 -> 392 -> 336`, always with `upper_bound_resize`. Fallback is permitted only on CUDA out-of-memory. Any other error aborts. The first successful resolution is frozen for all nine scenes.

## Primary metric and decision

Primary support is OBSERVED in oracle, equalized VGGT and DA3 simultaneously.

DA3 passes only if all three hold:

- median `(DA3 - oracle)` common-OBSERVED PSNR > **-3.0 dB**;
- median `(DA3 - equalized VGGT)` common-OBSERVED PSNR >= **+3.0 dB**;
- DA3 beats equalized VGGT in at least **7/9** scenes.

These are the same quality thresholds used for G1-A. MoGe-2 may be retained as a secondary development reference but cannot alter the pass rule.

If DA3 passes, freeze it before designing a fresh rank-4 confirmation. If DA3 fails, rank 4 stays sealed and the bounded rank-3 model-search lane stops; any later repair must be a newly declared hypothesis on fresh evidence.

This remains an oracle-scale/frame diagnostic, not an end-to-end single-image metric-scale result.

Authoritative protocol record: issue #19.
