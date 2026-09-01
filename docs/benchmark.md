# RefWorldBench v0

## Goal

Measure whether an image-conditioned generated world remains faithful to its source and persistent under movement, revisitation, and editing.

## Sample types

### Type A — single-image open-world

Only the reference image is available. Useful for source-anchor fidelity, semantic persistence, edit locality, runtime and subjective novel-view consistency.

### Type B — held-out multi-view

One image is given to the generator; additional calibrated views are withheld for evaluation. This is the strongest protocol for measuring the Anchor Fidelity Curve.

### Type C — video / 360 ground truth

A reference frame is given; a registered video or panorama provides dense novel-view evidence.

## Required system output

At least one of:

- renderable scene/world asset;
- splat / radiance representation;
- mesh/PBR world;
- interactive endpoint with deterministic camera control.

Optional but highly valuable:

- collision mesh;
- semantic entity graph;
- edit API;
- persistent state snapshot/restore.

## Camera protocol

Recover or receive anchor camera `C0`.

Evaluate:

- yaw: ±2°, ±5°, ±10°, ±20°, ±45°;
- pitch: ±2°, ±5°, ±10°;
- translation: ±0.05 m, ±0.10 m, ±0.25 m, ±0.5 m, ±1.0 m where scale is known;
- canonical loop: forward → orbit salient object → return to C0;
- long excursion → revisit C0.

## Metric groups

### A. Exact anchor fidelity

At `C0`:

- PSNR;
- SSIM / MS-SSIM;
- LPIPS;
- DINOv2 or equivalent feature similarity;
- edge/structure agreement;
- optional segmentation-aware regional scores.

A camera optimizer may refine `C0`, but the amount of refinement must be reported.

### B. Anchor Fidelity Curve (AFC)

For each camera displacement `d`, compute perceptual similarity to held-out ground truth where available.

Report:

- `AFC_yaw(d)`;
- `AFC_translation(d)`;
- area under the curve over agreed intervals;
- degradation slope near zero;
- failure radius at chosen quality thresholds.

This distinguishes “perfect billboard at source view” from a world whose nearby geometry is actually coherent.

### C. Loop consistency

Render the same camera before and after a closed navigation path.

Measure:

- image similarity;
- entity pose/state equality;
- geometry/camera drift;
- stochastic variance across repeated loops.

### D. Semantic persistence

If the system exposes entities:

- stable entity IDs across camera paths;
- stable transform/shape/appearance attributes;
- stable relations (`on`, `inside`, `next_to`, room membership);
- state snapshot → mutate → restore equivalence.

If no semantic API exists, a vision-language tracker can provide a weaker proxy score, clearly labeled as inferred rather than native persistence.

### E. Edit locality

Apply a bounded edit to entity `e`.

Score:

- target edit success;
- target identity preservation;
- unintended change outside target mask / spatial support;
- anchor-view preservation for unaffected regions;
- persistence after leaving and revisiting.

### F. Geometry / physics

Where ground truth exists:

- depth error;
- normal error;
- camera pose error;
- collision coverage;
- free-space violations;
- navigation reachability.

### G. Runtime

Report:

- generation time;
- first interactive frame;
- asset/world size;
- VRAM/RAM;
- FPS at fixed resolutions;
- mobile/browser viability;
- export time.

## Composite scores

Do **not** hide failure behind one aggregate leaderboard number in v0.

Primary plots should remain separate. If a composite is later needed, publish the weighting and sensitivity analysis.

## Minimum benchmark report

```json
{
  "system": "example",
  "sample": "scene-001",
  "anchor": {
    "psnr": 0,
    "ssim": 0,
    "lpips": 0
  },
  "afc": {
    "yaw_deg": {"0": 1.0, "2": 0.0, "5": 0.0, "10": 0.0, "20": 0.0}
  },
  "loop": {},
  "semantic_persistence": {},
  "edit_locality": {},
  "runtime": {}
}
```

## Human evaluation

Human pairwise preference remains useful because perceptual metrics can reward blurry or semantically wrong views.

Ask separately:

1. Which world better matches the reference?
2. Which world feels more spatially coherent while moving?
3. Which world better preserves object identity on revisit?
4. Which edit feels more local and controlled?

Never collapse those into “which looks better?”
