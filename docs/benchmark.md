# RefWorldBench v0

## Goal

Measure a narrow intersection that general world-model benchmarks do not isolate cleanly: whether an image-conditioned persistent world preserves a supplied observation as a calibrated visual anchor, remains locally correct under known 3D camera displacement, and exposes stable semantic/edit state over time.

RefWorldBench is **not** intended to be a general world-quality benchmark. 4DWorldBench, WorldExam and related suites already cover broad perceptual, physical, control and spatial-consistency dimensions.

The current `Anchor Fidelity Curve (AFC)` is a working diagnostic name, not a claimed novel metric.

## Sample types

### Type A — single-image open-world

Only the reference image is available.

Directly valid for:

- exact source-anchor fidelity after camera registration;
- semantic persistence;
- edit locality;
- runtime/portability;
- relative revisit tests where the system itself produces a rollout.

**Important limitation:** there is no photographic ground truth for arbitrary off-anchor views. Do not label a perceptual-stability curve against hallucinated/pseudo-target views as held-out novel-view fidelity.

### Type B — held-out calibrated multi-view

One image is given to the generator; additional calibrated views are withheld for evaluation.

This is the primary protocol for geometric displacement/fidelity analysis because `I_gt(C_d)` is actually observed.

### Type C — registered video / 360 ground truth

A reference frame is given; a calibrated/registered video or panorama provides denser path evidence and loop/revisit targets.

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

Capabilities that are not exposed must be reported as **unsupported**, not silently scored as zero.

## Canonical camera convention

RefWorldBench v0 uses an explicit canonical camera payload:

- right-handed OpenGL camera coordinates;
- local `-Z` forward;
- camera-to-world extrinsics;
- explicit 3×3 pinhole intrinsics;
- zero skew in the v0 Spark renderer path.

Historical/vendor assets may use other conventions; all conversions must be recorded and tested.

## Camera protocol

Recover or receive anchor camera `C0`.

Evaluate, where ground truth/scale permits:

- yaw: ±2°, ±5°, ±10°, ±20°, ±45°;
- pitch: ±2°, ±5°, ±10°;
- translation: ±0.05 m, ±0.10 m, ±0.25 m, ±0.5 m, ±1.0 m;
- compact rotate-away/rotate-back loop;
- rotation + translation loop;
- long excursion → revisit `C0`.

Camera recovery must be treated as a separate optimization problem. Report initial camera, optimized camera, optimization delta and final residual. Do not alter world geometry/appearance during camera-registration evaluation.

## Metric groups

### A. Exact anchor fidelity

At `C0` compare the rendered world against the actual supplied image.

Candidate metrics:

- PSNR;
- SSIM / MS-SSIM;
- LPIPS;
- DINOv2 or equivalent feature similarity;
- edge/structure agreement;
- optional segmentation-aware regional scores.

A camera optimizer may refine `C0`, but the amount of refinement must be reported.

### B. Held-out displacement fidelity / working AFC diagnostic

For **Type B/C** samples, render at calibrated held-out cameras `C_d` and compare to observed `I_gt(C_d)`.

Report separately by displacement family:

- `AFC_yaw(d)`;
- `AFC_pitch(d)`;
- `AFC_translation(d)`;
- normalized area under the chosen interval;
- degradation slope near zero;
- failure radius at declared quality thresholds.

Do not combine angular and metric displacement into a single axis without a declared normalization.

The purpose is diagnostic: distinguish a source-view solution that reproduces `C0` but collapses immediately from one whose nearby geometry/appearance generalizes to real held-out observations.

### C. Revisit-selective memory

Raw first-visit ↔ return similarity is **not sufficient**. R2M-Bench shows that this can be inflated by generic temporal stability, repetitive content, slow/failed motion or a rollout that barely changes.

For compatible interactive/video systems, use same-rollout relative controls inspired by R2M-Bench:

1. identify first-visit/return pair `R`;
2. select a gap-matched non-revisit pair `B` as a temporal baseline;
3. select a short-range pair `S` as a local-consistency reference;
4. compute/import a revisit advantage such as MemoryGain;
5. compute/import a normalized relative score such as NMR where the assumptions fit.

RefWorldBench should reproduce the established metric exactly before proposing a variant.

For explicit/exported 3D worlds, add deterministic state checks that video-only benchmarks cannot observe directly:

- render equality at identical camera before/after path;
- entity ID/attribute equality;
- world-state hash/snapshot equality where supported;
- geometry-transform equality;
- persistence across reload/reconnect where supported.

Keep these explicit-state checks separate from R2M-style video-memory scores.

### D. Semantic persistence

If the system exposes entities, measure:

- stable entity IDs across paths/reload;
- stable transform/shape/appearance attributes;
- stable relations (`on`, `inside`, `next_to`, room membership);
- mutable application-state persistence;
- snapshot → mutate → restore equivalence;
- edit/history provenance where exposed.

If no semantic API exists, a tracker/VLM can provide a weaker inferred proxy, clearly labeled as inferred rather than native persistence.

### E. Edit locality

Apply a bounded edit to entity/region `e`.

Score separately:

- target edit success;
- target identity preservation;
- unintended visual change outside target mask/spatial support;
- unintended semantic/state change outside the target entity/subgraph;
- exact-anchor preservation for unaffected regions;
- persistence after leave-and-return;
- restoration error after snapshot restore where supported.

### F. Geometry / physics

Where ground truth exists:

- depth error;
- normal error;
- camera pose error;
- collision coverage;
- free-space violations;
- navigation reachability.

Broad physical-realism evaluation should defer to/generalize from established benchmarks rather than be reinvented here.

### G. Runtime / portability

Report:

- generation time;
- first interactive frame;
- asset/world size;
- VRAM/RAM;
- FPS at fixed resolutions;
- browser/mobile viability;
- export time;
- renderer/version and hardware/software backend.

## Renderer control

For exported Marble SPZ baselines, the v0 deterministic path is pinned separately under `renderer/` using Spark + Three.js + Playwright.

Record:

- exact Spark/Three/Playwright versions;
- browser build;
- hardware or SwiftShader backend;
- output resolution;
- DPR (v0 = 1);
- antialiasing state (v0 = off);
- color management/tone mapping;
- SPZ export tier/hash;
- camera convention/conversion.

Do not change splat representation (for example SPZ → another LoD/radiance format) in the primary fidelity path unless the conversion itself is separately measured.

## Composite scores

Do **not** hide failure behind one aggregate leaderboard number in v0.

Primary plots remain separate. If a composite is later justified, publish weighting, normalization and sensitivity analysis.

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
  "held_out_displacement": {
    "yaw_deg": {"0": 1.0, "2": 0.0, "5": 0.0, "10": 0.0, "20": 0.0}
  },
  "revisit": {
    "raw_similarity": null,
    "memory_gain": null,
    "normalized_memory_ratio": null
  },
  "semantic_persistence": {},
  "edit_locality": {},
  "runtime": {}
}
```

The machine-readable contract lives in `schemas/report.schema.json`; update that schema when metric fields graduate from experimental to required.

## Human evaluation

Human pairwise evaluation remains useful because perceptual metrics can reward blurry or semantically wrong views.

Ask separately:

1. Which render better matches the actual source/held-out observation?
2. Which world feels more spatially coherent while moving?
3. Which world better preserves scene/object identity specifically on revisit?
4. Which edit is more local and controlled?

Never collapse these into “which looks better?”

## Prior-art constraints on claims

Before claiming a new benchmark/metric contribution, compare directly against:

- R2M-Bench — relative revisit memory;
- ViewBench / ViewRope — loop closure/geometric drift;
- WorldExam — Scene Revisit and 3D Consistency;
- 4DWorldBench — general 3D/4D world evaluation;
- Ref4D-VideoBench — reference-based evaluation;
- novel-view synthesis / inverse-rendering / reconstruction metrics outside the “world model” naming cluster.

See [`literature.md`](literature.md).
