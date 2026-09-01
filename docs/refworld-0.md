# RefWorld-0 — evidence-preserving local world construction

RefWorld-0 is the smallest method experiment justified by the project's thesis.

It is **not** a new foundation world model. It changes only the part of the open baseline most directly connected to the reference-fidelity problem: how hidden / newly visible content is proposed and incorporated into one canonical world.

## Hypothesis

Given one supplied image `I0` and recovered source camera `C0`, replacing one-shot hidden-panorama hallucination with **geometry-guided nearby-view proposals whose real evidence cannot be repainted** will improve local world coherence without sacrificing source-view fidelity.

The null result is useful. If this does not improve held-out calibrated views, the evidence-preserving proposal mechanism is not sufficient and should not be promoted as a method contribution.

## Evidence boundary

RefWorld-0 distinguishes three pixel origins:

```text
OBSERVED   real supplied observation projected into a target camera
GENERATED  synthesized where direct observed support is absent
UNRESOLVED no accepted evidence or generated proposal
```

`src/refworld/evidence.py` enforces the core invariant:

> A repaint backend may propose a full frame, but its output cannot replace OBSERVED support.

This is stronger than asking a generative model to preserve the source through prompt/guidance alone.

## Persistence is a different axis

Epistemic origin and world-history commitment must not be conflated.

For example, an object can be:

```text
evidence_origin   = generated
resolution_state  = resolved
commitment_state  = committed
exposure_state    = seen
```

Once exposed/committed, the world may be required to keep that generated hypothesis stable. It does **not** thereby become observed ground truth.

The persistent world-state schema records these axes separately.

## Data-leakage rule

For a Type-B calibrated benchmark scene:

- the one frozen anchor image is method input;
- its calibration may be used when the experimental protocol explicitly supplies it;
- frozen held-out RGB images are **evaluation only**;
- held-out RGB/depth must not influence proposal generation, reconstruction, camera refinement, hyperparameter selection for that scene, or anchor optimization;
- held-out camera poses may only be used for final evaluation unless a separately declared development split explicitly permits otherwise.

For the BlendedMVS bootstrap, treat all source views selected from the first `pair.txt` record as sealed evaluation views for the first RefWorld-0 comparison.

The bootstrap is pipeline research, not the eventual publication test set. Before a paper claim, freeze a separate development/test protocol so hyperparameter work does not consume the final test evidence.

## Pipeline v0

```text
I0, C0
 │
 ├─ geometry prior
 │    ├─ depth / point map
 │    ├─ visibility
 │    └─ confidence
 │
 ▼
deterministic proposal cameras
 │
 ├─ scale-free yaw / pitch
 └─ depth-normalized translations
 │
 ▼
geometry warp
 │
 ├─ warped RGB
 ├─ observed mask
 └─ confidence map
 │
 ▼
generative repaint
 │
 ├─ candidate RGB
 └─ validity mask
 │
 ▼
evidence-preserving compositor
 │
 ├─ observed copied exactly
 ├─ generated only on non-observed support
 └─ unresolved retained explicitly
 │
 ▼
source + proposal set
 │
 ▼
canonical reconstruction
 │
 ├─ 3DGS appearance
 ├─ optional mesh/collision geometry
 └─ provenance links
 │
 ▼
reference-constrained optimization
```

The first candidate geometry/repaint backend is inspired by WorldForge's VGGT warp + diffusion repaint pipeline. Backend identity is explicit in each proposal and may later be replaced.

## Deterministic camera neighborhood

`src/refworld/neighborhood.py` separates:

### Scale-free rotations

Useful even when monocular metric scale is unavailable:

```text
yaw   ±2°, ±5°, ±10°
pitch ±2°, ±5°
```

These primarily test/expand angular coverage and do not provide parallax by themselves.

### Depth-normalized translations

Translations are expressed as a declared fraction of a chosen reference-depth statistic, e.g.:

```text
±0.02 × reference_depth
±0.05 × reference_depth
```

If the depth statistic is monocular/arbitrarily scaled, these are **ratios**, not meters. The metadata preserves that distinction.

For calibrated real evaluation views, use their actual cameras instead of coercing them into these synthetic displacement labels.

## Proposal contract

A warper implements:

```python
warp(observations, target_camera) -> WarpResult
```

with:

- RGB;
- boolean observed mask;
- `[0,1]` confidence map;
- backend identifier;
- metadata.

A repainter implements:

```python
repaint(warp, target_camera, seed=...) -> RepaintResult
```

with:

- candidate RGB;
- boolean validity mask;
- backend identifier;
- seed;
- metadata.

`build_view_proposal` produces:

- deterministic proposal ID from lineage + camera + backends + seed;
- evidence-preserved RGB;
- per-pixel provenance;
- warp confidence;
- evidence counts/fractions;
- hashes for all relevant arrays;
- array-free metadata suitable for JSON.

Schema: `schemas/view-proposal.schema.json`.

## Objective semantics

The proposal images are **pseudo-evidence**, not equivalent to real observations.

Let provenance weight map `w(x)` be:

```text
OBSERVED   confidence(x) × w_observed
GENERATED  w_generated
UNRESOLVED 0
```

with:

```text
w_generated <= w_observed
w_unresolved = 0
```

The exact source-anchor render receives its own direct loss against the real source image and should not be diluted into a large pool of generated proposal pixels.

A conceptual objective is:

```text
L = λ_anchor L_anchor(I0, R(W,C0))
  + λ_proposal L_weighted_proposal
  + λ_geometry L_geometry
  + λ_regularization L_reg
```

where `L_weighted_proposal` uses provenance/confidence and the calibrated held-out RGB images are absent from optimization.

`src/refworld/objectives.py` currently provides dependency-light provenance weights + weighted L1 as a semantics test, not a claim that L1 will be the final perceptual objective.

## Required ablations

At minimum compare:

1. **WorldGen unchanged** — one-shot panorama completion.
2. **Warp only** — geometric target views, holes remain.
3. **Warp + unrestricted repaint** — generated candidate allowed to repaint the whole target frame.
4. **Warp + evidence-preserving repaint** — generated fill cannot overwrite observed support.
5. **Evidence-preserving proposals + canonical reconstruction, no anchor optimization.**
6. **Same + source-anchor optimization.**

If compute permits, compare at least two repaint priors/backends so conclusions do not reduce to one model's taste.

## Primary evaluation

Keep these separate:

### Exact source

- camera-registration residual;
- PSNR / SSIM / LPIPS / feature similarity;
- structure/edge metrics if useful.

### Held-out calibrated views

- similarity at each actual camera;
- view-direction angle from source;
- source-unit / metric camera-center distance as appropriate;
- curve/AUC only when the displacement axis is meaningful.

### Geometry / contradiction

- depth/normal disagreement where ground truth exists;
- cross-view reprojection consistency;
- floaters / duplicate geometry where measurable.

### Evidence behavior

- observed fraction;
- generated fraction;
- unresolved fraction;
- generator overlap-attempt fraction;
- any observed-pixel mutation is a hard failure.

### Runtime

- generation time;
- peak VRAM;
- artifact sizes;
- reconstruction/optimization time.

## Failure criteria

RefWorld-0 fails as a method idea if one or more of these hold systematically:

- source-anchor fidelity improves only by degrading held-out views;
- evidence-preserving repaint performs no better than unrestricted repaint;
- proposal reconstruction amplifies geometry contradictions;
- generated views dominate optimization despite lower epistemic confidence;
- quality gains depend on manually selecting favorable seeds/views;
- the open baseline already performs equivalently within measurement noise.

## What success would justify next

Only after a measurable visual benefit should the project add the deeper persistent-world machinery:

1. semantic entity extraction / stable IDs;
2. progressive near-to-far completion;
3. explicit hidden-space hypotheses;
4. commit-on-exposure world history;
5. edit locality / snapshot / restore.

That sequence keeps the research falsifiable: each added architectural idea must solve an observed failure rather than being added because it sounds like a complete world model.
