# Open-first research stack

RefWorldBench must not require a proprietary world-generation API to pursue its core research question.

The default research path is now **open/local first**. Proprietary systems such as World Labs Marble remain useful external baselines, but they are not infrastructure dependencies.

## Decision

The initial open stack is deliberately modular:

1. **WorldGen** as the first persistent 3D baseline / assembly reference.
2. **WorldForge-style VGGT warping + diffusion repainting** as the first candidate replacement for hidden-view / novel-view synthesis.
3. **RefWorld canonical state + anchor constraints** as the research layer owned by this repository.
4. **HY-World 2.0** as a heavy architectural/reference baseline, not a default dependency.

The goal is not to fork an entire foundation model immediately. The goal is to own the layer where the supplied observation becomes a hard constraint on a persistent, editable world.

## Why WorldGen first

Pinned upstream:

```text
repo: ZiYang-xie/WorldGen
commit: 7ce7b2767fdf31e2727b69a2e61e2e950e3a017f
license: Apache-2.0 (repository code)
```

At this commit WorldGen provides:

- image → scene generation;
- text → scene generation;
- 360° exploration / loop closure;
- Gaussian-splat output with `.ply` serialization;
- mesh output;
- arbitrary camera rendering through its viewer/runtime;
- low-VRAM mode documented around 10 GB VRAM;
- DA-2-based panorama depth;
- optional ml-sharp-based 3DGS generation.

Its image-to-scene path is useful because it already factorizes the problem in a way we can modify:

```text
reference image
  → single-view depth
  → map observed image into equirectangular panorama
  → fill unobserved panorama
  → panorama depth
  → 3DGS / mesh
```

Critically, the current image-to-panorama fill stage uses `FLUX.1-Fill-dev` plus a WorldGen LoRA. That is a replaceable module, not something RefWorld needs to treat as part of the permanent architecture.

### License boundary

WorldGen's repository code is Apache-2.0, but a runnable configuration also relies on external models/dependencies with their own licenses and access terms. The current documented image-to-scene path loads Black Forest Labs FLUX.1-Fill-dev. Do not describe the complete dependency closure as Apache-2.0 simply because the top-level repository is.

Every baseline report must record:

- WorldGen git commit;
- every checkpoint/model identifier;
- checkpoint license/access terms where known;
- low-VRAM / sharp / mesh switches;
- generation seed;
- library/CUDA versions;
- output content hashes.

## Why WorldForge is relevant

Pinned upstream:

```text
repo: Westlake-AGI-Lab/WorldForge
commit: ee573a051715a451b806a90e21462f23308faac4
license: Apache-2.0 (repository code)
```

WorldForge is not itself the persistent-world representation we need. Its value is the **novel-view synthesis mechanism**:

```text
single image
  → VGGT geometry / camera-aware warp
  → explicit holes / uncertainty mask
  → video-diffusion repainting under controlled camera motion
  → photorealistic neighboring views
```

That is closer to the RefWorld problem than asking a panorama model to hallucinate the entire hidden hemisphere in one shot.

The first method experiment should therefore compare at least:

- one-shot panorama completion;
- geometry-warp + repaint neighboring views;
- progressive near-to-far completion using generated neighboring observations.

WorldForge currently supports Wan2.1 and LongCat-Video backends. Those checkpoints retain their own licenses and compute requirements, so they remain swappable implementation choices.

## HY-World 2.0

HY-World 2.0 is strong prior art for the overall pipeline:

```text
panorama generation
→ trajectory planning
→ world expansion
→ multi-view reconstruction / 3DGS optimization
```

Its released world-generation code is valuable for architecture study, but it is not the default RefWorld dependency:

- the documented full pipeline recommends at least four GPUs and was tested with eight H20 GPUs;
- model sizes are substantially heavier than WorldGen;
- its community license has restrictions that make it a poor foundation for a generally reusable open research stack.

Use it as a comparison/reference implementation unless a particular ablation justifies the cost/license burden.

## RefWorld-owned architecture

The target architecture is:

```text
I_ref
  │
  ├── geometry prior
  │     ├── depth / rays
  │     ├── source camera
  │     └── confidence
  │
  ├── observed-region mask ───────────────────────────────┐
  │                                                       │
  ▼                                                       │
near-view proposal generator                              │
  ├── geometric warp                                      │
  ├── unresolved-region mask                              │
  └── generative repaint                                  │
  │                                                       │
  ▼                                                       │
canonical world builder                                   │
  ├── 3DGS / radiance appearance                          │
  ├── mesh / collision geometry                           │
  ├── semantic entities                                   │
  ├── uncertainty field                                   │
  └── observation / edit history                          │
  │                                                       │
  ▼                                                       │
reference-constrained optimization ◀──────────────────────┘
  ├── exact-anchor image loss
  ├── calibrated held-out-view loss
  ├── depth / normal / geometry regularization
  ├── cross-view semantic consistency
  └── edit / state persistence constraints
```

The reference is not merely a prompt. Pixels/rays supported by the supplied observation become evidence that later world completion should not casually overwrite.

## Hidden-space policy

The key RefWorld distinction is to represent epistemic status explicitly:

```text
observed      = constrained by real input / calibrated evidence
resolved      = generated hypothesis already shown to the user
hypothesized  = candidate hidden-space completion not yet committed
unknown       = not yet represented
```

A first implementation does not need a full probabilistic generative model. It can maintain masks/confidence plus deterministic provenance. The research question is whether postponing commitment improves global consistency and anchor preservation.

## EXP-000: open baseline

Before changing WorldGen, run it unchanged at the pinned commit.

Required variants:

1. `worldgen-default`: image-to-scene, splat output, no optional background inpaint, no ml-sharp.
2. `worldgen-sharp`: same input/config with ml-sharp enabled if its dependencies can be reproduced.
3. optional mesh output for geometry diagnostics; do not substitute mesh appearance for splat appearance in primary visual scoring.

For every run persist:

- exact input hash;
- upstream commit;
- model/checkpoint identifiers;
- seed;
- panorama intermediate;
- output PLY/mesh hashes;
- generation time / peak VRAM;
- renderer metadata;
- recovered source camera and residual;
- exact-anchor + held-out-view report.

## First method candidate: RefWorld-0

`RefWorld-0` should be intentionally small. Do not train a foundation model.

Start by replacing only WorldGen's hidden-view completion stage:

1. recover source geometry/camera;
2. generate a small deterministic camera neighborhood around the source;
3. warp observed pixels into those cameras with confidence masks;
4. repaint only unresolved regions using a pretrained image/video prior;
5. reconstruct one canonical 3D representation from the source + generated neighborhood;
6. optimize it with a high-weight exact-anchor constraint and lower-weight held-out/geometry constraints;
7. measure whether source fidelity and local novel-view fidelity improve over unchanged WorldGen.

Only add semantic persistence / uncertainty-frontier machinery after this visual hypothesis is measured.

## Compute boundary

The research no longer requires a vendor API key, but it does require GPU inference.

Current practical tiers:

- WorldGen low-VRAM path: roughly 10 GB VRAM according to upstream documentation;
- WorldForge + 14B video models: substantially heavier; exact requirement depends on backend/offload/precision;
- HY-World 2.0 full worldgen: multi-GPU research-server class.

A CPU-only environment can develop/test benchmark plumbing but cannot credibly execute these generation baselines.
