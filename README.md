# Reference Worlds

**R&D on reference-anchored persistent world synthesis.**

> Can a system turn one reference image into a freely explorable world while treating the original image as a measurable visual anchor — and keep objects, identity, edits, and state coherent over time?

This repository is a research scaffold and benchmark proposal, not a claim that the problem is solved or that its current metrics are novel.

## Why this exists

Single-image world generation moved quickly in 2025–2026. Google DeepMind's Genie 3 demonstrates interactive world simulation, while World Labs' Marble generates persistent, navigable 3D worlds from images/text/video and exports splats and meshes. VGGT, Depth Anything 3, SAM 3D Objects, TRELLIS.2 and related systems make strong geometry/object pieces increasingly accessible.

Evaluation has also moved quickly. 4DWorldBench, WorldExam, ViewBench, R2M-Bench and related work already evaluate broad world quality, 3D consistency, loop closure and revisit memory. Reference-based evaluation itself also has prior art such as Ref4D-VideoBench.

So the useful question is narrower than either “can we make a world from an image?” or “can we test whether it remembers revisits?”

**The target here is the intersection of calibrated source-reference fidelity + persistent explicit world state + local editability.**

A generated world should ideally:

1. reproduce the supplied reference view extremely closely after camera registration;
2. remain coherent as the camera moves through calibrated 3D displacements from that view;
3. preserve revisit-specific memory beyond generic temporal stability;
4. expose stable, addressable entities rather than only visually plausible pixels;
5. support localized edits without unrelated world drift;
6. persist those identities/edits across navigation, reload or snapshot/restore where supported.

Focused prior-art map: [`docs/literature.md`](docs/literature.md).

## Working diagnostic: the Anchor Fidelity Curve

Let `I_ref` be the reference image, `W` a generated world, and `R(W, C)` a renderer at camera `C`.

At the recovered/reference camera `C0`:

```text
R(W, C0) ≈ I_ref
```

For held-out calibrated multi-view data, evaluate similarity as the camera moves away from the input camera:

```text
AFC(d) = similarity(I_gt(C_d), R(W, C_d))
```

where `d` can be angular or metric camera displacement from `C0`.

The working **Anchor Fidelity Curve (AFC)** reports quantities such as:

- source-anchor score;
- normalized area under the displacement/fidelity curve;
- near-anchor degradation slope;
- threshold failure radius.

This is currently a **diagnostic name, not a novelty claim**. Before publication we must search the novel-view/inverse-rendering literature more deeply and show that this protocol exposes failures or changes rankings beyond established metrics.

For single-image-only cases, only the exact source view has direct photographic ground truth; off-anchor scores must not pretend hallucinated views are observed truth.

## Working hypothesis

A strong persistent system may be hybrid:

```text
reference image
    │
    ├─ geometry prior (VGGT / DA3 / equivalent)
    ├─ object + semantic decomposition
    ├─ world proposal / completion (Marble-class model or generative prior)
    │
    ▼
canonical persistent world
    ├─ geometry / collision representation
    ├─ appearance representation (splats / radiance / PBR)
    ├─ semantic entity graph
    ├─ uncertainty for unseen regions
    └─ persistent state + edit history
    │
    ▼
reference-constrained optimization
    │
    ▼
interactive runtime
```

Three principles:

- **Observed regions are constraints.** The source image should not be treated as loose inspiration.
- **Unseen regions are hypotheses.** They may be generated, but should not be confused with recovered ground truth.
- **Observed history becomes persistent.** Once something is exposed or edited, silent identity/state drift should be measurable.

## Research questions

### RQ1 — Exact source-anchor fidelity
How accurately can a generated world reproduce the actual input observation after generation, export, camera registration, expansion and editing?

### RQ2 — Local novel-view fidelity
On held-out calibrated views, how quickly does fidelity degrade as the camera moves ±2°, ±5°, ±10°, ±20° or known metric distances from the anchor?

### RQ3 — Revisit-selective persistence
When a camera returns, does the world preserve prior content **more than expected from generic temporal stability or failed/slow motion**? RefWorldBench should import/adapt R2M-Bench-style same-rollout relative controls rather than rely on raw return similarity.

### RQ4 — Semantic persistence
Can an object have a stable ID, transform, appearance, relations, permissions/application state and edit provenance independent of the renderer?

### RQ5 — Edit locality
If one object/region is changed, can the system preserve unaffected visual and semantic state?

### RQ6 — Uncertainty-aware completion
Does delaying commitment on genuinely unseen space improve global coherence compared with one-shot hidden-scene hallucination?

## Baselines and adjacent work

| Family | Candidate | What it gives us | Relevance here |
|---|---|---|---|
| Persistent generated world | World Labs Marble 1.1 / 1.1 Plus | Image→persistent 3D world, edit/expand, splat/mesh export | First falsification baseline for source-anchor + explicit export tests |
| Interactive world model | Google DeepMind Genie 3 | Real-time interactive visual simulation | Frontier evidence for interactive consistency; different output/state model |
| Revisit benchmark | R2M-Bench | Relative revisit-memory metrics with same-rollout controls | Should inform/replace naive raw loop similarity |
| Loop-closure benchmark/method | ViewBench / ViewRope | Controlled return trajectories + geometry-aware consistency | Prior art for loop closure and geometric drift |
| Revisit method | Closing the Loop | Historical latent retrieval + geometric correspondences | Method baseline/idea for long-horizon visual memory |
| General benchmark | WorldExam | Scene Revisit, 3D Consistency, reactivity | Prevents overclaiming generic spatial consistency |
| General benchmark | 4DWorldBench | Perceptual/alignment/physics/4D consistency | Prevents overclaiming general world evaluation |
| Reference-based evaluation | Ref4D-VideoBench | Reference-conditioned video evaluation | Prevents claiming reference evaluation itself as novel |
| Geometry | VGGT / VGGT-Omega | Cameras, depth, point maps, tracks | Camera/geometry prior |
| Geometry | Depth Anything 3 | Spatially consistent geometry/depth/pose | Camera/geometry prior |
| Object reconstruction | SAM 3D Objects | Object shape/texture/layout | Candidate semantic entity bootstrap |
| Image→3D asset | TRELLIS.2 | High-fidelity PBR assets | Candidate explicit object representation |
| Rendering | Spark / Three.js | Deterministic SPZ delivery/capture path | Benchmark renderer, not world inference |

See [`docs/landscape.md`](docs/landscape.md) and [`docs/literature.md`](docs/literature.md).

## Proposed benchmark: RefWorldBench

RefWorldBench is deliberately diagnostic rather than a second general “world quality” leaderboard.

Each compatible system receives one or more reference images and returns a persistent world representation or deterministic interactive endpoint.

The benchmark keeps these axes separate:

1. **Exact anchor fidelity** — recovered source camera.
2. **Held-out local novel-view fidelity** — calibrated camera displacement when ground truth exists.
3. **Revisit-selective memory** — relative revisit metrics plus explicit loop/state checks.
4. **Semantic persistence** — stable entity identity, attributes and application state.
5. **Edit locality** — target success vs collateral visual/semantic drift.
6. **Runtime/portability** — generation, asset size, render performance, export/runtime support.

Full protocol: [`docs/benchmark.md`](docs/benchmark.md). Dataset/license triage: [`docs/datasets.md`](docs/datasets.md).

## First experiments

- **EXP-001 — Marble falsification baseline:** generate from one rights-cleared reference, materialize a splat/mesh, recover `C0`, render controlled perturbations, and produce source-anchor + held-out-view measurements.
- **EXP-002 — Anchor correction:** only if EXP-001 exposes a systematic source/local-view failure, optimize camera/local appearance/geometry while regularizing held-out novel-view consistency.
- **EXP-003 — Near-to-far completion:** compare one-shot hidden-space completion with progressive completion ordered by distance from observed evidence.
- **EXP-004 — Semantic overlay:** attach stable entity IDs/state to a generated visual world; navigate away/revisit; edit one entity and measure collateral drift.
- **EXP-005 — Uncertainty frontier:** retain explicit uncertainty for unseen space and commit hypotheses only as they become observed.

See [`research/roadmap.md`](research/roadmap.md).

## Repository layout

```text
reference-worlds/
├── README.md
├── docs/
│   ├── architecture.md
│   ├── benchmark.md
│   ├── datasets.md
│   ├── landscape.md
│   ├── literature.md
│   └── marble.md
├── renderer/
│   ├── app.mjs
│   ├── capture.mjs
│   ├── projection.mjs
│   └── package.json
├── research/
│   └── roadmap.md
├── schemas/
│   ├── report.schema.json
│   └── world-state.schema.json
├── src/refworld/
│   ├── adapters/
│   ├── camera.py
│   └── metrics.py
├── tests/
└── examples/
```

## Current executable scaffold

The Python package includes deterministic source metrics, AFC summaries, canonical OpenGL camera perturbations, a renderer-neutral adapter protocol, secret-safe World Labs API/media handling, signed-export materialization with SHA-256 provenance, and public-report sanitization.

The JS renderer subproject pins Spark/Three/Playwright and separates projection math from renderer state.

```bash
python -m pytest
cd renderer && npm test
```

No GitHub Actions are required; reproducibility commands are intended to run explicitly.

## What would count as a real result?

Not “it looks cool.”

A useful result would be evidence such as:

> On a rights-cleared calibrated multi-view set, system A and system B have similar exact-anchor fidelity, but one degrades substantially faster over 0–20° held-out camera displacement; relative revisit-memory scores differ after controlling for generic temporal stability; and only one preserves stable semantic/edit state after leave-and-return.

A useful **negative** result is equally valid: if Marble already satisfies the visual hypothesis strongly, stop building another visual world generator and narrow the project to semantic/edit persistence or benchmark tooling.

## Key prior art / starting points

- Focused literature map: [`docs/literature.md`](docs/literature.md)
- R2M-Bench: https://arxiv.org/abs/2608.27328
- ViewBench / ViewRope: https://arxiv.org/abs/2602.07854
- Closing the Loop: https://arxiv.org/abs/2607.21848
- WorldExam: https://arxiv.org/abs/2608.02603
- 4DWorldBench: https://openaccess.thecvf.com/content/CVPR2026/html/Lu_4DWorldBench_A_Comprehensive_Evaluation_Framework_for_3D4D_World_Generation_Models_CVPR_2026_paper.html
- Ref4D-VideoBench: https://openaccess.thecvf.com/content/CVPR2026/html/Wei_Ref4D-VideoBench_Four-Dimensional_Reference-Based_Evaluation_of_Text-to-Video_Generative_Models_CVPR_2026_paper.html
- World Labs Marble/API: https://docs.worldlabs.ai/
- Google DeepMind Genie: https://deepmind.google/models/genie/
- VGGT: https://github.com/facebookresearch/vggt
- Depth Anything 3: https://github.com/ByteDance-Seed/Depth-Anything-3

## License

MIT for original code and documentation in this repository. External models, checkpoints, services, datasets and generated assets retain their own licenses and terms.
