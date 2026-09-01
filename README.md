# Reference Worlds

**R&D on reference-anchored persistent world synthesis.**

> Can a system turn one reference image into a freely explorable world while treating the original image as a hard visual anchor — and keep objects, identity, edits, and state coherent indefinitely?

This repository is a research scaffold and benchmark proposal, not a claim that the problem is solved.

## Why this exists

Single-image world generation moved quickly in 2025–2026. Systems such as Google DeepMind's Genie 3 generate interactive worlds in real time, while World Labs' Marble generates persistent, navigable 3D worlds from images/text/video and can export splats and meshes. Geometry systems such as VGGT, Depth Anything 3, SAM 3D Objects, and TRELLIS.2 make strong pieces of the pipeline increasingly accessible.

That means the useful open question is narrower than “can we make a world from an image?”

**The target here is reference fidelity + persistence + semantics at the same time.**

A generated world should:

1. reproduce the reference view extremely closely;
2. remain coherent as the camera moves away from that view;
3. return to the same state after long excursions and loop closure;
4. expose stable, addressable entities rather than only visually plausible pixels;
5. support localized edits without unrelated world drift;
6. run as a persistent interactive environment, not a short generated video.

## Core research object: the Anchor Fidelity Curve

Let `I_ref` be the reference image, `W` a generated world, and `R(W, C)` a renderer at camera `C`.

At the recovered/reference camera `C0`:

```text
R(W, C0) ≈ I_ref
```

Rather than report only one source-view score, evaluate fidelity as the camera moves away from the anchor:

```text
AFC(d) = similarity(I_expected(d), R(W, C_d))
```

where `d` can be angular displacement, translation, or path distance.

For datasets with held-out multi-view ground truth, `I_expected(d)` is observed imagery. For single-image-only cases, the anchor point is directly measurable and the surrounding curve is assessed through geometry, cycle consistency, perceptual stability, and human/vision-model preference.

The **Anchor Fidelity Curve (AFC)** makes a familiar failure mode visible: a world can look perfect at the source camera and collapse immediately when the viewer moves 10 degrees.

## Working hypothesis

A strong system will likely be hybrid:

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
anchor-constrained optimization
    │
    ▼
interactive runtime
```

Three principles:

- **Observed regions are constraints.** The source image should not be treated as loose inspiration.
- **Unseen regions are hypotheses.** They may be generated, but should not be confused with recovered ground truth.
- **Observed history becomes persistent.** Once the viewer sees or changes something, it should not silently morph on revisit.

## Research questions

### RQ1 — Anchor fidelity
How close can a generated world reproduce the exact input view after full world generation, editing, expansion, and optimization?

### RQ2 — Local novel-view consistency
How quickly does fidelity degrade as the camera moves ±2°, ±5°, ±10°, ±20°, or 0.1–1 m from the anchor?

### RQ3 — Loop closure
If the user travels around geometry and returns to a previously observed camera, does the world reproduce the same scene?

### RQ4 — Semantic persistence
Can an object have a stable ID, transform, appearance, permissions, and application state independent of the renderer?

### RQ5 — Edit locality
If one object is changed, can the system preserve the rest of the world without global visual drift?

### RQ6 — Uncertainty-aware completion
Does delaying commitment on unseen space improve global coherence compared with one-shot hidden-scene hallucination?

## Baselines to test

| Family | Candidate | What it gives us | Main gap for this project |
|---|---|---|---|
| Generative persistent world | World Labs Marble 1.1 / 1.1 Plus | Image→persistent 3D world, edit/expand, splat/mesh export | Need explicit anchor-fidelity + semantic persistence evaluation |
| Autoregressive world model | Google DeepMind Genie 3 | Real-time interactive photorealistic world simulation | Shorter horizon, limited explicit/exportable canonical state |
| Geometry | VGGT / VGGT-Omega | Cameras, depth, point maps, tracks | Geometry estimate, not complete persistent world |
| Geometry | Depth Anything 3 | Spatially consistent depth/pose from arbitrary views | Same |
| Object reconstruction | SAM 3D Objects | Full object shape/texture/layout from image | Object-level; scene/world continuity remains |
| Image→3D asset | TRELLIS.2 | High-fidelity PBR assets from a photo | Object/asset generation, not coherent whole-world persistence |
| Rendering | gsplat / SparkJS | Efficient Gaussian splat runtime | Representation/runtime, not world inference |

See [`docs/landscape.md`](docs/landscape.md).

## Proposed benchmark: RefWorldBench

Each system receives one or more reference images and returns a persistent world representation or an interactive world endpoint.

The benchmark scores six axes:

1. **Anchor fidelity** — exact recovered source camera.
2. **Anchor neighborhood fidelity** — controlled camera perturbations.
3. **Spatial / loop consistency** — revisit and path closure.
4. **Semantic persistence** — stable entity identity and state.
5. **Edit locality** — targeted edit vs unintended global drift.
6. **Runtime viability** — generation time, load time, FPS, memory, portability.

Full protocol: [`docs/benchmark.md`](docs/benchmark.md).

## First experiments

- **EXP-001 — Marble baseline:** generate a world from one reference, export its splat/mesh, recover the anchor camera, render controlled perturbations, and produce an AFC.
- **EXP-002 — Anchor correction:** optimize camera + local appearance/geometry against the source image while regularizing novel-view consistency.
- **EXP-003 — Near-to-far completion:** compare one-shot hidden-space completion with progressive completion ordered by camera distance from observed evidence.
- **EXP-004 — Semantic overlay:** attach stable entities/object IDs to a generated visual world; leave and revisit; edit one entity and measure collateral drift.
- **EXP-005 — Uncertainty frontier:** keep unseen regions unresolved until approached, then commit them into immutable world history.

See [`research/roadmap.md`](research/roadmap.md).

## Repository layout

```text
reference-worlds/
├── README.md
├── docs/
│   ├── architecture.md
│   ├── benchmark.md
│   └── landscape.md
├── research/
│   └── roadmap.md
├── schemas/
│   └── world-state.schema.json
├── src/refworld/
│   ├── __init__.py
│   └── metrics.py
├── tests/
│   └── test_metrics.py
└── examples/
    └── sample_manifest.json
```

## Minimal evaluation code

The initial Python package intentionally stays small. It gives us deterministic reference-image metrics and a common result shape while the heavier perceptual/geometry metrics are developed.

```bash
python -m pytest
```

Later benchmark adapters should be model-specific while emitting the same neutral result schema.

## What would count as a real result?

Not “it looks cool.”

A useful milestone would be something like:

> On a held-out multi-view reference set, the proposed method preserves source-view LPIPS within X of the input at the exact anchor, degrades more slowly than baseline across 0–20° perturbations, closes camera loops with less appearance drift, and preserves stable semantic entities after localized edits.

Until we can make a statement like that with reproducible evidence, this is exploratory R&D.

## Prior art / starting points

- Google DeepMind — Genie 3: https://deepmind.google/models/genie/
- World Labs — Marble: https://www.worldlabs.ai/blog/marble-world-model
- World Labs — World API: https://www.worldlabs.ai/blog/announcing-the-world-api
- World Labs — SparkJS: https://docs.worldlabs.ai/api/examples
- Meta/Oxford — VGGT: https://github.com/facebookresearch/vggt
- ByteDance Seed — Depth Anything 3: https://github.com/ByteDance-Seed/Depth-Anything-3
- Meta — SAM 3D Objects: https://github.com/facebookresearch/sam-3d-objects
- Microsoft — TRELLIS.2: https://github.com/microsoft/TRELLIS.2
- 4DWorldBench (CVPR 2026): https://openaccess.thecvf.com/content/CVPR2026/html/Lu_4DWorldBench_A_Comprehensive_Evaluation_Framework_for_3D4D_World_Generation_Models_CVPR_2026_paper.html

## License

MIT for original code and documentation in this repository. External models, checkpoints, services, datasets, and generated assets retain their own licenses and terms.
