# Reference Worlds

**R&D on reference-anchored persistent world synthesis.**

> Can one reference image become a freely explorable world while remaining a measurable visual constraint — and can that world preserve object identity, edits, and state over time?

This repository is a research scaffold, benchmark and method-development workspace. It does **not** claim that the problem is solved or that its current metrics are novel.

## Open-first principle

The core R&D does **not** require a proprietary world-generation API.

The current execution order is:

```text
EXP-000  open local baseline (WorldGen)
   ↓
measure exact source + held-out local views
   ↓
RefWorld-0: replace only the failing stage
   ↓
progressive completion / semantic persistence / uncertainty
   ↓
optional external comparison (Marble / other proprietary systems)
```

World Labs Marble remains a useful external baseline, but it is no longer infrastructure and does not gate the project.

Open stack and exact upstream pins: [`docs/open-stack.md`](docs/open-stack.md).  
Local execution path: [`docs/quickstart.md`](docs/quickstart.md).

## Why this exists

Single-image world generation moved quickly in 2025–2026. Systems now demonstrate:

- image/text → navigable persistent 3D scenes;
- real-time interactive world simulation;
- high-quality camera/depth/point reconstruction;
- controlled novel-view synthesis;
- Gaussian-splat and mesh outputs that can persist indefinitely.

That means the useful research question is no longer simply:

> Can an image become a world?

The narrower target is:

> **Can a generated world treat the supplied observation as durable evidence, preserve calibrated local novel views, and expose persistent explicit state that remains editable and addressable?**

A strong system should ideally:

1. reproduce the supplied reference view closely after honest camera registration;
2. remain coherent at calibrated neighboring cameras;
3. preserve revisit-specific memory beyond generic temporal stability;
4. expose stable semantic entities rather than only similar pixels;
5. support localized edits without unrelated visual/semantic drift;
6. persist identities/edits across navigation, reload or snapshot/restore where supported;
7. distinguish observed evidence from generated hidden-space hypotheses.

Focused prior art: [`docs/literature.md`](docs/literature.md).

## The root ambiguity

A single image does not determine one unique hidden 3D world.

If `I_ref` is the observation and `W` a candidate world:

```text
R(W, C0) ≈ I_ref
```

can hold for many different worlds `W`.

So RefWorld does **not** claim to recover an unknowable ground-truth backside of every object. Instead:

```text
observed      → constrained by real evidence
resolved      → generated hypothesis already exposed/committed
hypothesized  → candidate hidden completion
unknown       → not yet represented
```

The research problem is to construct a coherent world that respects observed evidence while making hidden-space commitments explicit and measurable.

## Working diagnostic: source-anchor / displacement fidelity

At recovered source camera `C0`:

```text
R(W, C0) ≈ I_ref
```

For calibrated held-out data:

```text
fidelity(d) = similarity(I_gt(C_d), R(W, C_d))
```

where `d` is the actual camera displacement from the source observation.

The current code includes an **Anchor Fidelity Curve (AFC)** summary:

- normalized area under the curve;
- near-anchor slope;
- threshold failure radius.

`AFC` is a **working diagnostic name, not a novelty claim**. Real arbitrary held-out cameras are reported by actual pose separation (`view_direction_angle_deg`, camera-center distance), not mislabeled as pure yaw.

For single-image-only data, only the exact source camera has photographic ground truth. Generated off-axis images are never treated as observed truth.

## Open baseline: WorldGen

The first baseline is pinned to:

```text
ZiYang-xie/WorldGen
commit 7ce7b2767fdf31e2727b69a2e61e2e950e3a017f
repository license: Apache-2.0
```

At that commit the image-to-scene pipeline is approximately:

```text
reference image
→ single-view depth
→ project observed pixels into a panorama
→ generatively fill the unobserved panorama
→ panorama depth
→ Gaussian splat / mesh
```

Why it is useful:

- modifiable code;
- image-to-scene path;
- 360° scene representation;
- PLY splat and mesh outputs;
- documented low-VRAM mode around 10 GB;
- clear internal stages that can be ablated.

Important license boundary: WorldGen's repository is Apache-2.0, but the current runnable image-to-scene path uses external checkpoints such as FLUX.1-Fill-dev and DA-2 under their own terms. Every experiment records the full checkpoint/dependency provenance.

### RefWorld WorldGen runner

`refworld-worldgen-run` mirrors the pinned WorldGen image-to-scene path but makes benchmark-critical state explicit:

- seed is exposed instead of hidden behind the upstream `seed=42` default;
- panorama intermediate is retained;
- input/panorama/PLY hashes are recorded;
- actual WorldGen commit is verified;
- checkpoints/configuration are recorded;
- CUDA/GPU/PyTorch metadata and peak VRAM are recorded;
- artifacts use output-relative paths.

The heavy WorldGen/CUDA environment remains separate from the lightweight benchmark package.

## Candidate RefWorld-0 method

Do **not** train a new foundation world model first.

The first method should replace only the most likely bottleneck: **hidden/novel-view completion**.

```text
I_ref
  │
  ├── geometry / camera prior
  │
  ▼
controlled nearby cameras
  │
  ├── warp observed pixels geometrically
  ├── retain visibility/confidence masks
  └── generatively repaint unresolved regions
  │
  ▼
source + synthesized neighborhood
  │
  ▼
canonical 3D reconstruction
  │
  ├── 3DGS / radiance appearance
  ├── mesh / collision geometry
  └── explicit observation provenance
  │
  ▼
reference-constrained optimization
```

WorldForge is the first open implementation reference for the **geometry warp + diffusion repaint** component. It is not itself the persistent-world representation.

The key falsification criterion is held-out multi-view performance: improving the source camera while making nearby real cameras worse is source overfitting, not progress.

## Longer-term canonical world

If RefWorld-0 improves the visual hypothesis, the world representation can grow into:

```text
canonical persistent world
├── appearance representation (3DGS / radiance / PBR)
├── geometry / collision representation
├── stable semantic entity graph
├── observation provenance
├── hidden-space uncertainty state
├── edit history
└── snapshot / persistent application state
```

Three principles:

- **Observed regions are constraints.** The source is not loose inspiration.
- **Unseen regions are hypotheses.** Generated completion is not recovered ground truth.
- **Observed history becomes persistent.** Once something is exposed or edited, silent drift should be measurable.

## Research questions

### RQ1 — Exact source-anchor fidelity
How accurately can a generated world reproduce the supplied observation after generation, export, registration, expansion and editing?

### RQ2 — Calibrated local novel-view fidelity
How quickly does agreement degrade at real neighboring cameras?

### RQ3 — Revisit-selective persistence
Does the system preserve revisited content more than expected from generic temporal stability or failed/slow motion? RefWorld imports R2M-Bench-style relative controls rather than relying on raw return-frame similarity.

### RQ4 — Semantic persistence
Can an object retain stable ID, transform, appearance, relations, application state and edit provenance independent of the renderer?

### RQ5 — Edit locality
Can one object/region change without unrelated visual or semantic state drifting?

### RQ6 — Uncertainty-aware completion
Does postponing commitment in genuinely unseen space improve global coherence compared with one-shot hidden-world hallucination?

## Baselines and adjacent work

| Family | Candidate | Role here |
|---|---|---|
| Open persistent 3D baseline | WorldGen | **Primary EXP-000 baseline** and modifiable assembly reference |
| Controlled novel-view generation | WorldForge | Reference implementation for VGGT warp + diffusion repaint |
| Heavy open/source-available world model | HY-World 2.0 | Architecture / later comparison; not default due compute/license burden |
| Proprietary persistent world | World Labs Marble | Optional external comparison after open baseline |
| Interactive video world model | Google DeepMind Genie 3 | Frontier evidence for interactive consistency; different state model |
| Revisit benchmark | R2M-Bench | Relative revisit-memory evaluation |
| Loop closure | ViewBench / ViewRope | Controlled return trajectories / geometric consistency prior art |
| Revisit method | Closing the Loop | Historical latent retrieval + geometry correspondence prior art |
| General world evaluation | WorldExam / 4DWorldBench | Prevents overclaiming generic world-quality evaluation |
| Geometry | VGGT / Depth Anything 3 | Camera / geometry priors |
| Object reconstruction | SAM 3D Objects / TRELLIS.2 | Candidate explicit semantic-object representations |
| Rendering | Spark / Three.js | Shared deterministic PLY/SPZ benchmark renderer |

See [`docs/landscape.md`](docs/landscape.md), [`docs/literature.md`](docs/literature.md), and [`docs/open-stack.md`](docs/open-stack.md).

## RefWorldBench

RefWorldBench is diagnostic rather than a second generic “world quality” leaderboard.

It keeps these axes separate:

1. **Exact anchor fidelity** — recovered source camera.
2. **Held-out local novel-view fidelity** — calibrated real neighboring views.
3. **Revisit-selective memory** — relative revisit controls + explicit state checks.
4. **Semantic persistence** — stable identity, attributes and application state.
5. **Edit locality** — target success vs collateral drift.
6. **Runtime / portability** — generation time, memory, asset size, renderability/export.

Protocol: [`docs/benchmark.md`](docs/benchmark.md).  
Dataset policy: [`docs/datasets.md`](docs/datasets.md).

## Frozen bootstrap data

The first public Type-B bootstrap uses BlendedMVS under its published CC BY 4.0 license.

Scene selection was frozen **before** running any baseline:

```text
all 7 official validation scenes in official order
+ first 3 non-validation scenes from the official master list
= 10 scenes
```

For each scene, the first published `pair.txt` reference becomes the single input image and its listed source views become held-out candidates.

Manifest:

```text
datasets/blendedmvs-bootstrap-v0.json
```

This is a pipeline-debugging set, not yet a complete domain benchmark for futuristic reflective/biophilic architecture.

## Experiments

- **EXP-000 — WorldGen open baseline:** unchanged pinned local baseline, then `ml-sharp` quality variant.
- **EXP-001 — Marble external comparison:** optional apples-to-apples comparison using the same inputs/schema.
- **EXP-002 — RefWorld-0:** controlled warp + repaint neighboring views + canonical reconstruction + source-anchor constraint.
- **EXP-003 — Progressive near-to-far completion:** compare one-shot vs staged hidden-space commitment.
- **EXP-004 — Semantic persistence:** stable entity IDs/state/edit locality over leave-and-return.
- **EXP-005 — Uncertainty frontier:** retain unresolved hidden state until evidence/interaction forces commitment.

Roadmap: [`research/roadmap.md`](research/roadmap.md).

## Current executable scaffold

Python:

- deterministic source-image metrics;
- curve summaries;
- R2M-style MemoryGain / Dynamic Range / NMR primitives;
- canonical OpenGL camera representation and perturbations;
- OpenCV-PnP camera registration with explicit convention conversion;
- MVSNet/BlendedMVS parser + deterministic preparation command;
- strict JSON report writer + schema validation;
- renderer-neutral baseline adapter protocol;
- local WorldGen runner/adapter;
- optional secret-safe Marble API/export adapter.

Renderer:

- Spark `2.1.0`;
- Three.js `0.180.0`;
- Playwright `1.62.1`;
- PLY/SPZ shared asset path;
- DPR 1;
- antialias off;
- explicit canonical camera payload;
- projection math separated and unit-tested.

Useful commands:

```bash
python -m pytest
cd renderer && npm test
refworld-validate-report examples/synthetic-report.json
refworld-prepare-blendedmvs /path/to/BlendedMVS
refworld-worldgen-run --help
```

The current ChatGPT execution environment is CPU-only, so heavy open model inference must run in a CUDA-capable environment. That is a **compute boundary, not an API boundary**.

## What would count as a real result?

Not “it looks cool.”

A useful result looks like:

> On the frozen calibrated set, unchanged WorldGen and RefWorld-0 have similar source scores at `C0`, but RefWorld-0 degrades more slowly across real held-out camera displacement without worsening camera-registration residual; the improvement survives repeated seeds and does not come from source-view overfitting.

A useful negative result is equally valid:

> The open baseline already performs strongly on source/local-view fidelity; method work narrows to explicit semantics/edit persistence rather than building another visual generator.

## Key prior art / starting points

- Open stack decision: [`docs/open-stack.md`](docs/open-stack.md)
- WorldGen: https://github.com/ZiYang-xie/WorldGen
- WorldForge: https://github.com/Westlake-AGI-Lab/WorldForge
- HY-World 2.0: https://github.com/Tencent-Hunyuan/HY-World-2.0
- R2M-Bench: https://arxiv.org/abs/2608.27328
- ViewBench / ViewRope: https://arxiv.org/abs/2602.07854
- Closing the Loop: https://arxiv.org/abs/2607.21848
- WorldExam: https://arxiv.org/abs/2608.02603
- 4DWorldBench: https://openaccess.thecvf.com/content/CVPR2026/html/Lu_4DWorldBench_A_Comprehensive_Evaluation_Framework_for_3D4D_World_Generation_Models_CVPR_2026_paper.html
- World Labs Marble/API: https://docs.worldlabs.ai/
- Google DeepMind Genie: https://deepmind.google/models/genie/
- VGGT: https://github.com/facebookresearch/vggt
- Depth Anything 3: https://github.com/ByteDance-Seed/Depth-Anything-3

## License

MIT for original code and documentation in this repository. External repositories, model checkpoints, datasets, generated artifacts and services retain their own licenses and access terms. A permissive top-level repository license must not be assumed to cover its full dependency/checkpoint closure.
