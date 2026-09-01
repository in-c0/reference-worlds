# Focused literature map — reference fidelity, revisit memory, persistent worlds

Last reviewed: 2026-09-01.

This note exists to keep **Reference Worlds / RefWorldBench** from claiming novelty for problems that the 2026 world-model literature already studies directly.

## Bottom line

The project should **not** claim novelty for:

- generic world-model visual quality evaluation;
- generic 3D/4D consistency evaluation;
- loop-closure / scene-revisit evaluation;
- the observation that absolute revisit similarity can be misleading;
- cyclic camera trajectories as a consistency test.

The still-interesting intersection is narrower:

> **Given an actual supplied observation, how well does a generated persistent world preserve that observation as a calibrated visual anchor under controlled geometric displacement, while also exposing stable semantic identity/state and supporting localized edits without collateral drift?**

The `Anchor Fidelity Curve (AFC)` is therefore a **working diagnostic**, not currently a claimed novel metric. A novelty claim would require a broader search plus experiments showing that existing reference-conditioned/novel-view benchmarks do not already instantiate the same protocol.

---

## R2M-Bench — revisit memory needs relative controls

**R2M-Bench: Evaluating Revisit Memory via Relative Consistency in Interactive Video World Models** (Gu et al., 2026)

- arXiv: https://arxiv.org/abs/2608.27328
- code: https://github.com/AMAP-ML/R2MBench

R2M-Bench directly evaluates revisit-selective memory. Its key warning is important for RefWorldBench: a high first-visit ↔ return similarity score does **not** prove memory if the rollout barely changed or the renderer is generically stable.

R2M-Bench compares each revisit pair against same-rollout controls and introduces:

- **MemoryGain (MG)** — revisit advantage over a gap-matched non-revisit temporal baseline;
- **Normalized Memory Ratio (NMR)** — normalizes that advantage using a short-range consistency reference.

### Consequence for RefWorldBench

Do not use raw loop-return similarity as the primary revisit score. For video/interactive world models, import or adapt the R2M relative-control idea. RefWorldBench may add explicit-3D/state-specific diagnostics, but should not relabel revisit memory as a new contribution.

---

## ViewBench / ViewRope — loop-closure trajectories and geometry-aware consistency

**Geometry-Aware Rotary Position Embedding for Consistent Video World Model** (Xiang et al., 2026)

- arXiv: https://arxiv.org/abs/2602.07854
- OpenReview: https://openreview.net/forum?id=eXgmwOOvlR
- ViewBench dataset: https://github.com/jedward225/viewbench-dataset

The work identifies long-horizon spatial drift in camera-controlled video world models and proposes a geometry-aware positional encoding (ViewRope). Its associated **ViewBench** explicitly measures loop-closure fidelity/geometric drift using controlled camera trajectories.

The public ViewBench release includes:

- pure rotation return trajectories;
- rotation + translation exploration trajectories;
- per-frame camera-to-world poses;
- depth-based geometric overlap annotations.

Its release is non-commercial / CC BY-NC-style and contains third-party UE5-rendered content, so it is useful as research prior art and possibly a local comparison set, but should not silently become the redistributable default RefWorldBench corpus.

### Consequence for RefWorldBench

Reuse the trajectory concepts and compare protocols. The distinct question is not “does the model close a loop?” but whether a **specific source observation remains a calibrated anchor** and whether explicit persistent state survives the loop.

---

## Closing the Loop — revisit consistency as a method, not only a metric

**Closing the Loop: Training-Free Revisit Consistency for Autoregressive Generative Rendering** (Ma et al., 2026)

- arXiv: https://arxiv.org/abs/2607.21848
- project: https://wenchao-m.github.io/ClosetheLoop.github.io/

This work targets exactly the long-horizon revisit failure caused by bounded autoregressive context. It retrieves pose-matched historical latent chunks and uses pose/depth correspondence to bias attention toward geometrically corresponding regions.

### Consequence for RefWorldBench

Any proposed “persistent visual memory” mechanism should be compared conceptually against explicit historical retrieval / geometric correspondence approaches. A new method needs to contribute beyond simply caching previously seen appearance.

---

## WorldExam — scene revisit and 3D consistency are already benchmark tasks

**WorldExam: Benchmarking World Models from Apparent Appearance to Inherent Reactivity** (Yang et al., 2026)

- arXiv: https://arxiv.org/abs/2608.02603
- project: https://worldexam.github.io/

WorldExam evaluates controllable video world models across Visual Quality, Control Adherence, Spatial Consistency, and World Reactivity. Its spatial-consistency level already includes **Scene Revisit** and **3D Consistency**.

### Consequence for RefWorldBench

Do not position RefWorldBench as the first benchmark of revisit or world consistency. Instead, position it as a specialized protocol for **reference-conditioned persistent reconstruction/synthesis**, especially when the output is an exportable scene or addressable world rather than only a generated video stream.

---

## 4DWorldBench — general world-generation evaluation is already broad

**4DWorldBench: A Comprehensive Evaluation Framework for 3D/4D World Generation Models** (Lu et al., CVPR 2026)

- paper: https://openaccess.thecvf.com/content/CVPR2026/html/Lu_4DWorldBench_A_Comprehensive_Evaluation_Framework_for_3D4D_World_Generation_Models_CVPR_2026_paper.html

4DWorldBench covers Perceptual Quality, Condition-4D Alignment, Physical Realism, and 4D Consistency across image/video/text-conditioned tasks.

### Consequence for RefWorldBench

RefWorldBench should remain deliberately narrow and diagnostic. It should not collapse into a second general-purpose “world quality” leaderboard.

---

## Ref4D-VideoBench — reference-based evaluation is not itself novel

**Ref4D-VideoBench: Four-Dimensional Reference-Based Evaluation of Text-to-Video Generative Models** (Wei et al., CVPR 2026)

- paper: https://openaccess.thecvf.com/content/CVPR2026/html/Wei_Ref4D-VideoBench_Four-Dimensional_Reference-Based_Evaluation_of_Text-to-Video_Generative_Models_CVPR_2026_paper.html
- code: https://github.com/TAILab-W/Ref4D-VideoBench

Ref4D-VideoBench argues for reference-based evaluation rather than no-reference judging, using reference videos to provide evidence for semantic, motion, event-temporal, and world-knowledge consistency.

### Consequence for RefWorldBench

“Reference-based evaluation” alone is not a novelty claim. Our narrower object is **geometrically calibrated displacement from a supplied source view into a persistent 3D/interactive world**.

---

## InfiniteNature-Zero — cyclic camera trajectories have older precedent

**InfiniteNature-Zero: Learning Perpetual View Generation of Natural Scenes from Single Images** (Li et al., ECCV 2022)

- paper/project: https://research.google/pubs/infinitenature-zero-learning-perpetual-view-generation-of-natural-scenes-from-single-images/
- arXiv: https://arxiv.org/abs/2207.11148

InfiniteNature-Zero already used virtual camera trajectories including cyclic ones to encourage stable view generation from single-image training collections.

### Consequence for RefWorldBench

Cyclic trajectories are useful protocol machinery, not a new research contribution.

---

## Current defensible research gap

After this literature pass, the strongest candidate gap is the **joint requirement**:

1. a real input observation is treated as an explicit measurable anchor;
2. camera displacement from that anchor is calibrated in 3D, not just temporal frame distance;
3. held-out views are used where available to separate source-view overfitting from actual local geometry;
4. revisit metrics use relative controls (R2M-style) rather than raw return similarity alone;
5. persistent outputs expose or are paired with stable semantic entities/state;
6. edits are tested for locality and persistence;
7. exported/explicit world representations are evaluated separately from purely autoregressive video streams.

This is a **hypothesis about an under-measured intersection**, not yet a confirmed novelty claim.

## Publication discipline

Before claiming a new metric or benchmark contribution:

- search specifically for source-view / reference-camera / novel-view fidelity curves;
- compare against novel-view synthesis, inverse-rendering and 3D reconstruction evaluation literature, not only “world model” papers;
- reproduce at least one established revisit metric (e.g. R2M-style relative controls);
- demonstrate that the proposed AFC/reporting protocol changes system rankings or exposes failures not captured by existing metrics;
- publish negative results if Marble or another baseline already satisfies the hypothesized gap.
