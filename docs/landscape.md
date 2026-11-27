# Research landscape — September 2026

This document tracks technologies directly relevant to **reference-anchored persistent world synthesis**. The goal is not to reimplement the world-model stack; it is to identify a defensible gap after current systems already generate, revisit and evaluate impressive worlds.

For focused benchmark/memory prior art, see [`literature.md`](literature.md).

## Frontier baseline: persistent generated worlds

### World Labs Marble

Marble is the closest production baseline to this project. World Labs describes it as generating high-fidelity, persistent 3D worlds from a single image, multiple images, video, text, or coarse 3D structure. Generated worlds can be explored persistently and exported as Gaussian splats; Marble also exposes collider meshes and high-quality mesh export.

For RefWorldBench, this makes Marble the first system to falsify the project against. If Marble already preserves a supplied observation strongly after camera registration and on calibrated held-out nearby views, while its persistent representation is sufficient for application semantics/editing, a new visual world-generation method may be unnecessary.

The key question is not whether Marble makes an explorable world; it already does. The question is how an actual source observation behaves as a **measured anchor** after generation/export/editing and whether persistent application state can be attached without collateral visual/semantic drift.

Sources:
- https://docs.worldlabs.ai/
- https://docs.worldlabs.ai/api
- https://docs.worldlabs.ai/marble/export/gaussian-splat/index
- https://docs.worldlabs.ai/marble/export/mesh

### Google DeepMind Genie 3

Genie 3 establishes a different frontier: real-time interactive visual world simulation. Its output/state model differs from an exportable explicit 3D reconstruction.

RefWorldBench therefore treats Genie-class systems as evidence that long-horizon interactive visual consistency is achievable while separately asking what can be measured about canonical/editable/addressable world state.

Sources:
- https://deepmind.google/models/genie/
- https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/

## Geometry priors

### VGGT / VGGT-Omega

VGGT predicts camera parameters, point maps, depth maps, and 3D point tracks from one to many views. It is a strong candidate for recovering the source camera and obtaining a geometry prior before any anchor-constrained optimization.

Source: https://github.com/facebookresearch/vggt

### Depth Anything 3

Depth Anything 3 predicts spatially consistent geometry from arbitrary visual inputs, with or without known poses, using a unified depth-ray representation. It is another camera/geometry prior and includes streaming support for long sequences.

Source: https://github.com/ByteDance-Seed/Depth-Anything-3

## Object and material reconstruction

### SAM 3D Objects

SAM 3D Objects reconstructs full object geometry, texture and pose/layout from a single image, including occluded/cluttered cases. It can provide entity-level candidates for a semantic overlay, though its role is object reconstruction rather than persistent whole-world synthesis.

Source: https://github.com/facebookresearch/sam-3d-objects

### TRELLIS.2

TRELLIS.2 is an image-to-3D asset model producing high-resolution geometry with PBR attributes. This is relevant when semantic entities need explicit editable assets rather than only radiance-field appearance.

Source: https://github.com/microsoft/TRELLIS.2

## Runtime representations

### Gaussian splats

Marble exports SPZ/PLY Gaussian splats and documents integration with Spark/Three.js and DCC/game-engine tooling. Splats are attractive for preserving high-frequency appearance, but they do not by themselves solve semantic identity, edit locality, collision semantics or hidden-space uncertainty.

### Mesh / hybrid world representation

A likely persistent application architecture is hybrid: splat/radiance appearance for visual fidelity; mesh/collider geometry for interaction; and a separate semantic world graph for stable object identity/application state.

RefWorldBench should not assume this is the only architecture. It should measure exposed capabilities rather than reward a representation by name.

## Existing evaluation work

### 4DWorldBench — CVPR 2026

4DWorldBench evaluates world-generation models across perceptual quality, condition-to-4D alignment, physical realism and 4D consistency over image/video/text-conditioned 3D/4D tasks.

Therefore RefWorldBench must **not** claim to be a first/general world-generation benchmark.

Source: https://openaccess.thecvf.com/content/CVPR2026/html/Lu_4DWorldBench_A_Comprehensive_Evaluation_Framework_for_3D4D_World_Generation_Models_CVPR_2026_paper.html

### WorldExam — scene revisit and 3D consistency

WorldExam already includes **Scene Revisit** and **3D Consistency** in a broader hierarchy spanning visual quality, control adherence, spatial consistency and world reactivity.

Source: https://arxiv.org/abs/2608.02603

### ViewBench / ViewRope — loop closure and geometric drift

ViewBench provides controlled rotate-away/return and rotation+translation loop trajectories with camera poses/depth-overlap information. The associated ViewRope method addresses long-horizon view consistency using geometry-aware attention/position encoding.

Sources:
- https://arxiv.org/abs/2602.07854
- https://github.com/jedward225/viewbench-dataset

### R2M-Bench — relative revisit memory

R2M-Bench demonstrates that raw first-visit ↔ revisit similarity is confounded by generic temporal stability and failed/slow motion. It compares revisits to same-rollout controls and introduces MemoryGain and Normalized Memory Ratio.

RefWorldBench should reproduce/import this logic for video/interactive revisit scoring rather than claim raw loop similarity as a contribution.

Sources:
- https://arxiv.org/abs/2608.27328
- https://github.com/AMAP-ML/R2MBench

### Closing the Loop — revisit consistency as a method

Closing the Loop uses pose-matched historical latent retrieval plus geometric correspondences to improve long-horizon revisit consistency without retraining.

Source: https://arxiv.org/abs/2607.21848

### Ref4D-VideoBench — reference-based evaluation

Ref4D-VideoBench uses reference videos for fine-grained video evaluation. This means “reference-based evaluation” itself is not a defensible novelty claim.

Source: https://openaccess.thecvf.com/content/CVPR2026/html/Wei_Ref4D-VideoBench_Four-Dimensional_Reference-Based_Evaluation_of_Text-to-Video_Generative_Models_CVPR_2026_paper.html

### InfiniteNature-Zero — cyclic trajectories are older prior art

InfiniteNature-Zero used virtual camera trajectories including cyclic ones to train stable perpetual view generation from single-image collections.

Source: https://arxiv.org/abs/2207.11148

## The remaining candidate gap

After the 2026 literature pass, the most defensible hypothesis is not a single isolated capability. It is a joint protocol/application requirement:

> **Can a generated persistent world preserve an actual supplied observation as a calibrated visual anchor under real 3D camera displacement, while also maintaining explicit semantic identity/state and localized edits over navigation/revisit?**

The candidate measurement intersection is:

1. exact source-view reconstruction after independent camera registration;
2. held-out calibrated local novel-view fidelity as camera displacement increases;
3. R2M-style relative revisit memory rather than naive return similarity;
4. explicit semantic entity/state persistence where available;
5. edit locality and collateral visual/semantic drift;
6. anchor preservation after world expansion/editing;
7. representation/runtime portability for persistent applications.

The working Anchor Fidelity Curve is a **diagnostic**, not yet a novelty claim. A method contribution is justified only if established systems fail this narrower joint test in a systematic, reproducible way.
