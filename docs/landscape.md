# Research landscape — September 2026

This document tracks technologies directly relevant to **reference-anchored persistent world synthesis**. The goal is not to reimplement the entire world-model stack; it is to identify the gap left after current systems already generate impressive worlds.

## Frontier baseline: persistent generated worlds

### World Labs Marble

Marble is the closest production baseline to this project. World Labs describes it as generating high-fidelity, persistent 3D worlds from a single image, multiple images, video, text, or coarse 3D structure. Generated worlds can be explored persistently and exported as Gaussian splats; Marble also exposes collider meshes and high-quality GLB mesh export.

For RefWorldBench, this makes Marble the first system to falsify the project against. If Marble already preserves a source image extremely closely under controlled viewpoint perturbations, loop closure, revisits, and localized edits, then a new world-generation method may be unnecessary.

What is not currently exposed as a standard evaluation target is an explicit **source-reference anchor constraint**: how accurately does the generated world reproduce the exact input observation, how rapidly does that agreement decay away from the recovered source camera, and does it remain stable after expansion/editing/revisit?

Sources:
- https://docs.worldlabs.ai/
- https://docs.worldlabs.ai/api
- https://docs.worldlabs.ai/marble/export/gaussian-splat/index
- https://docs.worldlabs.ai/marble/export/mesh

### Google DeepMind Genie 3

Genie 3 establishes a different frontier: real-time interactive world simulation. Google reports 20–24 FPS at 720p, with consistency over sustained interaction and recall of previously seen details.

Its output model is conceptually different from Marble/exportable reconstruction. RefWorldBench therefore treats Genie 3 primarily as evidence that long-horizon interactive visual consistency is achievable, while asking whether comparable systems expose a canonical, editable, addressable world state suitable for persistent applications.

Sources:
- https://deepmind.google/models/genie/
- https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/

## Geometry priors

### VGGT / VGGT-Omega

VGGT predicts camera parameters, point maps, depth maps, and 3D point tracks from one to many views. It is a strong candidate for recovering the source camera and obtaining a geometry prior before anchor-constrained optimization. The project page also points to VGGT-Omega as its 2026 successor.

Source: https://github.com/facebookresearch/vggt

### Depth Anything 3

Depth Anything 3 predicts spatially consistent geometry from arbitrary visual inputs, with or without known poses, using a unified depth-ray representation. It is another strong camera/geometry prior and includes a streaming mode for long sequences.

Source: https://github.com/ByteDance-Seed/Depth-Anything-3

## Object and material reconstruction

### SAM 3D Objects

SAM 3D Objects reconstructs full object geometry, texture, pose/layout from a single image and explicitly targets occlusion and clutter. It can provide entity-level candidates for a semantic overlay, though its role is object reconstruction rather than persistent whole-world synthesis.

Source: https://github.com/facebookresearch/sam-3d-objects

### TRELLIS.2

TRELLIS.2 is a 4B-parameter image-to-3D asset model using O-Voxel structured latents and produces high-resolution geometry with PBR surface attributes. This is relevant when persistent semantic entities need explicit editable assets rather than only radiance-field appearance.

Source: https://github.com/microsoft/TRELLIS.2

## Runtime representations

### Gaussian splats

Marble exports SPZ/PLY Gaussian splats and documents integration with Spark/Three.js, Unity, Unreal, Blender, and Houdini. Splats are attractive for preserving high-frequency appearance, but they do not by themselves solve entity identity, edit locality, collision semantics, or hidden-space uncertainty.

### Mesh / hybrid world representation

Marble can export collider meshes and high-quality meshes. A likely RefWorld architecture is hybrid: splat/radiance appearance for visual fidelity; mesh/collider geometry for interaction; and a separate semantic world graph for stable object identity and application state.

## Existing evaluation work

### 4DWorldBench — CVPR 2026

4DWorldBench evaluates world-generation models across perceptual quality, condition-to-4D alignment, physical realism, and 4D consistency over image/video/text-conditioned 3D/4D tasks. It is important prior art and means RefWorldBench should **not** claim to be a general world-generation benchmark.

The narrower proposed contribution is measurement of phenomena that matter specifically when a reference image should behave as a persistent hard anchor:

1. exact source-view reconstruction after world generation;
2. fidelity decay as the camera moves away from the source view;
3. loop/revisit fidelity after long excursions;
4. stable semantic entity identity and state;
5. edit locality and collateral visual drift;
6. anchor preservation after world expansion or editing.

Source: https://openaccess.thecvf.com/content/CVPR2026/html/Lu_4DWorldBench_A_Comprehensive_Evaluation_Framework_for_3D4D_World_Generation_Models_CVPR_2026_paper.html

## The remaining gap

The strongest current systems make the original question — “can one image become an explorable world?” — too broad to be useful. The more defensible question is:

> **Can a generated world treat a supplied observation as a measurable hard anchor while also maintaining stable geometry, appearance, entity identity, edits, and state through arbitrary navigation and revisit?**

This repository starts benchmark-first. A new generation architecture is only justified if existing systems fail that narrower test in a meaningful, reproducible way.
