# Dataset candidates and redistribution policy

RefWorldBench needs calibrated or registered held-out views, but the repository must not quietly redistribute images whose licenses do not permit it.

This document records the initial dataset triage. It is not legal advice; contributors remain responsible for complying with each dataset's current terms.

## Recommended public bootstrap: BlendedMVS / BlendedMVG

The official BlendedMVS repository states that **BlendedMVS and BlendedMVG are licensed under CC BY 4.0**.

Why it is useful:

- multi-view imagery and reconstruction-oriented scene data;
- known use in multi-view stereo evaluation;
- permissive attribution-based public license;
- enough geometric structure to debug camera recovery and AFC implementation.

Limitations:

- it is not a perfect proxy for photorealistic architectural concept imagery;
- benchmark results on it should not be generalized to reflective interiors, foliage-heavy spaces, or highly stylized images without additional strata.

Source: https://github.com/YoYo000/BlendedMVS

## Research-only / noncommercial candidate: CO3D

CO3D contains roughly 1.5M camera-annotated frames from nearly 19k object videos with point-cloud annotations for a subset. Its dataset agreement is **CC BY-NC 4.0**.

Useful for:

- controlled object-centric camera paths;
- occlusion/revisit tests;
- semantic identity experiments;
- local novel-view consistency around salient objects.

Policy for this repo:

- do not make CO3D a required dependency for the public/commercially-neutral benchmark core;
- keep it as an optional noncommercial research adapter and clearly propagate attribution/license requirements.

Source: https://ai.meta.com/datasets/co3d-downloads/

## Restricted-access candidate: ScanNet++

ScanNet++ provides 1000+ high-fidelity indoor scenes with laser scans, DSLR imagery, RGB-D streams, poses, and novel-view-synthesis benchmarks. Access requires an application and acceptance of the ScanNet++ Terms of Use.

Useful for:

- high-quality indoor architecture;
- geometry + appearance evaluation;
- large room-scale loop trajectories.

Policy for this repo:

- do not redistribute ScanNet++ data;
- an adapter may consume a user-provided local ScanNet++ root after the user has obtained access under its terms.

Source: https://scannetpp.mlsg.cit.tum.de/scannetpp/

## Geometry test candidate: ETH3D

ETH3D is widely used for calibrated multi-view geometry. The open-source dataset-pipeline code has a permissive BSD-style license, but code licensing must not be conflated with the terms covering the imagery itself.

Policy:

- verify the current dataset terms separately before redistributing any image subset;
- until then, support local-path evaluation only.

Source: https://www.eth3d.net/

## Reference-image-specific evaluation

The visual references that motivated this project may be copyrighted images from architecture/design portfolios. They can be useful **privately** for exploratory evaluation, but they should not be committed to this repository unless redistribution rights are explicit.

For public benchmark scenes that resemble those difficult cases, prefer one of:

1. self-captured multi-view sequences with explicit contributor permission;
2. CC BY / CC0 imagery with calibrated neighboring views;
3. synthetic scenes built entirely from redistribution-compatible assets;
4. licensed datasets used through local adapters without copying their files into the repo.

## v0 dataset plan

Start small:

- 5–10 BlendedMVS scenes to validate camera recovery + AFC machinery;
- a tiny contributor-captured interior set with known camera path, if available;
- optional CO3D object sequences for semantic persistence experiments;
- optional ScanNet++ adapter for stronger room-scale evaluation.

Only expand after the metrics demonstrate that they separate meaningful failure modes.
