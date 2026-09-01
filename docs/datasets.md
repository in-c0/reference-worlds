# Dataset candidates and redistribution policy

RefWorldBench needs calibrated or registered held-out views, but the repository must not quietly redistribute images whose licenses do not permit it.

This document records the initial dataset triage and the frozen v0 bootstrap split. It is not legal advice; contributors remain responsible for complying with each dataset's current terms.

## Public bootstrap: BlendedMVS

The official BlendedMVS repository states that **BlendedMVS and BlendedMVG are licensed under CC BY 4.0**. The original BlendedMVS set contains 113 scenes; the low-resolution set is 768×576 and uses MVSNet input format with per-image cameras, `pair.txt`, images and rendered depth maps.

Official source: https://github.com/YoYo000/BlendedMVS

Why it is useful:

- calibrated multi-view imagery;
- per-view MVSNet camera files;
- ranked overlapping source views in `pair.txt`;
- rendered depth maps;
- permissive attribution-based public license;
- enough geometric structure to debug camera recovery and held-out novel-view evaluation.

Limitations:

- it is not a perfect proxy for photorealistic architectural concept imagery;
- reconstruction units should not be called meters unless scale is independently established;
- the highest-overlap `pair.txt` neighbors are arbitrary camera poses, not controlled pure-yaw/pitch perturbations;
- results should not be generalized to mirrors, glass-heavy interiors, dense foliage or highly stylized concept art without additional strata.

### Frozen 10-scene bootstrap split

Manifest: [`../datasets/blendedmvs-bootstrap-v0.json`](../datasets/blendedmvs-bootstrap-v0.json)

The split was frozen **before any Marble outputs were evaluated**.

Selection rule:

1. take every scene in upstream `project_lists/validation_list.txt` in published order;
2. that official validation list contains seven scenes;
3. append the first scenes in upstream `project_lists/BlendedMVS.txt` that are not already in the validation list until there are 10 scenes;
4. never replace a scene because a model result is inconvenient or visually poor.

The resulting scene IDs are:

1. `5b7a3890fc8fcf6781e2593a`
2. `5c189f2326173c3a09ed7ef3`
3. `5b950c71608de421b1e7318f`
4. `5a6400933d809f1d8200af15`
5. `59d2657f82ca7774b1ec081d`
6. `5ba19a8a360c7c30c1c169df`
7. `59817e4a1bd4b175e7038d19`
8. `5c1f33f1d33e1f2e4aa6dda4`
9. `5bfe5ae0fe0ea555e6a969ca`
10. `5bff3c5cfe0ea555e6bcbf3a`

Upstream list Git blob SHAs are recorded in the manifest so future upstream edits cannot silently change the bootstrap selection.

### Deterministic per-scene view rule

For each scene:

1. parse `cams/pair.txt`;
2. take the **first published reference record** as the single image supplied to the world generator;
3. retain every source view listed for that record as held-out candidates;
4. parse the corresponding `*_cam.txt` files;
5. compute the actual relative camera geometry from calibration;
6. evaluate each held-out render against its observed image.

Do not select the anchor or held-out views by inspecting generated results.

MVSNet `pair.txt` stores each reference image followed by its top-ranked source views and scores. Its camera files store `E=[R|t]`, intrinsics and depth range. The official COLMAP→MVSNet converter writes COLMAP world→camera extrinsics; RefWorldBench converts those OpenCV-style world→camera matrices explicitly to its canonical OpenGL camera→world convention.

For arbitrary real neighboring views, report axes such as:

- `view_direction_angle_deg`;
- `center_distance_source_units`.

Do **not** relabel arbitrary neighboring views as `yaw_deg`. Do **not** use `translation_m` unless physical scale is independently verified.

Code: `src/refworld/datasets/mvsnet.py`.

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

- verify current dataset terms separately before redistributing any image subset;
- until then, support local-path evaluation only.

Source: https://www.eth3d.net/

## Reference-image-specific evaluation

The visual references that motivated this project may be copyrighted architecture/design portfolio images. They can be useful privately for exploratory evaluation, but they should not be committed to this repository unless redistribution rights are explicit.

For public benchmark scenes that resemble those difficult cases, prefer:

1. self-captured multi-view sequences with explicit contributor permission;
2. CC BY / CC0 imagery with calibrated neighboring views;
3. synthetic scenes built entirely from redistribution-compatible assets;
4. licensed datasets used through local adapters without copying their files into the repo.

## v0 dataset plan

1. **BlendedMVS bootstrap 10** — pipeline correctness, camera conversion, source/held-out visual metrics and Marble falsification.
2. **Small rights-cleared indoor/biophilic set** — closer to the motivating architectural domain, with deliberately captured calibrated neighboring views.
3. Optional CO3D — object-centric semantic persistence.
4. Optional ScanNet++ — room-scale high-fidelity evaluation through local-only adapter.

Only expand after metrics demonstrate useful separation between systems.
