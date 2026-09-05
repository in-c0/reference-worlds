# EXP-006 Collaborative Futures reference slice

This document freezes the first concrete LifeOS product slice after the bounded EXP-002 learned-geometry lane closed negative.

## Reference

- owner-selected reference: `LifeOS Studio: Collaborative Futures.png`
- source location: private owner ChatGPT library; binary is **not committed**
- dimensions: `1672 x 941`
- SHA-256: `4ee7a137e577378a02600ac8a32dc89a7c8409120273622227ad972cc5aff61a`
- provenance role: visual specification only; it is not assumed to depict a hidden real building

The image was selected because one frame contains all three EXP-006 semantic roles: a central architectural collaboration surface, an explicit XUXI Room project surface, and a system/world-model instrument in one shared atrium/research-workspace composition.

## Bound semantic entities

1. `lifeos.atrium.collaboration-table` — architectural/work-surface role.
2. `lifeos.project.xuxi-room` — project/artifact role.
3. `lifeos.system.world-model` — system/instrument role.

All three are marked `observed` only because they are visibly constrained by the supplied reference. Their 3D geometry remains `hypothesized`; this binding does **not** claim recovered metric geometry.

The project entity maps to the real implementation record `https://github.com/in-c0/lifeos-local-ai/pull/113`. The mapping deliberately does not copy project status, evidence, disclosure or permissions into RefWorld state. Those fields remain authoritative on the LifeOS/GitHub side.

## Frozen R0 renderer hypothesis

Do not return to monocular model shopping. EXP-006 R0 uses an **authored layered anchor proxy**:

- renderer id: `refworld.exp006.layered-proxy-v0`;
- authored hero camera: identity pose, `60.0°` horizontal FOV;
- hero intrinsics for 1672×941: `fx = fy = 1447.9944751275816`, `cx = 835.5`, `cy = 470.0`;
- neighboring proxy camera translations: `tx = -0.04` and `tx = +0.04`;
- coarse fronto-parallel layers are manually fixed from the selected composition before output review;
- the owner reference is projected as explicit `observed` appearance on those proxy layers;
- disoccluded pixels are filled with a fixed unknown colour and remain non-observed; there is no inpainting in R0;
- renderer assets cannot own semantic IDs, project truth or application authority;
- no metric reconstruction claim is made.

This is an image-based persistent-world product hypothesis, not an extension of G1 learned geometry.

## Automated R0 gate

Before any output review, the automated gate is frozen as all of:

1. reference SHA-256 matches the bound value;
2. hero frame is an exact pixel match to the reference;
3. both neighboring views retain at least `0.90` observed-pixel coverage;
4. the bounded semantic edit changes its target;
5. entity ID set remains stable after snapshot/reload;
6. collateral semantic drift count is exactly `0`;
7. no rank-4 BlendedMVS target is touched.

Passing this gate does **not** establish visual success. Human review of both neighboring views is mandatory because coverage cannot detect ugly parallax, duplicated semantics or implausible proxy structure.

## Full first renderer gate

Before any broader scene generation, require one deterministic run that demonstrates:

1. the reference hash is verified before rendering;
2. the authored hero camera is frozen and reproducible;
3. the hero view renders with an explicit observed/unknown provenance mask;
4. two small neighboring viewpoints render without catastrophic structural failure under human review;
5. the three entity IDs survive neighboring views and an occlusion/revisit trace;
6. `lifeos.system.world-model.panel_mode` changes from `overview` to `project-focus` with zero unrelated semantic drift;
7. snapshot/reload preserves the edit and IDs;
8. returning to the hero camera preserves reference-anchor fidelity;
9. no rank-4 BlendedMVS target is consumed.

The first renderer implementation may fail this gate. A failure should be attributed to proxy geometry, anchor projection, view continuity, provenance separation, edit locality or persistence—not repaired by silently changing the reference or relabeling generated support as observed.
