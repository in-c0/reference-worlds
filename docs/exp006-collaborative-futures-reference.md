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

Do not return to monocular model shopping. EXP-006 R0 used an **authored layered anchor proxy**:

- renderer id: `refworld.exp006.layered-proxy-v0`;
- authored hero camera: identity pose, `60.0°` horizontal FOV;
- hero intrinsics for 1672×941: `fx = fy = 1447.9944751275816`, `cx = 835.5`, `cy = 470.0`;
- neighboring proxy camera translations: `tx = -0.04` and `tx = +0.04`;
- coarse fronto-parallel layers were manually fixed from the selected composition before output review;
- the owner reference was projected as explicit `observed` appearance on those proxy layers;
- disoccluded pixels were filled with a fixed unknown colour and remained non-observed; there was no inpainting in R0;
- renderer assets could not own semantic IDs, project truth or application authority;
- no metric reconstruction claim was made.

This is an image-based persistent-world product hypothesis, not an extension of G1 learned geometry.

## R0 outcome

Owner execution produced:

- hero exact reference match: `true`;
- neighbor observed fractions: left `0.9910`, right `0.9910`;
- stable IDs after bounded edit + snapshot/reload: `true`;
- collateral semantic drift: `0`;
- automated gate: `PASS`;
- rank-4: sealed and untouched.

Human visual review failed R0. Both neighboring images showed obvious internal vertical/diagonal black tears and a bottom horizontal seam. The failure was traced to source ownership carving: foreground polygons were removed from the background source mask, then foreground/background planes moved by different parallax shifts, exposing the carved polygon boundaries as black unknown support inside the scene.

Classification: **R0 automated PASS, human visual FAIL**. Do not merge/accept R0 as a visual pass.

## Frozen R1 repair

R1 is intentionally narrower than a geometry retune. The following remain unchanged from R0:

- same bound reference image and hash;
- same hero camera and `60.0°` horizontal FOV;
- same `tx = ±0.04` neighboring camera translations;
- same authored layer polygons;
- same authored layer depths;
- same semantic bindings and bounded edit;
- same `>= 0.90` minimum observed-pixel coverage gate;
- same no-rank4 / no-metric-reconstruction boundary.

Only display/provenance separation changes:

- renderer id becomes `refworld.exp006.layered-proxy-v1`;
- the display starts from a full-reference background proxy shifted at the frozen background depth instead of a background texture with foreground holes carved out;
- pixels revealed from underneath foreground layers or from offscreen edge padding are **display hypotheses**, not observed evidence;
- foreground RGB edges are feathered with a fixed `3.0 px` Gaussian radius to remove hard slice boundaries;
- alpha-transition pixels are conservatively excluded from observed support unless their shifted hard mask has alpha `>= 0.95`;
- any pixel affected by a foreground alpha `> 0.05` is first removed from the prior observed claim before near-hard foreground support may restore it;
- R1 writes both `observed-mask` and complementary `hypothesized-mask` images;
- no generative inpainting is used.

This means visual continuity is allowed to use a deterministic hypothesis, while the provenance mask remains the authority for what is actually supported by the reference projection.

## Automated R1 gate

Before human output review, R1 must satisfy all of:

1. reference SHA-256 matches the bound value;
2. hero frame is an exact pixel match to the reference;
3. both neighboring views retain at least `0.90` observed-pixel coverage under the stricter feather-aware provenance mask;
4. the bounded semantic edit changes its target;
5. entity ID set remains stable after snapshot/reload;
6. collateral semantic drift count is exactly `0`;
7. no rank-4 BlendedMVS target is touched.

Passing this gate still does **not** establish visual success.

## Human R1 visual gate

Review both neighboring RGB views at normal scale. R1 passes visually only if:

- the R0 black internal seams/diagonal wedges are absent;
- there are no obvious hard layer-slice boundaries through the atrium, people, table or HUD surfaces;
- no large duplicated/ghosted structures appear from the hypothesized background proxy;
- the central collaboration table, world-model HUD and architecture remain structurally readable under both small camera moves;
- remaining hypothesized support is acceptable as a preview and remains explicitly visible in the separate provenance mask rather than being relabeled observed.

A human failure after R1 should identify the remaining visual mechanism before any further change. Do not silently retune geometry, change the reference, or consume rank-4 evidence.

## Full first renderer gate

Before any broader scene generation, require one deterministic run that demonstrates:

1. the reference hash is verified before rendering;
2. the authored hero camera is frozen and reproducible;
3. the hero view renders with explicit observed/hypothesized provenance;
4. two small neighboring viewpoints render without catastrophic structural failure under human review;
5. the three entity IDs survive neighboring views and an occlusion/revisit trace;
6. `lifeos.system.world-model.panel_mode` changes from `overview` to `project-focus` with zero unrelated semantic drift;
7. snapshot/reload preserves the edit and IDs;
8. returning to the hero camera preserves reference-anchor fidelity;
9. no rank-4 BlendedMVS target is consumed.

A failure should be attributed to proxy geometry, anchor projection, view continuity, provenance separation, edit locality or persistence—not repaired by silently changing the reference or relabeling generated support as observed.
