# EXP-006 deterministic interaction/revisit trace v0

This slice closes the next product-handoff gap after R1 earned a **narrow visual pass** for the frozen tiny neighboring-view motion.

It does not change the R1 renderer, reference binding, semantic schema, LifeOS mapping, layer geometry, camera, or benchmark evidence.

## Frozen input

- reference: private owner image `LifeOS Studio: Collaborative Futures.png`
- SHA-256: `4ee7a137e577378a02600ac8a32dc89a7c8409120273622227ad972cc5aff61a`
- size: `1672 x 941`
- canonical binding: `examples/exp006_collaborative_futures_binding.json`
- renderer: merged `refworld.exp006.layered-proxy-v1`
- neighboring motion: `tx = ±0.04`
- deliberate out-of-view camera: `tx = +1.60`
- out-of-view target: `lifeos.project.xuxi-room`

`tx=+1.60` is not a visual-quality acceptance view. It exists only to create an explicit identity-continuity test in which one semantic entity's authored image layer is fully outside the viewport.

## Frozen 12-step trace

1. load hero camera;
2. score hero before edit;
3. visit neighbor-left;
4. visit neighbor-right;
5. inspect and record all three native semantic entities;
6. move to `tx=+1.60` and require `lifeos.project.xuxi-room` visibility to reach zero;
7. return and verify camera navigation did not mutate canonical semantic state;
8. apply the already-bound edit `lifeos.system.world-model.panel_mode = project-focus`;
9. measure target change, unrelated semantic drift, and hero visual collateral drift;
10. navigate away and return, requiring the edit to survive;
11. serialize/reload canonical state, requiring stable IDs and persisted edit;
12. return to hero and re-score reference fidelity.

## Pass rule

The automated interaction gate passes only if all are true:

- exactly the 12 frozen steps are recorded;
- pre-edit hero is an exact reference match;
- both small neighboring views retain at least `0.90` observed support;
- the predeclared project entity becomes fully out of view at the away camera;
- returning from camera navigation leaves semantic state unchanged;
- the bounded target edit changes the target;
- collateral semantic drift count is exactly `0`;
- hero visual collateral drift caused by the semantic-only edit is exactly `0`;
- the edit survives away/return navigation;
- IDs remain stable after deterministic snapshot/reload;
- the edit survives reload;
- final hero is an exact reference match;
- final hero equals pre-edit hero;
- rank-4 remains untouched and no metric-reconstruction claim is introduced.

## Important limitation

R1 does **not** yet render the semantic `panel_mode` value into pixels. The bounded edit is therefore a persistent semantic-state proof, not a visible UI-state transition. The trace reports `semantic_edit_visually_mapped_in_r1: false` explicitly.

This is acceptable for this v0 handoff trace because EXP-006 requires stable native IDs/state, persistence, edit locality and reference-anchor fidelity; it must not hide renderer coupling gaps. A later product slice may bind selected application state to appearance, but that should be a new explicit claim rather than silently added here.

## Output package

The owner runner writes:

- `EXP006-INTERACTION-TRACE-V0.json`
- `interaction-trace.jsonl`
- `hero-before.png`
- `neighbor-left.png`
- `neighbor-right.png`
- `away.png`
- `hero-after.png`
- `world-before.json`
- `world-edited.json`
- `world-reloaded.json`

The private reference image remains outside the repository.
