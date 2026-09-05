# EXP-006 follow-up: semantic state-driven visual binding v0

## Purpose

EXP-006 passed the bounded LifeOS handoff gate, but its canonical semantic edit was intentionally not rendered as a visible state change. This follow-up closes that one coupling gap before any preview-only LifeOS spatial tier is wired into a product surface.

Tracker: issue #26.

## Frozen state transition

Target entity:

`lifeos.system.world-model`

Canonical state field:

`panel_mode: overview -> project-focus`

The LifeOS/canonical world state remains authoritative. The visual adapter consumes that state; it does not own or invent project truth.

## Frozen visual method

- accepted Collaborative Futures reference, SHA-256 `4ee7a137e577378a02600ac8a32dc89a7c8409120273622227ad972cc5aff61a`;
- accepted R1 renderer is imported unchanged;
- no camera, layer polygon/depth, neighbor translation or provenance threshold is retuned;
- a deterministic UI patch is authored inside the existing `world-model` layer support;
- the patch moves by the accepted world-model layer pixel shift in the two R1 neighbor cameras;
- only `panel_mode=project-focus` activates the patch;
- `panel_mode=overview` delegates to unmodified R1 output;
- state-driven changed pixels are classified as `state-generated-edit` and explicitly removed from OBSERVED provenance;
- base R1 hypothesized support remains separate;
- no generative inpainting and no metric-geometry claim.

Source patch rectangle, normalized to the accepted reference:

`(x0=0.535, y0=0.245, x1=0.695, y1=0.325)`

It is clipped to the accepted `world-model` polygon before projection.

## Frozen automated gate

All checks must pass:

1. pre-edit hero equals the frozen reference pixel-for-pixel;
2. the canonical target semantic state changes;
3. collateral semantic drift is exactly zero;
4. edited hero contains at least one visible changed pixel;
5. changed pixels outside the declared target support are exactly zero in hero and both neighbors;
6. changed pixels still classified OBSERVED are exactly zero in hero and both neighbors;
7. all pixels outside declared target support remain bitwise unchanged in hero and both neighbors;
8. snapshot/reload preserves `panel_mode=project-focus` and re-derives the identical edited hero;
9. reverting `panel_mode=overview` restores the exact reference hero;
10. both accepted neighboring cameras show a non-empty state-driven visual change.

No threshold may be weakened after seeing output.

## Human gate

Even if the automated gate passes, inspect:

- `hero-edited.png`;
- `neighbor-left-edited.png`;
- `neighbor-right-edited.png`.

The edit should read as a deliberate stateful UI change attached to the world-model surface. Reject if it appears detached, crosses unrelated scene content, becomes visually dominant, or destroys the accepted neighboring-view continuity.

## Evidence boundary

This follow-up does not:

- consume fresh benchmark evidence;
- touch BlendedMVS rank-4;
- alter accepted R1 renderer code;
- change the semantic schema or LifeOS mapping;
- claim arbitrary semantic-to-visual mapping;
- claim general 3D reconstruction;
- constitute production LifeOS integration.

A pass means only that one canonical LifeOS state field can deterministically drive one bounded visible change while reference/provenance discipline is preserved.
