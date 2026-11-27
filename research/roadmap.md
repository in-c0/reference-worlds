# Research roadmap

## Phase 0 — establish open baselines before inventing a method

The core R&D must remain executable without a proprietary world-generation API.

### EXP-000 — open local baseline: WorldGen

**Question:** How well does an existing modifiable open pipeline preserve the supplied image once it becomes a persistent 3D world?

Pinned first baseline:

```text
ZiYang-xie/WorldGen
commit 7ce7b2767fdf31e2727b69a2e61e2e950e3a017f
```

Protocol:

1. use the frozen rights-cleared BlendedMVS bootstrap set;
2. run WorldGen image-to-scene locally from the single frozen anchor image;
3. preserve the generated panorama intermediate;
4. export the 3DGS PLY and optional mesh;
5. recover/register the reference camera;
6. render the source camera and calibrated held-out cameras;
7. compute source metrics and held-out displacement/fidelity curves;
8. record generation time, peak VRAM, checkpoints, licenses, git commit and artifact hashes;
9. repeat the same source with `ml-sharp` enabled if that dependency is reproducible.

Important: WorldGen repository code is Apache-2.0, but its documented image-to-scene path currently uses FLUX.1-Fill-dev and other external models. Record the full dependency/checkpoint closure rather than calling the complete runtime Apache-2.0.

Exit criterion:

- If the open baseline already preserves the source + nearby calibrated views strongly, narrow the method work.
- If the main failure is hidden-view completion, proceed to RefWorld-0 rather than replacing unrelated parts of the stack.

### EXP-001 — optional proprietary comparison: Marble

Marble remains scientifically useful as an external SOTA comparison, but it is not a prerequisite for the core project.

Run it only when credentials/access are convenient and compare against the same frozen inputs/report schema.

Questions:

- Does Marble outperform the open baseline on exact-anchor and held-out local views?
- Does it expose editable/semantic state that the open stack lacks?
- Does its ranking change after camera-registration error is separated from world-generation error?

A Marble result must never gate development of the open method.

### EXP-002 — RefWorld-0: reference-constrained local world construction

Do **not** train a foundation model first.

Start with one surgical replacement: hidden/novel-view completion.

Candidate pipeline:

1. source geometry/camera from VGGT / DA3 / equivalent;
2. deterministic nearby camera set;
3. geometry-based source-image warping with explicit visibility/confidence masks;
4. generative repaint only where evidence is absent;
5. canonical reconstruction from source + synthesized neighboring observations;
6. exact-anchor constrained optimization.

WorldForge is the first implementation reference for step 3–4 because it combines VGGT-based warping with diffusion repainting under explicit camera control.

Ablations:

- unchanged WorldGen panorama completion;
- warp only (holes remain);
- warp + repaint;
- source + generated near views without anchor optimization;
- source + generated near views with anchor optimization;
- appearance-only anchor correction;
- geometry-only anchor correction;
- joint correction.

Measure source improvement against held-out-view degradation. A method that overfits the source camera and worsens nearby calibrated views fails.

### EXP-003 — near-to-far / progressive completion

Compare:

- one-shot hidden-world completion;
- all near views generated independently;
- progressive camera-neighborhood expansion where later views condition on committed earlier observations.

Primary evidence:

- exact-anchor preservation;
- held-out calibrated novel-view fidelity;
- contradiction rate / geometry disagreement;
- revisit-selective memory using established relative controls.

### EXP-004 — semantic persistence layer

Take the best visual baseline and add an explicit entity graph.

Test:

- stable IDs after occlusion;
- object/room relations;
- one-object edits;
- snapshot/restore;
- browser/runtime reload/reconnect;
- collateral semantic and visual drift.

Do not confuse a vision tracker that re-identifies similar pixels with native persistent identity.

### EXP-005 — uncertainty frontier

Compare committing all hidden space at generation time against leaving unseen space explicitly unresolved until approached.

Minimal state vocabulary:

- `observed`;
- `resolved`;
- `hypothesized`;
- `unknown`.

Measure:

- contradictions;
- anchor breakage;
- local-view degradation;
- revisit inconsistency;
- editability;
- amount of irreversible hallucinated state.

## Phase 1 — benchmark release

Release:

- fixed rights-cleared inputs;
- frozen scene/view selection rules;
- held-out calibrated multi-view data where available;
- camera path definitions;
- metric implementation;
- open baseline adapters;
- optional proprietary adapters;
- reproducible reports.

Do not ship copyrighted design references as benchmark inputs unless redistribution rights are explicit.

## Phase 2 — method paper candidate

Only pursue a method paper if experiments establish a measurable gap and a method improves it.

Candidate thesis, subject to evidence:

> Reference-constrained progressive world completion improves exact source fidelity and calibrated local novel-view consistency while enabling persistent explicit world state.

The wording must follow results, not precede them.

See also [`docs/open-stack.md`](../docs/open-stack.md).
