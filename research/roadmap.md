# Research roadmap

## Phase 0 — falsify novelty

Before building a model, determine whether existing systems already satisfy the proposed benchmark.

### EXP-001 — Marble baseline

**Question:** How well does Marble preserve a single source image quantitatively?

Protocol:

1. choose 10–30 reference scenes spanning interiors, architecture, foliage, reflective materials and clutter;
2. generate Marble 1.1/1.1 Plus worlds from one image each;
3. export splat + collider/mesh where available;
4. recover the exact reference camera;
5. render anchor + controlled perturbations;
6. calculate AFC and loop consistency;
7. record generation/export/runtime cost.

Exit criterion:

- If Marble is already excellent, shift research toward semantic persistence + benchmark tooling.
- If it fails sharply near the anchor, proceed to anchor correction.

### EXP-002 — Anchor correction

Optimize the generated world against the source camera.

Ablations:

- camera only;
- appearance only;
- geometry only;
- camera + appearance;
- full joint optimization.

Measure source improvement vs novel-view degradation.

### EXP-003 — Near-to-far completion

Compare:

- one-shot world completion;
- progressive camera-neighborhood expansion.

Primary metric: AFC slope and held-out novel-view consistency.

### EXP-004 — Semantic persistence layer

Take the best visual baseline and add an explicit entity graph.

Test:

- stable IDs after occlusion;
- loop closure;
- one-object edits;
- snapshot/restore;
- browser reload/reconnect.

### EXP-005 — Uncertainty frontier

Compare committing all hidden space at generation time against resolving it progressively as the user approaches.

Measure:

- contradictions;
- anchor breakage;
- loop inconsistency;
- user preference;
- editability.

## Phase 1 — benchmark release

Release:

- fixed input set with rights-cleared images;
- held-out multi-view set where possible;
- camera path definitions;
- metric implementation;
- baseline adapters;
- reproducible reports.

Do not ship copyrighted design references as benchmark inputs unless redistribution rights are explicit.

## Phase 2 — method paper candidate

Only pursue a method paper if experiments establish a measurable gap and a method improves it.

Possible paper thesis:

> Anchor-constrained progressive world completion improves reference fidelity and local novel-view consistency while preserving persistent semantic state.

The wording should follow evidence, not precede it.
