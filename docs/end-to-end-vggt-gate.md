# End-to-end VGGT gate before EXP-006

Purpose: replace oracle source geometry with a genuine single-image geometry path once, measure what breaks, and then move into the LifeOS handoff.

## Frozen scope

Use one frozen BlendedMVS scene/view already in the benchmark manifest. Input to the method is the anchor RGB only. Held-out RGB/depth remains evaluation-only and is unavailable until candidate generation is sealed.

Pipeline:

1. anchor RGB → pinned VGGT source camera/depth/confidence;
2. register monocular geometry to the benchmark camera frame without using held-out target RGB/depth;
3. resolve the otherwise-unidentified monocular translation/scale explicitly and record the transform;
4. produce one predeclared nearby target view using the same RefWorld strict warp/provenance semantics;
5. run one already-pinned repaint backend without tuning against held-out evidence;
6. compare unrestricted B vs evidence-preserved C;
7. only after sealing B/C, materialize the held-out target RGB and score both;
8. report source-anchor fidelity, registration residual, observed fraction, full-frame/observed-support C-B delta, and all geometry/scale diagnostics.

## Required diagnostics

Record separately:

- VGGT raw camera/depth/confidence outputs and hashes;
- source camera registration before/after parameters;
- scale/translation alignment method and numeric transform;
- reprojection/source-anchor residual before and after registration;
- any pixels rejected for invalid/low-confidence geometry;
- target camera displacement in calibrated benchmark units;
- B/C outputs and provenance maps;
- held-out score revealed only after candidate seal.

## Interpretation

PASS does not require matching the oracle result. The gate succeeds as an engineering/research handoff if it cleanly establishes one of:

- evidence preservation remains beneficial with realistic VGGT geometry;
- geometry/scale error overwhelms the benefit and is therefore the next bounded blocker;
- the effect becomes ambiguous and EXP-006 must treat near-view geometry as a quality risk.

Do not expand this into a multi-scene VGGT campaign before EXP-006 unless the single case reveals a specific defect whose correction is necessary for the LifeOS bridge.
