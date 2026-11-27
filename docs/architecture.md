# Candidate architecture

This is a hypothesis to test, not an implementation commitment.

## Canonical world state

```text
World
├── anchors
│   └── reference camera(s) + source observations
├── geometry
│   ├── navigable/collision mesh
│   └── uncertain hidden geometry
├── appearance
│   ├── splats / radiance field
│   └── PBR assets for explicit objects
├── entities
│   ├── stable IDs
│   ├── transforms
│   ├── semantic labels
│   ├── relations
│   └── permissions / application state
├── observations
│   └── immutable history of what has been exposed
└── edits
    └── append-only operations + checkpoints
```

## Pipeline

### Stage 1 — observation analysis

Input one reference image.

Estimate:

- camera intrinsics;
- depth / point map;
- normals;
- major architectural planes;
- object masks and identities;
- illumination cues;
- material categories;
- confidence / ambiguity.

### Stage 2 — world proposal

Generate or reconstruct a broad world proposal using one or more baselines.

Examples:

- Marble world generation;
- VGGT/DA3 geometry + splat completion;
- scene layout + per-object SAM 3D/TRELLIS assets;
- future world model API.

### Stage 3 — anchor alignment

Solve camera/world registration such that the candidate re-render aligns with the source.

Optimize, with regularization:

```text
L = λ_anchor L_perceptual(R(W,C0), I_ref)
  + λ_geom L_geometry
  + λ_consistency L_novel_view
  + λ_edit L_locality
  + λ_prior L_world_prior
```

A high `λ_anchor` is intentional: the observed source view is evidence; hidden space is not.

### Stage 4 — near-to-far consistency expansion

Do not optimize every unseen viewpoint equally from the start.

Expand the trusted region gradually:

```text
0° → 2° → 5° → 10° → 20° → wider world
```

This is motivated by the observation that image-to-3D systems tend to preserve views near the input much better than distant novel views.

At each band:

1. render candidate views;
2. detect geometry/identity inconsistencies;
3. update the canonical world;
4. re-check all earlier trusted views;
5. reject changes that break anchor constraints.

### Stage 5 — semantic compiler

Attach stable entities to the appearance/geometry world.

A table/chair/device should have one persistent identity even if represented by multiple splat clusters, mesh colliders, textures, and application objects.

### Stage 6 — uncertainty frontier

Unseen regions should carry uncertainty rather than false certainty.

Possible states:

- `observed` — constrained by source/held-out imagery;
- `resolved` — generated and already exposed to the user;
- `hypothesized` — generated but not yet exposed;
- `unknown` — intentionally unresolved.

Once exposed, state is committed to persistent history unless an explicit edit changes it.

## Representation question

The experiment should compare:

1. pure splat;
2. pure explicit mesh/PBR;
3. hybrid splat + collider mesh;
4. hybrid splat + explicit semantic object assets.

Hypothesis: (4) will be strongest for persistent applications even when a pure splat looks better initially.
