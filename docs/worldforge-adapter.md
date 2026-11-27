# WorldForge integration notes

Pinned reference implementation:

```text
Westlake-AGI-Lab/WorldForge
ee573a051715a451b806a90e21462f23308faac4
```

WorldForge is the first candidate backend/reference for RefWorld-0's controlled near-view synthesis. It is **not** treated as the canonical persistent-world representation.

## What the released VGGT path provides

At the pinned commit, `vggt/run_warp.py`:

1. loads `facebook/VGGT-1B`;
2. estimates source camera extrinsics/intrinsics;
3. estimates depth + depth confidence;
4. resizes depth/confidence to the original image;
5. constructs camera trajectories (`left`, `right`, `up`, `down`, `forward`, `backward`);
6. calls `warp_single_img`;
7. saves warped RGB frames + masks + camera interpolation descriptions.

This maps naturally onto RefWorld's `WarpBackend` concept.

## Critical provenance caveat: crack filling

The released CLI's `validate_args` forces:

```text
fill_cracks = True
skip_outlier_detection = False
use_fast_outlier_detection = True
```

and `warp_single_img` includes crack/depth/outlier processing.

Therefore a final saved WorldForge mask must **not** automatically be equated with RefWorld `OBSERVED` provenance. A pixel introduced by crack filling / interpolation is inferred image support, even if it is visually plausible.

### RefWorld rule

The hard `OBSERVED` mask must be derived from direct geometric projection of real source pixels **before** image-space crack filling/inpainting/repainting.

Preferred v0 integration:

```text
VGGT
  ├─ source camera
  ├─ source depth
  └─ depth confidence
        │
        ▼
RefWorld strict geometric z-buffer warp
        ├─ observed RGB
        ├─ OBSERVED mask
        └─ confidence
        │
        ├──────────────┐
        │              │
        ▼              ▼
optional crack     WorldForge-style
fill candidate     diffusion repaint
        │              │
        └──────┬───────┘
               ▼
      non-observed proposal
               │
               ▼
RefWorld evidence-preserving compositor
```

This keeps the WorldForge idea while making evidence provenance auditable.

An alternative implementation may modify/call WorldForge's lower-level warper to expose both:

- `raw_projected_mask`;
- `post_fill_mask`.

But the former—not the latter—is the RefWorld `OBSERVED` support.

## Camera convention boundary

WorldForge/VGGT code uses the camera conventions native to its VGGT utilities. RefWorld's external contract is:

```text
right-handed OpenGL camera-to-world
+X right
+Y up
-Z forward
image +v down
```

The adapter must convert explicitly and test the conversion with known synthetic cameras. Do not infer convention from matrix shape.

The RefWorld target camera should remain the source of truth. WorldForge CLI `direction`/`degree` controls are convenient demos but are not sufficiently general to define the benchmark camera contract.

For RefWorld-0, prefer one of:

1. call lower-level WorldForge/VGGT warping with exact target matrices; or
2. adapt its trajectory generation only when it can be proven to reproduce the requested RefWorld camera.

Do not silently replace a requested camera with the nearest `left/right/up/down` preset.

## Confidence semantics

VGGT exposes depth confidence. RefWorld stores a `[0,1]` warp confidence map.

The adapter must document the normalization that maps VGGT's native confidence values into `[0,1]`; do not min-max normalize each image independently without recording it, because that destroys cross-scene meaning.

Until a principled calibration is selected, confidence may be used as a binary filtering threshold and surviving direct projections assigned weight 1.0, with the threshold recorded.

## Repaint boundary

WorldForge currently supports Wan2.1 and LongCat-Video inference paths.

A RefWorld repaint adapter returns:

```text
candidate RGB
validity mask
backend ID
seed
configuration / checkpoint provenance
```

It may generate a full frame, but `build_view_proposal` applies candidate pixels only where hard observation support is absent.

Ablate both:

- unrestricted full-frame repaint;
- evidence-preserving repaint.

That ablation directly tests whether hard evidence preservation helps rather than assuming it does.

## Required adapter provenance

Record at least:

- WorldForge git commit;
- VGGT checkpoint;
- video model + checkpoint;
- source image hash;
- source-camera/depth inference settings;
- depth-confidence filtering rule;
- exact target RefWorld camera;
- any trajectory conversion;
- crack-fill policy;
- repaint seed;
- guidance/resampling parameters;
- generated candidate hash;
- strict observed-mask hash;
- final proposal/provenance hashes.

## First GPU implementation order

1. run VGGT geometry on one rights-cleared source;
2. convert VGGT camera to RefWorld canonical convention;
3. feed VGGT depth into `PinholeWarpBackend`;
4. compare its raw warp against WorldForge's visual warp;
5. plug one repaint backend into `RepaintBackend`;
6. construct `ViewProposal`;
7. verify observed pixels are bitwise unchanged;
8. only then reconstruct a canonical world from the proposal set.
