# Deterministic Spark capture harness

This subproject renders a materialized Marble SPZ from a supplied canonical RefWorld camera.

Pinned runtime:

- `@sparkjsdev/spark` 2.1.0
- `three` 0.180.0
- `playwright` 1.62.1
- Chromium installed by that Playwright release

The harness forces `deviceScaleFactor=1`, disables WebGL antialiasing, uses SwiftShader through ANGLE for the benchmark capture path, manually settles Spark frames, and uses no interactive camera controls.

Install locally:

```bash
cd renderer
npm install
npx playwright install chromium
```

Capture from repository root-relative assets:

```bash
cd renderer
npm run capture -- \
  --asset outputs/exp001/scene001/world-500k.spz \
  --camera outputs/exp001/scene001/anchor-camera.json \
  --out outputs/exp001/scene001/renders/anchor.png \
  --width 1280 \
  --height 720
```

Camera JSON uses the canonical `opengl-camera-to-world` convention from `refworld.camera`. Intrinsics are a row-major 3×3 pinhole matrix in pixel units with +u right / +v down. Extrinsics are row-major 4×4 camera-to-world with +X right / +Y up / -Z forward.

The renderer currently targets non-streamed SPZ benchmark exports. Do not convert to `.rad` for scoring unless the benchmark explicitly defines and pins the conversion, because LoD/streaming changes the rendered representation.
