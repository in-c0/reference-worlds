# Open GPU smoke runbook

This is the shortest path from the public repository to the first real open RefWorld artifacts.

It requires an NVIDIA GPU host with Docker + NVIDIA Container Toolkit. It does **not** require a World Labs/Marble API key or any proprietary world-generation API.

The smoke run is diagnostic, not a research result. It establishes:

```text
real source image
  → pinned VGGT camera/depth/raw confidence
  → source-only 3DGS PLY
  → deterministic geometric near-view warps
```

No hidden-view repaint occurs yet.

## 1. Build the pinned VGGT image

From the repository root:

```bash
docker build -f docker/vggt.Dockerfile -t refworld-vggt .
```

The image pins:

- `facebookresearch/vggt` commit `a288dd0f14786c93483e45524328726ab7b1b4ce`;
- PyTorch `2.3.1` + torchvision `0.18.1` on CUDA 12.1;
- the lightweight RefWorld package and method/test extras.

VGGT model weights (`facebook/VGGT-1B`) are intentionally **not** baked into the image. They download at runtime under their own upstream terms. Mount a Hugging Face cache for repeatability and speed.

## 2. Choose one rights-cleared source image

For a first smoke test, use one image that can be retained privately or redistributed legally. The full frozen BlendedMVS bootstrap is not required just to prove the pipeline.

Example host paths:

```bash
export REF_IMAGE="$PWD/private-data/smoke/reference.jpg"
mkdir -p "$PWD/outputs/smoke/source-01" "$HOME/.cache/huggingface"
```

`private-data/` is ignored and should remain uncommitted.

## 3. Run source geometry + splat + warp-only neighborhood

Mount the source image read-only and mount the output directory back into the repository's `outputs/` tree so the deterministic renderer can consume it afterward:

```bash
docker run --rm --gpus all \
  -v "$REF_IMAGE:/data/reference.jpg:ro" \
  -v "$PWD/outputs/smoke/source-01:/workspace/reference-worlds/outputs/smoke/source-01" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  refworld-vggt \
  bash scripts/run-vggt-smoke.sh \
    /data/reference.jpg \
    outputs/smoke/source-01
```

By default the script runs the focused CPU contract tests before starting VGGT inference. Set `RUN_TESTS=0` only when deliberately skipping that gate:

```bash
docker run --rm --gpus all \
  -e RUN_TESTS=0 \
  ...
```

For a scale-free-only first pass, set `ROTATION_ONLY=1`; otherwise the script also produces translations expressed as fractions of the median predicted source depth. Those ratios are **not meters**.

## 4. Expected artifacts

A successful run writes roughly:

```text
outputs/smoke/source-01/
├── source-geometry/
│   ├── source-geometry.safe.json
│   ├── depth.npy
│   └── confidence-raw.npy
├── source-splat/
│   ├── source-splat.safe.json
│   ├── source-splat.ply
│   └── source-camera.json
└── warp-only/
    ├── warp-only.safe.json
    └── cam-*/
        ├── proposal.png
        ├── provenance.npy
        ├── warp-confidence.npy
        └── proposal.json
```

The safe manifests contain hashes and output-relative paths. Raw VGGT confidence is preserved as a ranking score; v0 does not normalize it per image or pretend it is a calibrated probability.

## 5. Render and score the source-only splat before adding generation

Install the pinned renderer dependencies on the host checkout:

```bash
cd renderer
npm install
npx playwright install chromium
```

Then render using the exact command printed by `run-vggt-smoke.sh`, conceptually:

```bash
npm run capture -- \
  --asset outputs/smoke/source-01/source-splat/source-splat.ply \
  --camera outputs/smoke/source-01/source-splat/source-camera.json \
  --out outputs/smoke/source-01/source-splat/source-render.png \
  --width <SOURCE_WIDTH> \
  --height <SOURCE_HEIGHT>
```

`capture.mjs` intentionally resolves/serves files only under the repository root.

Return to the repository root and score the exact source camera:

```bash
cd ..
refworld-score-anchor \
  --reference "$REF_IMAGE" \
  --render outputs/smoke/source-01/source-splat/source-render.png \
  --output outputs/smoke/source-01/source-splat/source-anchor-score.json
```

The score artifact records MAE, MSE and PSNR plus source/render hashes. These bootstrap metrics are a plumbing gate, not sufficient final perceptual evaluation; serious experiments should add SSIM/MS-SSIM, LPIPS/foundation features and visual inspection under a declared protocol.

### Interpretation

Compare `source-render.png` against the supplied source image **before** introducing a repaint model.

If the anchor is substantially wrong here, investigate:

- VGGT camera/intrinsics remapping;
- predicted depth;
- OpenCV→OpenGL convention conversion;
- source RGB-D→3DGS conversion;
- Spark PLY interpretation;
- renderer projection/camera setup.

Do **not** blame hidden-view synthesis yet: none has occurred.

If the source-only anchor is healthy, proceed to the warp-only views. Those deliberately show disocclusion holes. That gives us the exact support a repaint backend must fill.

Do not invent a universal PSNR pass threshold before seeing a small calibration set. The first purpose of this gate is differential diagnosis and regression detection.

## 6. Add any generator without changing evidence semantics

A GPU NVS/video model may produce a candidate RGB image for one target view. RefWorld imports it through:

```bash
refworld-compose-candidate \
  --warp-view outputs/smoke/source-01/warp-only/<CAM_ID> \
  --candidate /path/to/model-target.png \
  --backend '<explicit-model-or-run-id>' \
  --seed 0 \
  --output outputs/smoke/source-01/composed/<CAM_ID>
```

That emits both:

- the unrestricted model candidate (ablation B), and
- the evidence-preserved proposal (ablation C).

The compositor structurally prevents generator pixels from overwriting geometric `OBSERVED` support and records overlap attempts.

Held-out benchmark images must never be supplied as repaint candidates or optimization inputs.

## 7. What this smoke run does not prove

A successful run does **not** establish that RefWorld-0 improves novel-view fidelity or that any metric is novel. It only proves the open plumbing necessary to run those experiments reproducibly.

The next scientific step is to run the same frozen source/held-out cameras across:

1. unchanged WorldGen baseline;
2. source-only 3DGS diagnostic;
3. geometry-only RefWorld warp;
4. unrestricted repaint;
5. evidence-preserving repaint;
6. canonical-world reconstruction;
7. source-anchor optimization without held-out leakage.
