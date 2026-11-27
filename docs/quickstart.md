# Local quickstart

This is the shortest reproducible path from a clean checkout to the first **open/local EXP-000** baseline.

A proprietary world-generation API is **not required**.

No GitHub Actions are required.

## 1. Install RefWorldBench

```bash
git clone https://github.com/in-c0/reference-worlds.git
cd reference-worlds

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e '.[dev]'

cd renderer
npm install
cd ..
```

Run lightweight checks:

```bash
python -m pytest -q
cd renderer && npm test && cd ..
refworld-validate-report examples/synthetic-report.json
```

The committed synthetic report is tooling-only and is not a benchmark result.

## 2. Prepare a CUDA WorldGen environment

EXP-000 pins:

```text
ZiYang-xie/WorldGen
commit 7ce7b2767fdf31e2727b69a2e61e2e950e3a017f
```

WorldGen's upstream documentation describes a low-VRAM mode around 10 GB VRAM. A CPU-only environment cannot run the baseline credibly.

Create a separate environment so heavy world-generation dependencies do not contaminate the lightweight benchmark environment:

```bash
git clone --recursive https://github.com/ZiYang-xie/WorldGen.git
cd WorldGen
git checkout 7ce7b2767fdf31e2727b69a2e61e2e950e3a017f
git submodule update --init --recursive

conda create -n refworld-worldgen python=3.11 -y
conda activate refworld-worldgen

# Install a CUDA-enabled torch/torchvision build appropriate for the machine.
pip install torch torchvision
pip install .
pip install git+https://github.com/EnVision-Research/DA-2.git#subdirectory=src --no-deps
pip install git+https://github.com/facebookresearch/pytorch3d.git --no-build-isolation
```

Then install RefWorldBench into that same GPU environment so the explicit-seed runner is available:

```bash
cd /path/to/reference-worlds
pip install -e .
```

### Model-access boundary

WorldGen repository code is Apache-2.0, but its current image-to-scene path loads external checkpoints under their own terms, including:

- `haodongli/DA-2`;
- `black-forest-labs/FLUX.1-Fill-dev`;
- `LeoXie/WorldGen` image-to-scene LoRA;
- a Nunchaku quantized FLUX transformer in low-VRAM mode.

Accept/login to any gated model terms required by those upstream projects. This is **model access**, not a paid world-generation API dependency.

For the optional `worldgen-sharp` variant also install the pinned WorldGen `ml-sharp` submodule dependencies according to upstream instructions.

## 3. Smoke-test one rights-cleared image

Use a rights-cleared local image first; do not wait for the full benchmark dataset to prove the GPU path.

```bash
conda activate refworld-worldgen

refworld-worldgen-run \
  --worldgen-root /path/to/WorldGen \
  --reference /path/to/reference.jpg \
  --output /path/to/reference-worlds/outputs/exp000/smoke/seed-42 \
  --seed 42 \
  --resolution 1600 \
  --low-vram \
  --no-use-sharp \
  --no-inpaint-bg \
  --no-return-mesh
```

The runner fails closed if the WorldGen checkout is not at the pinned commit unless `--allow-unpinned-worldgen` is supplied. Do not use that override for benchmark runs.

The runner intentionally preserves:

```text
run.safe.json
panorama.png
world-splat.ply
```

`run.safe.json` records:

- input SHA-256;
- actual + expected WorldGen commit;
- explicit seed;
- resolution/quality switches;
- checkpoint identifiers;
- Python/PyTorch/CUDA/GPU metadata;
- model-init and generation timing;
- peak allocated/reserved GPU memory;
- relative artifact paths + hashes.

The source prompt is not stored verbatim; only whether it was empty and its SHA-256 are recorded.

### Why seed 42?

At the pinned WorldGen commit, `gen_pano_fill_image` has an internal default seed of `42`, but `WorldGen.generate_world()` does not expose that seed. The RefWorld runner mirrors the same image-to-panorama logic and passes the seed explicitly. EXP-000 begins with seed 42 for every scene so the baseline corresponds to upstream default behavior without hidden randomness.

If variance later matters, predeclare repeated seeds for **all** scenes rather than selectively rerunning bad results.

## 4. Render the WorldGen PLY through the common benchmark renderer

Spark 2.1.0 supports Gaussian-splat `.ply` as well as `.spz`, so WorldGen and Marble do not need separate visual scoring renderers.

The existing `renderer/` harness is pinned to:

```text
@sparkjsdev/spark 2.1.0
three 0.180.0
playwright 1.62.1
DPR 1
antialias off
canonical RefWorld camera payload
```

Use the resulting `world-splat.ply` as the renderer asset. The renderer must receive a canonical camera; source-camera registration remains a separate evaluation stage rather than being guessed inside the viewer.

## 5. Obtain the frozen BlendedMVS bootstrap data

After the one-image smoke test works, use the official **BlendedMVS low-resolution v1.0.0** release:

- https://github.com/YoYo000/BlendedMVS

The frozen scene selection is committed in:

```text
datasets/blendedmvs-bootstrap-v0.json
```

Do not substitute scenes based on model output quality.

## 6. Prepare deterministic Type-B metadata

Assuming the local dataset root contains the BlendedMVS scene-ID folders:

```bash
refworld-prepare-blendedmvs /path/to/BlendedMVS \
  --output outputs/blendedmvs-bootstrap-v0.prepared.json
```

This command is designed to:

- verify every required anchor/held-out image, camera and depth map;
- select the first `pair.txt` reference record per scene;
- record every listed held-out source view;
- convert MVSNet/OpenCV W2C calibration into RefWorldBench's canonical OpenGL C2W convention;
- compute actual pose separation;
- hash files;
- write metadata only.

It does **not** copy dataset images into this repository.

## 7. Run EXP-000 on the frozen scenes

For each scene, use exactly the anchor image selected by the prepared metadata and seed 42.

Primary baseline:

```text
worldgen-default
low_vram=true where needed
use_sharp=false
inpaint_bg=false
return_mesh=false
seed=42
```

Secondary quality baseline:

```text
worldgen-sharp
same input/seed/settings except use_sharp=true
```

Do not mix `worldgen-default` and `worldgen-sharp` results under one system name.

## 8. Evaluation

For each generated PLY:

1. recover/refine the source camera `C0` **without changing the world**;
2. render the exact source camera with the pinned Spark renderer;
3. score against the actual source image;
4. render every calibrated BlendedMVS held-out camera selected by the frozen protocol;
5. score only against observed held-out images;
6. report real pose separation (`view_direction_angle_deg`, `center_distance_source_units`);
7. keep camera-registration residual separate from generation error;
8. validate the final JSON against `schemas/report.schema.json`.

Do not call arbitrary held-out camera pairs `yaw_deg`, and do not call dataset reconstruction units meters unless physical scale has been independently established.

## 9. What happens after EXP-000

If unchanged WorldGen is already excellent near the supplied observation, narrow RefWorld-0.

If the dominant failure is hidden-view completion, the first method change is **not another whole world model**. Replace only that stage:

```text
source geometry
→ controlled nearby-view warp
→ unresolved-region mask
→ diffusion repaint
→ canonical reconstruction
→ strong source-anchor optimization
```

WorldForge is the first open implementation reference for the warp+repaint portion.

See [`open-stack.md`](open-stack.md) and [`../research/roadmap.md`](../research/roadmap.md).

## 10. Optional external comparison: Marble

World Labs Marble is retained as EXP-001 because it is a useful SOTA comparison. It is optional and does not gate the open research loop.

If/when a World Labs credential is available, use `refworld-marble-stage1` and the same frozen scenes/report schema. See [`marble.md`](marble.md).
