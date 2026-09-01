# Local quickstart

This is the shortest reproducible path from a clean checkout to the first real EXP-001 Marble baseline.

No GitHub Actions are required.

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e '.[dev]'

cd renderer
npm install
cd ..
```

Run local checks:

```bash
python -m pytest -q
cd renderer && npm test && cd ..
refworld-validate-report examples/synthetic-report.json
```

The committed synthetic report is tooling-only and is not a benchmark result.

## 2. Obtain the frozen BlendedMVS bootstrap data

Use the official BlendedMVS low-resolution v1.0.0 release described at:

- https://github.com/YoYo000/BlendedMVS

The frozen scene selection is already committed in:

```text
datasets/blendedmvs-bootstrap-v0.json
```

Do not substitute scenes based on model output quality.

## 3. Prepare metadata only

Assuming the local dataset root contains the BlendedMVS scene-ID folders:

```bash
refworld-prepare-blendedmvs /path/to/BlendedMVS \
  --output outputs/blendedmvs-bootstrap-v0.prepared.json
```

This command:

- verifies every required anchor/held-out image, camera and depth map;
- selects the first `pair.txt` reference record per scene;
- records all listed held-out source views;
- converts camera calibration into RefWorldBench's canonical OpenGL C2W convention;
- computes actual pose separation;
- hashes files;
- writes metadata only.

It does **not** copy dataset images into this repository.

Inspect the prepared JSON to obtain each scene's exact anchor image/view ID.

## 4. Configure World Labs locally

Create a World Labs API key in your own secure environment and expose it only as an environment variable:

```bash
export WORLDLABS_API_KEY='...'
```

Never commit `.env`, credentials, raw World responses, or signed export URLs.

## 5. Run EXP-001 stage 1 for one frozen scene

Use the exact anchor image identified by the prepared metadata:

```bash
refworld-marble-stage1 \
  /path/to/BlendedMVS/<SCENE_ID>/blended_images/<ANCHOR_ID>.jpg \
  outputs/exp001/<SCENE_ID>/seed-0 \
  --display-name refworld-exp001-<SCENE_ID>-seed0 \
  --model marble-1.1 \
  --seed 0 \
  --spz-tier 500k
```

EXP-001 defaults to `disable_recaption=true` so the first reference-fidelity run is not intentionally transformed through an extra recaptioning step. The stage manifest records this choice.

The command will:

1. upload the local image using World Labs' signed media flow;
2. submit seeded image→world generation;
3. poll the operation until complete;
4. materialize the requested SPZ and collider locally;
5. compute content hashes;
6. write `stage1.safe.json` with output-relative paths and sanitized world metadata.

It deliberately does **not** save the raw World response.

## 6. Evaluation stage

The remaining EXP-001 evaluation path is:

1. recover/refine the source camera `C0` without changing world appearance;
2. render the exported SPZ with the pinned Spark renderer;
3. score the actual source image at `C0`;
4. render the BlendedMVS held-out cameras;
5. compare against the observed held-out images;
6. report fidelity against real pose separation (`view_direction_angle_deg`, source-unit camera-center distance);
7. validate the final report against `schemas/report.schema.json`.

Do not call arbitrary held-out camera pairs `yaw_deg`, and do not call dataset reconstruction units meters unless scale has been independently established.

## 7. Seed policy

The initial falsification pass uses **seed 0** for every scene. This prevents scene-dependent seed cherry-picking.

If generation variance appears material, run a predeclared repeated-seed study such as seeds `[0, 1, 2]` for **all** selected scenes rather than selectively rerunning failures.

## 8. Stop/narrow rule

EXP-001 is designed to kill unnecessary research quickly.

If Marble already gives strong exact-anchor and held-out local-view fidelity after honest camera registration, do not build another image→world generator merely to have one. Narrow Reference Worlds toward the remaining measurable gaps: semantic identity/state, edit locality, persistence, or benchmark tooling.
