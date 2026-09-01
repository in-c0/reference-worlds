# Marble baseline runbook

World Labs Marble is the first falsification baseline for RefWorldBench. This document keeps the experiment reproducible without committing credentials, signed asset URLs, or non-redistributable inputs.

Official API docs: https://docs.worldlabs.ai/api

## 1. Obtain credentials locally

World Labs requires an API key sent in the `WLT-Api-Key` header. Store it outside Git.

For shells that support environment variables:

```bash
export WORLDLABS_API_KEY='...'
```

Do **not** paste the key into benchmark manifests, issue comments, notebooks committed to Git, or public reports.

The Python client reads only `WORLDLABS_API_KEY`:

```python
from refworld.adapters.marble_api import MarbleClient

client = MarbleClient.from_env()
```

## 2. Generate from a public image URL

```python
operation = client.generate_image_uri(
    "https://example.org/rights-cleared-reference.jpg",
    display_name="refworld-exp001-scene001",
    model="marble-1.1",
)

world = client.wait_operation(operation["operation_id"])
```

Use `marble-1.1-plus` only when the experiment protocol calls for it; always record the exact model and configuration.

## 3. Generate from a local image

The documented flow is:

1. `POST /marble/v1/media-assets:prepare_upload`;
2. upload bytes to the returned signed storage URL;
3. `POST /marble/v1/worlds:generate` using the returned `media_asset_id`.

The client implements this as:

```python
operation = client.generate_image_file(
    "private-data/scene001/reference.jpg",
    display_name="refworld-exp001-scene001",
)
world = client.wait_operation(operation["operation_id"])
```

Security invariant: the Marble API key is sent to `api.worldlabs.ai` control-plane requests and is **not copied to the signed media upload URL**. This is covered by a mock test.

## 4. Never commit the raw World response

Completed World objects may contain:

- signed SPZ export URLs;
- collider/HQ mesh URLs;
- panorama/thumbnail URLs;
- source prompt information.

Those are useful locally but inappropriate as durable public benchmark metadata.

Use the whitelist helper for committed metadata:

```python
from refworld.adapters.marble_api import public_world_summary

safe = public_world_summary(world)
```

It records stable identifiers/model/timestamps and coarse asset availability while dropping prompts and all asset URLs.

## 5. Export handling

The benchmark adapter should materialize required exports under ignored local storage, for example:

```text
outputs/
└── exp001/
    └── scene001/
        ├── world.safe.json
        ├── world.spz
        ├── collider.glb
        └── renders/
```

Do not treat a signed URL as a permanent artifact. Download the required export while valid, hash the bytes, and record the content hash plus non-secret provenance in the benchmark report.

The World API exposes SPZ files in 100k, 500k, and full-resolution tiers plus a collider GLB. Pin the chosen tier in every benchmark report.

## 6. Deterministic rendering requirements

Before comparing source/novel views, pin:

- renderer implementation + version;
- output resolution;
- camera convention;
- tone mapping / color management;
- splat quality/export tier;
- world-to-renderer coordinate conversion;
- export format/version when known.

**Current Marble exports use an OpenGL coordinate system for splats and meshes.** World Labs release notes state that older generations used OpenCV coordinates, so the benchmark must record the export/model vintage and never infer convention from the product name alone. A synthetic camera/convention fixture should fail loudly if axes are flipped or handedness is wrong.

This is especially important for historical exports: an AFC computed from the wrong camera convention is invalid evidence about world fidelity.

## 7. EXP-001 stopping rule

Do not build an anchor-correction model merely because Marble produces an imperfect example.

Run enough rights-cleared scenes to establish whether there is a systematic, measurable failure:

- poor exact-anchor reconstruction after camera registration;
- steep Anchor Fidelity Curve near zero;
- loop/revisit drift;
- expansion/edit collateral drift;
- semantic persistence not exposed or unstable.

If Marble already performs strongly on the visual tests, narrow the research effort rather than duplicating world generation.
