import hashlib
import json
from pathlib import Path

import pytest

from refworld.adapters.marble_exports import MaterializedAsset
from refworld.experiments.exp001 import run_marble_stage1


class FakeClient:
    def __init__(self):
        self.generate_args = None
        self.waited = None

    def generate_image_file(self, path, **kwargs):
        self.generate_args = (Path(path), kwargs)
        return {"operation_id": "op-123", "done": False}

    def wait_operation(self, operation_id):
        self.waited = operation_id
        return {
            "world_id": "world-123",
            "display_name": "fixture",
            "model": "marble-1.1",
            "assets": {
                "splats": {"spz_urls": {"500k": "https://signed.example/world.spz?token=SECRET"}},
                "mesh": {"collider_mesh_url": "https://signed.example/collider.glb?token=SECRET"},
            },
            "world_prompt": {"image_prompt": {"uri": "https://private.example/input"}},
        }


def fake_materializer(world, destination, *, spz_tiers, include_collider):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    spz = destination / "world-500k.spz"
    spz.write_bytes(b"spz")
    records = [
        MaterializedAsset(
            kind="spz",
            tier="500k",
            path=str(spz),
            sha256=hashlib.sha256(b"spz").hexdigest(),
            size_bytes=3,
        )
    ]
    if include_collider:
        collider = destination / "collider.glb"
        collider.write_bytes(b"glb")
        records.append(
            MaterializedAsset(
                kind="collider",
                tier=None,
                path=str(collider),
                sha256=hashlib.sha256(b"glb").hexdigest(),
                size_bytes=3,
            )
        )
    return records


def test_stage1_records_seed_hashes_and_only_portable_safe_metadata(tmp_path: Path):
    image = tmp_path / "reference.jpg"
    image.write_bytes(b"reference-bytes")
    output = tmp_path / "run"
    client = FakeClient()

    manifest = run_marble_stage1(
        image,
        output,
        display_name="fixture",
        model="marble-1.1",
        seed=42,
        client=client,
        materializer=fake_materializer,
    )

    assert client.waited == "op-123"
    _, args = client.generate_args
    assert args["seed"] == 42
    assert args["disable_recaption"] is True
    assert manifest["input"]["sha256"] == hashlib.sha256(b"reference-bytes").hexdigest()
    assert manifest["exports"][0]["path"] == "exports/world-500k.spz"
    assert manifest["exports"][1]["path"] == "exports/collider.glb"

    written = (output / "stage1.safe.json").read_text()
    parsed = json.loads(written)
    assert parsed == manifest
    assert "token=SECRET" not in written
    assert "private.example" not in written
    assert str(tmp_path) not in written


def test_stage1_rejects_materializer_path_outside_output(tmp_path: Path):
    image = tmp_path / "reference.jpg"
    image.write_bytes(b"reference")
    outside = tmp_path / "outside.spz"
    outside.write_bytes(b"spz")

    def outside_materializer(*args, **kwargs):
        return [
            MaterializedAsset(
                kind="spz",
                tier="500k",
                path=str(outside),
                sha256=hashlib.sha256(b"spz").hexdigest(),
                size_bytes=3,
            )
        ]

    with pytest.raises(ValueError, match="inside the experiment output"):
        run_marble_stage1(
            image,
            tmp_path / "run",
            display_name="fixture",
            client=FakeClient(),
            materializer=outside_materializer,
        )
