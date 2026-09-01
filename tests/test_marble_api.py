import json
from pathlib import Path

import pytest

from refworld.adapters.marble_api import MarbleClient, public_world_summary


class FakeResponse:
    def __init__(self, payload=None):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        if self.payload is None:
            return b""
        return json.dumps(self.payload).encode()


def test_generate_image_uri_uses_documented_endpoint_seed_and_redacts_secret():
    seen = []

    def fake_urlopen(req):
        seen.append(req)
        return FakeResponse({"operation_id": "op-1", "done": False})

    client = MarbleClient("super-secret", urlopen=fake_urlopen)
    result = client.generate_image_uri(
        "https://example.com/reference.jpg",
        display_name="bench-scene",
        model="marble-1.1",
        seed=17,
    )
    assert result["operation_id"] == "op-1"
    assert seen[0].full_url.endswith("/marble/v1/worlds:generate")
    payload = json.loads(seen[0].data)
    assert payload["world_prompt"]["type"] == "image"
    assert payload["world_prompt"]["image_prompt"]["source"] == "uri"
    assert payload["seed"] == 17
    assert "super-secret" not in repr(client)


def test_local_upload_never_leaks_api_key_to_signed_storage(tmp_path: Path):
    image = tmp_path / "scene.jpg"
    image.write_bytes(b"jpeg-bytes")
    seen = []

    def fake_urlopen(req):
        seen.append(req)
        if req.full_url.endswith("/marble/v1/media-assets:prepare_upload"):
            return FakeResponse({
                "media_asset": {"media_asset_id": "asset-1"},
                "upload_info": {
                    "upload_url": "https://storage.example/signed",
                    "upload_method": "PUT",
                    "required_headers": {"x-test-required": "yes"},
                },
            })
        if req.full_url == "https://storage.example/signed":
            return FakeResponse()
        if req.full_url.endswith("/marble/v1/worlds:generate"):
            return FakeResponse({"operation_id": "op-1", "done": False})
        raise AssertionError(req.full_url)

    client = MarbleClient("super-secret", urlopen=fake_urlopen)
    result = client.generate_image_file(image, display_name="local-scene", seed=0)
    assert result["operation_id"] == "op-1"

    prepare, upload, generate = seen
    assert prepare.get_header("Wlt-api-key") == "super-secret"
    assert upload.get_header("Wlt-api-key") is None
    assert upload.get_header("Content-type") == "image/jpeg"
    assert upload.get_header("X-test-required") == "yes"
    assert upload.data == b"jpeg-bytes"
    generated = json.loads(generate.data)
    prompt = generated["world_prompt"]["image_prompt"]
    assert prompt == {"source": "media_asset", "media_asset_id": "asset-1"}
    assert generated["seed"] == 0


def test_invalid_seed_is_rejected_before_request():
    client = MarbleClient("key", urlopen=lambda _: (_ for _ in ()).throw(AssertionError("network should not run")))
    with pytest.raises(ValueError):
        client.generate_image_uri("https://example.com/a.jpg", display_name="bad", seed=-1)
    with pytest.raises(ValueError):
        client.generate_image_uri("https://example.com/a.jpg", display_name="bad", seed=2**32)
    with pytest.raises(TypeError):
        client.generate_image_uri("https://example.com/a.jpg", display_name="bad", seed=True)


def test_wait_operation_returns_completed_world():
    responses = iter([
        {"operation_id": "op-1", "done": False},
        {"operation_id": "op-1", "done": True, "error": None, "response": {"world_id": "w-1"}},
    ])

    def fake_urlopen(req):
        return FakeResponse(next(responses))

    client = MarbleClient("key", urlopen=fake_urlopen)
    world = client.wait_operation("op-1", poll_seconds=0, sleeper=lambda _: None)
    assert world["world_id"] == "w-1"


def test_image_uri_rejects_local_paths():
    client = MarbleClient("key", urlopen=lambda _: FakeResponse({}))
    with pytest.raises(ValueError):
        client.generate_image_uri("reference.jpg", display_name="bad")


def test_public_world_summary_drops_signed_urls_and_prompt_material():
    raw = {
        "world_id": "w-1",
        "display_name": "scene",
        "model": "marble-1.1",
        "world_prompt": {"image_prompt": {"uri": "https://private.example/input.jpg"}},
        "assets": {
            "splats": {"spz_urls": ["https://storage.example/file.spz?secret=token"]},
            "mesh": {"collider_mesh_url": "https://storage.example/collider.glb?secret=token"},
        },
    }
    safe = public_world_summary(raw)
    encoded = json.dumps(safe)
    assert safe["world_id"] == "w-1"
    assert safe["asset_capabilities"] == {"splats": True, "mesh": True, "imagery": False}
    assert "secret=token" not in encoded
    assert "private.example" not in encoded
    assert "world_prompt" not in safe
