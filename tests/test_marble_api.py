import json

import pytest

from refworld.adapters.marble_api import MarbleClient


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def test_generate_image_uri_uses_documented_endpoint_and_redacts_secret():
    seen = []

    def fake_urlopen(req):
        seen.append(req)
        return FakeResponse({"operation_id": "op-1", "done": False})

    client = MarbleClient("super-secret", urlopen=fake_urlopen)
    result = client.generate_image_uri(
        "https://example.com/reference.jpg",
        display_name="bench-scene",
        model="marble-1.1",
    )
    assert result["operation_id"] == "op-1"
    assert seen[0].full_url.endswith("/marble/v1/worlds:generate")
    payload = json.loads(seen[0].data)
    assert payload["world_prompt"]["type"] == "image"
    assert payload["world_prompt"]["image_prompt"]["source"] == "uri"
    assert "super-secret" not in repr(client)


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
