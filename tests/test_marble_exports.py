import hashlib
import json
from pathlib import Path
from urllib import error

import pytest

from refworld.adapters.marble_exports import (
    MarbleExportError,
    available_exports,
    materialize_exports,
)


class BinaryResponse:
    def __init__(self, data: bytes):
        self._data = data
        self._offset = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size=-1):
        if self._offset >= len(self._data):
            return b""
        if size is None or size < 0:
            size = len(self._data) - self._offset
        chunk = self._data[self._offset:self._offset + size]
        self._offset += len(chunk)
        return chunk


def sample_world():
    return {
        "world_id": "w-1",
        "assets": {
            "splats": {
                "spz_urls": {
                    "100k": "https://signed.example/100k?token=secret-a",
                    "500k": "https://signed.example/500k?token=secret-b",
                    "full_res": "https://signed.example/full?token=secret-c",
                }
            },
            "mesh": {"collider_mesh_url": "https://signed.example/collider?token=secret-d"},
        },
    }


def test_materializer_downloads_hashes_and_returns_no_urls(tmp_path: Path):
    payloads = {
        "https://signed.example/500k?token=secret-b": b"spz-bytes",
        "https://signed.example/collider?token=secret-d": b"glb-bytes",
    }
    seen = []

    def fake_urlopen(req):
        seen.append(req)
        return BinaryResponse(payloads[req.full_url])

    records = materialize_exports(sample_world(), tmp_path, urlopen=fake_urlopen)
    assert (tmp_path / "world-500k.spz").read_bytes() == b"spz-bytes"
    assert (tmp_path / "collider.glb").read_bytes() == b"glb-bytes"
    assert records[0].sha256 == hashlib.sha256(b"spz-bytes").hexdigest()
    assert records[1].sha256 == hashlib.sha256(b"glb-bytes").hexdigest()
    encoded = json.dumps([record.as_dict() for record in records])
    assert "signed.example" not in encoded
    assert "token=secret" not in encoded
    assert all(req.get_header("Wlt-api-key") is None for req in seen)


def test_available_exports_is_url_free_even_for_wrapped_world():
    availability = available_exports({"world": sample_world()})
    assert availability == {
        "spz:100k": True,
        "spz:500k": True,
        "spz:full_res": True,
        "collider": True,
    }
    assert "http" not in json.dumps(availability)


def test_unknown_tier_is_rejected_before_network(tmp_path: Path):
    with pytest.raises(ValueError):
        materialize_exports(sample_world(), tmp_path, spz_tiers=("2m",), urlopen=lambda _: None)


def test_download_failure_does_not_leak_signed_url(tmp_path: Path):
    signed = "https://signed.example/500k?token=super-secret"
    world = sample_world()
    world["assets"]["splats"]["spz_urls"]["500k"] = signed

    def fail(req):
        raise error.URLError("network down")

    with pytest.raises(MarbleExportError) as exc:
        materialize_exports(world, tmp_path, spz_tiers=("500k",), include_collider=False, urlopen=fail)
    assert signed not in str(exc.value)
    assert "super-secret" not in str(exc.value)
    assert not (tmp_path / "world-500k.spz.part").exists()
