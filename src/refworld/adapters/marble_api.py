"""Minimal World Labs Marble API client used by the baseline adapter.

This module intentionally covers the control-plane JSON calls and the documented
media-upload handoff. Export downloading and SPZ rendering belong in separate
components so API credentials never need to enter the renderer or benchmark
report.
"""

from __future__ import annotations

import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib import error, request


DEFAULT_BASE_URL = "https://api.worldlabs.ai"
DEFAULT_MODEL = "marble-1.1"
API_KEY_ENV = "WORLDLABS_API_KEY"
MAX_SEED = 2**32 - 1


class MarbleApiError(RuntimeError):
    pass


def public_world_summary(world: Mapping[str, Any]) -> dict[str, Any]:
    """Return a report-safe summary without prompts or signed asset URLs."""

    assets = world.get("assets")
    assets = assets if isinstance(assets, dict) else {}
    splats = assets.get("splats")
    mesh = assets.get("mesh")
    imagery = assets.get("imagery")
    return {
        "world_id": world.get("world_id") or world.get("id"),
        "display_name": world.get("display_name"),
        "model": world.get("model"),
        "created_at": world.get("created_at"),
        "updated_at": world.get("updated_at"),
        "asset_capabilities": {
            "splats": isinstance(splats, dict) and bool(splats),
            "mesh": isinstance(mesh, dict) and bool(mesh),
            "imagery": isinstance(imagery, dict) and bool(imagery),
        },
    }


def _validated_seed(seed: int | None) -> int | None:
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer or None")
    if seed < 0 or seed > MAX_SEED:
        raise ValueError(f"seed must satisfy 0 <= seed <= {MAX_SEED}")
    return seed


class MarbleClient:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        urlopen: Callable[..., Any] = request.urlopen,
    ) -> None:
        if not api_key.strip():
            raise ValueError("World Labs API key cannot be empty")
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._urlopen = urlopen

    @classmethod
    def from_env(cls, **kwargs: Any) -> "MarbleClient":
        key = os.environ.get(API_KEY_ENV, "")
        if not key:
            raise RuntimeError(f"set {API_KEY_ENV}; never commit API keys to the repository")
        return cls(key, **kwargs)

    def __repr__(self) -> str:
        return f"MarbleClient(base_url={self.base_url!r}, api_key=<redacted>)"

    def _json(self, method: str, path: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url}{path}",
            data=body,
            method=method,
            headers={
                "WLT-Api-Key": self._api_key,
                "Accept": "application/json",
                **({"Content-Type": "application/json"} if body is not None else {}),
            },
        )
        try:
            with self._urlopen(req) as response:
                raw = response.read()
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
            raise MarbleApiError(f"World Labs HTTP {exc.code}: {detail[:500]}") from exc
        except error.URLError as exc:
            raise MarbleApiError(f"World Labs request failed: {exc.reason}") from exc

        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MarbleApiError("World Labs returned a non-JSON response") from exc
        if not isinstance(decoded, dict):
            raise MarbleApiError("World Labs returned an unexpected JSON payload")
        return decoded

    def _generate_image_prompt(
        self,
        image_prompt: Mapping[str, Any],
        *,
        display_name: str,
        model: str,
        text_prompt: str | None,
        disable_recaption: bool | None,
        seed: int | None,
    ) -> dict[str, Any]:
        world_prompt: dict[str, Any] = {"type": "image", "image_prompt": dict(image_prompt)}
        if text_prompt is not None:
            world_prompt["text_prompt"] = text_prompt
        if disable_recaption is not None:
            world_prompt["disable_recaption"] = disable_recaption

        payload: dict[str, Any] = {
            "display_name": display_name,
            "model": model,
            "world_prompt": world_prompt,
        }
        seed = _validated_seed(seed)
        if seed is not None:
            payload["seed"] = seed
        return self._json("POST", "/marble/v1/worlds:generate", payload)

    def generate_image_uri(
        self,
        image_uri: str,
        *,
        display_name: str,
        model: str = DEFAULT_MODEL,
        text_prompt: str | None = None,
        disable_recaption: bool | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        if not image_uri.startswith(("https://", "http://")):
            raise ValueError("image_uri must be an HTTP(S) URL; local files require the media upload flow")
        return self._generate_image_prompt(
            {"source": "uri", "uri": image_uri},
            display_name=display_name,
            model=model,
            text_prompt=text_prompt,
            disable_recaption=disable_recaption,
            seed=seed,
        )

    def generate_image_asset(
        self,
        media_asset_id: str,
        *,
        display_name: str,
        model: str = DEFAULT_MODEL,
        text_prompt: str | None = None,
        disable_recaption: bool | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        if not media_asset_id.strip():
            raise ValueError("media_asset_id cannot be empty")
        return self._generate_image_prompt(
            {"source": "media_asset", "media_asset_id": media_asset_id},
            display_name=display_name,
            model=model,
            text_prompt=text_prompt,
            disable_recaption=disable_recaption,
            seed=seed,
        )

    def prepare_media_upload(
        self,
        file_name: str,
        *,
        kind: str = "image",
        extension: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if kind not in {"image", "video"}:
            raise ValueError("kind must be 'image' or 'video'")
        if not file_name.strip():
            raise ValueError("file_name cannot be empty")
        suffix = Path(file_name).suffix.lstrip(".")
        ext = (extension or suffix).lstrip(".")
        if not ext:
            raise ValueError("media extension cannot be empty")
        return self._json(
            "POST",
            "/marble/v1/media-assets:prepare_upload",
            {
                "file_name": file_name,
                "kind": kind,
                "extension": ext,
                "metadata": dict(metadata or {}),
            },
        )

    def upload_media_file(
        self,
        file_path: str | Path,
        *,
        kind: str = "image",
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(path)

        prepared = self.prepare_media_upload(path.name, kind=kind, metadata=metadata)
        media_asset = prepared.get("media_asset")
        upload_info = prepared.get("upload_info")
        if not isinstance(media_asset, dict) or not isinstance(upload_info, dict):
            raise MarbleApiError("prepare_upload response is missing media_asset or upload_info")

        media_asset_id = media_asset.get("media_asset_id") or media_asset.get("id")
        upload_url = upload_info.get("upload_url")
        upload_method = str(upload_info.get("upload_method") or "PUT").upper()
        required_headers = upload_info.get("required_headers") or {}
        if not isinstance(media_asset_id, str) or not media_asset_id:
            raise MarbleApiError("prepare_upload response has no media_asset_id")
        if not isinstance(upload_url, str) or not upload_url.startswith(("https://", "http://")):
            raise MarbleApiError("prepare_upload response has no valid upload_url")
        if not isinstance(required_headers, dict):
            raise MarbleApiError("prepare_upload required_headers must be an object")

        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        upload_req = request.Request(
            upload_url,
            data=path.read_bytes(),
            method=upload_method,
            headers={"Content-Type": content_type, **{str(k): str(v) for k, v in required_headers.items()}},
        )
        try:
            with self._urlopen(upload_req) as response:
                response.read()
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
            raise MarbleApiError(f"media upload HTTP {exc.code}: {detail[:500]}") from exc
        except error.URLError as exc:
            raise MarbleApiError(f"media upload failed: {exc.reason}") from exc
        return media_asset_id

    def generate_image_file(
        self,
        file_path: str | Path,
        *,
        display_name: str,
        model: str = DEFAULT_MODEL,
        text_prompt: str | None = None,
        disable_recaption: bool | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        media_asset_id = self.upload_media_file(file_path, kind="image")
        return self.generate_image_asset(
            media_asset_id,
            display_name=display_name,
            model=model,
            text_prompt=text_prompt,
            disable_recaption=disable_recaption,
            seed=seed,
        )

    def get_operation(self, operation_id: str) -> dict[str, Any]:
        if not operation_id:
            raise ValueError("operation_id cannot be empty")
        return self._json("GET", f"/marble/v1/operations/{operation_id}")

    def get_world(self, world_id: str) -> dict[str, Any]:
        if not world_id:
            raise ValueError("world_id cannot be empty")
        return self._json("GET", f"/marble/v1/worlds/{world_id}")

    def wait_operation(
        self,
        operation_id: str,
        *,
        timeout_seconds: float = 900.0,
        poll_seconds: float = 5.0,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> dict[str, Any]:
        if timeout_seconds <= 0 or poll_seconds < 0:
            raise ValueError("timeout_seconds must be positive and poll_seconds non-negative")

        deadline = time.monotonic() + timeout_seconds
        while True:
            operation = self.get_operation(operation_id)
            if operation.get("done") is True:
                if operation.get("error"):
                    raise MarbleApiError(f"World Labs operation failed: {operation['error']}")
                response = operation.get("response")
                if not isinstance(response, dict):
                    raise MarbleApiError("completed World Labs operation has no world response")
                return response
            if time.monotonic() >= deadline:
                raise TimeoutError(f"World Labs operation {operation_id} did not complete in time")
            sleeper(poll_seconds)
