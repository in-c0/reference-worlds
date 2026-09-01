"""Minimal World Labs Marble API client used by the baseline adapter.

This module intentionally covers only control-plane JSON calls. Downloading
signed exports and rendering SPZ assets belong in separate components so API
credentials never need to enter the renderer or benchmark report.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Mapping
from urllib import error, request


DEFAULT_BASE_URL = "https://api.worldlabs.ai"
DEFAULT_MODEL = "marble-1.1"
API_KEY_ENV = "WORLDLABS_API_KEY"


class MarbleApiError(RuntimeError):
    pass


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

    def generate_image_uri(
        self,
        image_uri: str,
        *,
        display_name: str,
        model: str = DEFAULT_MODEL,
        text_prompt: str | None = None,
        disable_recaption: bool | None = None,
    ) -> dict[str, Any]:
        if not image_uri.startswith(("https://", "http://")):
            raise ValueError("image_uri must be an HTTP(S) URL; local files require the media upload flow")

        world_prompt: dict[str, Any] = {
            "type": "image",
            "image_prompt": {"source": "uri", "uri": image_uri},
        }
        if text_prompt is not None:
            world_prompt["text_prompt"] = text_prompt
        if disable_recaption is not None:
            world_prompt["disable_recaption"] = disable_recaption

        return self._json(
            "POST",
            "/marble/v1/worlds:generate",
            {"display_name": display_name, "model": model, "world_prompt": world_prompt},
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
