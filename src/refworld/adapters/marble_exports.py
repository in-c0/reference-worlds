"""Materialize Marble's temporary signed exports into local benchmark assets.

This module deliberately does not know the World Labs API key. It accepts a
World payload that already contains signed asset URLs, downloads selected files,
and returns only local paths + content hashes suitable for benchmark provenance.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib import error, request


KNOWN_SPZ_TIERS = ("100k", "500k", "full_res")


class MarbleExportError(RuntimeError):
    pass


@dataclass(frozen=True)
class MaterializedAsset:
    kind: str
    tier: str | None
    path: str
    sha256: str
    size_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _world_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = payload.get("world")
    return nested if isinstance(nested, Mapping) else payload


def available_exports(world: Mapping[str, Any]) -> dict[str, bool]:
    """Return non-secret availability flags only."""

    world = _world_payload(world)
    assets = world.get("assets")
    assets = assets if isinstance(assets, Mapping) else {}
    splats = assets.get("splats")
    splats = splats if isinstance(splats, Mapping) else {}
    spz_urls = splats.get("spz_urls")
    spz_urls = spz_urls if isinstance(spz_urls, Mapping) else {}
    mesh = assets.get("mesh")
    mesh = mesh if isinstance(mesh, Mapping) else {}
    return {
        **{f"spz:{tier}": isinstance(spz_urls.get(tier), str) and bool(spz_urls.get(tier)) for tier in KNOWN_SPZ_TIERS},
        "collider": isinstance(mesh.get("collider_mesh_url"), str) and bool(mesh.get("collider_mesh_url")),
    }


def _signed_export_urls(world: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Internal-only extraction of temporary URLs. Never serialize this result."""

    world = _world_payload(world)
    assets = world.get("assets")
    if not isinstance(assets, Mapping):
        raise MarbleExportError("World has no assets payload")
    splats = assets.get("splats")
    splats = splats if isinstance(splats, Mapping) else {}
    spz_urls = splats.get("spz_urls")
    spz_urls = spz_urls if isinstance(spz_urls, Mapping) else {}
    mesh = assets.get("mesh")
    mesh = mesh if isinstance(mesh, Mapping) else {}
    return spz_urls, mesh


def _download_signed(
    url: str,
    destination: Path,
    *,
    urlopen: Callable[..., Any],
    label: str,
    chunk_bytes: int = 1024 * 1024,
) -> MaterializedAsset:
    if not url.startswith(("https://", "http://")):
        raise MarbleExportError(f"{label} has no valid signed URL")
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.sha256()
    size = 0
    req = request.Request(url, method="GET")
    try:
        with self_closing(urlopen(req)) as response, partial.open("wb") as out:
            while True:
                chunk = response.read(chunk_bytes)
                if not chunk:
                    break
                out.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        os.replace(partial, destination)
    except (error.HTTPError, error.URLError, OSError) as exc:
        partial.unlink(missing_ok=True)
        # Signed URLs may embed credentials. Never include req.full_url / exc URL text.
        raise MarbleExportError(f"failed to download {label}") from exc

    kind, _, tier = label.partition(":")
    return MaterializedAsset(
        kind=kind,
        tier=tier or None,
        path=str(destination),
        sha256=digest.hexdigest(),
        size_bytes=size,
    )


class self_closing:
    """Tiny context wrapper for urllib/fake responses with or without __enter__."""

    def __init__(self, response: Any) -> None:
        self.response = response

    def __enter__(self) -> Any:
        enter = getattr(self.response, "__enter__", None)
        return enter() if enter is not None else self.response

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        exit_ = getattr(self.response, "__exit__", None)
        if exit_ is not None:
            return exit_(exc_type, exc, tb)
        close = getattr(self.response, "close", None)
        if close is not None:
            close()
        return False


def materialize_exports(
    world: Mapping[str, Any],
    destination: str | Path,
    *,
    spz_tiers: Sequence[str] = ("500k",),
    include_collider: bool = True,
    urlopen: Callable[..., Any] = request.urlopen,
) -> list[MaterializedAsset]:
    """Download selected Marble exports and return URL-free provenance records."""

    unknown = [tier for tier in spz_tiers if tier not in KNOWN_SPZ_TIERS]
    if unknown:
        raise ValueError(f"unknown SPZ tier(s): {', '.join(unknown)}")

    root = Path(destination)
    spz_urls, mesh = _signed_export_urls(world)
    records: list[MaterializedAsset] = []

    for tier in spz_tiers:
        url = spz_urls.get(tier)
        if not isinstance(url, str) or not url:
            raise MarbleExportError(f"World does not expose requested SPZ tier {tier}")
        records.append(
            _download_signed(
                url,
                root / f"world-{tier}.spz",
                urlopen=urlopen,
                label=f"spz:{tier}",
            )
        )

    if include_collider:
        url = mesh.get("collider_mesh_url")
        if not isinstance(url, str) or not url:
            raise MarbleExportError("World does not expose a collider mesh")
        records.append(
            _download_signed(
                url,
                root / "collider.glb",
                urlopen=urlopen,
                label="collider",
            )
        )

    return records
