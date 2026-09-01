"""EXP-001 stage 1: generate and materialize a Marble baseline world.

This stage intentionally stops before camera registration/render scoring. It
creates a reproducible, URL-free local artifact bundle that later evaluation
can re-render many times without paying for another world generation.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Sequence

from ..adapters.marble_api import DEFAULT_MODEL, MarbleClient, public_world_summary
from ..adapters.marble_exports import MaterializedAsset, materialize_exports
from ..reporting import write_report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_asset(asset: MaterializedAsset, destination: Path) -> dict[str, Any]:
    root = destination.resolve()
    path = Path(asset.path).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError("materialized export must remain inside the experiment output directory") from exc
    return {
        "kind": asset.kind,
        "tier": asset.tier,
        "path": relative.as_posix(),
        "sha256": asset.sha256,
        "size_bytes": asset.size_bytes,
    }


def run_marble_stage1(
    reference_image: str | Path,
    output_dir: str | Path,
    *,
    display_name: str,
    model: str = DEFAULT_MODEL,
    seed: int = 0,
    disable_recaption: bool = True,
    spz_tiers: Sequence[str] = ("500k",),
    include_collider: bool = True,
    client: MarbleClient | None = None,
    materializer: Callable[..., list[MaterializedAsset]] = materialize_exports,
) -> dict[str, Any]:
    """Generate one Marble world and materialize benchmark-safe local exports."""

    source = Path(reference_image)
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    client = client or MarbleClient.from_env()
    operation = client.generate_image_file(
        source,
        display_name=display_name,
        model=model,
        disable_recaption=disable_recaption,
        seed=seed,
    )
    operation_id = operation.get("operation_id")
    if not isinstance(operation_id, str) or not operation_id:
        raise RuntimeError("World Labs generation response has no operation_id")

    world = client.wait_operation(operation_id)
    export_dir = destination / "exports"
    assets = materializer(
        world,
        export_dir,
        spz_tiers=tuple(spz_tiers),
        include_collider=include_collider,
    )

    manifest = {
        "version": "0.1",
        "experiment": "EXP-001",
        "stage": "marble-generation-export",
        "input": {
            "file_name": source.name,
            "sha256": _sha256(source),
            "size_bytes": source.stat().st_size,
        },
        "generation": {
            "provider": "World Labs",
            "system": "Marble",
            "model_requested": model,
            "seed": seed,
            "disable_recaption": disable_recaption,
            "display_name": display_name,
            "operation_id": operation_id,
        },
        "world": public_world_summary(world),
        "exports": [_portable_asset(asset, destination) for asset in assets],
        "next_stage": "camera-registration-and-render-evaluation",
    }
    write_report(destination / "stage1.safe.json", manifest)
    return manifest
