#!/usr/bin/env python3
"""Build a source-only Gaussian-splat diagnostic from verified source geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from refworld.source_geometry import load_source_geometry
from refworld.splats import rgbd_to_gaussian_arrays, write_gaussian_ply


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build source-only diagnostic 3DGS PLY")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--source-geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-splats", type=int, default=500_000)
    parser.add_argument("--footprint-scale", type=float, default=0.58)
    parser.add_argument("--thickness-ratio", type=float, default=0.05)
    parser.add_argument("--opacity", type=float, default=0.99)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reference = args.reference.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not reference.is_file():
        raise FileNotFoundError(reference)

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("refworld-source-splat requires Pillow: pip install 'refworld-bench[method]'") from exc

    geometry = load_source_geometry(args.source_geometry)
    reference_sha = _sha256_file(reference)
    if reference_sha != geometry.input_sha256:
        raise ValueError("reference image SHA-256 does not match verified source geometry")

    rgb = np.asarray(Image.open(reference).convert("RGB"), dtype=np.uint8)
    if rgb.shape[:2] != (geometry.height, geometry.width):
        raise ValueError("reference dimensions do not match source geometry")

    vertices, conversion = rgbd_to_gaussian_arrays(
        rgb,
        geometry.depth,
        geometry.camera,
        max_splats=args.max_splats,
        footprint_scale=args.footprint_scale,
        thickness_ratio=args.thickness_ratio,
        opacity=args.opacity,
    )
    ply_path = write_gaussian_ply(output / "source-splat.ply", vertices)

    source_geometry_manifest = Path(args.source_geometry).resolve()
    if source_geometry_manifest.is_dir():
        source_geometry_manifest = source_geometry_manifest / "source-geometry.safe.json"

    manifest: dict[str, Any] = {
        "version": "0.1",
        "stage": "refworld-source-splat",
        "purpose": "source-anchor camera/depth/export/renderer diagnostic; not a hidden-view generator",
        "input": {
            "reference_file_name": reference.name,
            "reference_sha256": reference_sha,
            "source_geometry_manifest_sha256": _sha256_file(source_geometry_manifest),
            "source_geometry_backend": geometry.backend,
        },
        "camera": {
            "intrinsics": list(geometry.camera.intrinsics),
            "extrinsics": list(geometry.camera.extrinsics),
            "convention": geometry.camera.convention,
        },
        "conversion": conversion,
        "representation": {
            "format": "3DGS PLY",
            "ply_encoding": "binary_little_endian",
            "color": "degree-0 spherical harmonics from source RGB",
            "scale_storage": "natural-log axis scales",
            "rotation_storage": "normalized quaternion wxyz",
            "opacity_storage": "logit",
            "source_support_only": True,
            "generated_content": False,
        },
        "confidence_policy": {
            "raw_source_confidence_used": False,
            "reason": "primary v0 diagnostic isolates camera/depth/export/rendering without an uncalibrated confidence transform",
        },
        "artifact": {
            "path": ply_path.relative_to(output).as_posix(),
            "sha256": _sha256_file(ply_path),
            "size_bytes": ply_path.stat().st_size,
        },
    }
    manifest_path = output / "source-splat.safe.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
