"""Verified source-geometry artifacts for RefWorld-0.

Geometry inference runs in heavyweight GPU environments, but the benchmark core
only consumes a small safe manifest plus local NumPy artifacts. This module
verifies confinement, hashes, shapes and camera semantics before inferred depth
can participate in an OBSERVED geometric warp.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .adapters.base import Camera
from .camera import OPENGL_C2W


@dataclass(frozen=True)
class SourceGeometry:
    backend: str
    input_sha256: str
    width: int
    height: int
    camera: Camera
    depth: np.ndarray
    confidence_raw: np.ndarray
    metadata: dict[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_artifact(root: Path, record: dict[str, Any]) -> Path:
    relative = Path(str(record.get("path", "")))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("source-geometry artifact path must be non-empty and output-relative")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("source-geometry artifact escaped its output directory") from exc
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    expected = str(record.get("sha256", ""))
    if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
        raise ValueError("source-geometry artifact must declare a lowercase SHA-256")
    actual = _sha256_file(resolved)
    if actual != expected:
        raise ValueError(f"source-geometry artifact SHA-256 mismatch for {relative.as_posix()}")
    return resolved


def load_source_geometry(path: str | Path) -> SourceGeometry:
    """Load and verify ``source-geometry.safe.json`` and its local artifacts."""

    manifest_path = Path(path).resolve()
    if manifest_path.is_dir():
        manifest_path = manifest_path / "source-geometry.safe.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    root = manifest_path.parent.resolve()

    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("source-geometry manifest must be a JSON object")
    if manifest.get("stage") != "refworld-source-geometry":
        raise ValueError("not a RefWorld source-geometry manifest")
    backend = str(manifest.get("backend", "")).strip()
    if not backend:
        raise ValueError("source-geometry backend must be explicit")

    input_meta = manifest.get("input")
    if not isinstance(input_meta, dict):
        raise ValueError("source-geometry input metadata is required")
    width = int(input_meta.get("width", 0))
    height = int(input_meta.get("height", 0))
    input_sha256 = str(input_meta.get("sha256", ""))
    if width <= 0 or height <= 0:
        raise ValueError("source-geometry input dimensions must be positive")
    if len(input_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in input_sha256):
        raise ValueError("source-geometry input SHA-256 is invalid")

    camera_meta = manifest.get("camera")
    if not isinstance(camera_meta, dict):
        raise ValueError("source-geometry camera is required")
    camera = Camera(
        intrinsics=tuple(float(v) for v in camera_meta.get("intrinsics", [])),
        extrinsics=tuple(float(v) for v in camera_meta.get("extrinsics", [])),
        convention=str(camera_meta.get("convention", "")),
    )
    if camera.convention != OPENGL_C2W:
        raise ValueError(f"source geometry must use {OPENGL_C2W}")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("source-geometry artifacts must be a list")
    by_kind: dict[str, dict[str, Any]] = {}
    for record in artifacts:
        if not isinstance(record, dict):
            raise ValueError("source-geometry artifact record must be an object")
        kind = str(record.get("kind", ""))
        if kind in by_kind:
            raise ValueError(f"duplicate source-geometry artifact kind: {kind}")
        by_kind[kind] = record

    try:
        depth_record = by_kind["source-depth-npy"]
        confidence_record = by_kind["source-confidence-raw-npy"]
    except KeyError as exc:
        raise ValueError(f"missing required source-geometry artifact: {exc.args[0]}") from exc

    depth_path = _safe_artifact(root, depth_record)
    confidence_path = _safe_artifact(root, confidence_record)
    depth = np.load(depth_path, allow_pickle=False)
    confidence_raw = np.load(confidence_path, allow_pickle=False)

    expected_shape = (height, width)
    if depth.shape != expected_shape or confidence_raw.shape != expected_shape:
        raise ValueError(
            "source-geometry arrays must match input HxW: "
            f"depth={depth.shape}, confidence={confidence_raw.shape}, expected={expected_shape}"
        )
    if not np.issubdtype(depth.dtype, np.floating) or not np.issubdtype(confidence_raw.dtype, np.floating):
        raise ValueError("source-geometry depth/confidence arrays must be floating point")
    if not np.all(np.isfinite(depth)) or np.any(depth <= 0.0):
        raise ValueError("source-geometry depth must contain only finite positive optical-axis depth")
    if not np.all(np.isfinite(confidence_raw)):
        raise ValueError("source-geometry raw confidence must be finite")

    geometry_meta = manifest.get("geometry")
    if not isinstance(geometry_meta, dict):
        raise ValueError("source-geometry semantics metadata is required")
    confidence_calibration = geometry_meta.get("confidence_calibration")
    if confidence_calibration is not None:
        raise ValueError(
            "v0 loader expects raw uncalibrated VGGT confidence; calibrated confidence must use an explicit later schema/version"
        )

    return SourceGeometry(
        backend=backend,
        input_sha256=input_sha256,
        width=width,
        height=height,
        camera=camera,
        depth=np.asarray(depth, dtype=np.float32),
        confidence_raw=np.asarray(confidence_raw, dtype=np.float32),
        metadata=manifest,
    )
