"""Dependency-light pinhole RGB-D forward warp for RefWorld-0.

This is intentionally a simple geometric baseline, not a novel renderer. It
projects real source pixels into a target camera, z-buffers collisions, and
leaves disocclusions explicitly unresolved for a later repaint backend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..adapters.base import Camera
from ..camera import OPENGL_C2W
from ..proposals import ObservationView, WarpResult


def _camera_matrices(camera: Camera) -> tuple[np.ndarray, np.ndarray]:
    if camera.convention != OPENGL_C2W:
        raise ValueError(f"expected camera convention {OPENGL_C2W}")
    c2w = np.asarray(camera.extrinsics, dtype=np.float64).reshape(4, 4)
    if not np.all(np.isfinite(c2w)):
        raise ValueError("camera extrinsics must be finite")
    try:
        w2c = np.linalg.inv(c2w)
    except np.linalg.LinAlgError as exc:
        raise ValueError("camera-to-world matrix is not invertible") from exc
    return c2w, w2c


def _intrinsics(camera: Camera) -> tuple[float, float, float, float]:
    k = np.asarray(camera.intrinsics, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(k)):
        raise ValueError("camera intrinsics must be finite")
    fx, skew, cx = k[0]
    _, fy, cy = k[1]
    if fx <= 0 or fy <= 0:
        raise ValueError("camera focal lengths must be positive")
    if abs(float(skew)) > 1e-9:
        raise ValueError("pinhole warp v0 requires zero skew")
    if not np.allclose(k[2], [0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("camera K bottom row must be [0,0,1]")
    return float(fx), float(fy), float(cx), float(cy)


def _validate_source(rgb: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(rgb)
    d = np.asarray(depth)
    if image.ndim != 3 or image.shape[2] not in (1, 3, 4) or image.size == 0:
        raise ValueError("rgb must be a non-empty HxWxC array")
    if image.dtype == np.dtype("O"):
        raise ValueError("rgb cannot use object dtype")
    if d.shape != image.shape[:2]:
        raise ValueError("depth must be HxW matching rgb")
    if not np.issubdtype(d.dtype, np.floating):
        raise ValueError("depth must use a floating dtype")
    if np.any(np.isinf(d)):
        raise ValueError("depth cannot contain infinities")
    return image, d.astype(np.float64, copy=False)


def forward_warp_rgbd(
    rgb: np.ndarray,
    depth: np.ndarray,
    source_camera: Camera,
    target_camera: Camera,
    *,
    min_depth: float = 1e-6,
    backend_name: str = "pinhole-forward@0.1",
) -> WarpResult:
    """Forward-project an RGB-D observation into a target pinhole camera.

    Depth semantics are **positive optical-axis depth**, not Euclidean ray
    distance. Under the canonical OpenGL camera convention, a valid source point
    has camera-space ``z = -depth``.

    Projection uses nearest-pixel rasterization plus a deterministic z-buffer.
    This deliberately produces holes at disocclusions; it does not inpaint.
    """

    image, d = _validate_source(rgb, depth)
    if not math.isfinite(float(min_depth)) or min_depth <= 0:
        raise ValueError("min_depth must be finite and positive")
    if not backend_name.strip():
        raise ValueError("backend_name cannot be empty")

    height, width = image.shape[:2]
    sfx, sfy, scx, scy = _intrinsics(source_camera)
    tfx, tfy, tcx, tcy = _intrinsics(target_camera)
    source_c2w, _ = _camera_matrices(source_camera)
    _, target_w2c = _camera_matrices(target_camera)

    vv, uu = np.indices((height, width), dtype=np.float64)
    valid_depth = np.isfinite(d) & (d > float(min_depth))
    source_indices = np.flatnonzero(valid_depth.reshape(-1))

    output = np.zeros_like(image)
    observed = np.zeros((height, width), dtype=bool)
    confidence = np.zeros((height, width), dtype=np.float32)

    if source_indices.size == 0:
        return WarpResult(
            rgb=output,
            observed_mask=observed,
            confidence=confidence,
            backend=backend_name,
            metadata={
                "depth_semantics": "positive-optical-axis-depth",
                "rasterization": "nearest-z-buffer",
                "valid_source_pixels": 0,
                "projected_in_bounds": 0,
                "visible_target_pixels": 0,
                "collision_count": 0,
            },
        )

    flat_d = d.reshape(-1)[source_indices]
    flat_u = uu.reshape(-1)[source_indices]
    flat_v = vv.reshape(-1)[source_indices]

    # Canonical OpenGL camera: +X right, +Y up, -Z forward. Image +v is down.
    x = (flat_u - scx) / sfx * flat_d
    y = -(flat_v - scy) / sfy * flat_d
    z = -flat_d
    homogeneous = np.stack([x, y, z, np.ones_like(z)], axis=1)

    world = (source_c2w @ homogeneous.T).T
    target = (target_w2c @ world.T).T
    forward_depth = -target[:, 2]

    front = np.isfinite(forward_depth) & (forward_depth > float(min_depth))
    target = target[front]
    forward_depth = forward_depth[front]
    source_indices = source_indices[front]

    if source_indices.size:
        projected_u = tfx * (target[:, 0] / forward_depth) + tcx
        projected_v = -tfy * (target[:, 1] / forward_depth) + tcy
        pixel_u = np.rint(projected_u).astype(np.int64)
        pixel_v = np.rint(projected_v).astype(np.int64)

        in_bounds = (
            np.isfinite(projected_u)
            & np.isfinite(projected_v)
            & (pixel_u >= 0)
            & (pixel_u < width)
            & (pixel_v >= 0)
            & (pixel_v < height)
        )
        pixel_u = pixel_u[in_bounds]
        pixel_v = pixel_v[in_bounds]
        forward_depth = forward_depth[in_bounds]
        source_indices = source_indices[in_bounds]
    else:
        pixel_u = np.empty(0, dtype=np.int64)
        pixel_v = np.empty(0, dtype=np.int64)

    projected_in_bounds = int(source_indices.size)
    if projected_in_bounds:
        target_linear = pixel_v * width + pixel_u
        # Primary key target pixel, secondary nearest depth, tertiary source
        # pixel index for deterministic exact-depth ties.
        order = np.lexsort((source_indices, forward_depth, target_linear))
        sorted_targets = target_linear[order]
        first = np.ones(order.size, dtype=bool)
        first[1:] = sorted_targets[1:] != sorted_targets[:-1]
        winners = order[first]

        win_target = target_linear[winners]
        win_source = source_indices[winners]
        out_flat = output.reshape(-1, image.shape[2])
        src_flat = image.reshape(-1, image.shape[2])
        out_flat[win_target] = src_flat[win_source]
        observed.reshape(-1)[win_target] = True
        confidence.reshape(-1)[win_target] = 1.0

    visible = int(np.count_nonzero(observed))
    return WarpResult(
        rgb=output,
        observed_mask=observed,
        confidence=confidence,
        backend=backend_name,
        metadata={
            "depth_semantics": "positive-optical-axis-depth",
            "rasterization": "nearest-z-buffer",
            "valid_source_pixels": int(np.count_nonzero(valid_depth)),
            "projected_in_bounds": projected_in_bounds,
            "visible_target_pixels": visible,
            "collision_count": max(0, projected_in_bounds - visible),
        },
    )


@dataclass
class PinholeWarpBackend:
    """Single-observation baseline implementation of the WarpBackend protocol."""

    depth_by_observation_id: Mapping[str, np.ndarray]
    name: str = "pinhole-forward@0.1"

    def warp(
        self,
        observations: Sequence[ObservationView],
        target_camera: Camera,
    ) -> WarpResult:
        if len(observations) != 1:
            raise ValueError("PinholeWarpBackend v0 requires exactly one source observation")
        observation = observations[0]
        if observation.observation_id not in self.depth_by_observation_id:
            raise KeyError(f"no depth for observation {observation.observation_id!r}")
        return forward_warp_rgbd(
            observation.image,
            self.depth_by_observation_id[observation.observation_id],
            observation.camera,
            target_camera,
            backend_name=self.name,
        )
