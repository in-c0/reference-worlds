"""Explicit Gaussian-splat construction from one calibrated RGB-D observation.

This is a diagnostic representation, not a hidden-view generator. Every splat
comes from a supplied image pixel and inferred source depth. It is useful for
checking source-camera/depth/export/renderer correctness before generative world
completion is introduced.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .adapters.base import Camera
from .camera import OPENGL_C2W, view_direction

_SH_C0 = 0.28209479177387814


def _rotation_matrices_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Convert proper Nx3x3 rotation matrices to normalized wxyz quaternions."""

    matrices = np.asarray(rotation, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError("rotation must be Nx3x3")
    out = np.empty((matrices.shape[0], 4), dtype=np.float64)
    for i, m in enumerate(matrices):
        trace = float(np.trace(m))
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (m[2, 1] - m[1, 2]) / s
            qy = (m[0, 2] - m[2, 0]) / s
            qz = (m[1, 0] - m[0, 1]) / s
        else:
            index = int(np.argmax(np.diag(m)))
            if index == 0:
                s = math.sqrt(max(0.0, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) * 2.0
                qw = (m[2, 1] - m[1, 2]) / s
                qx = 0.25 * s
                qy = (m[0, 1] + m[1, 0]) / s
                qz = (m[0, 2] + m[2, 0]) / s
            elif index == 1:
                s = math.sqrt(max(0.0, 1.0 + m[1, 1] - m[0, 0] - m[2, 2])) * 2.0
                qw = (m[0, 2] - m[2, 0]) / s
                qx = (m[0, 1] + m[1, 0]) / s
                qy = 0.25 * s
                qz = (m[1, 2] + m[2, 1]) / s
            else:
                s = math.sqrt(max(0.0, 1.0 + m[2, 2] - m[0, 0] - m[1, 1])) * 2.0
                qw = (m[1, 0] - m[0, 1]) / s
                qx = (m[0, 2] + m[2, 0]) / s
                qy = (m[1, 2] + m[2, 1]) / s
                qz = 0.25 * s
        q = np.asarray([qw, qx, qy, qz], dtype=np.float64)
        norm = float(np.linalg.norm(q))
        if not math.isfinite(norm) or norm <= 1e-12:
            raise ValueError("failed to construct a finite rotation quaternion")
        out[i] = q / norm
    return out


def rgbd_to_gaussian_arrays(
    rgb: np.ndarray,
    depth: np.ndarray,
    camera: Camera,
    *,
    max_splats: int = 500_000,
    footprint_scale: float = 0.58,
    thickness_ratio: float = 0.05,
    opacity: float = 0.99,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Convert one source RGB-D image into standard 3DGS PLY vertex fields."""

    image = np.asarray(rgb)
    d = np.asarray(depth, dtype=np.float64)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("rgb must be uint8 HxWx3")
    if d.shape != image.shape[:2]:
        raise ValueError("depth must match rgb HxW")
    if not np.all(np.isfinite(d)) or np.any(d <= 0.0):
        raise ValueError("depth must contain finite positive optical-axis values")
    if camera.convention != OPENGL_C2W:
        raise ValueError(f"expected {OPENGL_C2W}")
    view_direction(camera)  # canonical extrinsic validation
    k = np.asarray(camera.intrinsics, dtype=np.float64).reshape(3, 3)
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    if fx <= 0 or fy <= 0 or abs(float(k[0, 1])) > 1e-9:
        raise ValueError("source-splat v0 requires positive focal lengths and zero skew")
    if max_splats <= 0:
        raise ValueError("max_splats must be positive")
    if not (0.0 < footprint_scale <= 2.0):
        raise ValueError("footprint_scale must lie in (0,2]")
    if not (0.0 < thickness_ratio <= 1.0):
        raise ValueError("thickness_ratio must lie in (0,1]")
    if not (0.0 < opacity < 1.0):
        raise ValueError("opacity must lie in (0,1)")

    height, width = d.shape
    total = height * width
    stride = max(1, int(math.ceil(math.sqrt(total / max_splats))))
    vv, uu = np.indices((height, width), dtype=np.float64)
    selected = np.zeros((height, width), dtype=bool)
    selected[::stride, ::stride] = True
    flat = np.flatnonzero(selected.reshape(-1))

    z = d.reshape(-1)[flat]
    u = uu.reshape(-1)[flat]
    v = vv.reshape(-1)[flat]
    x = (u - cx) / fx * z
    y = -(v - cy) / fy * z
    cam_points = np.stack([x, y, -z, np.ones_like(z)], axis=1)

    c2w = np.asarray(camera.extrinsics, dtype=np.float64).reshape(4, 4)
    world_points = (c2w @ cam_points.T).T[:, :3]
    camera_rotation = c2w[:3, :3]

    ray_cam = np.stack([x, y, -z], axis=1)
    ray_cam /= np.linalg.norm(ray_cam, axis=1, keepdims=True)
    ray_world = ray_cam @ camera_rotation.T
    ray_world /= np.linalg.norm(ray_world, axis=1, keepdims=True)

    up_world = camera_rotation[:, 1]
    up = np.broadcast_to(up_world, ray_world.shape)
    x_axis = np.cross(ray_world, up)
    x_norm = np.linalg.norm(x_axis, axis=1, keepdims=True)
    degenerate = x_norm[:, 0] <= 1e-8
    if np.any(degenerate):
        fallback = np.broadcast_to(camera_rotation[:, 0], ray_world.shape)
        x_axis[degenerate] = fallback[degenerate]
        x_norm = np.linalg.norm(x_axis, axis=1, keepdims=True)
    x_axis /= x_norm
    y_axis = np.cross(ray_world, x_axis)
    y_axis /= np.linalg.norm(y_axis, axis=1, keepdims=True)
    rotation = np.stack([x_axis, y_axis, ray_world], axis=2)
    quaternions = _rotation_matrices_to_quaternion_wxyz(rotation)

    sx = np.maximum(z / fx * footprint_scale * stride, 1e-6)
    sy = np.maximum(z / fy * footprint_scale * stride, 1e-6)
    sz = np.maximum(np.minimum(sx, sy) * thickness_ratio, 1e-6)
    scales = np.stack([sx, sy, sz], axis=1)

    colors = image.reshape(-1, 3)[flat].astype(np.float64) / 255.0
    f_dc = (colors - 0.5) / _SH_C0
    opacity_logit = math.log(opacity / (1.0 - opacity))

    names = [
        "x", "y", "z", "nx", "ny", "nz",
        "f_dc_0", "f_dc_1", "f_dc_2",
        "opacity", "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    dtype = np.dtype([(name, "<f4") for name in names])
    vertices = np.empty(flat.size, dtype=dtype)
    vertices["x"], vertices["y"], vertices["z"] = world_points.T.astype(np.float32)
    vertices["nx"] = 0.0
    vertices["ny"] = 0.0
    vertices["nz"] = 0.0
    for channel in range(3):
        vertices[f"f_dc_{channel}"] = f_dc[:, channel].astype(np.float32)
    vertices["opacity"] = np.float32(opacity_logit)
    log_scales = np.log(scales).astype(np.float32)
    for axis in range(3):
        vertices[f"scale_{axis}"] = log_scales[:, axis]
    for component in range(4):
        vertices[f"rot_{component}"] = quaternions[:, component].astype(np.float32)

    metadata = {
        "source_pixels": int(total),
        "splat_count": int(flat.size),
        "sampling_stride": int(stride),
        "footprint_scale": float(footprint_scale),
        "thickness_ratio": float(thickness_ratio),
        "opacity": float(opacity),
    }
    return vertices, metadata


def write_gaussian_ply(path: str | Path, vertices: np.ndarray) -> Path:
    """Write standard 3DGS fields as binary little-endian PLY."""

    output = Path(path)
    required = [
        "x", "y", "z", "nx", "ny", "nz",
        "f_dc_0", "f_dc_1", "f_dc_2",
        "opacity", "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    if vertices.dtype.names is None or list(vertices.dtype.names) != required:
        raise ValueError("vertices do not use the expected RefWorld 3DGS field layout")
    for name in required:
        if vertices.dtype[name] != np.dtype("<f4"):
            raise ValueError(f"PLY field {name} must be little-endian float32")

    output.parent.mkdir(parents=True, exist_ok=True)
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        "comment Generated by reference-worlds source RGB-D diagnostic",
        f"element vertex {len(vertices)}",
    ]
    header_lines.extend(f"property float {name}" for name in required)
    header_lines.append("end_header")
    header = ("\n".join(header_lines) + "\n").encode("ascii")
    with output.open("wb") as handle:
        handle.write(header)
        vertices.tofile(handle)
    return output
