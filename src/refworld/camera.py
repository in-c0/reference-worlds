"""Canonical camera math for RefWorldBench.

Benchmark cameras use a right-handed OpenGL camera-to-world transform:
local +X = right, +Y = up, -Z = view direction. Matrices are flattened row-major
at API boundaries but manipulated as 4x4 NumPy arrays internally.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .adapters.base import Camera


OPENGL_C2W = "opengl-camera-to-world"


def _matrix(camera: Camera) -> np.ndarray:
    if camera.convention != OPENGL_C2W:
        raise ValueError(f"expected {OPENGL_C2W}, got {camera.convention!r}")
    matrix = np.asarray(camera.extrinsics, dtype=np.float64).reshape(4, 4)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("camera extrinsics must be finite")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("camera-to-world matrix must have homogeneous bottom row [0,0,0,1]")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("camera rotation must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6):
        raise ValueError("camera rotation must be proper right-handed rotation")
    return matrix


def _camera_like(camera: Camera, matrix: np.ndarray) -> Camera:
    return Camera(
        intrinsics=camera.intrinsics,
        extrinsics=tuple(float(v) for v in matrix.reshape(-1)),
        convention=OPENGL_C2W,
    )


def rotation_y(degrees: float) -> np.ndarray:
    radians = math.radians(float(degrees))
    c, s = math.cos(radians), math.sin(radians)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def rotation_x(degrees: float) -> np.ndarray:
    radians = math.radians(float(degrees))
    c, s = math.cos(radians), math.sin(radians)
    return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def yaw(camera: Camera, degrees: float) -> Camera:
    """Rotate orientation around the camera's local +Y axis; position is fixed."""

    matrix = _matrix(camera).copy()
    matrix[:3, :3] = matrix[:3, :3] @ rotation_y(degrees)
    return _camera_like(camera, matrix)


def pitch(camera: Camera, degrees: float) -> Camera:
    """Rotate orientation around the camera's local +X axis; position is fixed."""

    matrix = _matrix(camera).copy()
    matrix[:3, :3] = matrix[:3, :3] @ rotation_x(degrees)
    return _camera_like(camera, matrix)


def translate_local(camera: Camera, xyz: Iterable[float]) -> Camera:
    """Translate in camera-local axes while preserving orientation."""

    delta = np.asarray(tuple(xyz), dtype=np.float64)
    if delta.shape != (3,) or not np.all(np.isfinite(delta)):
        raise ValueError("local translation must contain exactly three finite values")
    matrix = _matrix(camera).copy()
    matrix[:3, 3] += matrix[:3, :3] @ delta
    return _camera_like(camera, matrix)


def view_direction(camera: Camera) -> tuple[float, float, float]:
    """Return the normalized world-space view direction (-local Z)."""

    rotation = _matrix(camera)[:3, :3]
    direction = rotation @ np.asarray([0.0, 0.0, -1.0])
    return tuple(float(v) for v in direction)
