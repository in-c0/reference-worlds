"""Camera registration primitives for RefWorldBench.

The canonical benchmark camera is right-handed OpenGL camera-to-world:
+X right, +Y up, -Z forward. 2D image coordinates remain conventional pixels:
+u right, +v down.

OpenCV PnP is used only as an optional solver. Its returned world-to-camera pose
uses OpenCV camera axes (+X right, +Y down, +Z forward), so conversion is explicit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .adapters.base import Camera
from .camera import OPENGL_C2W, _matrix

_CV_FROM_GL_3 = np.diag([1.0, -1.0, -1.0])
_CV_FROM_GL_4 = np.eye(4, dtype=np.float64)
_CV_FROM_GL_4[:3, :3] = _CV_FROM_GL_3


@dataclass(frozen=True)
class RegistrationResult:
    camera: Camera
    reprojection_rmse_px: float
    point_count: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "reprojection_rmse_px": self.reprojection_rmse_px,
            "point_count": self.point_count,
        }


def _intrinsic_matrix(intrinsics: Iterable[float]) -> np.ndarray:
    k = np.asarray(tuple(intrinsics), dtype=np.float64)
    if k.size != 9:
        raise ValueError("intrinsics must contain 9 values")
    k = k.reshape(3, 3)
    if not np.all(np.isfinite(k)):
        raise ValueError("intrinsics must be finite")
    if k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError("focal lengths must be positive")
    if not np.allclose(k[2], [0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("v0 pinhole intrinsics require bottom row [0,0,1]")
    return k


def project_world_points(camera: Camera, world_points: Sequence[Sequence[float]]) -> np.ndarray:
    """Project world points to +u-right/+v-down pixels using the canonical camera."""

    c2w = _matrix(camera)
    k = _intrinsic_matrix(camera.intrinsics)
    points = np.asarray(world_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("world_points must be a non-empty Nx3 array")
    if not np.all(np.isfinite(points)):
        raise ValueError("world_points must be finite")

    w2c_gl = np.linalg.inv(c2w)
    homogeneous = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    camera_gl = (w2c_gl @ homogeneous.T).T[:, :3]
    camera_cv = (_CV_FROM_GL_3 @ camera_gl.T).T

    depth = camera_cv[:, 2]
    if np.any(depth <= 0.0):
        raise ValueError("all projected points must lie in front of the camera")

    normalized = camera_cv[:, :2] / depth[:, None]
    u = k[0, 0] * normalized[:, 0] + k[0, 1] * normalized[:, 1] + k[0, 2]
    v = k[1, 1] * normalized[:, 1] + k[1, 2]
    return np.column_stack([u, v])


def _opencv_w2c_to_opengl_c2w(rotation_cv: np.ndarray, translation_cv: np.ndarray) -> np.ndarray:
    w2c_cv = np.eye(4, dtype=np.float64)
    w2c_cv[:3, :3] = rotation_cv
    w2c_cv[:3, 3] = translation_cv.reshape(3)
    w2c_gl = _CV_FROM_GL_4 @ w2c_cv
    return np.linalg.inv(w2c_gl)


def opencv_w2c_to_camera(
    extrinsic_w2c: Sequence[Sequence[float]],
    intrinsics: Iterable[float] | Sequence[Sequence[float]],
) -> Camera:
    """Convert an OpenCV camera-from-world pose into RefWorld's canonical camera.

    ``extrinsic_w2c`` may be 3x4 or homogeneous 4x4. OpenCV camera axes are
    +X right, +Y down, +Z forward. The returned camera is right-handed OpenGL
    camera-to-world (+X right, +Y up, -Z forward).
    """

    extrinsic = np.asarray(extrinsic_w2c, dtype=np.float64)
    if extrinsic.shape == (3, 4):
        w2c_cv = np.eye(4, dtype=np.float64)
        w2c_cv[:3] = extrinsic
    elif extrinsic.shape == (4, 4):
        w2c_cv = extrinsic.copy()
        if not np.allclose(w2c_cv[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
            raise ValueError("OpenCV homogeneous extrinsic must end with [0,0,0,1]")
    else:
        raise ValueError("OpenCV extrinsic must be 3x4 or 4x4")
    if not np.all(np.isfinite(w2c_cv)):
        raise ValueError("OpenCV extrinsic must be finite")

    rotation = w2c_cv[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("OpenCV extrinsic rotation must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-5):
        raise ValueError("OpenCV extrinsic rotation must be proper")

    k = np.asarray(intrinsics, dtype=np.float64)
    if k.size != 9:
        raise ValueError("intrinsics must contain 9 values")
    k = _intrinsic_matrix(k.reshape(-1))

    c2w_gl = np.linalg.inv(_CV_FROM_GL_4 @ w2c_cv)
    return Camera(
        intrinsics=tuple(float(v) for v in k.reshape(-1)),
        extrinsics=tuple(float(v) for v in c2w_gl.reshape(-1)),
        convention=OPENGL_C2W,
    )


def recover_camera_pnp(
    world_points: Sequence[Sequence[float]],
    image_points: Sequence[Sequence[float]],
    intrinsics: Iterable[float],
    *,
    distortion: Iterable[float] | None = None,
    refine_lm: bool = True,
) -> RegistrationResult:
    """Recover canonical OpenGL camera-to-world pose from 3D↔2D correspondences.

    This is deliberately a registration primitive, not a correspondence finder.
    Pixel coordinates use +u right / +v down. The solver's OpenCV pose is
    converted explicitly to RefWorldBench's OpenGL camera convention.
    """

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "recover_camera_pnp requires the optional camera dependency: "
            "pip install 'refworld-bench[camera]'"
        ) from exc

    object_points = np.asarray(world_points, dtype=np.float64)
    pixels = np.asarray(image_points, dtype=np.float64)
    if object_points.ndim != 2 or object_points.shape[1] != 3:
        raise ValueError("world_points must be an Nx3 array")
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("image_points must be an Nx2 array")
    if object_points.shape[0] != pixels.shape[0]:
        raise ValueError("world_points and image_points must contain the same number of points")
    if object_points.shape[0] < 6:
        raise ValueError("at least six 3D↔2D correspondences are required")
    if not np.all(np.isfinite(object_points)) or not np.all(np.isfinite(pixels)):
        raise ValueError("correspondences must be finite")

    k = _intrinsic_matrix(intrinsics)
    if distortion is None:
        dist = np.zeros((5, 1), dtype=np.float64)
    else:
        dist = np.asarray(tuple(distortion), dtype=np.float64).reshape(-1, 1)
        if dist.size not in {4, 5, 8, 12, 14} or not np.all(np.isfinite(dist)):
            raise ValueError("distortion must contain 4, 5, 8, 12, or 14 finite coefficients")

    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        pixels,
        k,
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("OpenCV solvePnP failed to recover a camera")

    if refine_lm and hasattr(cv2, "solvePnPRefineLM"):
        rvec, tvec = cv2.solvePnPRefineLM(object_points, pixels, k, dist, rvec, tvec)

    rotation_cv, _ = cv2.Rodrigues(rvec)
    camera = opencv_w2c_to_camera(
        np.column_stack([rotation_cv, tvec.reshape(3)]),
        k,
    )

    reprojected = project_world_points(camera, object_points)
    rmse = float(np.sqrt(np.mean(np.sum(np.square(reprojected - pixels), axis=1))))
    if not math.isfinite(rmse):
        raise RuntimeError("camera reprojection error is non-finite")

    return RegistrationResult(
        camera=camera,
        reprojection_rmse_px=rmse,
        point_count=int(object_points.shape[0]),
    )
