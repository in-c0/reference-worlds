"""MVSNet/BlendedMVS metadata parsing.

MVSNet camera files store an extrinsic ``E=[R|t]`` in the convention used by
its COLMAP converter: world-to-camera with OpenCV camera axes (+X right,
+Y down, +Z forward). RefWorldBench converts this explicitly to its canonical
right-handed OpenGL camera-to-world convention (+X right, +Y up, -Z forward).

Published camera matrices are text calibrations and can contain small rounding
errors. Once a source rotation has passed the dataset-level near-rotation checks,
we project it to the nearest proper SO(3) matrix before constructing a canonical
camera. This prevents a calibration that is valid to the source precision from
later failing RefWorld's stricter canonical rotation invariant. The correction
magnitude is retained as metadata for auditability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..adapters.base import Camera
from ..camera import OPENGL_C2W, view_direction

_CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])


@dataclass(frozen=True)
class PairRecord:
    reference_id: int
    source_ids: tuple[int, ...]
    scores: tuple[float, ...]


@dataclass(frozen=True)
class MVSNetCamera:
    camera: Camera
    depth_min: float
    depth_interval: float
    depth_num: float | None = None
    depth_max: float | None = None
    source_convention: str = "opencv-world-to-camera"
    rotation_orthonormalization_frobenius: float = 0.0


def parse_pair_text(text: str) -> tuple[PairRecord, ...]:
    """Parse MVSNet ``pair.txt`` preserving published record/source order."""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("pair.txt is empty")
    try:
        expected = int(lines[0])
    except ValueError as exc:
        raise ValueError("pair.txt first line must be the reference-view count") from exc
    if expected < 1:
        raise ValueError("pair.txt reference-view count must be positive")
    if len(lines) != 1 + 2 * expected:
        raise ValueError(
            f"pair.txt declares {expected} records but contains {(len(lines) - 1) // 2} complete records"
        )

    records: list[PairRecord] = []
    seen_refs: set[int] = set()
    for index in range(expected):
        try:
            reference_id = int(lines[1 + 2 * index])
        except ValueError as exc:
            raise ValueError(f"invalid reference id in pair record {index}") from exc
        if reference_id < 0 or reference_id in seen_refs:
            raise ValueError("pair.txt reference ids must be unique non-negative integers")
        seen_refs.add(reference_id)

        tokens = lines[2 + 2 * index].split()
        if not tokens:
            raise ValueError(f"missing source list for reference {reference_id}")
        try:
            count = int(tokens[0])
        except ValueError as exc:
            raise ValueError(f"invalid source count for reference {reference_id}") from exc
        if count < 0 or len(tokens) != 1 + 2 * count:
            raise ValueError(f"malformed source list for reference {reference_id}")

        source_ids: list[int] = []
        scores: list[float] = []
        for j in range(count):
            try:
                source_id = int(tokens[1 + 2 * j])
                score = float(tokens[2 + 2 * j])
            except ValueError as exc:
                raise ValueError(f"invalid source entry for reference {reference_id}") from exc
            if source_id < 0 or source_id == reference_id:
                raise ValueError("source ids must be non-negative and differ from the reference id")
            if not math.isfinite(score):
                raise ValueError("view-selection scores must be finite")
            source_ids.append(source_id)
            scores.append(score)

        records.append(PairRecord(reference_id, tuple(source_ids), tuple(scores)))

    return tuple(records)


def _numeric_rows(lines: Sequence[str], start: int, rows: int, cols: int, label: str) -> np.ndarray:
    parsed: list[list[float]] = []
    for row in lines[start : start + rows]:
        values = row.split()
        if len(values) != cols:
            raise ValueError(f"{label} row must contain {cols} values")
        try:
            numbers = [float(value) for value in values]
        except ValueError as exc:
            raise ValueError(f"{label} contains a non-numeric value") from exc
        if not all(math.isfinite(value) for value in numbers):
            raise ValueError(f"{label} must be finite")
        parsed.append(numbers)
    if len(parsed) != rows:
        raise ValueError(f"{label} is incomplete")
    return np.asarray(parsed, dtype=np.float64)


def _nearest_proper_rotation(rotation: np.ndarray) -> tuple[np.ndarray, float]:
    """Project an already-near-valid 3x3 rotation to the nearest SO(3) matrix."""

    u, _, vt = np.linalg.svd(np.asarray(rotation, dtype=np.float64))
    proper = u @ vt
    if float(np.linalg.det(proper)) < 0.0:
        # This branch is not expected after the source determinant check, but
        # keeps the projection mathematically well-defined and right-handed.
        u = u.copy()
        u[:, -1] *= -1.0
        proper = u @ vt
    correction = float(np.linalg.norm(proper - rotation, ord="fro"))
    return proper, correction


def parse_camera_text(text: str) -> MVSNetCamera:
    """Parse a MVSNet ``*_cam.txt`` file into the canonical benchmark camera."""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    try:
        extrinsic_index = lines.index("extrinsic")
        intrinsic_index = lines.index("intrinsic")
    except ValueError as exc:
        raise ValueError("camera file must contain extrinsic and intrinsic sections") from exc
    if intrinsic_index <= extrinsic_index + 4:
        raise ValueError("camera extrinsic section is malformed")

    w2c_cv = _numeric_rows(lines, extrinsic_index + 1, 4, 4, "extrinsic")
    if not np.allclose(w2c_cv[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError("camera extrinsic bottom row must be [0,0,0,1]")
    rotation = w2c_cv[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("camera extrinsic rotation must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-5):
        raise ValueError("camera extrinsic rotation must be a proper rotation")

    # BlendedMVS/MVSNet calibration files are decimal text. A rotation that is
    # valid to their published precision can be a few e-6 away from SO(3), while
    # RefWorld canonical cameras deliberately enforce a stricter 1e-6 invariant.
    # Normalize only after the source matrix has already passed the 1e-5 checks.
    normalized_rotation, rotation_correction = _nearest_proper_rotation(rotation)
    w2c_cv = w2c_cv.copy()
    w2c_cv[:3, :3] = normalized_rotation

    k = _numeric_rows(lines, intrinsic_index + 1, 3, 3, "intrinsic")
    if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1], atol=1e-8):
        raise ValueError("invalid pinhole intrinsic matrix")

    depth_index = intrinsic_index + 4
    if depth_index >= len(lines):
        raise ValueError("camera file is missing depth parameters")
    try:
        depth = [float(value) for value in lines[depth_index].split()]
    except ValueError as exc:
        raise ValueError("depth parameters must be numeric") from exc
    if len(depth) not in {2, 3, 4} or not all(math.isfinite(value) for value in depth):
        raise ValueError("depth parameters must contain 2, 3, or 4 finite values")
    if depth[0] <= 0 or depth[1] <= 0:
        raise ValueError("depth_min and depth_interval must be positive")

    # p_cv = E_cv p_world; p_gl = A p_cv, where A flips camera Y/Z.
    # Therefore E_gl = A E_cv and canonical C2W_gl = inv(E_gl).
    w2c_gl = _CV_TO_GL @ w2c_cv
    c2w_gl = np.linalg.inv(w2c_gl)

    camera = Camera(
        intrinsics=tuple(float(value) for value in k.reshape(-1)),
        extrinsics=tuple(float(value) for value in c2w_gl.reshape(-1)),
        convention=OPENGL_C2W,
    )
    # Force canonical validation here so imported-camera failures happen at the
    # representation boundary, not later during pose metrics or rendering.
    view_direction(camera)

    return MVSNetCamera(
        camera=camera,
        depth_min=float(depth[0]),
        depth_interval=float(depth[1]),
        depth_num=None if len(depth) < 3 else float(depth[2]),
        depth_max=None if len(depth) < 4 else float(depth[3]),
        rotation_orthonormalization_frobenius=rotation_correction,
    )


def camera_pose_separation(a: Camera, b: Camera) -> dict[str, float]:
    """Return view-direction angle and camera-center distance in source units."""

    da = np.asarray(view_direction(a), dtype=np.float64)
    db = np.asarray(view_direction(b), dtype=np.float64)
    dot = float(np.clip(np.dot(da, db), -1.0, 1.0))
    angle = math.degrees(math.acos(dot))

    a_matrix = np.asarray(a.extrinsics, dtype=np.float64).reshape(4, 4)
    b_matrix = np.asarray(b.extrinsics, dtype=np.float64).reshape(4, 4)
    center_distance = float(np.linalg.norm(a_matrix[:3, 3] - b_matrix[:3, 3]))
    return {
        "view_direction_angle_deg": angle,
        "center_distance_source_units": center_distance,
    }
