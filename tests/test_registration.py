import numpy as np
import pytest

from refworld.adapters.base import Camera
from refworld.camera import OPENGL_C2W
from refworld.registration import project_world_points, recover_camera_pnp


def _rotation_y(degrees):
    a = np.deg2rad(degrees)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _rotation_x(degrees):
    a = np.deg2rad(degrees)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _camera():
    rotation = _rotation_y(17.0) @ _rotation_x(-9.0)
    c2w = np.eye(4)
    c2w[:3, :3] = rotation
    c2w[:3, 3] = [1.2, 0.8, 2.3]
    intrinsics = (
        900.0, 0.0, 640.0,
        0.0, 880.0, 360.0,
        0.0, 0.0, 1.0,
    )
    return Camera(intrinsics, tuple(c2w.reshape(-1)), OPENGL_C2W)


def _world_points(camera):
    rng = np.random.default_rng(1234)
    camera_points = np.column_stack([
        rng.uniform(-1.2, 1.2, 30),
        rng.uniform(-0.9, 0.9, 30),
        rng.uniform(-8.0, -3.0, 30),
    ])
    c2w = np.asarray(camera.extrinsics).reshape(4, 4)
    return (c2w[:3, :3] @ camera_points.T).T + c2w[:3, 3]


def test_projection_and_pnp_recover_known_camera():
    expected = _camera()
    world = _world_points(expected)
    pixels = project_world_points(expected, world)
    got = recover_camera_pnp(world, pixels, expected.intrinsics)

    expected_matrix = np.asarray(expected.extrinsics).reshape(4, 4)
    got_matrix = np.asarray(got.camera.extrinsics).reshape(4, 4)

    assert got.point_count == len(world)
    assert got.reprojection_rmse_px < 1e-6
    assert got_matrix[:3, 3] == pytest.approx(expected_matrix[:3, 3], abs=1e-6)
    assert got_matrix[:3, :3] == pytest.approx(expected_matrix[:3, :3], abs=1e-6)


def test_project_rejects_points_behind_camera():
    camera = _camera()
    c2w = np.asarray(camera.extrinsics).reshape(4, 4)
    behind_camera = np.array([[0.0, 0.0, 1.0]])
    world = (c2w[:3, :3] @ behind_camera.T).T + c2w[:3, 3]
    with pytest.raises(ValueError):
        project_world_points(camera, world)


def test_pnp_requires_sufficient_correspondences():
    camera = _camera()
    world = _world_points(camera)[:5]
    pixels = project_world_points(camera, world)
    with pytest.raises(ValueError):
        recover_camera_pnp(world, pixels, camera.intrinsics)
