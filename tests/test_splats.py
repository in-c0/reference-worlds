import numpy as np
import pytest

from refworld.adapters.base import Camera
from refworld.camera import OPENGL_C2W
from refworld.splats import rgbd_to_gaussian_arrays, write_gaussian_ply


def _camera():
    return Camera(
        intrinsics=(2.0, 0.0, 0.5, 0.0, 2.0, 0.5, 0.0, 0.0, 1.0),
        extrinsics=(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ),
        convention=OPENGL_C2W,
    )


def test_source_splat_places_pixels_in_canonical_camera_space():
    rgb = np.asarray(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )
    depth = np.full((2, 2), 2.0, dtype=np.float32)

    vertices, metadata = rgbd_to_gaussian_arrays(rgb, depth, _camera(), max_splats=4)

    assert metadata["splat_count"] == 4
    assert metadata["sampling_stride"] == 1
    np.testing.assert_allclose(
        [vertices["x"][0], vertices["y"][0], vertices["z"][0]],
        [-0.5, 0.5, -2.0],
        atol=1e-6,
    )
    quaternions = np.column_stack(
        [vertices[f"rot_{index}"] for index in range(4)]
    )
    np.testing.assert_allclose(np.linalg.norm(quaternions, axis=1), 1.0, atol=1e-5)


def test_source_splat_encodes_degree_zero_sh_color():
    rgb = np.asarray([[[255, 128, 0]]], dtype=np.uint8)
    depth = np.asarray([[1.0]], dtype=np.float32)

    vertices, _ = rgbd_to_gaussian_arrays(rgb, depth, _camera(), max_splats=1)

    c0 = 0.28209479177387814
    expected = (rgb[0, 0].astype(np.float64) / 255.0 - 0.5) / c0
    actual = np.asarray([vertices[f"f_dc_{i}"][0] for i in range(3)])
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert vertices["opacity"][0] > 0.0


def test_source_splat_budget_uses_deterministic_grid_stride():
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    depth = np.ones((4, 4), dtype=np.float32)

    vertices, metadata = rgbd_to_gaussian_arrays(rgb, depth, _camera(), max_splats=4)

    assert metadata["sampling_stride"] == 2
    assert len(vertices) == 4


def test_write_gaussian_ply_has_expected_binary_layout(tmp_path):
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    depth = np.ones((2, 2), dtype=np.float32)
    vertices, _ = rgbd_to_gaussian_arrays(rgb, depth, _camera(), max_splats=4)

    path = write_gaussian_ply(tmp_path / "source.ply", vertices)
    payload = path.read_bytes()

    marker = b"end_header\n"
    assert payload.startswith(b"ply\nformat binary_little_endian 1.0\n")
    assert marker in payload
    body = payload.split(marker, 1)[1]
    assert len(body) == vertices.nbytes
    restored = np.frombuffer(body, dtype=vertices.dtype)
    np.testing.assert_array_equal(restored, vertices)


def test_source_splat_rejects_invalid_depth():
    rgb = np.zeros((1, 1, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="positive optical-axis"):
        rgbd_to_gaussian_arrays(rgb, np.asarray([[0.0]], dtype=np.float32), _camera())
