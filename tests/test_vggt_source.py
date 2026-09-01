import numpy as np

from refworld.camera import view_direction
from refworld.registration import opencv_w2c_to_camera, project_world_points
from refworld.runners.vggt_source import _original_from_square_transform, _remap_to_original


def test_opencv_identity_camera_projects_positive_z_to_image_center():
    k = np.asarray([[100.0, 0.0, 50.0], [0.0, 120.0, 40.0], [0.0, 0.0, 1.0]])
    w2c_cv = np.column_stack([np.eye(3), np.zeros(3)])

    camera = opencv_w2c_to_camera(w2c_cv, k)
    pixel = project_world_points(camera, [[0.0, 0.0, 2.0]])[0]

    np.testing.assert_allclose(pixel, [50.0, 40.0], atol=1e-9)
    np.testing.assert_allclose(view_direction(camera), [0.0, 0.0, 1.0], atol=1e-9)


def test_opencv_camera_center_is_preserved_during_convention_conversion():
    k = np.eye(3)
    # OpenCV camera-from-world with R=I and t=-C for C=(1,2,3).
    w2c_cv = np.column_stack([np.eye(3), np.asarray([-1.0, -2.0, -3.0])])

    camera = opencv_w2c_to_camera(w2c_cv, k)
    c2w = np.asarray(camera.extrinsics).reshape(4, 4)

    np.testing.assert_allclose(c2w[:3, 3], [1.0, 2.0, 3.0], atol=1e-9)


def test_square_loader_inverse_transform_uses_pixel_center_mapping():
    # Original 2x2 pixels occupy model coordinates [1,1] through [2,2].
    h, width, height = _original_from_square_transform(
        np.asarray([1.0, 1.0, 3.0, 3.0, 2.0, 2.0])
    )

    assert (width, height) == (2, 2)
    np.testing.assert_allclose(
        h,
        [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
        atol=1e-12,
    )


def test_remap_to_original_extracts_the_unpadded_source_region():
    model_map = np.arange(16, dtype=np.float32).reshape(4, 4)
    h, width, height = _original_from_square_transform(
        np.asarray([1.0, 1.0, 3.0, 3.0, 2.0, 2.0])
    )

    restored = _remap_to_original(model_map, h, width, height)

    np.testing.assert_allclose(restored, model_map[1:3, 1:3], atol=1e-6)


def test_opencv_conversion_rejects_improper_rotation():
    k = np.eye(3)
    bad = np.eye(4)
    bad[0, 0] = -1.0

    try:
        opencv_w2c_to_camera(bad, k)
    except ValueError as exc:
        assert "proper" in str(exc)
    else:
        raise AssertionError("expected improper rotation to be rejected")
