import numpy as np

from refworld.adapters.base import Camera
from refworld.camera import OPENGL_C2W
from refworld.proposals import ObservationView
from refworld.warps.pinhole import PinholeWarpBackend, forward_warp_rgbd


def _camera(*, fx=10.0, fy=10.0, cx=1.0, cy=1.0, tx=0.0):
    return Camera(
        intrinsics=(fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0),
        extrinsics=(
            1.0, 0.0, 0.0, tx,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ),
        convention=OPENGL_C2W,
    )


def test_identity_warp_reproduces_every_valid_source_pixel_exactly():
    rgb = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)
    depth = np.ones((3, 4), dtype=np.float32)
    camera = _camera(cx=1.5, cy=1.0)

    result = forward_warp_rgbd(rgb, depth, camera, camera)

    assert np.array_equal(result.rgb, rgb)
    assert np.all(result.observed_mask)
    assert np.all(result.confidence == 1.0)
    assert result.metadata["valid_source_pixels"] == 12
    assert result.metadata["projected_in_bounds"] == 12
    assert result.metadata["visible_target_pixels"] == 12
    assert result.metadata["collision_count"] == 0


def test_camera_translation_creates_expected_disocclusion_hole():
    # Moving the target camera +0.1 world units in X with fx=10 and depth=1
    # shifts the static plane left by exactly one target pixel.
    rgb = np.zeros((3, 4, 3), dtype=np.uint8)
    for u in range(4):
        rgb[:, u] = [u + 1, 0, 0]
    depth = np.ones((3, 4), dtype=np.float32)
    source = _camera(cx=1.5, cy=1.0)
    target = _camera(cx=1.5, cy=1.0, tx=0.1)

    result = forward_warp_rgbd(rgb, depth, source, target)

    assert np.all(result.observed_mask[:, :3])
    assert not np.any(result.observed_mask[:, 3])
    assert np.array_equal(result.rgb[:, 0], rgb[:, 1])
    assert np.array_equal(result.rgb[:, 1], rgb[:, 2])
    assert np.array_equal(result.rgb[:, 2], rgb[:, 3])
    assert np.all(result.rgb[:, 3] == 0)


def test_z_buffer_selects_nearest_source_when_pixels_collapse():
    # Narrow target focal length collapses all three source samples onto u=1.
    rgb = np.array([[[10, 0, 0], [20, 0, 0], [30, 0, 0]]], dtype=np.uint8)
    depth = np.array([[2.0, 1.0, 3.0]], dtype=np.float32)
    source = _camera(fx=4.0, fy=4.0, cx=1.0, cy=0.0)
    target = _camera(fx=1.0, fy=1.0, cx=1.0, cy=0.0)

    result = forward_warp_rgbd(rgb, depth, source, target)

    assert result.observed_mask.tolist() == [[False, True, False]]
    assert np.array_equal(result.rgb[0, 1], [20, 0, 0])
    assert result.metadata["projected_in_bounds"] == 3
    assert result.metadata["visible_target_pixels"] == 1
    assert result.metadata["collision_count"] == 2


def test_invalid_depth_stays_unobserved():
    rgb = np.full((2, 2, 3), 9, dtype=np.uint8)
    depth = np.array([[1.0, 0.0], [np.nan, 2.0]], dtype=np.float32)
    camera = _camera(fx=5, fy=5, cx=0.5, cy=0.5)

    result = forward_warp_rgbd(rgb, depth, camera, camera)

    assert result.observed_mask.tolist() == [[True, False], [False, True]]
    assert result.metadata["valid_source_pixels"] == 2


def test_backend_protocol_uses_depth_by_observation_id():
    image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    depth = np.ones((2, 2), dtype=np.float32)
    camera = _camera(fx=5, fy=5, cx=0.5, cy=0.5)
    observation = ObservationView("source", image, camera)
    backend = PinholeWarpBackend({"source": depth})

    result = backend.warp([observation], camera)
    assert np.array_equal(result.rgb, image)
    assert result.backend == "pinhole-forward@0.1"
