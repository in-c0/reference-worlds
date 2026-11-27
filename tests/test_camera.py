import numpy as np
import pytest

from refworld.adapters import Camera
from refworld.camera import OPENGL_C2W, pitch, translate_local, view_direction, yaw


def identity_camera() -> Camera:
    return Camera(
        intrinsics=(100.0, 0.0, 50.0, 0.0, 100.0, 50.0, 0.0, 0.0, 1.0),
        extrinsics=tuple(float(v) for v in np.eye(4).reshape(-1)),
        convention=OPENGL_C2W,
    )


def test_identity_camera_looks_down_negative_z():
    assert view_direction(identity_camera()) == pytest.approx((0.0, 0.0, -1.0))


def test_symmetric_yaw_is_reversible_and_keeps_position():
    camera = identity_camera()
    moved = yaw(yaw(camera, 10.0), -10.0)
    assert moved.extrinsics == pytest.approx(camera.extrinsics, abs=1e-9)
    yawed = yaw(camera, 90.0)
    assert np.asarray(yawed.extrinsics).reshape(4, 4)[:3, 3] == pytest.approx((0.0, 0.0, 0.0))
    assert view_direction(yawed) == pytest.approx((-1.0, 0.0, 0.0), abs=1e-9)


def test_pitch_is_reversible():
    camera = identity_camera()
    moved = pitch(pitch(camera, 7.5), -7.5)
    assert moved.extrinsics == pytest.approx(camera.extrinsics, abs=1e-9)


def test_local_translation_uses_camera_axes():
    camera = yaw(identity_camera(), 90.0)
    translated = translate_local(camera, (0.0, 0.0, -2.0))
    position = np.asarray(translated.extrinsics).reshape(4, 4)[:3, 3]
    # Forward is world -X after +90 local yaw, so -Z translation moves -X.
    assert position == pytest.approx((-2.0, 0.0, 0.0), abs=1e-9)


def test_wrong_convention_is_rejected():
    camera = Camera(
        intrinsics=identity_camera().intrinsics,
        extrinsics=identity_camera().extrinsics,
        convention="opencv-camera-to-world",
    )
    with pytest.raises(ValueError):
        yaw(camera, 2.0)
