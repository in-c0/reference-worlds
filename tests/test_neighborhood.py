import numpy as np
import pytest

from refworld.adapters.base import Camera
from refworld.camera import OPENGL_C2W, view_direction
from refworld.neighborhood import (
    depth_normalized_translation_neighborhood,
    rotational_neighborhood,
)


def _camera():
    return Camera(
        intrinsics=(100.0, 0.0, 2.0, 0.0, 100.0, 2.0, 0.0, 0.0, 1.0),
        extrinsics=(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ),
        convention=OPENGL_C2W,
    )


def _position(camera):
    return np.asarray(camera.extrinsics, dtype=float).reshape(4, 4)[:3, 3]


def test_rotational_neighborhood_is_deterministic_and_keeps_camera_center():
    anchor = _camera()
    a = rotational_neighborhood(anchor, yaw_degrees=(-2, 2), pitch_degrees=(-2, 2))
    b = rotational_neighborhood(anchor, yaw_degrees=(-2, 2), pitch_degrees=(-2, 2))
    assert [item.view_id for item in a] == [item.view_id for item in b]
    assert len({item.view_id for item in a}) == 4
    for item in a:
        assert np.allclose(_position(item.camera), [0, 0, 0])
        assert item.reference_depth is None
    assert not np.allclose(view_direction(a[0].camera), view_direction(anchor))


def test_depth_normalized_translation_uses_declared_depth_without_calling_it_metric():
    anchor = _camera()
    views = depth_normalized_translation_neighborhood(
        anchor,
        reference_depth=10.0,
        lateral_ratios=(-0.05, 0.05),
        forward_ratios=(0.02,),
    )
    assert len(views) == 3
    left, right, forward = views
    assert np.allclose(_position(left.camera), [-0.5, 0.0, 0.0])
    assert np.allclose(_position(right.camera), [0.5, 0.0, 0.0])
    assert np.allclose(_position(forward.camera), [0.0, 0.0, -0.2])
    assert all(v.displacement_kind == "local_translation_depth_ratio" for v in views)
    assert all(v.reference_depth == pytest.approx(10.0) for v in views)


def test_neighborhood_rejects_zero_duplicate_or_invalid_displacements():
    anchor = _camera()
    with pytest.raises(ValueError, match="non-zero"):
        rotational_neighborhood(anchor, yaw_degrees=(0,), pitch_degrees=())
    with pytest.raises(ValueError, match="duplicate"):
        rotational_neighborhood(anchor, yaw_degrees=(2, 2), pitch_degrees=())
    with pytest.raises(ValueError, match="positive"):
        depth_normalized_translation_neighborhood(anchor, reference_depth=0)
    with pytest.raises(ValueError, match="non-zero"):
        depth_normalized_translation_neighborhood(
            anchor,
            reference_depth=1,
            lateral_ratios=(0,),
            forward_ratios=(),
        )
