import pytest

from refworld.adapters import Camera, Unsupported


def test_camera_requires_explicit_matrix_shapes_and_convention():
    camera = Camera(
        intrinsics=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        extrinsics=(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        convention="opencv-camera-to-world",
    )
    assert camera.convention.startswith("opencv")

    with pytest.raises(ValueError):
        Camera((1.0,), tuple(range(16)), "opencv")
    with pytest.raises(ValueError):
        Camera(tuple(range(9)), tuple(range(16)), "")


def test_unsupported_is_distinct_from_a_failed_score():
    result = Unsupported("entities", "baseline exposes visual world only")
    assert result.capability == "entities"
    assert "visual" in result.reason
