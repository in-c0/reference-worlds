import numpy as np
import pytest

from refworld.datasets.mvsnet import camera_pose_separation, parse_camera_text, parse_pair_text
from refworld.registration import project_world_points


PAIR_TEXT = """3
0
2 1 0.9 2 0.8
1
2 0 0.9 2 0.7
2
1 0 0.8
"""

IDENTITY_CAMERA = """extrinsic
1 0 0 0
0 1 0 0
0 0 1 0
0 0 0 1

intrinsic
800 0 320
0 800 240
0 0 1

0.5 0.01 128 1.77
"""

TRANSLATED_CAMERA = """extrinsic
1 0 0 -1
0 1 0 0
0 0 1 0
0 0 0 1

intrinsic
800 0 320
0 800 240
0 0 1

0.5 0.01
"""


def test_pair_parser_preserves_reference_and_ranked_source_order():
    records = parse_pair_text(PAIR_TEXT)
    assert [record.reference_id for record in records] == [0, 1, 2]
    assert records[0].source_ids == (1, 2)
    assert records[0].scores == pytest.approx((0.9, 0.8))


def test_camera_parser_converts_identity_opencv_w2c_to_canonical_gl_c2w():
    parsed = parse_camera_text(IDENTITY_CAMERA)
    assert parsed.source_convention == "opencv-world-to-camera"
    assert parsed.depth_min == pytest.approx(0.5)
    assert parsed.depth_interval == pytest.approx(0.01)
    assert parsed.depth_num == pytest.approx(128)
    assert parsed.depth_max == pytest.approx(1.77)

    # OpenCV identity sees world +Z. Canonical OpenGL camera must therefore
    # look along world +Z because its local forward axis is -Z.
    pixels = project_world_points(parsed.camera, [[0.0, 0.0, 5.0]])
    assert pixels[0] == pytest.approx([320.0, 240.0])


def test_camera_center_and_pose_separation_follow_source_extrinsics():
    a = parse_camera_text(IDENTITY_CAMERA).camera
    b = parse_camera_text(TRANSLATED_CAMERA).camera
    separation = camera_pose_separation(a, b)
    assert separation["view_direction_angle_deg"] == pytest.approx(0.0)
    assert separation["center_distance_source_units"] == pytest.approx(1.0)


def test_pair_parser_rejects_declared_record_mismatch():
    with pytest.raises(ValueError):
        parse_pair_text("2\n0\n1 1 0.5\n")


def test_camera_parser_rejects_invalid_rotation():
    bad = IDENTITY_CAMERA.replace("1 0 0 0\n0 1 0 0", "2 0 0 0\n0 1 0 0", 1)
    with pytest.raises(ValueError, match="orthonormal"):
        parse_camera_text(bad)
