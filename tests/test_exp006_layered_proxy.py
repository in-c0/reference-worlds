import math

import numpy as np

from refworld.exp006_layered_proxy import (
    AUTHORED_HFOV_DEGREES,
    NEIGHBOR_TRANSLATION,
    authored_camera,
    render_triplet,
    render_view,
)


def _synthetic_reference(height=96, width=160):
    y, x = np.mgrid[0:height, 0:width]
    return np.stack(
        [
            (x * 3 + y) % 256,
            (x + y * 5) % 256,
            (x * 7 + y * 11) % 256,
        ],
        axis=-1,
    ).astype(np.uint8)


def test_authored_camera_is_reproducible_and_uses_frozen_hfov():
    camera = authored_camera(1672, 941)
    expected_focal = 1672.0 / (2.0 * math.tan(math.radians(AUTHORED_HFOV_DEGREES) / 2.0))
    assert camera["hfov_degrees"] == 60.0
    assert camera["intrinsics"][0] == expected_focal
    assert camera["intrinsics"][4] == expected_focal
    assert camera["intrinsics"][2] == 835.5
    assert camera["intrinsics"][5] == 470.0
    assert camera["extrinsics"] == [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]


def test_hero_partition_reconstructs_reference_exactly():
    reference = _synthetic_reference()
    hero, observed, shifts = render_view(reference, camera_tx=0.0)
    assert np.array_equal(hero, reference)
    assert np.all(observed)
    assert set(shifts.values()) == {0}


def test_neighbors_are_deterministic_and_leave_disocclusion_unknown():
    reference = _synthetic_reference()
    first = render_triplet(reference)
    second = render_triplet(reference)

    for name in first:
        rgb_a, observed_a, shifts_a = first[name]
        rgb_b, observed_b, shifts_b = second[name]
        assert np.array_equal(rgb_a, rgb_b)
        assert np.array_equal(observed_a, observed_b)
        assert shifts_a == shifts_b

    for name in ("neighbor-left", "neighbor-right"):
        _rgb, observed, shifts = first[name]
        assert 0.90 <= float(np.mean(observed)) < 1.0
        assert any(value != 0 for value in shifts.values())


def test_neighbor_translation_is_frozen_and_symmetric():
    reference = _synthetic_reference()
    _left_rgb, _left_observed, left_shifts = render_view(
        reference,
        camera_tx=-NEIGHBOR_TRANSLATION,
    )
    _right_rgb, _right_observed, right_shifts = render_view(
        reference,
        camera_tx=NEIGHBOR_TRANSLATION,
    )
    assert left_shifts.keys() == right_shifts.keys()
    for key in left_shifts:
        assert left_shifts[key] == -right_shifts[key]
