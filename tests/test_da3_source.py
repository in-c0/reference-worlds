from __future__ import annotations

import numpy as np
import pytest

from refworld.runners.da3_source import _map_intrinsics_to_original, _valid_aware_resize_depth


def test_map_intrinsics_to_original_scales_pixel_axes_independently():
    processed = np.asarray(
        [
            [420.0, 0.0, 252.0],
            [0.0, 430.0, 189.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    result = _map_intrinsics_to_original(
        processed,
        processed_width=504,
        processed_height=378,
        original_width=768,
        original_height=576,
    )
    np.testing.assert_allclose(
        result,
        [
            [640.0, 0.0, 384.0],
            [0.0, 655.2380952380952, 288.0],
            [0.0, 0.0, 1.0],
        ],
    )


def test_map_intrinsics_canonicalizes_only_tiny_skew():
    tiny = np.eye(3, dtype=np.float64)
    tiny[0, 0] = 400.0
    tiny[1, 1] = 410.0
    tiny[0, 1] = 5e-7
    tiny[1, 0] = -5e-7
    result = _map_intrinsics_to_original(
        tiny,
        processed_width=100,
        processed_height=100,
        original_width=100,
        original_height=100,
    )
    assert result[0, 1] == 0.0
    assert result[1, 0] == 0.0

    real_skew = tiny.copy()
    real_skew[0, 1] = 1e-3
    with pytest.raises(ValueError, match="zero-skew"):
        _map_intrinsics_to_original(
            real_skew,
            processed_width=100,
            processed_height=100,
            original_width=100,
            original_height=100,
        )


def test_valid_aware_resize_preserves_all_valid_constant_depth():
    depth = np.full((2, 3), 7.5, dtype=np.float32)
    valid = np.ones_like(depth, dtype=bool)
    resized, mask = _valid_aware_resize_depth(
        depth,
        valid,
        original_width=6,
        original_height=4,
    )
    assert mask.shape == (4, 6)
    assert np.all(mask)
    np.testing.assert_allclose(resized, 7.5)


def test_valid_aware_resize_does_not_cross_invalid_boundary():
    depth = np.asarray([[2.0, 2.0], [2.0, 100.0]], dtype=np.float32)
    valid = np.asarray([[True, True], [True, False]], dtype=bool)
    resized, mask = _valid_aware_resize_depth(
        depth,
        valid,
        original_width=4,
        original_height=4,
    )
    assert not mask[-1, -1]
    assert np.isnan(resized[-1, -1])
    assert np.all(resized[mask] < 3.0)


def test_valid_aware_resize_rejects_nonpositive_depth_inside_valid_support():
    depth = np.asarray([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    valid = np.ones_like(depth, dtype=bool)
    with pytest.raises(ValueError, match="finite and positive"):
        _valid_aware_resize_depth(
            depth,
            valid,
            original_width=2,
            original_height=2,
        )
