from __future__ import annotations

import numpy as np
import pytest

from refworld.runners.moge2_source import _pixel_intrinsics_from_normalized, _storage_safe_depth


def test_pixel_intrinsics_from_normalized_scales_axes_independently():
    k = np.asarray(
        [
            [1.25, 0.0, 0.50],
            [0.0, 1.50, 0.40],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    result = _pixel_intrinsics_from_normalized(k, width=768, height=576)
    np.testing.assert_allclose(
        result,
        [
            [960.0, 0.0, 384.0],
            [0.0, 864.0, 230.4],
            [0.0, 0.0, 1.0],
        ],
    )


def test_pixel_intrinsics_from_normalized_rejects_nonpositive_focal():
    k = np.eye(3, dtype=np.float64)
    k[0, 0] = 0.0
    with pytest.raises(ValueError, match="focal"):
        _pixel_intrinsics_from_normalized(k, width=768, height=576)


def test_storage_safe_depth_only_sanitizes_masked_out_values():
    depth = np.asarray([[2.0, -1.0], [4.0, np.nan]], dtype=np.float32)
    mask = np.asarray([[True, False], [True, False]])
    stored, count = _storage_safe_depth(depth, mask)
    assert count == 2
    assert stored[0, 0] == pytest.approx(2.0)
    assert stored[1, 0] == pytest.approx(4.0)
    assert np.all(np.isfinite(stored))
    assert np.all(stored > 0.0)


def test_storage_safe_depth_refuses_invalid_value_inside_valid_mask():
    depth = np.asarray([[2.0, -1.0], [4.0, 5.0]], dtype=np.float32)
    mask = np.asarray([[True, True], [True, False]])
    with pytest.raises(RuntimeError, match="valid"):
        _storage_safe_depth(depth, mask)
