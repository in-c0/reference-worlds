import numpy as np
import pytest

from refworld.geometry_scale import estimate_positive_depth_scale


def test_recovers_positive_scale_from_top_confidence_half() -> None:
    predicted = np.ones((4, 4), dtype=np.float32) * 2.0
    reference = predicted * 3.5
    confidence = np.arange(16, dtype=np.float32).reshape(4, 4)

    result = estimate_positive_depth_scale(
        predicted,
        reference,
        confidence,
        top_fraction=0.5,
        min_selected=8,
    )

    assert result.scale == pytest.approx(3.5)
    assert result.valid_count == 16
    assert result.selected_count == 8
    assert result.selected_fraction_of_valid == pytest.approx(0.5)
    assert result.ratio_mad == pytest.approx(0.0)


def test_low_confidence_outliers_do_not_change_frozen_top_half_fit() -> None:
    predicted = np.ones((4, 4), dtype=np.float32)
    reference = np.ones((4, 4), dtype=np.float32) * 2.0
    confidence = np.arange(16, dtype=np.float32).reshape(4, 4)

    # Corrupt exactly the lower-confidence half. The frozen estimator must rank
    # by confidence first and therefore recover the untouched high-confidence half.
    flat_reference = reference.reshape(-1)
    flat_reference[:8] = 100.0

    result = estimate_positive_depth_scale(
        predicted,
        reference,
        confidence,
        top_fraction=0.5,
        min_selected=8,
    )

    assert result.scale == pytest.approx(2.0)
    assert result.ratio_p10 == pytest.approx(2.0)
    assert result.ratio_p90 == pytest.approx(2.0)


def test_confidence_ties_use_stable_flattened_order() -> None:
    predicted = np.ones((2, 4), dtype=np.float32)
    reference = np.asarray([[2, 2, 2, 2], [9, 9, 9, 9]], dtype=np.float32)
    confidence = np.ones((2, 4), dtype=np.float32)

    result = estimate_positive_depth_scale(
        predicted,
        reference,
        confidence,
        top_fraction=0.5,
        min_selected=4,
    )

    # Stable descending argsort preserves flattened order for equal confidence,
    # selecting the first row exactly.
    assert result.scale == pytest.approx(2.0)
    assert result.selected_count == 4


def test_invalid_or_insufficient_inputs_fail_closed() -> None:
    good = np.ones((4, 4), dtype=np.float32)
    confidence = np.ones((4, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="identical shapes"):
        estimate_positive_depth_scale(good, good[:3], confidence)

    with pytest.raises(ValueError, match="no finite positive overlapping"):
        estimate_positive_depth_scale(good, np.zeros_like(good), confidence)

    with pytest.raises(ValueError, match="require at least"):
        estimate_positive_depth_scale(good, good, confidence, top_fraction=0.5, min_selected=9)

    with pytest.raises(ValueError, match="top_fraction"):
        estimate_positive_depth_scale(good, good, confidence, top_fraction=0.0)
