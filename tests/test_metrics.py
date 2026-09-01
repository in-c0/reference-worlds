import math

import numpy as np
import pytest

from refworld.metrics import anchor_metrics, psnr, summarize_curve


def test_identical_images_have_infinite_psnr():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    assert math.isinf(psnr(image, image))


def test_uint8_and_float_normalization_agree():
    ref = np.zeros((4, 4), dtype=np.uint8)
    got = np.full((4, 4), 255, dtype=np.uint8)
    metrics = anchor_metrics(ref, got)
    assert metrics.mae == pytest.approx(1.0)
    assert metrics.mse == pytest.approx(1.0)
    assert metrics.psnr == pytest.approx(0.0)


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError):
        psnr(np.zeros((4, 4)), np.zeros((5, 4)))


def test_curve_summary_reports_auc_slope_and_failure_radius():
    summary = summarize_curve(
        [0.0, 2.0, 5.0, 10.0],
        [1.0, 0.9, 0.7, 0.4],
        failure_threshold=0.5,
    )
    assert summary.normalized_auc == pytest.approx(0.705)
    assert summary.near_anchor_slope == pytest.approx(-0.05)
    assert summary.failure_radius == pytest.approx(10.0)


def test_curve_failure_radius_is_infinite_when_threshold_not_crossed():
    summary = summarize_curve([0.0, 1.0], [0.9, 0.8], failure_threshold=0.5)
    assert math.isinf(summary.failure_radius)


def test_curve_rejects_unsorted_or_duplicate_displacements():
    with pytest.raises(ValueError):
        summarize_curve([0.0, 2.0, 2.0], [1.0, 0.9, 0.8], failure_threshold=0.5)
    with pytest.raises(ValueError):
        summarize_curve([0.0, 2.0, 1.0], [1.0, 0.9, 0.8], failure_threshold=0.5)
