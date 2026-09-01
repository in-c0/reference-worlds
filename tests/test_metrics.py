import math

import numpy as np
import pytest

from refworld.metrics import anchor_metrics, psnr


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
