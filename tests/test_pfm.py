from __future__ import annotations

import numpy as np

from refworld.datasets.pfm import read_pfm


def test_read_pfm_restores_top_to_bottom_grayscale(tmp_path):
    expected = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    path = tmp_path / "depth.pfm"
    with path.open("wb") as handle:
        handle.write(b"Pf\n")
        handle.write(b"3 2\n")
        handle.write(b"-1.0\n")
        handle.write(np.flipud(expected).astype("<f4").tobytes())

    got = read_pfm(path)
    np.testing.assert_array_equal(got, expected)


def test_read_pfm_applies_positive_big_endian_scale(tmp_path):
    base = np.asarray([[1.5, 2.5]], dtype=np.float32)
    path = tmp_path / "scaled.pfm"
    with path.open("wb") as handle:
        handle.write(b"Pf\n")
        handle.write(b"2 1\n")
        handle.write(b"2.0\n")
        handle.write(np.flipud(base).astype(">f4").tobytes())

    got = read_pfm(path)
    np.testing.assert_allclose(got, base * 2.0, atol=0.0, rtol=0.0)
