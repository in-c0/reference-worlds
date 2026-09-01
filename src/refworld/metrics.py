"""Dependency-light anchor metrics.

These are intentionally basic bootstrap metrics. Serious benchmark reports should
also use perceptual metrics such as LPIPS and foundation-model feature similarity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class AnchorMetrics:
    mae: float
    mse: float
    psnr: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _as_unit_float(image: np.ndarray | Sequence) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim not in (2, 3):
        raise ValueError(f"expected HxW or HxWxC image, got shape {arr.shape}")
    arr = arr.astype(np.float64, copy=False)
    if arr.size == 0:
        raise ValueError("image cannot be empty")
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def psnr(reference: np.ndarray | Sequence, candidate: np.ndarray | Sequence) -> float:
    ref = _as_unit_float(reference)
    got = _as_unit_float(candidate)
    if ref.shape != got.shape:
        raise ValueError(f"shape mismatch: {ref.shape} vs {got.shape}")
    mse = float(np.mean(np.square(ref - got)))
    if mse == 0.0:
        return math.inf
    return 10.0 * math.log10(1.0 / mse)


def anchor_metrics(reference: np.ndarray | Sequence, candidate: np.ndarray | Sequence) -> AnchorMetrics:
    ref = _as_unit_float(reference)
    got = _as_unit_float(candidate)
    if ref.shape != got.shape:
        raise ValueError(f"shape mismatch: {ref.shape} vs {got.shape}")
    delta = ref - got
    mse = float(np.mean(np.square(delta)))
    mae = float(np.mean(np.abs(delta)))
    score = math.inf if mse == 0.0 else 10.0 * math.log10(1.0 / mse)
    return AnchorMetrics(mae=mae, mse=mse, psnr=score)
