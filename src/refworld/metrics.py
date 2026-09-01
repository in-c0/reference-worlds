"""Dependency-light metrics for reference-anchored world evaluation.

These bootstrap metrics intentionally avoid heavyweight vision dependencies.
Serious benchmark reports should add SSIM/MS-SSIM, LPIPS, foundation-model
feature similarity, geometry error, and human evaluation where appropriate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class AnchorMetrics:
    mae: float
    mse: float
    psnr: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class CurveSummary:
    """Summary of a similarity-vs-displacement curve.

    Similarity is assumed to be higher-is-better. ``normalized_auc`` divides
    trapezoidal area by the displacement span, so it stays in the same units as
    the similarity score. ``near_anchor_slope`` is the slope between the first
    two samples. ``failure_radius`` is the first sampled displacement whose
    score falls strictly below the requested threshold, or infinity if no
    failure is observed within the sampled range.
    """

    normalized_auc: float
    near_anchor_slope: float
    failure_radius: float

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


def summarize_curve(
    displacements: Iterable[float],
    similarities: Iterable[float],
    *,
    failure_threshold: float,
) -> CurveSummary:
    """Summarize an Anchor Fidelity Curve or another monotonic-axis curve.

    The function does not assume the scores are monotonic; non-monotonicity is
    itself useful evidence. Displacements must be finite, unique, non-negative,
    and strictly increasing. Similarities must be finite. At least two samples
    are required.
    """

    x = np.asarray(tuple(displacements), dtype=np.float64)
    y = np.asarray(tuple(similarities), dtype=np.float64)

    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("displacements and similarities must be equal-length 1D sequences")
    if x.size < 2:
        raise ValueError("at least two curve samples are required")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("curve samples must be finite")
    if np.any(x < 0.0):
        raise ValueError("displacements must be non-negative")
    if np.any(np.diff(x) <= 0.0):
        raise ValueError("displacements must be strictly increasing")
    if not math.isfinite(failure_threshold):
        raise ValueError("failure_threshold must be finite")

    span = float(x[-1] - x[0])
    if span <= 0.0:
        raise ValueError("curve displacement span must be positive")

    widths = np.diff(x)
    area = float(np.sum((y[:-1] + y[1:]) * 0.5 * widths))
    normalized_auc = area / span
    near_anchor_slope = float((y[1] - y[0]) / (x[1] - x[0]))

    below = np.flatnonzero(y < failure_threshold)
    failure_radius = math.inf if below.size == 0 else float(x[int(below[0])])

    return CurveSummary(
        normalized_auc=normalized_auc,
        near_anchor_slope=near_anchor_slope,
        failure_radius=failure_radius,
    )
