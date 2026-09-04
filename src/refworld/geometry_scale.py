"""Explicit scale-only calibration primitives for learned monocular depth.

A single monocular prediction does not identify the metric/source-unit scale of a
calibrated benchmark. This module deliberately exposes only a *single positive
multiplicative scalar* fit. It must not grow into an affine/per-region geometry
correction without a new protocol and claim boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DepthScaleEstimate:
    scale: float
    valid_count: int
    selected_count: int
    valid_fraction: float
    selected_fraction_of_valid: float
    ratio_mad: float
    relative_ratio_mad: float
    ratio_p10: float
    ratio_p90: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "scale": self.scale,
            "valid_count": self.valid_count,
            "selected_count": self.selected_count,
            "valid_fraction": self.valid_fraction,
            "selected_fraction_of_valid": self.selected_fraction_of_valid,
            "ratio_mad": self.ratio_mad,
            "relative_ratio_mad": self.relative_ratio_mad,
            "ratio_p10": self.ratio_p10,
            "ratio_p90": self.ratio_p90,
        }


def estimate_positive_depth_scale(
    predicted_depth: np.ndarray,
    reference_depth: np.ndarray,
    confidence_raw: np.ndarray,
    *,
    top_fraction: float = 0.5,
    min_selected: int = 64,
) -> DepthScaleEstimate:
    """Fit one positive scalar from predicted optical-axis depth to source units.

    Selection is deterministic and rank-only: among pixels where both depths are
    finite and positive, retain exactly the top ``top_fraction`` by raw confidence.
    Confidence ties are broken by flattened pixel order. Raw confidence is never
    interpreted as a calibrated probability.

    The estimator is intentionally limited to ``reference ~= scale * predicted``.
    No offset, spatial correction, focal correction or pose refinement is allowed.
    """

    predicted = np.asarray(predicted_depth, dtype=np.float64)
    reference = np.asarray(reference_depth, dtype=np.float64)
    confidence = np.asarray(confidence_raw, dtype=np.float64)
    if predicted.shape != reference.shape or predicted.shape != confidence.shape:
        raise ValueError(
            "predicted_depth, reference_depth and confidence_raw must have identical shapes"
        )
    if predicted.ndim != 2 or predicted.size == 0:
        raise ValueError("depth scale calibration requires non-empty HxW arrays")
    if not (0.0 < float(top_fraction) <= 1.0):
        raise ValueError("top_fraction must be in (0,1]")
    if int(min_selected) < 1:
        raise ValueError("min_selected must be positive")

    valid = (
        np.isfinite(predicted)
        & np.isfinite(reference)
        & np.isfinite(confidence)
        & (predicted > 0.0)
        & (reference > 0.0)
    )
    valid_indices = np.flatnonzero(valid.reshape(-1))
    valid_count = int(valid_indices.size)
    if valid_count == 0:
        raise ValueError("no finite positive overlapping depth pixels are available for scale calibration")

    selected_count = int(math.ceil(valid_count * float(top_fraction)))
    if selected_count < int(min_selected):
        raise ValueError(
            f"scale calibration selected only {selected_count} pixels; require at least {int(min_selected)}"
        )

    confidence_flat = confidence.reshape(-1)
    ranked = np.argsort(-confidence_flat[valid_indices], kind="stable")
    selected_indices = valid_indices[ranked[:selected_count]]

    predicted_flat = predicted.reshape(-1)
    reference_flat = reference.reshape(-1)
    ratios = reference_flat[selected_indices] / predicted_flat[selected_indices]
    if not np.all(np.isfinite(ratios)) or np.any(ratios <= 0.0):
        raise ValueError("depth-ratio calibration produced non-finite or non-positive values")

    scale = float(np.median(ratios))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("estimated depth scale must be finite and positive")

    mad = float(np.median(np.abs(ratios - scale)))
    p10, p90 = (float(v) for v in np.quantile(ratios, [0.10, 0.90]))
    return DepthScaleEstimate(
        scale=scale,
        valid_count=valid_count,
        selected_count=selected_count,
        valid_fraction=float(valid_count / predicted.size),
        selected_fraction_of_valid=float(selected_count / valid_count),
        ratio_mad=mad,
        relative_ratio_mad=float(mad / scale),
        ratio_p10=p10,
        ratio_p90=p90,
    )
