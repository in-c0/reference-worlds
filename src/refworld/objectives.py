"""Evidence-aware objective primitives for RefWorld-0.

These functions do not prescribe a final training objective. They encode one
research invariant: real observed support should dominate generated hypotheses,
and unresolved support must not silently contribute as if it were evidence.
"""

from __future__ import annotations

import math

import numpy as np

from .evidence import PixelProvenance


def evidence_weight_map(
    provenance: np.ndarray,
    *,
    observed_confidence: np.ndarray | None = None,
    observed_weight: float = 1.0,
    generated_weight: float = 0.1,
    unresolved_weight: float = 0.0,
) -> np.ndarray:
    """Convert epistemic provenance into a floating optimization-weight map.

    If ``observed_confidence`` is supplied, observed support is weighted by
    ``observed_weight * confidence``. Confidence is ignored outside OBSERVED
    pixels so a warper cannot accidentally assign evidence weight to generated
    or unresolved regions.
    """

    prov = np.asarray(provenance)
    if prov.ndim != 2 or prov.size == 0 or not np.issubdtype(prov.dtype, np.integer):
        raise ValueError("provenance must be a non-empty integer HxW array")

    valid_codes = {
        int(PixelProvenance.UNRESOLVED),
        int(PixelProvenance.OBSERVED),
        int(PixelProvenance.GENERATED),
    }
    unknown = {int(v) for v in np.unique(prov)} - valid_codes
    if unknown:
        raise ValueError(f"unknown provenance code(s): {sorted(unknown)}")

    for name, value in (
        ("observed_weight", observed_weight),
        ("generated_weight", generated_weight),
        ("unresolved_weight", unresolved_weight),
    ):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")

    if generated_weight > observed_weight:
        raise ValueError("generated_weight cannot exceed observed_weight by default contract")
    if unresolved_weight != 0.0:
        raise ValueError("unresolved_weight must remain zero; unresolved support is not evidence")

    weights = np.zeros(prov.shape, dtype=np.float64)
    generated = prov == int(PixelProvenance.GENERATED)
    observed = prov == int(PixelProvenance.OBSERVED)
    weights[generated] = float(generated_weight)

    if observed_confidence is None:
        weights[observed] = float(observed_weight)
    else:
        confidence = np.asarray(observed_confidence)
        if confidence.shape != prov.shape:
            raise ValueError("observed_confidence must match provenance shape")
        if not np.issubdtype(confidence.dtype, np.floating):
            raise ValueError("observed_confidence must use a floating dtype")
        if not np.all(np.isfinite(confidence)):
            raise ValueError("observed_confidence must be finite")
        if np.any(confidence < 0.0) or np.any(confidence > 1.0):
            raise ValueError("observed_confidence must lie in [0,1]")
        weights[observed] = float(observed_weight) * confidence[observed]

    return weights


def weighted_l1(
    reference: np.ndarray,
    prediction: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Return channel-aware weighted mean absolute error.

    The HxW weight map is broadcast across image channels. The denominator is
    the sum of pixel weights, not number of channels, while each pixel error is
    first averaged over channels. This makes the result comparable across RGB
    and RGBA inputs.
    """

    ref = np.asarray(reference, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)

    if ref.shape != pred.shape:
        raise ValueError("reference and prediction shapes must match")
    if ref.ndim not in (2, 3) or ref.size == 0:
        raise ValueError("reference/prediction must be non-empty HxW or HxWxC arrays")
    if w.shape != ref.shape[:2]:
        raise ValueError("weights must be HxW matching the image")
    if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(pred)):
        raise ValueError("reference and prediction must be finite")
    if not np.all(np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError("weights must be finite and non-negative")

    total_weight = float(np.sum(w))
    if total_weight <= 0.0:
        raise ValueError("weighted loss has no supported pixels")

    error = np.abs(ref - pred)
    if error.ndim == 3:
        error = np.mean(error, axis=2)
    return float(np.sum(error * w) / total_weight)
