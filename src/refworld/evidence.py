"""Pixel/ray provenance primitives for reference-conditioned view proposals.

The central invariant is simple: generative completion may fill missing support,
but must not overwrite support traced to a real supplied observation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Iterable

import numpy as np

from .adapters.base import Camera


class PixelProvenance(IntEnum):
    UNRESOLVED = 0
    OBSERVED = 1
    GENERATED = 2


@dataclass(frozen=True)
class EvidenceSummary:
    pixel_count: int
    observed_pixels: int
    generated_pixels: int
    unresolved_pixels: int
    overlap_attempt_pixels: int
    observed_fraction: float
    generated_fraction: float
    unresolved_fraction: float

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceComposition:
    image: np.ndarray
    provenance: np.ndarray
    summary: EvidenceSummary


def _validate_rgb(name: str, image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] not in (1, 3, 4):
        raise ValueError(f"{name} must be HxWxC with C in {{1,3,4}}, got {arr.shape}")
    if arr.shape[0] <= 0 or arr.shape[1] <= 0:
        raise ValueError(f"{name} cannot be empty")
    if arr.dtype == np.dtype("O"):
        raise ValueError(f"{name} cannot use object dtype")
    return arr


def _validate_mask(name: str, mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.shape != shape:
        raise ValueError(f"{name} shape {arr.shape} does not match image shape {shape}")
    if arr.dtype != np.bool_:
        raise ValueError(f"{name} must have boolean dtype")
    return arr


def summarize_provenance(
    provenance: np.ndarray,
    *,
    overlap_attempt_pixels: int = 0,
) -> EvidenceSummary:
    prov = np.asarray(provenance)
    if prov.ndim != 2 or prov.size == 0:
        raise ValueError("provenance must be a non-empty HxW array")
    if not np.issubdtype(prov.dtype, np.integer):
        raise ValueError("provenance must use an integer dtype")
    valid = {
        int(PixelProvenance.UNRESOLVED),
        int(PixelProvenance.OBSERVED),
        int(PixelProvenance.GENERATED),
    }
    values = {int(v) for v in np.unique(prov)}
    unknown = values - valid
    if unknown:
        raise ValueError(f"unknown provenance code(s): {sorted(unknown)}")
    if overlap_attempt_pixels < 0:
        raise ValueError("overlap_attempt_pixels cannot be negative")

    total = int(prov.size)
    observed = int(np.count_nonzero(prov == PixelProvenance.OBSERVED))
    generated = int(np.count_nonzero(prov == PixelProvenance.GENERATED))
    unresolved = total - observed - generated
    return EvidenceSummary(
        pixel_count=total,
        observed_pixels=observed,
        generated_pixels=generated,
        unresolved_pixels=unresolved,
        overlap_attempt_pixels=int(overlap_attempt_pixels),
        observed_fraction=observed / total,
        generated_fraction=generated / total,
        unresolved_fraction=unresolved / total,
    )


def compose_evidence_preserving_view(
    warped_rgb: np.ndarray,
    observed_mask: np.ndarray,
    *,
    generated_rgb: np.ndarray | None = None,
    generated_mask: np.ndarray | None = None,
    unresolved_value: int | float = 0,
) -> EvidenceComposition:
    """Compose one target view while preserving observed support bit-for-bit.

    ``warped_rgb`` is the geometry-projected appearance from real evidence.
    ``observed_mask`` marks pixels that are directly supported by that evidence.

    A repaint backend may return a full-frame ``generated_rgb``. Its pixels are
    applied only where observation support is absent. ``generated_mask`` can
    further restrict which generated pixels are considered valid. If omitted,
    the generated candidate is treated as proposing every pixel; attempted
    overlap with observed support is counted but never applied.

    Pixels supported by neither source remain explicitly UNRESOLVED.
    """

    warped = _validate_rgb("warped_rgb", warped_rgb)
    height, width, _ = warped.shape
    observed = _validate_mask("observed_mask", observed_mask, (height, width))

    if generated_rgb is None and generated_mask is not None:
        raise ValueError("generated_mask requires generated_rgb")

    if generated_rgb is None:
        generated = None
        proposed = np.zeros((height, width), dtype=bool)
    else:
        generated = _validate_rgb("generated_rgb", generated_rgb)
        if generated.shape != warped.shape:
            raise ValueError(
                f"generated_rgb shape {generated.shape} does not match warped_rgb {warped.shape}"
            )
        if generated.dtype != warped.dtype:
            raise ValueError(
                f"generated_rgb dtype {generated.dtype} must match warped_rgb dtype {warped.dtype}"
            )
        if generated_mask is None:
            proposed = np.ones((height, width), dtype=bool)
        else:
            proposed = _validate_mask("generated_mask", generated_mask, (height, width))

    overlap_attempt = int(np.count_nonzero(observed & proposed))
    generated_support = proposed & ~observed

    output = np.empty_like(warped)
    try:
        output[...] = unresolved_value
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("unresolved_value cannot be represented by image dtype") from exc

    output[observed] = warped[observed]
    if generated is not None:
        output[generated_support] = generated[generated_support]

    provenance = np.full(
        (height, width), int(PixelProvenance.UNRESOLVED), dtype=np.uint8
    )
    provenance[observed] = int(PixelProvenance.OBSERVED)
    provenance[generated_support] = int(PixelProvenance.GENERATED)

    # Hard invariant: no future refactor may accidentally let repaint change an
    # observed pixel.
    if not np.array_equal(output[observed], warped[observed]):
        raise AssertionError("evidence-preserving compositor modified observed pixels")

    return EvidenceComposition(
        image=output,
        provenance=provenance,
        summary=summarize_provenance(
            provenance, overlap_attempt_pixels=overlap_attempt
        ),
    )


def deterministic_proposal_id(
    *,
    parent_observation_ids: Iterable[str],
    target_camera: Camera,
    warp_backend: str,
    repaint_backend: str,
    seed: int | None,
) -> str:
    parents = tuple(str(item) for item in parent_observation_ids)
    if not parents or any(not item for item in parents):
        raise ValueError("at least one non-empty parent observation id is required")
    if not warp_backend.strip() or not repaint_backend.strip():
        raise ValueError("backend identifiers cannot be empty")

    payload = {
        "parents": parents,
        "target_camera": {
            "intrinsics": [float(v) for v in target_camera.intrinsics],
            "extrinsics": [float(v) for v in target_camera.extrinsics],
            "convention": target_camera.convention,
        },
        "warp_backend": warp_backend,
        "repaint_backend": repaint_backend,
        "seed": seed,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "view-" + hashlib.sha256(encoded).hexdigest()[:20]
