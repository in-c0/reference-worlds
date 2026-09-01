"""Backend-neutral near-view proposal contract for RefWorld-0."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from .adapters.base import Camera
from .evidence import (
    EvidenceComposition,
    compose_evidence_preserving_view,
    deterministic_proposal_id,
)


def hash_array(array: np.ndarray) -> str:
    """Hash array semantics (dtype + shape + contiguous bytes), not only bytes."""

    arr = np.asarray(array)
    if arr.dtype == np.dtype("O"):
        raise ValueError("cannot hash object arrays")
    contiguous = np.ascontiguousarray(arr)
    header = json.dumps(
        {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


@dataclass(frozen=True)
class ObservationView:
    """One real observation supplied to the proposal pipeline."""

    observation_id: str
    image: np.ndarray
    camera: Camera
    source_kind: str = "real-observation"

    def __post_init__(self) -> None:
        if not self.observation_id.strip():
            raise ValueError("observation_id cannot be empty")
        if self.source_kind != "real-observation":
            raise ValueError("ObservationView must originate from real-observation evidence")
        image = np.asarray(self.image)
        if image.ndim != 3 or image.shape[2] not in (1, 3, 4) or image.size == 0:
            raise ValueError("observation image must be a non-empty HxWxC array")
        if image.dtype == np.dtype("O"):
            raise ValueError("observation image cannot use object dtype")


@dataclass(frozen=True)
class WarpResult:
    rgb: np.ndarray
    observed_mask: np.ndarray
    confidence: np.ndarray
    backend: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        rgb = np.asarray(self.rgb)
        mask = np.asarray(self.observed_mask)
        confidence = np.asarray(self.confidence)
        if rgb.ndim != 3 or rgb.shape[2] not in (1, 3, 4) or rgb.size == 0:
            raise ValueError("warp rgb must be a non-empty HxWxC array")
        if mask.shape != rgb.shape[:2] or mask.dtype != np.bool_:
            raise ValueError("warp observed_mask must be boolean HxW matching rgb")
        if confidence.shape != rgb.shape[:2]:
            raise ValueError("warp confidence must be HxW matching rgb")
        if not np.issubdtype(confidence.dtype, np.floating):
            raise ValueError("warp confidence must use a floating dtype")
        if not np.all(np.isfinite(confidence)):
            raise ValueError("warp confidence must be finite")
        if np.any(confidence < 0.0) or np.any(confidence > 1.0):
            raise ValueError("warp confidence must lie in [0,1]")
        if not self.backend.strip():
            raise ValueError("warp backend cannot be empty")


@dataclass(frozen=True)
class RepaintResult:
    rgb: np.ndarray
    valid_mask: np.ndarray
    backend: str
    seed: int | None
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        rgb = np.asarray(self.rgb)
        mask = np.asarray(self.valid_mask)
        if rgb.ndim != 3 or rgb.shape[2] not in (1, 3, 4) or rgb.size == 0:
            raise ValueError("repaint rgb must be a non-empty HxWxC array")
        if mask.shape != rgb.shape[:2] or mask.dtype != np.bool_:
            raise ValueError("repaint valid_mask must be boolean HxW matching rgb")
        if not self.backend.strip():
            raise ValueError("repaint backend cannot be empty")


@dataclass(frozen=True)
class ViewProposal:
    proposal_id: str
    parent_observation_ids: tuple[str, ...]
    target_camera: Camera
    image: np.ndarray
    provenance: np.ndarray
    warp_confidence: np.ndarray
    warp_backend: str
    repaint_backend: str
    repaint_seed: int | None
    summary: Mapping[str, Any]
    hashes: Mapping[str, str]
    backend_metadata: Mapping[str, Any]

    def metadata_dict(self) -> dict[str, Any]:
        """Return array-free metadata suitable for a manifest/report."""

        return {
            "version": "0.1",
            "proposal_id": self.proposal_id,
            "parent_observation_ids": list(self.parent_observation_ids),
            "target_camera": {
                "intrinsics": [float(v) for v in self.target_camera.intrinsics],
                "extrinsics": [float(v) for v in self.target_camera.extrinsics],
                "convention": self.target_camera.convention,
            },
            "backends": {
                "warp": self.warp_backend,
                "repaint": self.repaint_backend,
                "repaint_seed": self.repaint_seed,
            },
            "evidence": dict(self.summary),
            "hashes": dict(self.hashes),
            "backend_metadata": dict(self.backend_metadata),
            "epistemic_origin": {
                "observed": "projected from real supplied observation(s)",
                "generated": "synthesized where direct observation support was absent",
                "unresolved": "no accepted evidence/proposal",
            },
        }


@runtime_checkable
class WarpBackend(Protocol):
    name: str

    def warp(
        self,
        observations: Sequence[ObservationView],
        target_camera: Camera,
    ) -> WarpResult:
        ...


@runtime_checkable
class RepaintBackend(Protocol):
    name: str

    def repaint(
        self,
        warp: WarpResult,
        target_camera: Camera,
        *,
        seed: int | None,
    ) -> RepaintResult:
        ...


def build_view_proposal(
    observations: Sequence[ObservationView],
    target_camera: Camera,
    warp: WarpResult,
    repaint: RepaintResult,
    *,
    unresolved_value: int | float = 0,
) -> ViewProposal:
    """Compose and hash a proposal without letting repaint overwrite evidence."""

    if not observations:
        raise ValueError("at least one real observation is required")
    parent_ids = tuple(obs.observation_id for obs in observations)
    if len(set(parent_ids)) != len(parent_ids):
        raise ValueError("parent observation IDs must be unique")
    if np.asarray(warp.rgb).shape != np.asarray(repaint.rgb).shape:
        raise ValueError("warp and repaint image shapes must match")
    if np.asarray(warp.rgb).dtype != np.asarray(repaint.rgb).dtype:
        raise ValueError("warp and repaint image dtypes must match")

    composition: EvidenceComposition = compose_evidence_preserving_view(
        warp.rgb,
        warp.observed_mask,
        generated_rgb=repaint.rgb,
        generated_mask=repaint.valid_mask,
        unresolved_value=unresolved_value,
    )

    proposal_id = deterministic_proposal_id(
        parent_observation_ids=parent_ids,
        target_camera=target_camera,
        warp_backend=warp.backend,
        repaint_backend=repaint.backend,
        seed=repaint.seed,
    )

    hashes = {
        "proposal_image_sha256": hash_array(composition.image),
        "provenance_sha256": hash_array(composition.provenance),
        "warp_rgb_sha256": hash_array(warp.rgb),
        "warp_observed_mask_sha256": hash_array(warp.observed_mask),
        "warp_confidence_sha256": hash_array(warp.confidence),
        "repaint_rgb_sha256": hash_array(repaint.rgb),
        "repaint_valid_mask_sha256": hash_array(repaint.valid_mask),
    }

    backend_metadata = {
        "warp": dict(warp.metadata),
        "repaint": dict(repaint.metadata),
    }

    return ViewProposal(
        proposal_id=proposal_id,
        parent_observation_ids=parent_ids,
        target_camera=target_camera,
        image=composition.image,
        provenance=composition.provenance,
        warp_confidence=np.asarray(warp.confidence).copy(),
        warp_backend=warp.backend,
        repaint_backend=repaint.backend,
        repaint_seed=repaint.seed,
        summary=composition.summary.as_dict(),
        hashes=hashes,
        backend_metadata=backend_metadata,
    )
