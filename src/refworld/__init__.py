"""Benchmark and method primitives for reference-anchored world synthesis."""

from .evidence import (
    EvidenceComposition,
    EvidenceSummary,
    PixelProvenance,
    compose_evidence_preserving_view,
    deterministic_proposal_id,
    summarize_provenance,
)
from .metrics import (
    AnchorMetrics,
    CurveSummary,
    RelativeRevisitSummary,
    anchor_metrics,
    psnr,
    summarize_curve,
    summarize_relative_revisit,
)
from .neighborhood import (
    NearViewCamera,
    depth_normalized_translation_neighborhood,
    rotational_neighborhood,
)
from .objectives import evidence_weight_map, weighted_l1
from .proposals import (
    ObservationView,
    RepaintBackend,
    RepaintResult,
    ViewProposal,
    WarpBackend,
    WarpResult,
    build_view_proposal,
    hash_array,
)
from .registration import RegistrationResult, project_world_points, recover_camera_pnp
from .reporting import json_safe, load_json, validate_report, write_report

__all__ = [
    "AnchorMetrics",
    "CurveSummary",
    "EvidenceComposition",
    "EvidenceSummary",
    "NearViewCamera",
    "ObservationView",
    "PixelProvenance",
    "RegistrationResult",
    "RelativeRevisitSummary",
    "RepaintBackend",
    "RepaintResult",
    "ViewProposal",
    "WarpBackend",
    "WarpResult",
    "anchor_metrics",
    "build_view_proposal",
    "compose_evidence_preserving_view",
    "depth_normalized_translation_neighborhood",
    "deterministic_proposal_id",
    "evidence_weight_map",
    "hash_array",
    "json_safe",
    "load_json",
    "psnr",
    "project_world_points",
    "recover_camera_pnp",
    "rotational_neighborhood",
    "summarize_curve",
    "summarize_provenance",
    "summarize_relative_revisit",
    "validate_report",
    "weighted_l1",
    "write_report",
]
