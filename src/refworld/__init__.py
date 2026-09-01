"""Small benchmark utilities for reference-anchored world synthesis."""

from .metrics import (
    AnchorMetrics,
    CurveSummary,
    RelativeRevisitSummary,
    anchor_metrics,
    psnr,
    summarize_curve,
    summarize_relative_revisit,
)
from .registration import RegistrationResult, project_world_points, recover_camera_pnp
from .reporting import json_safe, load_json, validate_report, write_report

__all__ = [
    "AnchorMetrics",
    "CurveSummary",
    "RelativeRevisitSummary",
    "RegistrationResult",
    "anchor_metrics",
    "json_safe",
    "load_json",
    "psnr",
    "project_world_points",
    "recover_camera_pnp",
    "summarize_curve",
    "summarize_relative_revisit",
    "validate_report",
    "write_report",
]
