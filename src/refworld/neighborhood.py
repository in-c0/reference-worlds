"""Deterministic camera neighborhoods for RefWorld-0 proposal generation.

Generated proposal cameras are method inputs, not ground truth. Calibrated
held-out cameras remain the evaluation target when available.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Iterable, Literal

from .adapters.base import Camera
from .camera import pitch, translate_local, yaw


@dataclass(frozen=True)
class NearViewCamera:
    view_id: str
    camera: Camera
    displacement_kind: Literal[
        "yaw_deg", "pitch_deg", "local_translation_depth_ratio"
    ]
    displacement_value: float
    axis: str
    reference_depth: float | None = None

    def metadata_dict(self) -> dict:
        return {
            "view_id": self.view_id,
            "displacement_kind": self.displacement_kind,
            "displacement_value": self.displacement_value,
            "axis": self.axis,
            "reference_depth": self.reference_depth,
            "camera": {
                "intrinsics": [float(v) for v in self.camera.intrinsics],
                "extrinsics": [float(v) for v in self.camera.extrinsics],
                "convention": self.camera.convention,
            },
        }


def _view_id(
    camera: Camera,
    *,
    kind: str,
    value: float,
    axis: str,
    reference_depth: float | None,
) -> str:
    payload = {
        "kind": kind,
        "value": float(value),
        "axis": axis,
        "reference_depth": reference_depth,
        "camera": {
            "intrinsics": list(camera.intrinsics),
            "extrinsics": list(camera.extrinsics),
            "convention": camera.convention,
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "cam-" + hashlib.sha256(encoded).hexdigest()[:16]


def rotational_neighborhood(
    anchor: Camera,
    *,
    yaw_degrees: Iterable[float] = (-10.0, -5.0, -2.0, 2.0, 5.0, 10.0),
    pitch_degrees: Iterable[float] = (-5.0, -2.0, 2.0, 5.0),
) -> tuple[NearViewCamera, ...]:
    """Return a deterministic scale-free rotational camera neighborhood."""

    items: list[NearViewCamera] = []
    seen: set[tuple[str, float]] = set()
    for kind, values, fn, axis in (
        ("yaw_deg", tuple(yaw_degrees), yaw, "local-y"),
        ("pitch_deg", tuple(pitch_degrees), pitch, "local-x"),
    ):
        for raw in values:
            value = float(raw)
            if not math.isfinite(value) or value == 0.0:
                raise ValueError(f"{kind} values must be finite and non-zero")
            key = (kind, value)
            if key in seen:
                raise ValueError(f"duplicate camera displacement: {key}")
            seen.add(key)
            camera = fn(anchor, value)
            items.append(
                NearViewCamera(
                    view_id=_view_id(
                        camera,
                        kind=kind,
                        value=value,
                        axis=axis,
                        reference_depth=None,
                    ),
                    camera=camera,
                    displacement_kind=kind,
                    displacement_value=value,
                    axis=axis,
                )
            )
    return tuple(items)


def depth_normalized_translation_neighborhood(
    anchor: Camera,
    *,
    reference_depth: float,
    lateral_ratios: Iterable[float] = (-0.05, -0.02, 0.02, 0.05),
    vertical_ratios: Iterable[float] = (),
    forward_ratios: Iterable[float] = (-0.05, -0.02, 0.02, 0.05),
) -> tuple[NearViewCamera, ...]:
    """Translate by a declared fraction of a scene-depth statistic.

    ``reference_depth`` may come from a monocular depth model and therefore does
    not imply metric scale. Reports must preserve the ratio/source-depth
    semantics unless the depth statistic is independently known to be metric.

    Local axes follow the canonical camera: +X right, +Y up, -Z forward.
    Positive ``forward_ratios`` move forward, hence local Z translation is
    ``-ratio * reference_depth``.
    """

    depth = float(reference_depth)
    if not math.isfinite(depth) or depth <= 0.0:
        raise ValueError("reference_depth must be finite and positive")

    specs = (
        ("lateral", tuple(lateral_ratios), (1.0, 0.0, 0.0), "local-x"),
        ("vertical", tuple(vertical_ratios), (0.0, 1.0, 0.0), "local-y"),
        ("forward", tuple(forward_ratios), (0.0, 0.0, -1.0), "local-minus-z"),
    )

    items: list[NearViewCamera] = []
    seen: set[tuple[str, float]] = set()
    for semantic_axis, ratios, basis, axis in specs:
        for raw in ratios:
            ratio = float(raw)
            if not math.isfinite(ratio) or ratio == 0.0:
                raise ValueError("translation ratios must be finite and non-zero")
            key = (semantic_axis, ratio)
            if key in seen:
                raise ValueError(f"duplicate translation displacement: {key}")
            seen.add(key)
            delta = tuple(component * ratio * depth for component in basis)
            camera = translate_local(anchor, delta)
            items.append(
                NearViewCamera(
                    view_id=_view_id(
                        camera,
                        kind="local_translation_depth_ratio",
                        value=ratio,
                        axis=semantic_axis,
                        reference_depth=depth,
                    ),
                    camera=camera,
                    displacement_kind="local_translation_depth_ratio",
                    displacement_value=ratio,
                    axis=semantic_axis,
                    reference_depth=depth,
                )
            )
    return tuple(items)
