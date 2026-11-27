"""No-fill repaint backend for the RefWorld-0 warp-only ablation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..adapters.base import Camera
from ..proposals import RepaintResult, WarpResult


@dataclass(frozen=True)
class NoRepaintBackend:
    name: str = "none@0.1"

    def repaint(
        self,
        warp: WarpResult,
        target_camera: Camera,
        *,
        seed: int | None,
    ) -> RepaintResult:
        if seed is not None:
            raise ValueError("NoRepaintBackend is deterministic and requires seed=None")
        rgb = np.zeros_like(np.asarray(warp.rgb))
        valid = np.zeros(np.asarray(warp.rgb).shape[:2], dtype=bool)
        return RepaintResult(
            rgb=rgb,
            valid_mask=valid,
            backend=self.name,
            seed=None,
            metadata={
                "policy": "no-fill",
                "meaning": "non-observed target support remains unresolved",
            },
        )
