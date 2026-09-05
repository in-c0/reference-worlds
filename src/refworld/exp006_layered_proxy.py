"""Deterministic authored layered proxy for the first EXP-006 LifeOS slice.

This is an image-based product handoff renderer, not recovered scene geometry.
The owner reference is partitioned into explicit fronto-parallel proxy layers.
At the hero camera the partition reconstructs the reference pixel-for-pixel. Small
lateral camera moves apply depth-dependent parallax; disoccluded support remains
unknown instead of being silently inpainted or relabeled observed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw


AUTHORED_HFOV_DEGREES = 60.0
NEIGHBOR_TRANSLATION = 0.04
UNKNOWN_RGB = np.array([18, 20, 20], dtype=np.uint8)


@dataclass(frozen=True)
class LayerSpec:
    name: str
    depth: float
    polygon_normalized: tuple[tuple[float, float], ...]
    entity_id: str | None = None


# Coordinates are authored against the selected Collaborative Futures composition.
# They are deliberately coarse. No target/depth supervision or learned geometry is
# used to define these planes.
DEFAULT_LAYER_SPECS: tuple[LayerSpec, ...] = (
    LayerSpec(
        name="collaboration-table",
        depth=3.2,
        polygon_normalized=((0.20, 0.57), (0.76, 0.51), (0.86, 0.91), (0.14, 0.96)),
        entity_id="lifeos.atrium.collaboration-table",
    ),
    LayerSpec(
        name="left-discussion-hud",
        depth=4.4,
        polygon_normalized=((0.17, 0.23), (0.38, 0.22), (0.37, 0.66), (0.15, 0.66)),
    ),
    LayerSpec(
        name="right-projects-hud",
        depth=4.4,
        polygon_normalized=((0.72, 0.12), (0.99, 0.12), (0.99, 0.92), (0.76, 0.92)),
        entity_id="lifeos.project.xuxi-room",
    ),
    LayerSpec(
        name="world-model",
        depth=5.8,
        polygon_normalized=((0.35, 0.21), (0.73, 0.21), (0.76, 0.55), (0.34, 0.55)),
        entity_id="lifeos.system.world-model",
    ),
)

BACKGROUND_DEPTH = 11.0


def authored_camera(width: int, height: int, *, hfov_degrees: float = AUTHORED_HFOV_DEGREES) -> dict:
    if width <= 1 or height <= 1:
        raise ValueError("image dimensions must be greater than one pixel")
    if not (1.0 < hfov_degrees < 179.0):
        raise ValueError("hfov_degrees must be in (1, 179)")
    focal = float(width) / (2.0 * math.tan(math.radians(hfov_degrees) / 2.0))
    return {
        "hfov_degrees": float(hfov_degrees),
        "intrinsics": [
            focal,
            0.0,
            (float(width) - 1.0) / 2.0,
            0.0,
            focal,
            (float(height) - 1.0) / 2.0,
            0.0,
            0.0,
            1.0,
        ],
        "extrinsics": [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ],
        "convention": "opengl-camera-to-world",
    }


def _polygon_mask(width: int, height: int, polygon: Iterable[tuple[float, float]]) -> np.ndarray:
    points = []
    for x_norm, y_norm in polygon:
        if not (0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0):
            raise ValueError("normalized polygon coordinates must be within [0, 1]")
        points.append((round(x_norm * (width - 1)), round(y_norm * (height - 1))))
    if len(points) < 3:
        raise ValueError("layer polygon must have at least three points")
    image = Image.new("1", (width, height), 0)
    ImageDraw.Draw(image).polygon(points, fill=1)
    return np.asarray(image, dtype=bool)


def exclusive_layer_masks(
    width: int,
    height: int,
    *,
    specs: tuple[LayerSpec, ...] = DEFAULT_LAYER_SPECS,
) -> list[tuple[LayerSpec, np.ndarray]]:
    """Return a complete, non-overlapping layer partition.

    Foreground layers claim overlapping pixels first. The background is exactly the
    complement, which guarantees exact hero reconstruction without pretending that
    pixels hidden behind foreground objects are known.
    """
    claimed = np.zeros((height, width), dtype=bool)
    exclusive: list[tuple[LayerSpec, np.ndarray]] = []
    for spec in sorted(specs, key=lambda item: item.depth):
        if spec.depth <= 0.0:
            raise ValueError("layer depth must be positive")
        raw = _polygon_mask(width, height, spec.polygon_normalized)
        mask = raw & ~claimed
        exclusive.append((spec, mask))
        claimed |= raw

    background = LayerSpec(
        name="atrium-background",
        depth=BACKGROUND_DEPTH,
        polygon_normalized=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        entity_id=None,
    )
    exclusive.append((background, ~claimed))
    return exclusive


def _shift_masked_rgb(
    reference: np.ndarray,
    mask: np.ndarray,
    *,
    dx: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape
    shifted_rgb = np.zeros_like(reference)
    shifted_mask = np.zeros_like(mask)
    if abs(dx) >= width:
        return shifted_rgb, shifted_mask

    if dx >= 0:
        src = slice(0, width - dx)
        dst = slice(dx, width)
    else:
        src = slice(-dx, width)
        dst = slice(0, width + dx)

    shifted_rgb[:, dst] = reference[:, src]
    shifted_mask[:, dst] = mask[:, src]
    return shifted_rgb, shifted_mask


def render_view(
    reference_rgb: np.ndarray,
    *,
    camera_tx: float,
    hfov_degrees: float = AUTHORED_HFOV_DEGREES,
    specs: tuple[LayerSpec, ...] = DEFAULT_LAYER_SPECS,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Render one pure-lateral authored-proxy camera view.

    `camera_tx` is an authored proxy-world translation. Layer motion follows
    dx ~= f * tx / depth. No hidden content is synthesized by this function.
    """
    reference = np.asarray(reference_rgb, dtype=np.uint8)
    if reference.ndim != 3 or reference.shape[2] != 3:
        raise ValueError("reference_rgb must be HxWx3 uint8-compatible")
    height, width = reference.shape[:2]
    camera = authored_camera(width, height, hfov_degrees=hfov_degrees)
    focal = float(camera["intrinsics"][0])

    output = np.broadcast_to(UNKNOWN_RGB, (height, width, 3)).copy()
    observed = np.zeros((height, width), dtype=bool)
    shifts: dict[str, int] = {}

    # Composite back to front after foreground-first ownership partitioning.
    layers = exclusive_layer_masks(width, height, specs=specs)
    for spec, mask in sorted(layers, key=lambda item: item[0].depth, reverse=True):
        dx = int(round(focal * float(camera_tx) / float(spec.depth)))
        shifts[spec.name] = dx
        shifted_rgb, shifted_mask = _shift_masked_rgb(reference, mask, dx=dx)
        output[shifted_mask] = shifted_rgb[shifted_mask]
        observed[shifted_mask] = True

    return output, observed, shifts


def render_triplet(reference_rgb: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray, dict[str, int]]]:
    return {
        "hero": render_view(reference_rgb, camera_tx=0.0),
        "neighbor-left": render_view(reference_rgb, camera_tx=-NEIGHBOR_TRANSLATION),
        "neighbor-right": render_view(reference_rgb, camera_tx=NEIGHBOR_TRANSLATION),
    }
