"""Deterministic authored layered proxy for the first EXP-006 LifeOS slice.

This is an image-based product handoff renderer, not recovered scene geometry.
R0 exposed black internal disocclusion seams because foreground ownership was carved
out of the background texture before depth-dependent shifts. R1 separates display
continuity from epistemic provenance: a deterministic full-reference background
proxy may preview newly exposed support, but only hard/near-hard projected source
support is labelled observed. Feather/fallback pixels remain hypothesized.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


AUTHORED_HFOV_DEGREES = 60.0
NEIGHBOR_TRANSLATION = 0.04
R1_FEATHER_RADIUS_PX = 3.0
R1_ALPHA_OBSERVED_THRESHOLD = 0.95
R1_ALPHA_AFFECTED_THRESHOLD = 0.05
UNKNOWN_RGB = np.array([18, 20, 20], dtype=np.uint8)


@dataclass(frozen=True)
class LayerSpec:
    name: str
    depth: float
    polygon_normalized: tuple[tuple[float, float], ...]
    entity_id: str | None = None


# Coordinates are authored against the selected Collaborative Futures composition.
# They are deliberately coarse. No target/depth supervision or learned geometry is
# used to define these planes. R1 intentionally leaves these polygons/depths frozen
# so the repair addresses seam presentation rather than post-hoc geometry tuning.
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


def authored_layer_masks(
    width: int,
    height: int,
    *,
    specs: tuple[LayerSpec, ...] = DEFAULT_LAYER_SPECS,
) -> list[tuple[LayerSpec, np.ndarray]]:
    layers: list[tuple[LayerSpec, np.ndarray]] = []
    for spec in specs:
        if spec.depth <= 0.0:
            raise ValueError("layer depth must be positive")
        layers.append((spec, _polygon_mask(width, height, spec.polygon_normalized)))
    return layers


def exclusive_layer_masks(
    width: int,
    height: int,
    *,
    specs: tuple[LayerSpec, ...] = DEFAULT_LAYER_SPECS,
) -> list[tuple[LayerSpec, np.ndarray]]:
    """Return the historical R0 ownership partition.

    Retained for audit/tests. R1 display rendering does not carve these masks out of
    the background texture because doing so produced visible black internal seams.
    """
    claimed = np.zeros((height, width), dtype=bool)
    exclusive: list[tuple[LayerSpec, np.ndarray]] = []
    for spec, raw in sorted(authored_layer_masks(width, height, specs=specs), key=lambda item: item[0].depth):
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


def _shift_mask(mask: np.ndarray, *, dx: int) -> np.ndarray:
    height, width = mask.shape
    shifted = np.zeros_like(mask)
    if abs(dx) >= width:
        return shifted

    if dx >= 0:
        src = slice(0, width - dx)
        dst = slice(dx, width)
    else:
        src = slice(-dx, width)
        dst = slice(0, width + dx)

    shifted[:, dst] = mask[:, src]
    return shifted


def _shift_full_rgb_edge_padded(reference: np.ndarray, *, dx: int) -> np.ndarray:
    """Shift a full-frame display proxy and edge-pad offscreen reveal.

    Padding/fallback is a display hypothesis only. Provenance is tracked separately
    by the observed mask, so edge-padded or hidden-under-foreground support can never
    become observed merely because it has pixels to show.
    """
    height, width = reference.shape[:2]
    shifted = np.empty_like(reference)
    if dx == 0:
        return reference.copy()
    if dx >= width:
        shifted[:] = reference[:, :1]
        return shifted
    if dx <= -width:
        shifted[:] = reference[:, -1:]
        return shifted

    if dx > 0:
        shifted[:, dx:] = reference[:, : width - dx]
        shifted[:, :dx] = reference[:, :1]
    else:
        k = -dx
        shifted[:, : width - k] = reference[:, k:]
        shifted[:, width - k :] = reference[:, -1:]
    return shifted


def _feather_alpha(mask: np.ndarray, *, radius_px: float) -> np.ndarray:
    if radius_px < 0.0:
        raise ValueError("feather radius must be non-negative")
    if radius_px == 0.0:
        return mask.astype(np.float32)
    image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    blurred = image.filter(ImageFilter.GaussianBlur(radius=float(radius_px)))
    return np.asarray(blurred, dtype=np.float32) / 255.0


def render_view(
    reference_rgb: np.ndarray,
    *,
    camera_tx: float,
    hfov_degrees: float = AUTHORED_HFOV_DEGREES,
    specs: tuple[LayerSpec, ...] = DEFAULT_LAYER_SPECS,
    feather_radius_px: float = R1_FEATHER_RADIUS_PX,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Render one pure-lateral authored-proxy camera view.

    Display continuity and epistemic provenance are intentionally separate:
    - RGB starts from a full-frame background proxy so R0 black seam placeholders do
      not appear inside the scene.
    - pixels sourced from underneath authored foreground layers, edge padding, and
      feather-transition bands remain non-observed/hypothesized.
    - only hard/near-hard source projections are labelled observed.
    """
    reference = np.asarray(reference_rgb, dtype=np.uint8)
    if reference.ndim != 3 or reference.shape[2] != 3:
        raise ValueError("reference_rgb must be HxWx3 uint8-compatible")
    height, width = reference.shape[:2]
    camera = authored_camera(width, height, hfov_degrees=hfov_degrees)
    focal = float(camera["intrinsics"][0])

    raw_layers = authored_layer_masks(width, height, specs=specs)
    shifts: dict[str, int] = {
        "atrium-background": int(round(focal * float(camera_tx) / BACKGROUND_DEPTH))
    }
    for spec, _mask in raw_layers:
        shifts[spec.name] = int(round(focal * float(camera_tx) / float(spec.depth)))

    if abs(float(camera_tx)) < 1e-12:
        return reference.copy(), np.ones((height, width), dtype=bool), shifts

    claimed_source = np.zeros((height, width), dtype=bool)
    for _spec, mask in raw_layers:
        claimed_source |= mask

    background_dx = shifts["atrium-background"]
    output = _shift_full_rgb_edge_padded(reference, dx=background_dx).astype(np.float32)
    observed = _shift_mask(~claimed_source, dx=background_dx)

    # Composite far-to-near. The full-frame texture is only a display fallback;
    # provenance follows the shifted hard masks and never the fallback pixels.
    for spec, source_mask in sorted(raw_layers, key=lambda item: item[0].depth, reverse=True):
        dx = shifts[spec.name]
        texture = _shift_full_rgb_edge_padded(reference, dx=dx).astype(np.float32)
        shifted_hard = _shift_mask(source_mask, dx=dx)
        alpha = _feather_alpha(shifted_hard, radius_px=feather_radius_px)

        output = (
            alpha[..., None] * texture
            + (1.0 - alpha[..., None]) * output
        )

        affected = alpha > R1_ALPHA_AFFECTED_THRESHOLD
        near_hard = shifted_hard & (alpha >= R1_ALPHA_OBSERVED_THRESHOLD)
        observed[affected] = False
        observed[near_hard] = True

    return np.clip(np.rint(output), 0, 255).astype(np.uint8), observed, shifts


def render_triplet(reference_rgb: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray, dict[str, int]]]:
    return {
        "hero": render_view(reference_rgb, camera_tx=0.0),
        "neighbor-left": render_view(reference_rgb, camera_tx=-NEIGHBOR_TRANSLATION),
        "neighbor-right": render_view(reference_rgb, camera_tx=NEIGHBOR_TRANSLATION),
    }
