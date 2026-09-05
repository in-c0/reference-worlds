"""Bounded semantic-state-driven visual binding for the accepted EXP-006 slice.

This module deliberately wraps the accepted R1 renderer instead of changing it.
`lifeos.system.world-model.panel_mode=project-focus` produces one deterministic
visual patch confined to the authored world-model support. Pixels changed by the
semantic edit are explicitly removed from OBSERVED provenance and reported as
state-generated edit support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from refworld.exp006_layered_proxy import (
    DEFAULT_LAYER_SPECS,
    NEIGHBOR_TRANSLATION,
    authored_layer_masks,
    render_view,
)
from refworld.semantic_handoff import (
    apply_entity_state_patch,
    semantic_drift_report,
    snapshot_roundtrip,
)


TARGET_ENTITY_ID = "lifeos.system.world-model"
TARGET_FIELD = "panel_mode"
OVERVIEW_MODE = "overview"
PROJECT_FOCUS_MODE = "project-focus"
TARGET_LAYER_NAME = "world-model"

# Authored wholly inside the accepted world-model polygon. This is a UI edit
# support, not inferred geometry.
FOCUS_PATCH_NORMALIZED = (0.535, 0.245, 0.695, 0.325)


@dataclass(frozen=True)
class StateVisualView:
    rgb: np.ndarray
    observed: np.ndarray
    hypothesized: np.ndarray
    state_generated: np.ndarray
    target_support: np.ndarray
    changed: np.ndarray
    shifts: dict[str, int]
    camera_tx: float
    panel_mode: str


def _entity_index(world_state: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    entities = world_state.get("entities")
    if not isinstance(entities, list):
        raise ValueError("world_state.entities must be a list")
    index: dict[str, Mapping[str, Any]] = {}
    for entity in entities:
        if not isinstance(entity, Mapping):
            raise ValueError("world_state entity must be an object")
        entity_id = entity.get("id")
        if not isinstance(entity_id, str) or not entity_id:
            raise ValueError("world_state entity id must be a non-empty string")
        index[entity_id] = entity
    return index


def panel_mode(world_state: Mapping[str, Any]) -> str:
    entity = _entity_index(world_state).get(TARGET_ENTITY_ID)
    if entity is None:
        raise KeyError(TARGET_ENTITY_ID)
    state = entity.get("state")
    if not isinstance(state, Mapping):
        raise ValueError("target entity state must be an object")
    mode = state.get(TARGET_FIELD)
    if mode not in {OVERVIEW_MODE, PROJECT_FOCUS_MODE}:
        raise ValueError(f"unsupported {TARGET_FIELD}: {mode!r}")
    return str(mode)


def _shift_mask(mask: np.ndarray, *, dx: int) -> np.ndarray:
    height, width = mask.shape
    shifted = np.zeros((height, width), dtype=bool)
    if abs(dx) >= width:
        return shifted
    if dx >= 0:
        shifted[:, dx:] = mask[:, : width - dx]
    else:
        k = -dx
        shifted[:, : width - k] = mask[:, k:]
    return shifted


def _shift_masked_rgb(rgb: np.ndarray, mask: np.ndarray, *, dx: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape
    shifted_rgb = np.zeros_like(rgb)
    shifted_mask = np.zeros((height, width), dtype=bool)
    if abs(dx) >= width:
        return shifted_rgb, shifted_mask
    if dx >= 0:
        shifted_rgb[:, dx:] = rgb[:, : width - dx]
        shifted_mask[:, dx:] = mask[:, : width - dx]
    else:
        k = -dx
        shifted_rgb[:, : width - k] = rgb[:, k:]
        shifted_mask[:, : width - k] = mask[:, k:]
    return shifted_rgb, shifted_mask


def _rect_mask(width: int, height: int) -> np.ndarray:
    x0n, y0n, x1n, y1n = FOCUS_PATCH_NORMALIZED
    x0 = int(round(x0n * (width - 1)))
    y0 = int(round(y0n * (height - 1)))
    x1 = int(round(x1n * (width - 1)))
    y1 = int(round(y1n * (height - 1)))
    mask = np.zeros((height, width), dtype=bool)
    mask[y0 : y1 + 1, x0 : x1 + 1] = True
    return mask


def source_target_support(width: int, height: int) -> np.ndarray:
    world_model_mask = None
    for spec, mask in authored_layer_masks(width, height):
        if spec.name == TARGET_LAYER_NAME:
            world_model_mask = mask
            break
    if world_model_mask is None:
        raise RuntimeError("accepted R1 renderer has no world-model layer")
    support = _rect_mask(width, height) & world_model_mask
    if not np.any(support):
        raise RuntimeError("state visual target support is empty")
    return support


def _project_focus_overlay(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    support = source_target_support(width, height)
    ys, xs = np.nonzero(support)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    image = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    border = max(1, int(round(width / 900.0)))
    margin = max(2, int(round(width / 700.0)))
    draw.rectangle((x0, y0, x1, y1), fill=(8, 26, 28), outline=(114, 235, 218), width=border)

    font = ImageFont.load_default()
    draw.text((x0 + margin, y0 + margin), "PROJECT FOCUS", fill=(228, 255, 249), font=font)

    # A deterministic compact focus indicator: three ranked project rails.
    inner_w = max(1, x1 - x0 - 2 * margin)
    rail_y = y0 + max(14, margin + 12)
    rail_h = max(2, int(round(height / 260.0)))
    gap = max(3, rail_h * 2)
    widths = (0.82, 0.58, 0.36)
    for i, fraction in enumerate(widths):
        yy = rail_y + i * gap
        if yy + rail_h <= y1 - margin:
            draw.rectangle(
                (x0 + margin, yy, x0 + margin + int(round(inner_w * fraction)), yy + rail_h),
                fill=(114, 235, 218),
            )

    overlay = np.asarray(image, dtype=np.uint8)
    return overlay, support


def render_state_view(
    reference_rgb: np.ndarray,
    world_state: Mapping[str, Any],
    *,
    camera_tx: float,
) -> StateVisualView:
    reference = np.asarray(reference_rgb, dtype=np.uint8)
    if reference.ndim != 3 or reference.shape[2] != 3:
        raise ValueError("reference_rgb must be HxWx3")
    height, width = reference.shape[:2]

    base_rgb, base_observed, shifts = render_view(reference, camera_tx=float(camera_tx))
    mode = panel_mode(world_state)
    empty = np.zeros((height, width), dtype=bool)

    if mode == OVERVIEW_MODE:
        return StateVisualView(
            rgb=base_rgb,
            observed=base_observed,
            hypothesized=~base_observed,
            state_generated=empty.copy(),
            target_support=empty.copy(),
            changed=empty.copy(),
            shifts=shifts,
            camera_tx=float(camera_tx),
            panel_mode=mode,
        )

    overlay_source, support_source = _project_focus_overlay(width, height)
    dx = int(shifts[TARGET_LAYER_NAME])
    overlay_shifted, support_shifted = _shift_masked_rgb(overlay_source, support_source, dx=dx)

    edited = base_rgb.copy()
    edited[support_shifted] = overlay_shifted[support_shifted]
    changed = np.any(edited != base_rgb, axis=2)
    if not np.any(changed):
        raise RuntimeError("project-focus edit produced no visible pixel change")
    if np.any(changed & ~support_shifted):
        raise RuntimeError("state visual edit escaped declared target support")

    observed = base_observed.copy()
    observed[changed] = False
    state_generated = changed.copy()
    hypothesized = ~(observed | state_generated)

    return StateVisualView(
        rgb=edited,
        observed=observed,
        hypothesized=hypothesized,
        state_generated=state_generated,
        target_support=support_shifted,
        changed=changed,
        shifts=shifts,
        camera_tx=float(camera_tx),
        panel_mode=mode,
    )


def _view_report(before: StateVisualView, after: StateVisualView) -> dict[str, Any]:
    if before.rgb.shape != after.rgb.shape:
        raise ValueError("view shapes differ")
    outside = ~after.target_support
    changed_outside = after.changed & outside
    changed_still_observed = after.changed & after.observed
    return {
        "camera_tx": float(after.camera_tx),
        "panel_mode": after.panel_mode,
        "target_support_fraction": float(np.mean(after.target_support)),
        "changed_fraction": float(np.mean(after.changed)),
        "changed_pixels": int(np.count_nonzero(after.changed)),
        "changed_outside_target_support": int(np.count_nonzero(changed_outside)),
        "changed_pixels_still_observed": int(np.count_nonzero(changed_still_observed)),
        "outside_target_exact": bool(np.array_equal(before.rgb[outside], after.rgb[outside])),
        "state_generated_fraction": float(np.mean(after.state_generated)),
        "observed_fraction": float(np.mean(after.observed)),
        "hypothesized_fraction": float(np.mean(after.hypothesized)),
        "layer_pixel_shifts": dict(after.shifts),
    }


def build_state_visual_binding(reference_rgb: np.ndarray, binding: Mapping[str, Any]) -> dict[str, Any]:
    reference = np.asarray(reference_rgb, dtype=np.uint8)
    before_world = binding.get("world_state")
    if not isinstance(before_world, Mapping):
        raise ValueError("binding.world_state must be an object")
    if panel_mode(before_world) != OVERVIEW_MODE:
        raise RuntimeError("frozen state-visual trace must start from panel_mode=overview")

    before_views = {
        "hero": render_state_view(reference, before_world, camera_tx=0.0),
        "neighbor-left": render_state_view(reference, before_world, camera_tx=-NEIGHBOR_TRANSLATION),
        "neighbor-right": render_state_view(reference, before_world, camera_tx=NEIGHBOR_TRANSLATION),
    }

    edited_world = apply_entity_state_patch(
        before_world,
        entity_id=TARGET_ENTITY_ID,
        patch={TARGET_FIELD: PROJECT_FOCUS_MODE},
        actor="owner-exp006-state-visual-v0",
    )
    drift = semantic_drift_report(before_world, edited_world, target_entity_id=TARGET_ENTITY_ID)
    edited_views = {
        "hero": render_state_view(reference, edited_world, camera_tx=0.0),
        "neighbor-left": render_state_view(reference, edited_world, camera_tx=-NEIGHBOR_TRANSLATION),
        "neighbor-right": render_state_view(reference, edited_world, camera_tx=NEIGHBOR_TRANSLATION),
    }

    reloaded_world = snapshot_roundtrip(edited_world)
    reloaded_hero = render_state_view(reference, reloaded_world, camera_tx=0.0)

    reverted_world = apply_entity_state_patch(
        reloaded_world,
        entity_id=TARGET_ENTITY_ID,
        patch={TARGET_FIELD: OVERVIEW_MODE},
        actor="owner-exp006-state-visual-v0-revert",
    )
    reverted_hero = render_state_view(reference, reverted_world, camera_tx=0.0)

    view_reports = {
        name: _view_report(before_views[name], edited_views[name])
        for name in ("hero", "neighbor-left", "neighbor-right")
    }

    checks = {
        "pre_edit_hero_exact_reference": bool(np.array_equal(before_views["hero"].rgb, reference)),
        "semantic_target_changed": bool(drift["target_changed"]),
        "zero_collateral_semantic_drift": drift["collateral_semantic_drift_count"] == 0,
        "hero_visible_change_exists": view_reports["hero"]["changed_pixels"] > 0,
        "zero_changed_pixels_outside_target_all_views": all(
            view_reports[name]["changed_outside_target_support"] == 0 for name in view_reports
        ),
        "zero_changed_pixels_still_observed_all_views": all(
            view_reports[name]["changed_pixels_still_observed"] == 0 for name in view_reports
        ),
        "outside_target_exact_all_views": all(
            bool(view_reports[name]["outside_target_exact"]) for name in view_reports
        ),
        "reload_rederives_identical_hero": bool(
            np.array_equal(edited_views["hero"].rgb, reloaded_hero.rgb)
            and np.array_equal(edited_views["hero"].state_generated, reloaded_hero.state_generated)
        ),
        "reload_preserves_panel_mode": panel_mode(reloaded_world) == PROJECT_FOCUS_MODE,
        "revert_restores_exact_reference_hero": bool(np.array_equal(reverted_hero.rgb, reference)),
        "neighbor_visible_change_exists": bool(
            view_reports["neighbor-left"]["changed_pixels"] > 0
            and view_reports["neighbor-right"]["changed_pixels"] > 0
        ),
    }

    report = {
        "version": "0.1",
        "stage": "refworld-exp006-state-visual-binding-v0",
        "target": {
            "entity_id": TARGET_ENTITY_ID,
            "field": TARGET_FIELD,
            "before": OVERVIEW_MODE,
            "after": PROJECT_FOCUS_MODE,
            "target_layer": TARGET_LAYER_NAME,
            "source_patch_normalized": list(FOCUS_PATCH_NORMALIZED),
        },
        "views": view_reports,
        "semantic_drift": drift,
        "persistence": {
            "reload_rederives_identical_hero": checks["reload_rederives_identical_hero"],
            "reload_preserves_panel_mode": checks["reload_preserves_panel_mode"],
            "revert_restores_exact_reference_hero": checks["revert_restores_exact_reference_hero"],
        },
        "provenance_rule": {
            "changed_pixels_class": "state-generated-edit",
            "changed_pixels_may_remain_observed": False,
            "base_r1_hypothesized_support_preserved": True,
        },
        "evidence_boundary": {
            "fresh_benchmark_evidence_consumed": False,
            "rank4_touched": False,
            "target_depth_read": False,
            "metric_reconstruction_claim": False,
            "accepted_r1_renderer_modified": False,
        },
        "gate_checks": checks,
        "automated_gate_passed": bool(all(checks.values())),
    }

    return {
        "report": report,
        "world_states": {
            "before": before_world,
            "edited": edited_world,
            "reloaded": reloaded_world,
            "reverted": reverted_world,
        },
        "views": {
            "before": before_views,
            "edited": edited_views,
            "reloaded_hero": reloaded_hero,
            "reverted_hero": reverted_hero,
        },
    }
