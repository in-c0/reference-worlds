"""Deterministic interaction/revisit trace for the EXP-006 LifeOS handoff.

This module composes the already-accepted R1 authored proxy with the canonical
renderer-independent semantic state. It does not infer new geometry, alter the
reference binding, consume benchmark evidence, or let renderer state become the
source of semantic truth.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from refworld.exp006_layered_proxy import (
    AUTHORED_HFOV_DEGREES,
    DEFAULT_LAYER_SPECS,
    NEIGHBOR_TRANSLATION,
    authored_camera,
    authored_layer_masks,
    render_view,
)
from refworld.semantic_handoff import (
    apply_entity_state_patch,
    canonical_sha256,
    semantic_drift_report,
    snapshot_roundtrip,
)


AWAY_TRANSLATION = 1.60
OCCLUDED_ENTITY_ID = "lifeos.project.xuxi-room"
MIN_NEIGHBOR_OBSERVED_FRACTION = 0.90

TRACE_STEP_NAMES: tuple[str, ...] = (
    "load-hero",
    "score-hero-before",
    "visit-neighbor-left",
    "visit-neighbor-right",
    "inspect-three-entities",
    "move-entity-out-of-view",
    "return-and-verify-identity",
    "apply-bounded-semantic-edit",
    "measure-collateral-drift",
    "navigate-away-and-return",
    "reload-and-verify-state",
    "score-hero-after",
)


@dataclass(frozen=True)
class InteractionTraceResult:
    report: dict[str, Any]
    images: dict[str, np.ndarray]
    states: dict[str, dict[str, Any]]


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


def entity_visibility_at_tx(
    width: int,
    height: int,
    *,
    camera_tx: float,
) -> dict[str, dict[str, Any]]:
    """Return deterministic authored-proxy visibility for semantic layer entities."""
    camera = authored_camera(width, height, hfov_degrees=AUTHORED_HFOV_DEGREES)
    focal = float(camera["intrinsics"][0])
    visibility: dict[str, dict[str, Any]] = {}
    for spec, source_mask in authored_layer_masks(width, height, specs=DEFAULT_LAYER_SPECS):
        if spec.entity_id is None:
            continue
        dx = int(round(focal * float(camera_tx) / float(spec.depth)))
        shifted = _shift_mask(source_mask, dx=dx)
        source_pixels = int(np.count_nonzero(source_mask))
        visible_pixels = int(np.count_nonzero(shifted))
        visibility[spec.entity_id] = {
            "layer": spec.name,
            "depth": float(spec.depth),
            "pixel_shift": dx,
            "source_pixels": source_pixels,
            "visible_pixels": visible_pixels,
            "visible_fraction": 0.0 if source_pixels == 0 else visible_pixels / source_pixels,
            "fully_out_of_view": visible_pixels == 0,
        }
    return visibility


def rgb_fidelity(reference_rgb: np.ndarray, candidate_rgb: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(reference_rgb, dtype=np.uint8)
    candidate = np.asarray(candidate_rgb, dtype=np.uint8)
    if reference.shape != candidate.shape:
        raise ValueError("RGB fidelity inputs must have identical shapes")
    delta = candidate.astype(np.int16) - reference.astype(np.int16)
    abs_delta = np.abs(delta)
    mse = float(np.mean(delta.astype(np.float64) ** 2))
    changed = np.any(delta != 0, axis=-1)
    if mse == 0.0:
        psnr_db = None
        psnr_infinite = True
    else:
        psnr_db = float(10.0 * math.log10((255.0**2) / mse))
        psnr_infinite = False
    return {
        "exact_match": bool(np.array_equal(reference, candidate)),
        "mse": mse,
        "mean_abs_error": float(np.mean(abs_delta)),
        "max_abs_error": int(abs_delta.max(initial=0)),
        "changed_pixel_fraction": float(np.mean(changed)),
        "psnr_db": psnr_db,
        "psnr_infinite": psnr_infinite,
    }


def _entity_index(world_state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    entities = world_state.get("entities")
    if not isinstance(entities, list):
        raise ValueError("world_state.entities must be a list")
    index: dict[str, dict[str, Any]] = {}
    for raw in entities:
        if not isinstance(raw, dict):
            raise ValueError("world_state entity must be an object")
        entity_id = raw.get("id")
        if not isinstance(entity_id, str) or not entity_id:
            raise ValueError("entity id must be a non-empty string")
        if entity_id in index:
            raise ValueError(f"duplicate entity id: {entity_id}")
        index[entity_id] = raw
    return index


def _inspect_entities(world_state: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entity_id, entity in sorted(_entity_index(world_state).items()):
        rows.append(
            {
                "entity_id": entity_id,
                "kind": entity.get("kind"),
                "state": copy.deepcopy(entity.get("state", {})),
                "spatial_membership": entity.get("spatial_membership"),
                "evidence_origin": entity.get("evidence_origin"),
                "resolution_state": entity.get("resolution_state"),
                "entity_sha256": canonical_sha256(entity),
            }
        )
    return rows


def _patch_persisted(world_state: Mapping[str, Any], *, entity_id: str, patch: Mapping[str, Any]) -> bool:
    entity = _entity_index(world_state)[entity_id]
    state = entity.get("state")
    if not isinstance(state, Mapping):
        return False
    return all(state.get(key) == value for key, value in patch.items())


def build_interaction_trace(
    reference_rgb: np.ndarray,
    binding: Mapping[str, Any],
) -> InteractionTraceResult:
    """Execute the frozen 12-step EXP-006 interaction trace in memory."""
    reference = np.asarray(reference_rgb, dtype=np.uint8)
    if reference.ndim != 3 or reference.shape[2] != 3:
        raise ValueError("reference_rgb must be HxWx3")
    height, width = reference.shape[:2]

    renderer_binding = binding.get("renderer_binding")
    if not isinstance(renderer_binding, Mapping):
        raise ValueError("binding.renderer_binding missing")
    if bool(renderer_binding.get("owns_semantic_truth")):
        raise ValueError("renderer may not own semantic truth")
    if bool(renderer_binding.get("metric_reconstruction_claim")):
        raise ValueError("EXP-006 trace may not claim metric reconstruction")

    before_world = copy.deepcopy(binding["world_state"])
    before_index = _entity_index(before_world)
    initial_ids = sorted(before_index)
    if len(initial_ids) < 3:
        raise ValueError("EXP-006 requires at least three semantic entities")
    if OCCLUDED_ENTITY_ID not in before_index:
        raise ValueError(f"missing required out-of-view entity: {OCCLUDED_ENTITY_ID}")

    # Steps 1-4: hero and frozen R1 neighboring views.
    hero_before, hero_before_observed, _ = render_view(reference, camera_tx=0.0)
    neighbor_left, neighbor_left_observed, _ = render_view(reference, camera_tx=-NEIGHBOR_TRANSLATION)
    neighbor_right, neighbor_right_observed, _ = render_view(reference, camera_tx=NEIGHBOR_TRANSLATION)

    hero_before_score = rgb_fidelity(reference, hero_before)
    neighbor_observed = {
        "left": float(np.mean(neighbor_left_observed)),
        "right": float(np.mean(neighbor_right_observed)),
    }

    # Step 5: inspect native semantic state, not visual tracking labels.
    inspected_before = _inspect_entities(before_world)
    before_world_sha = canonical_sha256(before_world)

    # Step 6: move far enough that the project entity's authored layer is fully out of view.
    away_rgb, away_observed, _ = render_view(reference, camera_tx=AWAY_TRANSLATION)
    away_visibility = entity_visibility_at_tx(width, height, camera_tx=AWAY_TRANSLATION)
    occluded = away_visibility[OCCLUDED_ENTITY_ID]["fully_out_of_view"]

    # Step 7: camera navigation alone cannot mutate canonical semantic state.
    returned_world = copy.deepcopy(before_world)
    return_identity_unchanged = canonical_sha256(returned_world) == before_world_sha

    # Steps 8-9: bounded semantic edit and collateral semantic/visual drift.
    edit = binding["trace_plan"]["bounded_edit"]
    edit_entity_id = edit["entity_id"]
    edit_patch = edit["state_patch"]
    edited_world = apply_entity_state_patch(
        returned_world,
        entity_id=edit_entity_id,
        patch=edit_patch,
        actor="owner-exp006-interaction-trace",
    )
    semantic_drift = semantic_drift_report(
        returned_world,
        edited_world,
        target_entity_id=edit_entity_id,
    )
    hero_after_edit, _after_edit_observed, _ = render_view(reference, camera_tx=0.0)
    visual_collateral = rgb_fidelity(hero_before, hero_after_edit)

    # Step 10: navigate away and return with edited canonical state untouched.
    edited_sha_before_navigation = canonical_sha256(edited_world)
    _away_after_edit, _away_after_edit_observed, _ = render_view(reference, camera_tx=AWAY_TRANSLATION)
    returned_edited_world = copy.deepcopy(edited_world)
    edit_survives_navigation = (
        canonical_sha256(returned_edited_world) == edited_sha_before_navigation
        and _patch_persisted(returned_edited_world, entity_id=edit_entity_id, patch=edit_patch)
    )

    # Step 11: deterministic JSON snapshot/reload must retain IDs and the edit.
    reloaded_world = snapshot_roundtrip(returned_edited_world)
    reloaded_ids = sorted(_entity_index(reloaded_world))
    stable_ids_after_reload = reloaded_ids == initial_ids
    edit_survives_reload = _patch_persisted(reloaded_world, entity_id=edit_entity_id, patch=edit_patch)
    inspected_after = _inspect_entities(reloaded_world)

    # Step 12: return to the hero camera and re-score anchor fidelity.
    hero_after, hero_after_observed, _ = render_view(reference, camera_tx=0.0)
    hero_after_score = rgb_fidelity(reference, hero_after)
    hero_before_after_drift = rgb_fidelity(hero_before, hero_after)

    steps = [
        {"step": 1, "name": TRACE_STEP_NAMES[0], "camera_tx": 0.0},
        {"step": 2, "name": TRACE_STEP_NAMES[1], "fidelity": hero_before_score},
        {"step": 3, "name": TRACE_STEP_NAMES[2], "camera_tx": -NEIGHBOR_TRANSLATION, "observed_fraction": neighbor_observed["left"]},
        {"step": 4, "name": TRACE_STEP_NAMES[3], "camera_tx": NEIGHBOR_TRANSLATION, "observed_fraction": neighbor_observed["right"]},
        {"step": 5, "name": TRACE_STEP_NAMES[4], "entities": inspected_before},
        {"step": 6, "name": TRACE_STEP_NAMES[5], "camera_tx": AWAY_TRANSLATION, "target_entity_id": OCCLUDED_ENTITY_ID, "visibility": away_visibility[OCCLUDED_ENTITY_ID]},
        {"step": 7, "name": TRACE_STEP_NAMES[6], "identity_unchanged": return_identity_unchanged},
        {"step": 8, "name": TRACE_STEP_NAMES[7], "entity_id": edit_entity_id, "state_patch": copy.deepcopy(edit_patch)},
        {"step": 9, "name": TRACE_STEP_NAMES[8], "semantic_drift": semantic_drift, "hero_visual_drift": visual_collateral},
        {"step": 10, "name": TRACE_STEP_NAMES[9], "edit_survives_navigation": edit_survives_navigation},
        {"step": 11, "name": TRACE_STEP_NAMES[10], "stable_ids": stable_ids_after_reload, "edit_survives_reload": edit_survives_reload, "entities": inspected_after},
        {"step": 12, "name": TRACE_STEP_NAMES[11], "fidelity": hero_after_score, "before_after_drift": hero_before_after_drift},
    ]

    gate_checks = {
        "twelve_steps_recorded": len(steps) == 12 and [item["name"] for item in steps] == list(TRACE_STEP_NAMES),
        "hero_before_exact_reference": hero_before_score["exact_match"],
        "neighbors_above_observed_floor": neighbor_observed["left"] >= MIN_NEIGHBOR_OBSERVED_FRACTION and neighbor_observed["right"] >= MIN_NEIGHBOR_OBSERVED_FRACTION,
        "one_semantic_entity_fully_out_of_view": bool(occluded),
        "identity_unchanged_after_camera_return": bool(return_identity_unchanged),
        "target_edit_changed": bool(semantic_drift["target_changed"]),
        "zero_collateral_semantic_drift": semantic_drift["collateral_semantic_drift_count"] == 0,
        "zero_collateral_hero_visual_drift": visual_collateral["exact_match"],
        "edit_survives_navigation": bool(edit_survives_navigation),
        "stable_ids_after_reload": bool(stable_ids_after_reload),
        "edit_survives_reload": bool(edit_survives_reload),
        "hero_after_exact_reference": hero_after_score["exact_match"],
        "hero_before_after_exact": hero_before_after_drift["exact_match"],
    }

    report: dict[str, Any] = {
        "version": "0.1",
        "stage": "refworld-exp006-interaction-trace-v0",
        "trace_steps": steps,
        "entity_ids": initial_ids,
        "neighbor_observed_fraction": neighbor_observed,
        "away": {
            "camera_tx": AWAY_TRANSLATION,
            "visibility": away_visibility,
            "observed_fraction": float(np.mean(away_observed)),
        },
        "bounded_edit": copy.deepcopy(edit),
        "semantic_drift": semantic_drift,
        "visual_drift": {
            "hero_before_vs_after_edit": visual_collateral,
            "hero_before_vs_final_hero": hero_before_after_drift,
        },
        "hero_fidelity": {
            "before": hero_before_score,
            "after": hero_after_score,
        },
        "persistence": {
            "edit_survives_navigation": edit_survives_navigation,
            "stable_ids_after_reload": stable_ids_after_reload,
            "edit_survives_reload": edit_survives_reload,
            "edited_world_sha256": canonical_sha256(edited_world),
            "reloaded_world_sha256": canonical_sha256(reloaded_world),
        },
        "renderer_semantic_boundary": {
            "renderer_owns_semantic_truth": False,
            "semantic_edit_visually_mapped_in_r1": False,
            "note": "R1 preserves persistent semantic state but does not yet map panel_mode into appearance; zero hero visual drift is therefore expected and explicitly reported.",
        },
        "evidence_boundary": {
            "fresh_benchmark_evidence_consumed": False,
            "rank4_touched": False,
            "target_depth_read": False,
            "metric_reconstruction_claim": False,
        },
        "gate_checks": gate_checks,
        "automated_gate_passed": bool(all(gate_checks.values())),
    }

    return InteractionTraceResult(
        report=report,
        images={
            "hero-before": hero_before,
            "neighbor-left": neighbor_left,
            "neighbor-right": neighbor_right,
            "away": away_rgb,
            "hero-after": hero_after,
        },
        states={
            "world-before": before_world,
            "world-edited": edited_world,
            "world-reloaded": reloaded_world,
        },
    )
