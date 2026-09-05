import json
from pathlib import Path

import numpy as np

from refworld.exp006_layered_proxy import authored_layer_masks
from refworld.exp006_state_visual import (
    OVERVIEW_MODE,
    PROJECT_FOCUS_MODE,
    TARGET_ENTITY_ID,
    TARGET_LAYER_NAME,
    build_state_visual_binding,
    render_state_view,
    source_target_support,
)
from refworld.semantic_handoff import apply_entity_state_patch


ROOT = Path(__file__).resolve().parents[1]


def _binding():
    return json.loads(
        (ROOT / "examples/exp006_collaborative_futures_binding.json").read_text(encoding="utf-8")
    )


def _reference(height=188, width=334):
    y, x = np.mgrid[0:height, 0:width]
    return np.stack(
        [
            (x * 3 + y * 5) % 256,
            (x * 7 + y) % 256,
            (x + y * 11) % 256,
        ],
        axis=-1,
    ).astype(np.uint8)


def test_target_patch_is_inside_accepted_world_model_layer():
    height, width = 188, 334
    support = source_target_support(width, height)
    world_model = next(
        mask for spec, mask in authored_layer_masks(width, height) if spec.name == TARGET_LAYER_NAME
    )
    assert np.any(support)
    assert not np.any(support & ~world_model)


def test_overview_mode_is_exact_r1_with_no_state_generated_pixels():
    reference = _reference()
    view = render_state_view(reference, _binding()["world_state"], camera_tx=0.0)
    assert view.panel_mode == OVERVIEW_MODE
    assert np.array_equal(view.rgb, reference)
    assert not np.any(view.changed)
    assert not np.any(view.state_generated)


def test_project_focus_changes_only_target_and_changed_pixels_are_not_observed():
    reference = _reference()
    before_world = _binding()["world_state"]
    edited_world = apply_entity_state_patch(
        before_world,
        entity_id=TARGET_ENTITY_ID,
        patch={"panel_mode": PROJECT_FOCUS_MODE},
    )
    before = render_state_view(reference, before_world, camera_tx=0.0)
    after = render_state_view(reference, edited_world, camera_tx=0.0)

    assert after.panel_mode == PROJECT_FOCUS_MODE
    assert np.any(after.changed)
    assert not np.any(after.changed & ~after.target_support)
    assert not np.any(after.changed & after.observed)
    assert np.array_equal(before.rgb[~after.target_support], after.rgb[~after.target_support])
    assert np.array_equal(after.changed, after.state_generated)


def test_full_binding_gate_persists_visual_state_and_reverts_exactly():
    reference = _reference()
    result = build_state_visual_binding(reference, _binding())
    report = result["report"]

    assert report["automated_gate_passed"] is True
    assert all(report["gate_checks"].values())
    assert report["semantic_drift"]["collateral_semantic_drift_count"] == 0
    assert report["target"]["before"] == OVERVIEW_MODE
    assert report["target"]["after"] == PROJECT_FOCUS_MODE
    assert report["provenance_rule"]["changed_pixels_may_remain_observed"] is False
    assert report["evidence_boundary"]["accepted_r1_renderer_modified"] is False

    edited = result["views"]["edited"]["hero"]
    reloaded = result["views"]["reloaded_hero"]
    reverted = result["views"]["reverted_hero"]
    assert np.array_equal(edited.rgb, reloaded.rgb)
    assert np.array_equal(reverted.rgb, reference)


def test_both_neighbors_receive_visible_local_state_change_without_escape():
    reference = _reference()
    result = build_state_visual_binding(reference, _binding())
    for name in ("neighbor-left", "neighbor-right"):
        report = result["report"]["views"][name]
        assert report["changed_pixels"] > 0
        assert report["changed_outside_target_support"] == 0
        assert report["changed_pixels_still_observed"] == 0
        assert report["outside_target_exact"] is True
