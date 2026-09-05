import json
from pathlib import Path

import numpy as np

from refworld.exp006_interaction_trace import (
    AWAY_TRANSLATION,
    OCCLUDED_ENTITY_ID,
    TRACE_STEP_NAMES,
    build_interaction_trace,
    entity_visibility_at_tx,
    rgb_fidelity,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_binding():
    return json.loads(
        (ROOT / "examples/exp006_collaborative_futures_binding.json").read_text(encoding="utf-8")
    )


def _synthetic_reference(height=188, width=334):
    y, x = np.mgrid[0:height, 0:width]
    return np.stack(
        [
            (x * 3 + y * 5) % 256,
            (x * 7 + y) % 256,
            (x + y * 11) % 256,
        ],
        axis=-1,
    ).astype(np.uint8)


def test_rgb_fidelity_exact_case_is_json_safe():
    image = _synthetic_reference()
    score = rgb_fidelity(image, image.copy())
    assert score["exact_match"] is True
    assert score["mse"] == 0.0
    assert score["changed_pixel_fraction"] == 0.0
    assert score["psnr_db"] is None
    assert score["psnr_infinite"] is True


def test_away_camera_fully_removes_predeclared_project_entity():
    visibility = entity_visibility_at_tx(1672, 941, camera_tx=AWAY_TRANSLATION)
    assert OCCLUDED_ENTITY_ID in visibility
    assert visibility[OCCLUDED_ENTITY_ID]["fully_out_of_view"] is True
    assert visibility[OCCLUDED_ENTITY_ID]["visible_pixels"] == 0


def test_trace_records_all_required_steps_and_persists_bounded_edit():
    result = build_interaction_trace(_synthetic_reference(), _load_binding())
    report = result.report

    assert [step["name"] for step in report["trace_steps"]] == list(TRACE_STEP_NAMES)
    assert len(report["trace_steps"]) == 12
    assert report["away"]["visibility"][OCCLUDED_ENTITY_ID]["fully_out_of_view"] is True
    assert report["semantic_drift"]["target_changed"] is True
    assert report["semantic_drift"]["collateral_semantic_drift_count"] == 0
    assert report["persistence"]["edit_survives_navigation"] is True
    assert report["persistence"]["stable_ids_after_reload"] is True
    assert report["persistence"]["edit_survives_reload"] is True

    edit = report["bounded_edit"]
    target = next(
        entity
        for entity in result.states["world-reloaded"]["entities"]
        if entity["id"] == edit["entity_id"]
    )
    for key, value in edit["state_patch"].items():
        assert target["state"][key] == value


def test_trace_hero_is_exact_before_and_after_and_has_no_collateral_visual_drift():
    result = build_interaction_trace(_synthetic_reference(), _load_binding())
    report = result.report

    assert report["hero_fidelity"]["before"]["exact_match"] is True
    assert report["hero_fidelity"]["after"]["exact_match"] is True
    assert report["visual_drift"]["hero_before_vs_after_edit"]["exact_match"] is True
    assert report["visual_drift"]["hero_before_vs_final_hero"]["exact_match"] is True
    assert np.array_equal(result.images["hero-before"], result.images["hero-after"])


def test_trace_gate_passes_without_claiming_visual_mapping_or_metric_geometry():
    result = build_interaction_trace(_synthetic_reference(), _load_binding())
    report = result.report

    assert report["automated_gate_passed"] is True
    assert all(report["gate_checks"].values())
    assert report["renderer_semantic_boundary"]["renderer_owns_semantic_truth"] is False
    assert report["renderer_semantic_boundary"]["semantic_edit_visually_mapped_in_r1"] is False
    assert report["evidence_boundary"] == {
        "fresh_benchmark_evidence_consumed": False,
        "rank4_touched": False,
        "target_depth_read": False,
        "metric_reconstruction_claim": False,
    }
