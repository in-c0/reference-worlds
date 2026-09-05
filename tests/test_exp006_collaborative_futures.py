import json
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SHA256 = "4ee7a137e577378a02600ac8a32dc89a7c8409120273622227ad972cc5aff61a"
XUXI_ROOM_RECORD = "https://github.com/in-c0/lifeos-local-ai/pull/113"


def _load(path: str):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_collaborative_futures_binding_validates_against_both_contracts():
    handoff_schema = _load("schemas/lifeos-handoff.schema.json")
    world_schema = _load("schemas/world-state.schema.json")
    binding = _load("examples/exp006_collaborative_futures_binding.json")

    jsonschema.validate(binding, handoff_schema)
    jsonschema.validate(binding["world_state"], world_schema)


def test_reference_is_bound_by_hash_without_metric_geometry_claim():
    binding = _load("examples/exp006_collaborative_futures_binding.json")
    reference = binding["reference_binding"]
    world = binding["world_state"]

    assert reference["status"] == "bound"
    assert reference["sha256"] == REFERENCE_SHA256
    assert world["anchors"][0]["observation_sha256"] == REFERENCE_SHA256
    assert world["anchors"][0]["camera_status"] == "pending-authored-proxy-registration"
    assert world["history"][0]["payload"]["binary_committed"] is False
    assert world["history"][0]["payload"]["metric_geometry_claim"] is False
    assert binding["renderer_binding"]["metric_reconstruction_claim"] is False


def test_three_observed_entities_keep_reference_provenance_and_only_project_is_externally_bound():
    binding = _load("examples/exp006_collaborative_futures_binding.json")
    observation_id = binding["reference_binding"]["observation_id"]
    entities = binding["world_state"]["entities"]
    mappings = {item["entity_id"]: item for item in binding["lifeos_mappings"]}

    assert len(entities) == 3
    assert {mapping["role"] for mapping in mappings.values()} == {
        "architectural",
        "project_artifact",
        "system_instrument",
    }
    for entity in entities:
        assert entity["evidence_origin"] == "observed"
        assert entity["resolution_state"] == "hypothesized"
        assert entity["provenance"]["observation_ids"] == [observation_id]

    project = mappings["lifeos.project.xuxi-room"]
    assert project["binding_status"] == "bound"
    assert project["external_id"] == XUXI_ROOM_RECORD
    assert "project_status" in project["authoritative_fields"]
    assert "project_status" not in next(
        entity["state"] for entity in entities if entity["id"] == "lifeos.project.xuxi-room"
    )

    pending = [mapping for mapping in mappings.values() if mapping["entity_id"] != "lifeos.project.xuxi-room"]
    assert all(mapping["binding_status"] == "pending" for mapping in pending)
    assert all(mapping["external_id"] is None for mapping in pending)


def test_renderer_cannot_own_semantic_truth_and_rank4_is_not_part_of_the_slice():
    binding = _load("examples/exp006_collaborative_futures_binding.json")
    renderer = binding["renderer_binding"]
    plan = " ".join(binding["trace_plan"]["ordered_steps"])

    assert renderer["owns_semantic_truth"] is False
    assert renderer["renderer_id"] is None
    assert "rank-4" not in plan.lower()
    assert binding["trace_plan"]["bounded_edit"] == {
        "entity_id": "lifeos.system.world-model",
        "state_patch": {"panel_mode": "project-focus"},
    }
