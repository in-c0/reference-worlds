import json
from pathlib import Path

import pytest

from refworld.semantic_handoff import (
    apply_entity_state_patch,
    canonical_sha256,
    semantic_drift_report,
    snapshot_roundtrip,
)


ROOT = Path(__file__).resolve().parents[1]
jsonschema = pytest.importorskip("jsonschema")


def _load(path: str):
    return json.loads((ROOT / path).read_text())


def test_exp006_scaffold_validates_against_both_contracts():
    handoff_schema = _load("schemas/lifeos-handoff.schema.json")
    world_schema = _load("schemas/world-state.schema.json")
    example = _load("examples/exp006_handoff_scaffold.json")

    jsonschema.Draft202012Validator.check_schema(handoff_schema)
    jsonschema.validate(example, handoff_schema)
    jsonschema.validate(example["world_state"], world_schema)

    world_ids = {entity["id"] for entity in example["world_state"]["entities"]}
    mapped_ids = {mapping["entity_id"] for mapping in example["lifeos_mappings"]}
    assert world_ids == mapped_ids
    assert {mapping["role"] for mapping in example["lifeos_mappings"]} == {
        "architectural",
        "project_artifact",
        "system_instrument",
    }
    assert all(mapping["authority"] == "lifeos" for mapping in example["lifeos_mappings"])
    assert example["renderer_binding"]["owns_semantic_truth"] is False


def test_bounded_edit_survives_snapshot_with_zero_unrelated_semantic_drift():
    example = _load("examples/exp006_handoff_scaffold.json")
    before = example["world_state"]

    after = apply_entity_state_patch(
        before,
        entity_id="lifeos.system.console",
        patch={"mode": "active"},
        actor="owner",
    )
    restored = snapshot_roundtrip(after)
    report = semantic_drift_report(
        before,
        restored,
        target_entity_id="lifeos.system.console",
    )

    assert report["stable_id_set"] is True
    assert report["target_changed"] is True
    assert report["collateral_semantic_drift_count"] == 0
    assert report["unrelated_changed_entity_ids"] == []
    assert restored["entities"][2]["state"]["mode"] == "active"
    assert restored["history"][-1]["op"] == "entity.state.patch"
    assert restored["history"][-1]["entity_id"] == "lifeos.system.console"
    assert restored["history"][-1]["payload"]["actor"] == "owner"


def test_snapshot_roundtrip_is_canonical_and_nonmutating():
    example = _load("examples/exp006_handoff_scaffold.json")
    before = example["world_state"]
    before_hash = canonical_sha256(before)
    restored = snapshot_roundtrip(before)

    assert restored == before
    assert canonical_sha256(restored) == before_hash


def test_missing_target_entity_is_rejected():
    example = _load("examples/exp006_handoff_scaffold.json")
    with pytest.raises(KeyError):
        apply_entity_state_patch(
            example["world_state"],
            entity_id="lifeos.missing",
            patch={"mode": "active"},
        )
