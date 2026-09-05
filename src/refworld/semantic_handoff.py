"""Renderer-independent persistence primitives for the EXP-006 LifeOS handoff.

These helpers deliberately operate only on canonical RefWorld world-state dictionaries.
They do not infer renderer state, project status, evidence truth, or permissions.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON-compatible data deterministically for equality/drift checks."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def snapshot_roundtrip(world_state: Mapping[str, Any]) -> dict[str, Any]:
    """Return the state reconstructed from its deterministic JSON snapshot."""
    return json.loads(canonical_json_bytes(world_state).decode("utf-8"))


def _entity_index(world_state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    entities = world_state.get("entities")
    if not isinstance(entities, list):
        raise ValueError("world_state.entities must be a list")
    index: dict[str, dict[str, Any]] = {}
    for entity in entities:
        if not isinstance(entity, dict):
            raise ValueError("world_state entity must be an object")
        entity_id = entity.get("id")
        if not isinstance(entity_id, str) or not entity_id:
            raise ValueError("world_state entity id must be a non-empty string")
        if entity_id in index:
            raise ValueError(f"duplicate world_state entity id: {entity_id}")
        index[entity_id] = entity
    return index


def apply_entity_state_patch(
    world_state: Mapping[str, Any],
    *,
    entity_id: str,
    patch: Mapping[str, Any],
    actor: str = "owner",
) -> dict[str, Any]:
    """Apply one bounded top-level patch to an entity's application state.

    The function does not mutate identity, geometry, epistemic fields, relations, or
    renderer bindings. Every edit is appended to canonical history with a stable
    before/after hash for audit and restore tests.
    """
    if not isinstance(entity_id, str) or not entity_id:
        raise ValueError("entity_id must be a non-empty string")
    if not isinstance(patch, Mapping) or not patch:
        raise ValueError("patch must be a non-empty mapping")

    updated = copy.deepcopy(dict(world_state))
    entities = _entity_index(updated)
    if entity_id not in entities:
        raise KeyError(entity_id)
    target = entities[entity_id]
    state = target.get("state", {})
    if not isinstance(state, dict):
        raise ValueError("target entity state must be an object")

    before_hash = canonical_sha256(state)
    for key, value in patch.items():
        if not isinstance(key, str) or not key:
            raise ValueError("state patch keys must be non-empty strings")
        state[key] = copy.deepcopy(value)
    target["state"] = state
    after_hash = canonical_sha256(state)

    history = updated.setdefault("history", [])
    if not isinstance(history, list):
        raise ValueError("world_state.history must be a list")
    next_seq = 0 if not history else max(int(item["seq"]) for item in history) + 1
    history.append(
        {
            "seq": next_seq,
            "op": "entity.state.patch",
            "entity_id": entity_id,
            "proposal_id": None,
            "payload": {
                "actor": actor,
                "patch": copy.deepcopy(dict(patch)),
                "before_state_sha256": before_hash,
                "after_state_sha256": after_hash,
            },
        }
    )
    return updated


def semantic_drift_report(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    target_entity_id: str,
) -> dict[str, Any]:
    """Report identity and non-target semantic drift for one bounded edit."""
    before_index = _entity_index(before)
    after_index = _entity_index(after)
    before_ids = tuple(sorted(before_index))
    after_ids = tuple(sorted(after_index))
    if target_entity_id not in before_index or target_entity_id not in after_index:
        raise KeyError(target_entity_id)

    changed_unrelated: list[str] = []
    for entity_id in before_ids:
        if entity_id == target_entity_id or entity_id not in after_index:
            continue
        if canonical_sha256(before_index[entity_id]) != canonical_sha256(after_index[entity_id]):
            changed_unrelated.append(entity_id)

    return {
        "stable_id_set": before_ids == after_ids,
        "before_ids": list(before_ids),
        "after_ids": list(after_ids),
        "target_entity_id": target_entity_id,
        "target_changed": canonical_sha256(before_index[target_entity_id])
        != canonical_sha256(after_index[target_entity_id]),
        "unrelated_changed_entity_ids": changed_unrelated,
        "collateral_semantic_drift_count": len(changed_unrelated),
    }
