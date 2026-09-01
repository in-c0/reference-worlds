import json
from pathlib import Path

import numpy as np
import pytest

jsonschema = pytest.importorskip("jsonschema")

from refworld.adapters.base import Camera
from refworld.camera import OPENGL_C2W
from refworld.proposals import ObservationView, RepaintResult, WarpResult, build_view_proposal


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str):
    return json.loads((ROOT / path).read_text())


def _camera():
    return Camera(
        intrinsics=(100.0, 0.0, 1.0, 0.0, 100.0, 1.0, 0.0, 0.0, 1.0),
        extrinsics=(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ),
        convention=OPENGL_C2W,
    )


def test_committed_world_state_example_validates():
    schema = _load("schemas/world-state.schema.json")
    example = _load("examples/sample_manifest.json")
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(example, schema)


def test_observed_entity_without_observation_provenance_is_invalid():
    schema = _load("schemas/world-state.schema.json")
    example = _load("examples/sample_manifest.json")
    broken = json.loads(json.dumps(example))
    broken["entities"][0].pop("provenance")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(broken, schema)


def test_view_proposal_metadata_validates_and_contains_no_arrays():
    schema = _load("schemas/view-proposal.schema.json")
    jsonschema.Draft202012Validator.check_schema(schema)

    camera = _camera()
    observation = ObservationView(
        "obs-1",
        np.zeros((2, 3, 3), dtype=np.uint8),
        camera,
    )
    observed = np.array([[True, True, False], [True, False, False]], dtype=bool)
    warp = WarpResult(
        rgb=np.zeros((2, 3, 3), dtype=np.uint8),
        observed_mask=observed,
        confidence=np.where(observed, 1.0, 0.0).astype(np.float32),
        backend="synthetic-warp@1",
        metadata={},
    )
    repaint = RepaintResult(
        rgb=np.full((2, 3, 3), 9, dtype=np.uint8),
        valid_mask=np.ones((2, 3), dtype=bool),
        backend="synthetic-repaint@1",
        seed=42,
        metadata={},
    )
    proposal = build_view_proposal([observation], camera, warp, repaint)
    metadata = proposal.metadata_dict()

    jsonschema.validate(metadata, schema)
    assert "image" not in metadata
    assert "provenance" not in metadata
