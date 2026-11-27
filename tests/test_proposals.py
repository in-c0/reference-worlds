import json

import numpy as np
import pytest

from refworld.adapters.base import Camera
from refworld.camera import OPENGL_C2W
from refworld.evidence import PixelProvenance
from refworld.proposals import (
    ObservationView,
    RepaintResult,
    WarpResult,
    build_view_proposal,
    build_view_proposal_from_parent_ids,
    hash_array,
)


def _camera(tx=0.0):
    return Camera(
        intrinsics=(120.0, 0.0, 2.0, 0.0, 120.0, 2.0, 0.0, 0.0, 1.0),
        extrinsics=(
            1.0, 0.0, 0.0, tx,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ),
        convention=OPENGL_C2W,
    )


def test_proposal_preserves_warped_observation_and_emits_array_free_metadata():
    source = ObservationView(
        observation_id="obs-1",
        image=np.zeros((3, 4, 3), dtype=np.uint8),
        camera=_camera(),
    )
    warped = np.zeros((3, 4, 3), dtype=np.uint8)
    observed = np.zeros((3, 4), dtype=bool)
    observed[:, :2] = True
    warped[observed] = [12, 34, 56]
    warp = WarpResult(
        rgb=warped,
        observed_mask=observed,
        confidence=np.where(observed, 0.95, 0.0).astype(np.float32),
        backend="synthetic-warp@1",
        metadata={"method": "unit-test"},
    )
    repaint = RepaintResult(
        rgb=np.full_like(warped, 200),
        valid_mask=np.ones((3, 4), dtype=bool),
        backend="synthetic-repaint@1",
        seed=42,
        metadata={"steps": 1},
    )

    proposal = build_view_proposal([source], _camera(0.1), warp, repaint)

    assert np.array_equal(proposal.image[observed], warped[observed])
    assert np.all(proposal.provenance[observed] == PixelProvenance.OBSERVED)
    assert np.all(proposal.provenance[~observed] == PixelProvenance.GENERATED)
    assert proposal.summary["overlap_attempt_pixels"] == int(observed.sum())

    metadata = proposal.metadata_dict()
    encoded = json.dumps(metadata, allow_nan=False)
    assert proposal.proposal_id in encoded
    assert metadata["parent_observation_ids"] == ["obs-1"]
    assert metadata["hashes"]["proposal_image_sha256"] == hash_array(proposal.image)
    assert "array(" not in encoded
    assert "image" not in metadata
    assert "provenance" not in metadata


def test_partial_repaint_remains_unresolved_in_proposal():
    source = ObservationView("obs", np.zeros((2, 3, 3), dtype=np.uint8), _camera())
    observed = np.zeros((2, 3), dtype=bool)
    observed[0, 0] = True
    warp = WarpResult(
        rgb=np.zeros((2, 3, 3), dtype=np.uint8),
        observed_mask=observed,
        confidence=np.where(observed, 1.0, 0.0).astype(np.float32),
        backend="warp",
        metadata={},
    )
    valid = np.zeros((2, 3), dtype=bool)
    valid[0, 1] = True
    repaint = RepaintResult(
        rgb=np.full((2, 3, 3), 8, dtype=np.uint8),
        valid_mask=valid,
        backend="repaint",
        seed=0,
        metadata={},
    )

    proposal = build_view_proposal([source], _camera(), warp, repaint)
    assert proposal.summary["observed_pixels"] == 1
    assert proposal.summary["generated_pixels"] == 1
    assert proposal.summary["unresolved_pixels"] == 4


def test_persisted_parent_ids_build_same_proposal_without_dummy_observation_image():
    observed = np.zeros((2, 3), dtype=bool)
    observed[:, 0] = True
    warp_rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    warp_rgb[observed] = [10, 20, 30]
    warp = WarpResult(
        warp_rgb,
        observed,
        np.where(observed, 1.0, 0.0).astype(np.float32),
        "warp",
        {"persisted": True},
    )
    repaint = RepaintResult(
        np.full_like(warp_rgb, 200),
        np.ones((2, 3), dtype=bool),
        "external-generator@1",
        7,
        {"run_id": "gpu-run-1"},
    )

    proposal = build_view_proposal_from_parent_ids(
        ["obs-persisted"], _camera(0.1), warp, repaint
    )

    assert proposal.parent_observation_ids == ("obs-persisted",)
    assert np.array_equal(proposal.image[observed], warp_rgb[observed])
    assert proposal.summary["overlap_attempt_pixels"] == int(observed.sum())
    assert proposal.repaint_backend == "external-generator@1"


def test_hash_array_includes_shape_and_dtype_semantics():
    bytes_same_a = np.array([1, 2, 3, 4], dtype=np.uint8)
    bytes_same_b = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    assert bytes_same_a.tobytes() == bytes_same_b.tobytes()
    assert hash_array(bytes_same_a) != hash_array(bytes_same_b)

    values = np.array([1, 2], dtype=np.uint16)
    as_bytes = np.frombuffer(values.tobytes(), dtype=np.uint8)
    assert values.tobytes() == as_bytes.tobytes()
    assert hash_array(values) != hash_array(as_bytes)


def test_warp_confidence_is_strictly_bounded_and_float():
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    mask = np.zeros((2, 2), dtype=bool)
    with pytest.raises(ValueError, match="floating"):
        WarpResult(rgb, mask, np.zeros((2, 2), dtype=np.uint8), "warp", {})
    with pytest.raises(ValueError, match="\[0,1\]"):
        WarpResult(rgb, mask, np.full((2, 2), 1.2, dtype=np.float32), "warp", {})


def test_duplicate_parent_observation_ids_are_rejected():
    obs_a = ObservationView("same", np.zeros((2, 2, 3), dtype=np.uint8), _camera())
    obs_b = ObservationView("same", np.zeros((2, 2, 3), dtype=np.uint8), _camera())
    warp = WarpResult(
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.zeros((2, 2), dtype=bool),
        np.zeros((2, 2), dtype=np.float32),
        "warp",
        {},
    )
    repaint = RepaintResult(
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.zeros((2, 2), dtype=bool),
        "repaint",
        0,
        {},
    )
    with pytest.raises(ValueError, match="unique"):
        build_view_proposal([obs_a, obs_b], _camera(), warp, repaint)


def test_persisted_parent_constructor_rejects_empty_ids():
    warp = WarpResult(
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.zeros((2, 2), dtype=bool),
        np.zeros((2, 2), dtype=np.float32),
        "warp",
        {},
    )
    repaint = RepaintResult(
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.zeros((2, 2), dtype=bool),
        "repaint",
        None,
        {},
    )
    with pytest.raises(ValueError, match="non-empty"):
        build_view_proposal_from_parent_ids([""], _camera(), warp, repaint)
