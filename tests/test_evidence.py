import numpy as np
import pytest

from refworld.adapters.base import Camera
from refworld.camera import OPENGL_C2W, yaw
from refworld.evidence import (
    PixelProvenance,
    compose_evidence_preserving_view,
    deterministic_proposal_id,
)


def _camera() -> Camera:
    return Camera(
        intrinsics=(100.0, 0.0, 2.0, 0.0, 100.0, 2.0, 0.0, 0.0, 1.0),
        extrinsics=(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ),
        convention=OPENGL_C2W,
    )


def test_full_frame_generator_cannot_overwrite_observed_pixels():
    warped = np.zeros((4, 5, 3), dtype=np.uint8)
    warped[1:3, 1:4] = [10, 20, 30]
    observed = np.zeros((4, 5), dtype=bool)
    observed[1:3, 1:4] = True
    generated = np.full((4, 5, 3), 255, dtype=np.uint8)

    result = compose_evidence_preserving_view(
        warped,
        observed,
        generated_rgb=generated,
    )

    assert np.array_equal(result.image[observed], warped[observed])
    assert np.all(result.image[~observed] == 255)
    assert np.all(result.provenance[observed] == PixelProvenance.OBSERVED)
    assert np.all(result.provenance[~observed] == PixelProvenance.GENERATED)
    assert result.summary.overlap_attempt_pixels == int(observed.sum())
    assert result.summary.unresolved_pixels == 0


def test_partial_generation_keeps_missing_support_unresolved():
    warped = np.zeros((3, 4, 3), dtype=np.uint8)
    observed = np.zeros((3, 4), dtype=bool)
    observed[0, 0] = True
    warped[0, 0] = [1, 2, 3]

    generated = np.full_like(warped, 99)
    generated_mask = np.zeros((3, 4), dtype=bool)
    generated_mask[0, 0] = True  # overlap attempt: must not replace evidence
    generated_mask[1, 1] = True
    generated_mask[2, 2] = True

    result = compose_evidence_preserving_view(
        warped,
        observed,
        generated_rgb=generated,
        generated_mask=generated_mask,
        unresolved_value=7,
    )

    assert np.array_equal(result.image[0, 0], [1, 2, 3])
    assert np.array_equal(result.image[1, 1], [99, 99, 99])
    assert np.array_equal(result.image[2, 2], [99, 99, 99])
    assert result.provenance[0, 0] == PixelProvenance.OBSERVED
    assert result.provenance[1, 1] == PixelProvenance.GENERATED
    assert result.provenance[0, 1] == PixelProvenance.UNRESOLVED
    assert np.array_equal(result.image[0, 1], [7, 7, 7])
    assert result.summary.overlap_attempt_pixels == 1
    assert result.summary.generated_pixels == 2
    assert result.summary.unresolved_pixels == 9


def test_without_generator_non_observed_support_stays_unresolved():
    warped = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    observed = np.array([[True, False, False], [False, True, False]], dtype=bool)

    result = compose_evidence_preserving_view(warped, observed)

    assert np.array_equal(result.image[observed], warped[observed])
    assert np.all(result.image[~observed] == 0)
    assert result.summary.observed_pixels == 2
    assert result.summary.generated_pixels == 0
    assert result.summary.unresolved_pixels == 4


def test_compositor_rejects_ambiguous_masks_and_dtype_changes():
    warped = np.zeros((2, 2, 3), dtype=np.uint8)
    observed_int = np.zeros((2, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="boolean"):
        compose_evidence_preserving_view(warped, observed_int)

    observed = np.zeros((2, 2), dtype=bool)
    generated = np.zeros((2, 2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="dtype"):
        compose_evidence_preserving_view(
            warped,
            observed,
            generated_rgb=generated,
        )

    with pytest.raises(ValueError, match="requires generated_rgb"):
        compose_evidence_preserving_view(
            warped,
            observed,
            generated_mask=np.ones((2, 2), dtype=bool),
        )


def test_proposal_id_is_deterministic_and_sensitive_to_lineage_camera_backend_seed():
    camera = _camera()
    kwargs = dict(
        parent_observation_ids=["source-001"],
        target_camera=camera,
        warp_backend="vggt-warp@abc123",
        repaint_backend="wan2.1-worldforge@def456",
        seed=42,
    )
    first = deterministic_proposal_id(**kwargs)
    second = deterministic_proposal_id(**kwargs)
    assert first == second
    assert first.startswith("view-")

    moved = deterministic_proposal_id(
        **{**kwargs, "target_camera": yaw(camera, 2.0)}
    )
    backend_changed = deterministic_proposal_id(
        **{**kwargs, "repaint_backend": "longcat-worldforge@def456"}
    )
    seed_changed = deterministic_proposal_id(**{**kwargs, "seed": 43})

    assert moved != first
    assert backend_changed != first
    assert seed_changed != first
