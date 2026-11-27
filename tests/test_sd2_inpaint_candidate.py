from refworld.runners.sd2_inpaint_candidate import (
    DEFAULT_CONTEXT_RADIUS,
    DEFAULT_GUIDANCE,
    DEFAULT_MAX_SIDE,
    DEFAULT_SEED,
    DEFAULT_STEPS,
    MODEL_REPO,
    _model_size,
)


def test_model_size_preserves_near_square_reference_without_exceeding_512():
    assert _model_size(1199, 1215, 512) == (504, 512)


def test_model_size_does_not_upscale_small_input():
    assert _model_size(320, 240, 512) == (320, 240)


def test_first_candidate_protocol_constants_are_frozen():
    assert MODEL_REPO == "sd2-community/stable-diffusion-2-inpainting"
    assert DEFAULT_SEED == 42
    assert DEFAULT_STEPS == 30
    assert DEFAULT_GUIDANCE == 4.0
    assert DEFAULT_CONTEXT_RADIUS == 16
    assert DEFAULT_MAX_SIDE == 512
