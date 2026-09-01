import numpy as np
import pytest

from refworld.evidence import PixelProvenance
from refworld.objectives import evidence_weight_map, weighted_l1


def test_evidence_weights_prioritize_observed_and_ignore_unresolved():
    provenance = np.array(
        [
            [PixelProvenance.OBSERVED, PixelProvenance.GENERATED],
            [PixelProvenance.UNRESOLVED, PixelProvenance.OBSERVED],
        ],
        dtype=np.uint8,
    )
    confidence = np.array([[1.0, 0.0], [0.0, 0.5]], dtype=np.float32)
    weights = evidence_weight_map(
        provenance,
        observed_confidence=confidence,
        observed_weight=1.0,
        generated_weight=0.1,
    )
    assert weights[0, 0] == pytest.approx(1.0)
    assert weights[0, 1] == pytest.approx(0.1)
    assert weights[1, 0] == pytest.approx(0.0)
    assert weights[1, 1] == pytest.approx(0.5)


def test_confidence_outside_observed_support_cannot_create_evidence_weight():
    provenance = np.array(
        [[PixelProvenance.GENERATED, PixelProvenance.UNRESOLVED]], dtype=np.uint8
    )
    confidence = np.ones((1, 2), dtype=np.float32)
    weights = evidence_weight_map(
        provenance,
        observed_confidence=confidence,
        generated_weight=0.2,
    )
    assert np.array_equal(weights, [[0.2, 0.0]])


def test_unresolved_weight_cannot_be_enabled_and_generated_cannot_dominate_observed():
    provenance = np.zeros((1, 1), dtype=np.uint8)
    with pytest.raises(ValueError, match="unresolved_weight must remain zero"):
        evidence_weight_map(provenance, unresolved_weight=0.01)
    with pytest.raises(ValueError, match="generated_weight cannot exceed"):
        evidence_weight_map(provenance, observed_weight=0.5, generated_weight=0.6)


def test_weighted_l1_ignores_unresolved_error_and_downweights_generated_error():
    reference = np.zeros((1, 3, 3), dtype=np.float64)
    prediction = np.array(
        [[[1.0, 1.0, 1.0], [10.0, 10.0, 10.0], [100.0, 100.0, 100.0]]]
    )
    provenance = np.array(
        [[PixelProvenance.OBSERVED, PixelProvenance.GENERATED, PixelProvenance.UNRESOLVED]],
        dtype=np.uint8,
    )
    weights = evidence_weight_map(provenance, generated_weight=0.1)
    # (1*1 + 10*0.1) / (1 + 0.1) = 2 / 1.1
    assert weighted_l1(reference, prediction, weights) == pytest.approx(2.0 / 1.1)


def test_weighted_l1_rejects_empty_support():
    reference = np.zeros((2, 2, 3))
    with pytest.raises(ValueError, match="no supported pixels"):
        weighted_l1(reference, reference, np.zeros((2, 2)))
