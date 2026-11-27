import pytest

from refworld.datasets.mvsnet import PairRecord
from refworld.runners.blendedmvs_oracle_pair import _select_held_out


def test_select_held_out_preserves_published_1_based_rank():
    record = PairRecord(
        reference_id=0,
        source_ids=(136, 158, 20),
        scores=(8.0, 7.0, 6.0),
    )

    assert _select_held_out(record, 1) == (136, 8.0)
    assert _select_held_out(record, 2) == (158, 7.0)
    assert _select_held_out(record, 3) == (20, 6.0)


def test_select_held_out_rejects_out_of_range_rank():
    record = PairRecord(reference_id=0, source_ids=(136,), scores=(8.0,))

    with pytest.raises(ValueError, match=r"\[1,1\]"):
        _select_held_out(record, 0)
    with pytest.raises(ValueError, match=r"\[1,1\]"):
        _select_held_out(record, 2)
