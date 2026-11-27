from refworld.runners.da3_source import _optional_scalar_int


def test_optional_scalar_int_accepts_real_scalars():
    assert _optional_scalar_int(False) == 0
    assert _optional_scalar_int(True) == 1
    assert _optional_scalar_int(0) == 0
    assert _optional_scalar_int(1) == 1
    assert _optional_scalar_int(1.0) == 1


def test_optional_scalar_int_rejects_missing_or_structured_values():
    assert _optional_scalar_int(None) is None
    assert _optional_scalar_int({}) is None
    assert _optional_scalar_int({"unexpected": 1}) is None
    assert _optional_scalar_int(0.5) is None
