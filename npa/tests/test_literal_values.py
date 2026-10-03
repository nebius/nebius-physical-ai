"""Scalar evidence rejects coercion while preserving valid literal values."""

import pytest

from npa.literal_values import require_boolean, require_integer, require_number


@pytest.mark.parametrize("value", [None, 0, 1, "true", "false", [], {}])
def test_boolean_rejects_non_literals(value):
    with pytest.raises(ValueError, match="actual_key must be a literal boolean"):
        require_boolean(value, field="actual_key")


@pytest.mark.parametrize("value", [True, False])
def test_boolean_preserves_literal(value):
    assert require_boolean(value, field="flag") is value


@pytest.mark.parametrize("value", [None, True, False, "1", 1.0, 1.5, [], {}])
def test_integer_rejects_coercion(value):
    with pytest.raises(ValueError, match="count must be a literal integer"):
        require_integer(value, field="count")


@pytest.mark.parametrize("value", [None, True, False, "0.01", [], {}])
def test_number_rejects_coercion(value):
    with pytest.raises(ValueError, match="distance must be a literal number"):
        require_number(value, field="distance")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_number_rejects_nonfinite(value):
    with pytest.raises(ValueError, match="distance must be finite"):
        require_number(value, field="distance")


@pytest.mark.parametrize("validator", [require_integer, require_number])
def test_bounds_and_large_integers(validator):
    for value in [0, 5, 10]:
        assert validator(value, field="count", minimum=0, maximum=10) == value
    assert validator(10**1000, field="count") == 10**1000
    with pytest.raises(ValueError, match="count must be at least 0"):
        validator(-1, field="count", minimum=0)
    with pytest.raises(ValueError, match="count must be at most 10"):
        validator(11, field="count", maximum=10)


def test_number_preserves_float():
    value = 0.01
    assert require_number(value, field="distance", minimum=0, maximum=1) is value
