"""Pinned SkyPilot Kubernetes shape semantics at the capacity boundary."""

from decimal import Decimal, localcontext

import pytest

from npa.orchestration.skypilot.resource_quantities import kubernetes_gpu_quantities


@pytest.mark.parametrize("resources,accelerator,expected", [
    ({"cpus": "8+", "memory": "32+"}, "B200:1", ("8000m", "32000000000")),
    ({"cpus": 8, "memory": 32}, "B200:1", ("8000m", "32000000000")),
    ({"cpus": "8+", "memory": "32768Mi+"}, "B200:1", ("8000m", "32000000000")),
    ({"cpus": "8+", "memory": "32gB+"}, "B200:1", ("8000m", "32000000000")),
    ({"cpus": ".5", "memory": ".5"}, "B200:1", ("500m", "500000000")),
    ({"cpus": "8.06+", "memory": "32.06+"}, "B200:1", ("8100m", "32100000000")),
    ({"cpus": "8.04+", "memory": "32.04+"}, "B200:1", ("8040m", "32040000000")),
    ({"cpus": "8.00001", "memory": "32.00000000001"}, "B200:1", ("8001m", "32000000001")),
    ({}, "B200:1", ("4000m", "16000000000")),
    ({}, "B200:2", ("8000m", "32000000000")),
    ({"cpus": "8+"}, "B200:1", ("8000m", "32000000000")),
    ({"cpus": "8.06+"}, "B200:1", ("8100m", "32400000000")),
    ({"memory": "32+"}, "B200:2", ("8000m", "32000000000")),
])
def test_actual_backend_units_defaults_and_rounding(resources, accelerator, expected):
    assert kubernetes_gpu_quantities(resources, accelerator=accelerator) == expected


@pytest.mark.parametrize("field", ["cpus", "memory"])
@pytest.mark.parametrize("value", [
    True, False, 0, -1, "-1+", "NaN", "Infinity", "1e9999", "8++", "8+1", [], {},
])
def test_invalid_declared_shapes_are_not_replaced_with_zero(field, value):
    with pytest.raises(ValueError):
        kubernetes_gpu_quantities({field: value}, accelerator="B200:1")


@pytest.mark.parametrize("resources", [
    {"memory": "4x"}, {"memory": "4x+"},
    {"instance_type": "8CPU--32GB--1B200"},
])
def test_unproven_gpu_shapes_fail_before_a_capacity_claim(resources):
    with pytest.raises(ValueError):
        kubernetes_gpu_quantities(resources, accelerator="B200:1")


@pytest.mark.parametrize("resources,expected", [
    ({"cpus": "8.000000000000000000000000000001+", "memory": "32+"}, ("8001m", "32000000000")),
    ({"cpus": "8+", "memory": "32.000000000000000000000000000001+"}, ("8000m", "32000000001")),
    ({"cpus": "8+", "memory": "32768.00000000000000000000000001Mi+"}, ("8000m", "32000000001")),
    ({"cpus": "8+", "memory": ".000001024000000000000000000000000001Mi+"}, ("8000m", "2")),
    ({"cpus": "8.000000000000000000000000000001+"}, ("8001m", "32000000001")),
])
def test_positive_fraction_beyond_decimal_precision_never_understates_capacity(resources, expected):
    assert kubernetes_gpu_quantities(resources, accelerator="B200:1") == expected


def test_capacity_ceiling_is_independent_of_ambient_decimal_precision():
    with localcontext() as context:
        context.prec = 6
        assert kubernetes_gpu_quantities(
            {"cpus": "8.00001", "memory": "32.00000000001"}, accelerator="B200:1",
        ) == ("8001m", "32000000001")


@pytest.mark.parametrize("resources", [
    {"cpus": "1e999999999"}, {"memory": "1e999999999"},
    {"memory": "1e999999999Ki"},
])
def test_extreme_overflow_rejected_before_expanding_decimal_fraction(monkeypatch, resources):
    from npa.orchestration.skypilot import resource_quantities

    original = resource_quantities.Fraction

    def bounded_fraction(value, *args):
        assert not isinstance(value, Decimal), "extreme exponent reached rational allocation"
        return original(value, *args)

    monkeypatch.setattr(resource_quantities, "Fraction", bounded_fraction)
    with pytest.raises(ValueError, match="positive finite"):
        kubernetes_gpu_quantities(resources, accelerator="B200:1")


@pytest.mark.parametrize("resources,expected", [
    ({"cpus": "1e-999999999"}, ("1m", "1")),
    ({"memory": "1e-999999999"}, ("4000m", "1")),
    ({"memory": "1e-999999999Pi"}, ("4000m", "1")),
])
def test_extreme_positive_underflow_preserves_minimum_without_expanding_fraction(monkeypatch, resources, expected):
    from npa.orchestration.skypilot import resource_quantities

    original = resource_quantities.Fraction

    def bounded_fraction(value, *args):
        assert not isinstance(value, Decimal), "extreme exponent reached rational allocation"
        return original(value, *args)

    monkeypatch.setattr(resource_quantities, "Fraction", bounded_fraction)
    assert kubernetes_gpu_quantities(resources, accelerator="B200:1") == expected


def test_large_raw_memory_can_normalize_to_a_finite_shape():
    cpu, memory = kubernetes_gpu_quantities({"cpus": 1, "memory": "1e310Ki"}, accelerator="B200:1")
    assert cpu == "1000m"
    assert int(memory) >= (10**310 // 1024**2) * 10**9


def test_small_positive_shape_without_spy_retains_capacity_minimum():
    assert kubernetes_gpu_quantities({"cpus": "1e-1000"}, accelerator="B200:1") == ("1m", "1")


def test_small_raw_memory_above_a_byte_after_suffix_keeps_exact_ceiling():
    assert kubernetes_gpu_quantities(
        {"cpus": 1, "memory": "9.53674316406250000000000000001e-16Pi"}, accelerator="B200:1",
    ) == ("1000m", "2")
