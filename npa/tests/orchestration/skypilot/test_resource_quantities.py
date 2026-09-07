"""Pinned SkyPilot Kubernetes shape semantics at the capacity boundary."""

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
