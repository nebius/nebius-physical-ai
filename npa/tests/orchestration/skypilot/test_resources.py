from __future__ import annotations

import pytest

from npa.orchestration.skypilot.resources import (
    DEFAULT_REGION,
    InvalidResourceSpecError,
    resources_for_npa_spec,
)


@pytest.mark.parametrize(
    ("gpu", "accelerator", "instance_type"),
    [
        ("h100", "H100:1", "gpu-h100-sxm_1gpu-16vcpu-200gb"),
        ("h200", "H200:1", "gpu-h200-sxm_1gpu-16vcpu-200gb"),
        ("l40s", "L40S:1", "gpu-l40s-d_1gpu-16vcpu-96gb"),
        ("rtx6000", "RTX6000:1", "gpu-rtx6000_1gpu-24vcpu-218gb"),
    ],
)
def test_resources_for_nebius_gpu_specs(gpu: str, accelerator: str, instance_type: str) -> None:
    resources = resources_for_npa_spec({"backend": "nebius", "gpu": gpu, "count": 1})

    assert resources["cloud"] == "nebius"
    assert resources["region"] == DEFAULT_REGION
    assert resources["instance_type"] == instance_type
    assert resources["accelerators"] == accelerator
    assert resources["autostop"] == {"idle_minutes": 5, "down": False}


def test_cpu_only_spec_returns_nebius_cpu_instance_type() -> None:
    resources = resources_for_npa_spec({"backend": "nebius", "cpus": 2, "memory_gb": 8})

    assert resources["cloud"] == "nebius"
    assert resources["region"] == DEFAULT_REGION
    assert resources["instance_type"] == "cpu-e2_2vcpu-8gb"
    assert "accelerators" not in resources


def test_kubernetes_backend_spec_returns_kubernetes_shape() -> None:
    resources = resources_for_npa_spec(
        {"backend": "kubernetes", "gpu": "h200", "count": 1, "cpus": 4, "memory_gb": 16}
    )

    assert resources == {
        "cloud": "kubernetes",
        "cpus": 4,
        "memory": 16,
        "accelerators": "H200:1",
    }


def test_invalid_gpu_type_raises_typed_error() -> None:
    with pytest.raises(InvalidResourceSpecError):
        resources_for_npa_spec({"gpu": "not-a-gpu"})


def test_unknown_resource_spec_key_raises_typed_error() -> None:
    with pytest.raises(InvalidResourceSpecError, match="unknown_key.*Valid keys"):
        resources_for_npa_spec({"backend": "nebius", "unknown_key": "ignored-before"})


@pytest.mark.parametrize("count", [0, -1])
def test_non_positive_counts_raise_typed_error(count: int) -> None:
    with pytest.raises(InvalidResourceSpecError):
        resources_for_npa_spec({"gpu": "h100", "count": count})


def test_default_region_is_applied() -> None:
    resources = resources_for_npa_spec({"backend": "nebius"})

    assert resources["region"] == DEFAULT_REGION
