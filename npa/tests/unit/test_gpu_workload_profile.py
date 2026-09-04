"""Explicit RTX rendering workload-profile contract."""

from __future__ import annotations

import pytest

from npa.cluster.gpu_workload_profile import (
    RTX_RENDERING_PLATFORM,
    RTX_RENDERING_PRESET,
    resolve_gpu_workload_profile,
)
from npa.cluster_backends.mk8s import desired_state
from npa.fleet.spec import ClusterSpec, NodePoolSpec


def test_unprofiled_selection_preserves_existing_defaults() -> None:
    selection = resolve_gpu_workload_profile(
        profile="",
        gpu_nodes=-1,
        gpu_platform="",
        gpu_preset="",
        gpu_driver_mode="",
    )
    assert selection.gpu_nodes == -1
    assert selection.gpu_driver_mode == ""
    assert selection.graphics_smoke is False


def test_rtx_rendering_profile_selects_complete_driver_and_health_contract() -> None:
    cluster = ClusterSpec(name="render", gpu_workload_profile="rtx-rendering")
    cluster.validate()
    plan = desired_state(cluster)

    assert plan["gpu_nodes"] == 1
    assert plan["gpu_platform"] == RTX_RENDERING_PLATFORM
    assert plan["gpu_preset"] == RTX_RENDERING_PRESET
    assert plan["gpu_driver_mode"] == "operator"
    assert plan["gpu_workload_profile"] == "rtx-rendering"
    assert plan["gpu_graphics_smoke"] is True


def test_rtx_rendering_profile_rejects_conflicting_platform() -> None:
    with pytest.raises(ValueError, match="requires platform"):
        ClusterSpec(
            name="render",
            gpu_workload_profile="rtx-rendering",
            gpu_nodes=NodePoolSpec(
                count=1,
                platform="gpu-h200-sxm",
                preset="8gpu-128vcpu-1600gb",
            ),
        )


def test_rtx_rendering_profile_does_not_bypass_nvswitch_operator_rejection() -> None:
    with pytest.raises(ValueError, match="requires platform"):
        ClusterSpec(
            name="render",
            gpu_workload_profile="rtx-rendering",
            gpu_nodes=NodePoolSpec(
                count=1,
                platform="gpu-b200-sxm",
                preset="8gpu-160vcpu-1792gb",
            ),
            allow_unsafe_nvswitch_operator=True,
        )
