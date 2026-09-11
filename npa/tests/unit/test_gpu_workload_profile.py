"""Explicit RTX rendering workload-profile contract."""

from __future__ import annotations

import pytest

from npa.cluster.gpu_workload_profile import (
    RTX_RENDERING_8GPU_PRESET,
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


def test_rtx_rendering_profile_accepts_eight_gpu_rtx_nodes() -> None:
    cluster = ClusterSpec(
        name="render",
        gpu_workload_profile="rtx-rendering",
        gpu_nodes=NodePoolSpec(
            count=3,
            platform=RTX_RENDERING_PLATFORM,
            preset=RTX_RENDERING_8GPU_PRESET,
        ),
    )

    cluster.validate()
    plan = desired_state(cluster)

    assert plan["gpu_nodes"] == 3
    assert plan["gpu_preset"] == RTX_RENDERING_8GPU_PRESET
    assert plan["gpu_driver_mode"] == "operator"
    assert plan["gpu_graphics_smoke"] is True
    assert plan["enable_gpu_cluster"] is False


def test_rtx_rendering_profile_preserves_exact_zonal_rtx_platform() -> None:
    cluster = ClusterSpec(
        name="render",
        gpu_workload_profile="rtx-rendering",
        gpu_nodes=NodePoolSpec(
            count=3,
            platform="gpu-rtx6000-a",
            preset=RTX_RENDERING_8GPU_PRESET,
        ),
    )

    cluster.validate()
    plan = desired_state(cluster)

    assert plan["gpu_platform"] == "gpu-rtx6000-a"
    assert plan["gpu_driver_mode"] == "operator"
    assert plan["gpu_graphics_smoke"] is True


def test_rtx_rendering_profile_rejects_unknown_rtx_platform_suffix() -> None:
    with pytest.raises(ValueError, match="requires platform"):
        ClusterSpec(
            name="render",
            gpu_workload_profile="rtx-rendering",
            gpu_nodes=NodePoolSpec(
                count=1,
                platform="gpu-rtx6000-preview",
                preset=RTX_RENDERING_PRESET,
            ),
        )


def test_rtx_rendering_profile_rejects_non_rtx_preset() -> None:
    with pytest.raises(ValueError, match="requires an RTX PRO 6000 preset"):
        ClusterSpec(
            name="render",
            gpu_workload_profile="rtx-rendering",
            gpu_nodes=NodePoolSpec(
                count=1,
                platform=RTX_RENDERING_PLATFORM,
                preset="8gpu-96vcpu-872gb",
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
