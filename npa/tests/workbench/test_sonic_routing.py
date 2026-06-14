from __future__ import annotations

import pytest

from npa.workbench.sonic.routing import (
    CPU,
    DATACENTER_HEADLESS,
    RT_CORE,
    UNKNOWN,
    SonicRoutingError,
    classify_gpu_target,
    is_datacenter_headless_target,
    is_rt_core_target,
    validate_gpu_routing,
    validate_render_gpu_target,
)


@pytest.mark.parametrize(
    "gpu_target, expected",
    [
        ("L40S", RT_CORE),
        ("gpu-l40s-a", RT_CORE),
        ("gpu-rtx6000", RT_CORE),
        ("NVIDIA RTX PRO 6000 Blackwell", RT_CORE),
        ("blackwell-sm_120", RT_CORE),
        ("sm_120", RT_CORE),
        ("H100", DATACENTER_HEADLESS),
        ("gpu-h100-sxm", DATACENTER_HEADLESS),
        ("h200", DATACENTER_HEADLESS),
        ("a100", DATACENTER_HEADLESS),
        ("b200", DATACENTER_HEADLESS),
        ("cpu", CPU),
        ("none", CPU),
        ("", UNKNOWN),
        ("some-future-gpu", UNKNOWN),
    ],
)
def test_classify_gpu_target(gpu_target: str, expected: str) -> None:
    assert classify_gpu_target(gpu_target) == expected


def test_rt_core_and_headless_helpers_are_consistent() -> None:
    assert is_rt_core_target("l40s") is True
    assert is_rt_core_target("h100") is False
    assert is_datacenter_headless_target("h100") is True
    assert is_datacenter_headless_target("l40s") is False


@pytest.mark.parametrize("gpu_target", ["l40s", "gpu-rtx6000", "blackwell sm_120"])
def test_validate_render_accepts_rt_core(gpu_target: str) -> None:
    assert validate_render_gpu_target(gpu_target)


def test_validate_render_allows_empty_for_default_fallback() -> None:
    assert validate_render_gpu_target("") == ""


@pytest.mark.parametrize("gpu_target", ["h100", "gpu-h200-sxm", "a100"])
def test_validate_render_rejects_datacenter_headless(gpu_target: str) -> None:
    with pytest.raises(SonicRoutingError) as excinfo:
        validate_render_gpu_target(gpu_target)
    message = str(excinfo.value)
    assert "RT-core" in message
    assert gpu_target in message


def test_validate_render_rejects_unknown_gpu() -> None:
    with pytest.raises(SonicRoutingError):
        validate_render_gpu_target("totally-unknown")


def test_retarget_requires_cpu() -> None:
    assert validate_gpu_routing(workload="retarget", gpu_target="cpu") == CPU
    assert validate_gpu_routing(workload="retarget", gpu_target="") == CPU
    with pytest.raises(SonicRoutingError):
        validate_gpu_routing(workload="retarget", gpu_target="h100")
    with pytest.raises(SonicRoutingError):
        validate_gpu_routing(workload="retarget", gpu_target="l40s")


@pytest.mark.parametrize("workload", ["finetune", "train", "mujoco-eval"])
def test_headless_workloads_accept_datacenter_and_rt_core(workload: str) -> None:
    assert validate_gpu_routing(workload=workload, gpu_target="h100") == (
        DATACENTER_HEADLESS
    )
    assert validate_gpu_routing(workload=workload, gpu_target="l40s") == RT_CORE


@pytest.mark.parametrize("workload", ["finetune", "train", "mujoco-eval"])
def test_headless_workloads_reject_cpu(workload: str) -> None:
    with pytest.raises(SonicRoutingError):
        validate_gpu_routing(workload=workload, gpu_target="cpu")


def test_isaac_render_rejects_h100_via_routing() -> None:
    with pytest.raises(SonicRoutingError):
        validate_gpu_routing(workload="isaac-render", gpu_target="h100")
    assert validate_gpu_routing(workload="isaac-render", gpu_target="l40s") == RT_CORE
    # No explicit target falls back to the RT-core default class.
    assert validate_gpu_routing(workload="isaac-render", gpu_target="") == RT_CORE


def test_unknown_workload_fails_loud() -> None:
    with pytest.raises(SonicRoutingError) as excinfo:
        validate_gpu_routing(workload="teleport", gpu_target="l40s")
    assert "unknown SONIC workload" in str(excinfo.value)
