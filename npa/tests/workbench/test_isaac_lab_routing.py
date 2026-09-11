from __future__ import annotations

import pytest

from npa.workbench.isaac_lab.routing import (
    HEADLESS_TRAIN,
    RENDER,
    IsaacLabRoutingError,
    classify_workload,
    task_requires_rt_cores,
    validate_gpu_routing,
    validate_render_gpu_target,
)


DATACENTER_TARGETS = (
    "gpu-h100-sxm",
    "gpu-h200-sxm",
    "gpu-b200-sxm",
    "gpu-b200-sxm-a",
    "gpu-b300-sxm",
)
RT_CORE_TARGETS = ("gpu-l40s-a", "gpu-l40s-d", "gpu-rtx6000", "gpu-rtx-pro-6000")


@pytest.mark.parametrize(
    "task",
    [
        "Isaac-Cartpole-RGB-Camera-Direct-v0",
        "Isaac-Cartpole-Depth-Camera-Direct-v0",
        "Isaac-Franka-Visual-Lift-v0",
        "Isaac-Repose-Cube-Tiled-v0",
        "isaac_cartpole_rgb_camera_direct_v0",
    ],
)
def test_camera_tasks_are_render_workloads(task: str) -> None:
    assert task_requires_rt_cores(task) is True
    assert classify_workload(task=task) == RENDER


@pytest.mark.parametrize(
    "task",
    [
        "Isaac-Reach-Franka-v0",
        "Isaac-Ant-v0",
        "Isaac-Velocity-Rough-G1-v0",
        "",
    ],
)
def test_state_based_tasks_are_headless_workloads(task: str) -> None:
    assert task_requires_rt_cores(task) is False
    assert classify_workload(task=task) == HEADLESS_TRAIN


def test_explicit_render_flags_override_a_state_based_task_name() -> None:
    assert classify_workload(task="Isaac-Ant-v0", capture_video=True) == RENDER
    assert classify_workload(task="Isaac-Ant-v0", enable_cameras=True) == RENDER


@pytest.mark.parametrize("target", DATACENTER_TARGETS)
def test_headless_training_runs_on_datacenter_parts(target: str) -> None:
    """The bug this fixes: nothing in state-based RL needs RT cores."""

    assert (
        validate_gpu_routing(workload=HEADLESS_TRAIN, gpu_target=target)
        == "datacenter-headless"
    )


@pytest.mark.parametrize("target", RT_CORE_TARGETS)
def test_headless_training_also_runs_on_rt_core_parts(target: str) -> None:
    assert validate_gpu_routing(workload=HEADLESS_TRAIN, gpu_target=target) == "rt-core"


@pytest.mark.parametrize("target", DATACENTER_TARGETS)
def test_render_is_rejected_on_datacenter_parts(target: str) -> None:
    with pytest.raises(IsaacLabRoutingError, match="no RT"):
        validate_gpu_routing(workload=RENDER, gpu_target=target)


@pytest.mark.parametrize("target", RT_CORE_TARGETS)
def test_render_is_allowed_on_rt_core_parts(target: str) -> None:
    assert validate_gpu_routing(workload=RENDER, gpu_target=target) == "rt-core"


def test_render_rejection_names_the_camera_task() -> None:
    with pytest.raises(IsaacLabRoutingError, match="camera or rendered observations"):
        validate_gpu_routing(
            workload=RENDER,
            gpu_target="gpu-b200-sxm",
            task="Isaac-Cartpole-RGB-Camera-Direct-v0",
        )


def test_headless_training_rejects_a_cpu_target() -> None:
    with pytest.raises(IsaacLabRoutingError, match="requires a GPU"):
        validate_gpu_routing(workload=HEADLESS_TRAIN, gpu_target="cpu")


def test_unknown_target_defers_to_the_caller_default() -> None:
    assert validate_gpu_routing(workload=HEADLESS_TRAIN, gpu_target="") == "unknown"
    assert validate_render_gpu_target("") == ""


def test_unknown_workload_fails_loud() -> None:
    with pytest.raises(IsaacLabRoutingError, match="unknown Isaac Lab workload"):
        validate_gpu_routing(workload="finetune", gpu_target="gpu-l40s-a")


def test_datacenter_blackwell_is_not_read_as_rt_core_from_the_family_name() -> None:
    """"Blackwell" spans both classes; a datacenter model number must win."""

    with pytest.raises(IsaacLabRoutingError):
        validate_gpu_routing(workload=RENDER, gpu_target="blackwell-b300")
    assert validate_gpu_routing(workload=RENDER, gpu_target="rtx-pro-6000-blackwell") == "rt-core"
