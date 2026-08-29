"""SONIC image resolution must respect the workload, not just the GPU target.

The variants differ in capability, not only in driver provisioning: the only
variant that matches a datacenter-Blackwell target serves MuJoCo evaluation and
cannot fine-tune. Resolving on the GPU alone therefore used to answer a
fine-tune request with the evaluation runtime, which is a wrong image rather
than a missing one.
"""

from __future__ import annotations

import pytest

from npa.deploy.images import (
    container_image_for_tool,
    sonic_image_entry,
    sonic_image_variant_for_gpu,
    sonic_image_variants,
    sonic_variant_workloads,
)
from npa.workbench.sonic.routing import (
    FINETUNE,
    ISAAC_RENDER,
    MUJOCO_EVAL,
    TRAIN,
)


def test_every_active_variant_declares_its_workloads() -> None:
    active = [
        variant
        for variant, entry in sonic_image_variants().items()
        if str(entry.get("status") or "active") == "active"
    ]
    assert active
    for variant in active:
        assert sonic_variant_workloads(variant)


@pytest.mark.parametrize("workload", [FINETUNE, TRAIN])
def test_datacenter_target_does_not_substitute_the_mujoco_variant(workload: str) -> None:
    with pytest.raises(ValueError) as excinfo:
        sonic_image_variant_for_gpu("gpu-b200", workload=workload)

    message = str(excinfo.value)
    assert "sonic-mujoco-runtime-fetch" in message
    assert workload in message
    # The operator is told what to do instead of being handed a different image.
    assert "--image" in message


def test_mujoco_eval_still_resolves_on_a_datacenter_target() -> None:
    assert (
        sonic_image_variant_for_gpu("gpu-b200", workload=MUJOCO_EVAL)
        == "sonic-mujoco-runtime-fetch"
    )


@pytest.mark.parametrize("workload", [FINETUNE, TRAIN, ISAAC_RENDER, MUJOCO_EVAL])
def test_rt_core_target_serves_every_sonic_workload(workload: str) -> None:
    assert (
        sonic_image_variant_for_gpu("gpu-rtx6000", workload=workload)
        == "sonic-k8s-host-mounted"
    )


def test_gpu_only_resolution_is_unchanged_for_callers_without_a_workload() -> None:
    assert sonic_image_variant_for_gpu("gpu-rtx6000") == "sonic-k8s-host-mounted"
    assert sonic_image_variant_for_gpu("gpu-b200") == "sonic-mujoco-runtime-fetch"


def test_explicit_variant_is_still_checked_against_the_workload() -> None:
    entry = sonic_image_entry(
        image_variant="sonic-mujoco-runtime-fetch", workload=MUJOCO_EVAL
    )
    assert entry["name"] == "npa-sonic-mujoco"

    with pytest.raises(ValueError, match="cannot serve workload"):
        sonic_image_entry(
            image_variant="sonic-mujoco-runtime-fetch", workload=FINETUNE
        )


def test_container_image_for_tool_threads_the_workload() -> None:
    ref = container_image_for_tool(
        "sonic",
        registry="registry.example",
        gpu_target="gpu-b200",
        workload=MUJOCO_EVAL,
    )
    assert "npa-sonic-mujoco" in ref

    with pytest.raises(ValueError, match="serves workload"):
        container_image_for_tool(
            "sonic",
            registry="registry.example",
            gpu_target="gpu-b200",
            workload=FINETUNE,
        )


def test_workload_selection_is_sonic_only() -> None:
    with pytest.raises(ValueError, match="only defined for SONIC"):
        container_image_for_tool(
            "lerobot", registry="registry.example", workload=TRAIN
        )
