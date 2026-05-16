from __future__ import annotations

import pytest

from .conftest import (
    assert_s3_has_objects,
    cleanup_workbench,
    deploy_byovm_args,
    npa_args,
    query_gpu_names,
)


pytestmark = pytest.mark.multi_gpu


def test_isaac_lab_byovm_multi_gpu_training_distributes_envs(
    byovm_target,
    run_npa,
    s3_prefix,
    unique_name,
) -> None:
    gpu_names = query_gpu_names(byovm_target)
    supported = any(
        "L40S" in name.upper() or "RTX PRO 6000" in name.upper()
        for name in gpu_names
    )
    if not supported:
        pytest.skip("Isaac Lab BYOVM multi-GPU test requires L40S or RTX PRO 6000 GPUs")

    requested_gpus = min(2, byovm_target.gpu_count)
    if requested_gpus < 2:
        pytest.skip("Isaac Lab multi-GPU training test requires at least two GPUs")

    name = f"isaac-lab-{unique_name}"
    output_uri = f"{s3_prefix}isaac-lab/train/"
    try:
        run_npa(deploy_byovm_args("isaac-lab", byovm_target, name, requested_gpus), timeout=3600)

        result = run_npa(
            [
                *npa_args("isaac-lab", byovm_target, name),
                "train",
                "--task",
                "Isaac-Reach-Franka-v0",
                "--num-envs",
                "128",
                "--steps",
                "50",
                "--output-path",
                output_uri,
            ],
            timeout=2400,
        )
        assert "ISAAC_LAB_TRAIN_COMPLETE" in result.stdout
        assert "ISAAC_LAB_MULTI_GPU_DEVICES" in result.stdout
        assert "cuda:0" in result.stdout
        assert "cuda:1" in result.stdout
        assert_s3_has_objects(output_uri)
    finally:
        cleanup_workbench(run_npa, "isaac-lab", byovm_target, name)
