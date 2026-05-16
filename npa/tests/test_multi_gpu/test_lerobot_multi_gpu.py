from __future__ import annotations

import pytest

from .conftest import (
    assert_s3_has_objects,
    assert_visible_gpus_used,
    cleanup_workbench,
    deploy_byovm_args,
    npa_args,
    parse_loss_values,
    run_with_gpu_poll,
)


pytestmark = pytest.mark.multi_gpu


@pytest.mark.parametrize("requested_gpus", [2, 4])
def test_lerobot_byovm_multi_gpu_training(
    byovm_target,
    npa_base_env,
    run_npa,
    s3_prefix,
    unique_name,
    requested_gpus: int,
) -> None:
    if byovm_target.gpu_count < requested_gpus:
        pytest.skip(f"target only has {byovm_target.gpu_count} GPU(s)")

    name = f"lerobot-{requested_gpus}-{unique_name}"
    job_name = f"act-{requested_gpus}gpu-{unique_name}"
    output_uri = f"{s3_prefix}lerobot/{requested_gpus}gpu/checkpoint/"
    try:
        run_npa(deploy_byovm_args("lerobot", byovm_target, name, requested_gpus), timeout=1800)
        result = run_with_gpu_poll(
            [
                *npa_args("lerobot", byovm_target, name),
                "train",
                "--policy-type",
                "act",
                "--dataset",
                "lerobot/aloha_sim_transfer_cube_human",
                "--job-name",
                job_name,
                "--steps",
                "50",
                "--batch-size",
                "8",
                "--gpu-count",
                str(requested_gpus),
                "--output-path",
                output_uri,
            ],
            target=byovm_target,
            env=npa_base_env,
            timeout=2400,
        )

        assert result.returncode == 0, result.stdout
        assert "NPA_TRAIN_COMPLETE" in result.stdout
        assert_visible_gpus_used(result.gpu_snapshots, requested_gpus)
        assert_s3_has_objects(output_uri)

        losses = parse_loss_values(result.stdout)
        if len(losses) >= 2:
            assert losses[-1] <= losses[0], f"expected loss to decrease, got first={losses[0]} last={losses[-1]}"
    finally:
        cleanup_workbench(run_npa, "lerobot", byovm_target, name)
