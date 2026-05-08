from __future__ import annotations

import statistics

import pytest

from .conftest import (
    assert_s3_has_objects,
    cleanup_workbench,
    deploy_byovm_args,
    npa_args,
    parse_fps_values,
    run_with_gpu_poll,
)


pytestmark = pytest.mark.multi_gpu


def _mean_fps(stdout: str) -> float:
    values = parse_fps_values(stdout)
    assert values, "expected Genesis FPS lines in output"
    return statistics.mean(values[-10:])


def test_genesis_byovm_parallel_simulation_scales(
    byovm_target,
    npa_base_env,
    run_npa,
    s3_prefix,
    unique_name,
) -> None:
    requested_gpus = min(2, byovm_target.gpu_count)
    if requested_gpus < 2:
        pytest.skip("Genesis multi-GPU simulation test requires at least two GPUs")

    name = f"genesis-{unique_name}"
    teacher_dir = f"/tmp/npa-multi-gpu-{unique_name}/teacher"
    checkpoint = f"{teacher_dir}/model.pt"
    single_uri = f"{s3_prefix}genesis/single/"
    multi_uri = f"{s3_prefix}genesis/{requested_gpus}gpu/"

    try:
        run_npa(deploy_byovm_args("genesis", byovm_target, name, 1), timeout=1800)
        train = run_with_gpu_poll(
            [
                *npa_args("genesis", byovm_target, name),
                "train-teacher",
                "--n-envs",
                "64",
                "--max-iterations",
                "1",
                "--output",
                teacher_dir,
                "--action-space",
                "cartesian",
            ],
            target=byovm_target,
            env=npa_base_env,
            timeout=1800,
        )
        assert train.returncode == 0, train.stdout

        single = run_with_gpu_poll(
            [
                *npa_args("genesis", byovm_target, name),
                "simulate",
                "--checkpoint",
                checkpoint,
                "--n-envs",
                "64",
                "--n-episodes",
                "0",
                "--allow-failure-demos",
                "--output-path",
                single_uri,
            ],
            target=byovm_target,
            env=npa_base_env,
            timeout=1800,
        )
        assert single.returncode == 0, single.stdout

        run_npa(deploy_byovm_args("genesis", byovm_target, name, requested_gpus), timeout=1800)
        multi = run_with_gpu_poll(
            [
                *npa_args("genesis", byovm_target, name),
                "simulate",
                "--checkpoint",
                checkpoint,
                "--n-envs",
                str(64 * requested_gpus),
                "--n-episodes",
                "0",
                "--allow-failure-demos",
                "--output-path",
                multi_uri,
            ],
            target=byovm_target,
            env=npa_base_env,
            timeout=2400,
        )
        assert multi.returncode == 0, multi.stdout

        single_fps = _mean_fps(single.stdout)
        multi_fps = _mean_fps(multi.stdout)
        assert multi_fps / single_fps > 1.5, f"expected >1.5x scaling, got {single_fps=:.2f}, {multi_fps=:.2f}"
        assert_s3_has_objects(single_uri)
        assert_s3_has_objects(multi_uri)
    finally:
        cleanup_workbench(run_npa, "genesis", byovm_target, name)
