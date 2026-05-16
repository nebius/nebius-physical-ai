from __future__ import annotations

import json

import pytest

from .conftest import (
    assert_s3_has_objects,
    cleanup_workbench,
    deploy_byovm_args,
    npa_args,
)


pytestmark = pytest.mark.multi_gpu


def test_cosmos_byovm_tensor_parallel_serving(
    byovm_target,
    run_npa,
    s3_prefix,
    unique_name,
) -> None:
    requested_gpus = 2
    if byovm_target.gpu_count < requested_gpus:
        pytest.skip("Cosmos tensor-parallel serving test requires at least two GPUs")

    name = f"cosmos-{unique_name}"
    output_uri = f"{s3_prefix}cosmos/output.json"
    try:
        run_npa(deploy_byovm_args("cosmos", byovm_target, name, requested_gpus), timeout=3600)

        status = run_npa([*npa_args("cosmos", byovm_target, name), "status", "--output", "json"], timeout=120)
        status_data = json.loads(status.stdout)
        assert status_data.get("server") == "up"

        result = run_npa(
            [
                *npa_args("cosmos", byovm_target, name),
                "infer",
                "--prompt",
                "a small robot arm sorting colored cubes on a table",
                "--output-path",
                output_uri,
                "--timeout",
                "1800",
                "--output-format",
                "json",
            ],
            timeout=2400,
        )
        data = json.loads(result.stdout)
        assert data.get("job_id")
        assert data.get("status") == "completed"
        assert data.get("saved_to") == output_uri or data.get("downloaded_to") == output_uri
        assert_s3_has_objects(output_uri)
    finally:
        cleanup_workbench(run_npa, "cosmos", byovm_target, name)
