from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pytest

from npa.clients.project_credentials import storage_env_for_project


pytestmark = pytest.mark.e2e

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _require_live_mode() -> None:
    if _truthy(os.environ.get("NPA_DRY_RUN", "")) or _truthy(os.environ.get("DRY_RUN", "")):
        pytest.skip("live sim-to-real e2e tests require writes; DRY_RUN is enabled")


def test_e2e_data_sync_and_vlm_eval_write_real_s3_artifacts(
    e2e_project: str | None,
    e2e_test_bucket: str,
    s3_helper: Any,
) -> None:
    source_key = "sim-to-real/source/episode-000.json"
    nested_source_key = "sim-to-real/source/nested/frame-000.json"
    s3_helper.client.put_object(
        Bucket=e2e_test_bucket,
        Key=source_key,
        Body=b'{"episode": 0}\n',
        Metadata={"sha256": hashlib.sha256(b'{"episode": 0}\n').hexdigest()},
    )
    s3_helper.client.put_object(
        Bucket=e2e_test_bucket,
        Key=nested_source_key,
        Body=b'{"frame": 0}\n',
    )

    source_uri = f"s3://{e2e_test_bucket}/sim-to-real/source/"
    imported_uri = f"s3://{e2e_test_bucket}/sim-to-real/run/imported/"
    sync_result = _run_npa(
        _workbench_data_command(e2e_project)
        + [
            "sync",
            "--input-path",
            source_uri,
            "--output-path",
            imported_uri,
            "--output",
            "json",
        ],
    )

    assert sync_result.returncode == 0, _format_result(sync_result)
    sync_payload = json.loads(sync_result.stdout)
    assert sync_payload["status"] == "synced"
    assert sync_payload["object_count"] == 2
    assert sorted(s3_helper.list_objects(e2e_test_bucket, "sim-to-real/run/imported/")) == [
        "sim-to-real/run/imported/episode-000.json",
        "sim-to-real/run/imported/nested/frame-000.json",
    ]

    eval_uri = f"s3://{e2e_test_bucket}/sim-to-real/run/vlm-eval/"
    eval_result = _run_npa(
        [
            "workbench",
            "vlm-eval",
            "run",
            "--input-path",
            imported_uri,
            "--output-path",
            eval_uri,
            "--task",
            "sim-to-real",
            "--backend",
            "stub",
            "--score",
            "0.91",
            "--success-threshold",
            "0.8",
            "--output",
            "json",
        ],
        env_overrides=storage_env_for_project(e2e_project),
    )

    assert eval_result.returncode == 0, _format_result(eval_result)
    eval_payload = json.loads(eval_result.stdout)
    assert eval_payload["status"] == "passed"
    assert eval_payload["passed"] is True
    assert eval_payload["written_uri"] == f"{eval_uri}vlm_eval_stub.json"
    eval_object = s3_helper.client.get_object(
        Bucket=e2e_test_bucket,
        Key="sim-to-real/run/vlm-eval/vlm_eval_stub.json",
    )
    written_eval = json.loads(eval_object["Body"].read().decode("utf-8"))
    assert written_eval["backend"] == "stub"
    assert written_eval["score"] == 0.91


def test_e2e_retargeting_and_mjlab_write_real_s3_artifacts(
    e2e_project: str | None,
    e2e_test_bucket: str,
    s3_helper: Any,
    tmp_path: Path,
) -> None:
    source_motion_key = "sonic-locomotion/source-motion/walk.pkl"
    source_motion = tmp_path / "walk.pkl"
    joblib.dump(
        {
            "walk": {
                "root_trans_offset": np.zeros((4, 3), dtype=np.float32),
                "pose_aa": np.zeros((4, 30, 3), dtype=np.float32),
                "dof": np.zeros((4, 29), dtype=np.float32),
                "root_rot": np.zeros((4, 4), dtype=np.float32),
                "fps": 30,
            }
        },
        source_motion,
    )
    s3_helper.client.upload_file(str(source_motion), e2e_test_bucket, source_motion_key)
    checkpoint_key = "sonic-locomotion/training/checkpoint_smoke.json"
    s3_helper.client.put_object(
        Bucket=e2e_test_bucket,
        Key=checkpoint_key,
        Body=b'{"format": "npa_sonic_serverless_smoke_v1", "status": "success"}\n',
    )

    storage_env = storage_env_for_project(e2e_project)
    source_uri = f"s3://{e2e_test_bucket}/sonic-locomotion/source-motion/"
    retargeted_uri = f"s3://{e2e_test_bucket}/sonic-locomotion/retargeted/"
    retarget_result = _run_npa(
        [
            "workbench",
            "sonic",
            "retargeting",
            "run",
            "--input-path",
            source_uri,
            "--output-path",
            retargeted_uri,
            "--source-format",
            "motion-lib",
            "--embodiment",
            "unitree-g1",
            "--frame-rate",
            "50",
            "--max-frames",
            "8",
            "--output",
            "json",
        ],
        env_overrides=storage_env,
    )

    assert retarget_result.returncode == 0, _format_result(retarget_result)
    retarget_payload = json.loads(retarget_result.stdout)
    assert retarget_payload["status"] == "retargeted"
    assert retarget_payload["artifact_kind"] == "robot_motion_lib"
    assert retarget_payload["artifact_uri"] == retargeted_uri
    assert retarget_payload["metadata_written_uri"] == f"{retargeted_uri}retargeting_result.json"
    retarget_object = s3_helper.client.get_object(
        Bucket=e2e_test_bucket,
        Key="sonic-locomotion/retargeted/retargeting_result.json",
    )
    retarget_metadata = json.loads(retarget_object["Body"].read().decode("utf-8"))
    assert retarget_metadata["embodiment"] == "unitree-g1"
    assert retarget_metadata["max_frames"] == 8
    retargeted_motion = s3_helper.client.get_object(
        Bucket=e2e_test_bucket,
        Key="sonic-locomotion/retargeted/walk.pkl",
    )
    assert retargeted_motion["ContentLength"] > 0

    mjlab_uri = f"s3://{e2e_test_bucket}/sonic-locomotion/mjlab/"
    mjlab_result = _run_npa(
        [
            "workbench",
            "mjlab",
            "eval",
            "--input-path",
            retargeted_uri,
            "--checkpoint",
            f"s3://{e2e_test_bucket}/{checkpoint_key}",
            "--output-path",
            mjlab_uri,
            "--suite",
            "locomotion",
            "--embodiment",
            "unitree-g1",
            "--episodes",
            "2",
            "--score",
            "0.9",
            "--success-threshold",
            "0.75",
            "--output",
            "json",
        ],
        env_overrides=storage_env,
    )

    assert mjlab_result.returncode == 0, _format_result(mjlab_result)
    mjlab_payload = json.loads(mjlab_result.stdout)
    assert mjlab_payload["status"] == "passed"
    assert mjlab_payload["backend"] == "mjlab"
    assert mjlab_payload["written_uri"] == f"{mjlab_uri}mjlab_eval.json"
    mjlab_object = s3_helper.client.get_object(
        Bucket=e2e_test_bucket,
        Key="sonic-locomotion/mjlab/mjlab_eval.json",
    )
    written_mjlab = json.loads(mjlab_object["Body"].read().decode("utf-8"))
    assert written_mjlab["passed"] is True
    assert written_mjlab["score"] == 0.9


# `test_e2e_workflow_yamls_cover_sim_to_real_and_sonic_contracts` used to live here.
# It asserted the raw shape of four SkyPilot templates (sim-to-real-loop,
# sonic-locomotion-finetuning, retargeting, mjlab-eval) from inside the *live* tier,
# so the check only ran when NPA_INTEGRATION_E2E was set. Those templates are being
# retired, and the same contract is now enforced OFFLINE, on the surface that
# survives, by tests that run in the default suite:
#
#   * npa/tests/workflows/test_sonic_locomotion_finetuning.py - each spec declares the
#     right toolRef and its resolved argv invokes the real CLI command;
#   * npa/tests/guardrails/test_three_tier_contract.py - CLI <-> SDK <-> spec/toolRef;
#   * npa/tests/guardrails/test_tool_catalog_argv.py - every argv flag exists;
#   * npa/tests/guardrails/test_spec_declared_outputs.py - `outputs:` names the file
#     the tool actually writes.
#
# Net effect: stronger coverage, and it no longer needs credentials to run.


def test_e2e_cli_smoke_surfaces_for_pipeline_tools() -> None:
    commands = [
        ["workbench", "data", "--help"],
        ["workbench", "vlm-eval", "status", "--output", "json"],
        ["workbench", "vlm-eval", "list", "--output", "json"],
        ["workbench", "sonic", "retargeting", "status", "--output", "json"],
        ["workbench", "sonic", "retargeting", "list", "--output", "json"],
        ["workbench", "sonic", "retargeting", "workflow", "--output", "json"],
        ["workbench", "mjlab", "status", "--output", "json"],
        ["workbench", "mjlab", "list", "--output", "json"],
        ["workbench", "mjlab", "workflow", "--output", "json"],
    ]

    for command in commands:
        result = _run_npa(command)
        assert result.returncode == 0, _format_result(result)
        if "--output" in command:
            payload = json.loads(result.stdout)
            assert payload


def _workbench_data_command(project: str | None) -> list[str]:
    command = ["workbench", "data"]
    if project:
        command.extend(["-p", project])
    return command


def _run_npa(
    args: list[str],
    *,
    env_overrides: dict[str, str] | None = None,
    timeout: int = 180,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    repo_src = ROOT / "npa" / "src"
    env["PYTHONPATH"] = str(repo_src) + os.pathsep + env.get("PYTHONPATH", "")
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-c", "from npa.cli.main import app; app()", *args],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _format_result(result: subprocess.CompletedProcess[str]) -> str:
    return (
        f"command failed with rc={result.returncode}\n"
        f"stdout:\n{result.stdout[-4000:]}\n"
        f"stderr:\n{result.stderr[-4000:]}"
    )
