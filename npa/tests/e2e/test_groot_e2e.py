from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

import boto3
import pytest

from npa.clients.serverless import EndpointNotFoundError, ServerlessClient

from ._serverless_images import (
    resolve_image,
    resolve_serverless_gpu_preset,
    resolve_serverless_gpu_type,
)


PROJECT_ALIAS = "eu-north1"
PROJECT_ID = "project-test-00000000000"
BUCKET = "your-bucket-name"
ENDPOINT_URL = "https://storage.eu-north1.nebius.cloud"
WORKBENCH_NAME = "h200"
GROOT_IMAGE = "cr.eu-north1.nebius.cloud/your-registry-id/npa-groot:0.1.0"
GROOT_MODEL_VARIANT = "nvidia/GR00T-N1.7-3B"
INPUT_PATH = "s3://your-bucket-name/w7all-staging/20260514T161525Z/checkpoint/"
DATASET_PATH = "s3://your-bucket-name/w7all-staging/20260514T161525Z/dataset/"
EMBODIMENT_TAG = "NEW_EMBODIMENT"
INFERENCE_MODE = "pytorch"
GPU_TYPE = "h200"
GPU_PRESET = "1gpu-16vcpu-200gb"
STEPS = 1
ACTION_HORIZON = 1
JOB_PREFIX = "npa-e2e-groot-infer"
POLL_INTERVAL = float(os.environ.get("NPA_E2E_GROOT_POLL_INTERVAL", "30"))
MAX_WAIT = float(os.environ.get("NPA_E2E_GROOT_MAX_WAIT", "7200"))
STARTING_WAIT = float(os.environ.get("NPA_E2E_GROOT_STARTING_WAIT", "3600"))
EXPECTED_MANIFEST_KEYS = {
    "action_horizon",
    "dataset_path",
    "duration_seconds",
    "embodiment_tag",
    "groot_import",
    "inference_mode",
    "input_path",
    "job",
    "model_variant",
    "status",
    "steps",
    "tool",
}


def test_groot_e2e_config_shape() -> None:
    test_id = "shape"
    output_path = _output_path(test_id)
    command = _submit_command(
        project_alias=PROJECT_ALIAS,
        workbench_name=WORKBENCH_NAME,
        project_id=PROJECT_ID,
        output_path=output_path,
        job_name=f"{JOB_PREFIX}-{test_id}",
    )

    assert GROOT_IMAGE == "cr.eu-north1.nebius.cloud/your-registry-id/npa-groot:0.1.0"
    assert GROOT_MODEL_VARIANT == "nvidia/GR00T-N1.7-3B"
    assert WORKBENCH_NAME == "h200"
    assert INPUT_PATH == "s3://your-bucket-name/w7all-staging/20260514T161525Z/checkpoint/"
    assert DATASET_PATH == "s3://your-bucket-name/w7all-staging/20260514T161525Z/dataset/"
    assert "--subnet-id" not in command
    assert command[:7] == ["workbench", "groot", "-p", PROJECT_ALIAS, "-n", WORKBENCH_NAME, "infer"]
    for flag in (
        "--runtime",
        "--project-id",
        "--input-path",
        "--dataset-path",
        "--output-path",
        "--gpu-type",
        "--gpu-count",
        "--gpu-preset",
        "--model-variant",
        "--steps",
        "--action-horizon",
        "--job-name",
        "--timeout",
        "--submit-only",
        "--output",
    ):
        assert flag in command
    assert _expected_artifact_names() == {
        "npa_groot_infer_results.json",
        "predicted_actions.json",
    }


@pytest.mark.e2e_serverless
def test_groot_serverless_infer(tmp_path: Path) -> None:
    _require_groot_e2e()
    test_id = f"w7groot-e2e-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}"
    artifacts_dir = Path("/tmp") / f"groot-e2e-artifacts-{test_id}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    project_alias = os.environ.get("NPA_E2E_PROJECT", PROJECT_ALIAS)
    workbench_name = os.environ.get("NPA_E2E_GROOT_WORKBENCH", WORKBENCH_NAME)
    project_id = os.environ.get("NPA_E2E_SERVERLESS_PROJECT", PROJECT_ID)
    bucket = os.environ.get("NPA_E2E_S3_BUCKET", BUCKET)
    endpoint_url = os.environ.get("NPA_E2E_S3_ENDPOINT", ENDPOINT_URL)
    expected_subnet = os.environ["NPA_E2E_EXPECTED_SUBNET"]
    access_key = os.environ["NPA_E2E_S3_ACCESS_KEY_ID"]
    secret_key = os.environ["NPA_E2E_S3_SECRET_ACCESS_KEY"]
    output_path = _output_path(test_id, bucket=bucket)
    job_name = f"{JOB_PREFIX}-{uuid.uuid4().hex[:8]}"
    command = _submit_command(
        project_alias=project_alias,
        workbench_name=workbench_name,
        project_id=project_id,
        output_path=output_path,
        job_name=job_name,
        image=resolve_image(os.environ.get("NPA_E2E_GROOT_IMAGE", GROOT_IMAGE)),
        gpu_type=resolve_serverless_gpu_type(
            os.environ.get("NPA_E2E_GROOT_GPU_TYPE", GPU_TYPE)
        ),
    )
    job_id = ""

    (artifacts_dir / "output-path.txt").write_text(output_path + "\n", encoding="utf-8")
    (artifacts_dir / "job-name.txt").write_text(job_name + "\n", encoding="utf-8")
    (artifacts_dir / "submit-command.json").write_text(
        json.dumps(command, indent=2) + "\n",
        encoding="utf-8",
    )

    try:
        submitted = _run_npa(command, timeout=int(os.environ.get("NPA_E2E_GROOT_SUBMIT_TIMEOUT", "600")))
        (artifacts_dir / "submit-stdout.txt").write_text(submitted.stdout, encoding="utf-8")
        (artifacts_dir / "submit-stderr.txt").write_text(submitted.stderr, encoding="utf-8")
        assert submitted.returncode == 0, _format_result(submitted)
        payload = json.loads(submitted.stdout)
        assert payload["status"] == "submitted"
        assert payload["job_name"] == job_name
        assert payload["output_path"] == output_path
        job_id = payload["job_id"]
        assert job_id
        (artifacts_dir / "job-id.txt").write_text(job_id + "\n", encoding="utf-8")

        client = ServerlessClient()
        submitted_info = _wait_for_visible_job(client, project_id, job_id)
        _write_job_capture(project_id, submitted_info, artifacts_dir, label="submitted")
        assert expected_subnet in json.dumps(submitted_info.raw, sort_keys=True), (
            f"configured subnet {expected_subnet} missing from submitted Job spec"
        )

        final = _poll_job(client, project_id, job_id, artifacts_dir)
        assert final.status == "succeeded", final.raw
        _write_job_capture(project_id, final, artifacts_dir, label="final")

        local_dir = artifacts_dir / "s3"
        _download_s3_prefix(output_path, local_dir, access_key, secret_key, endpoint_url)
        assert {path.name for path in local_dir.iterdir() if path.is_file()} >= _expected_artifact_names()

        manifest = json.loads((local_dir / "npa_groot_infer_results.json").read_text(encoding="utf-8"))
        predicted = json.loads((local_dir / "predicted_actions.json").read_text(encoding="utf-8"))
        _assert_manifest(manifest, job_name=job_name)
        assert set(predicted) == {"actions", "manifest"}
        assert predicted["actions"] == []
        assert predicted["manifest"] == manifest
    finally:
        if job_id or job_name:
            _cleanup_job(project_id, job_id or job_name, artifacts_dir)


def _require_groot_e2e() -> None:
    if os.environ.get("NPA_INTEGRATION_E2E") != "1":
        pytest.skip("NPA_INTEGRATION_E2E not set")
    for key in (
        "NPA_E2E_SERVERLESS_PROJECT",
        "NPA_E2E_EXPECTED_SUBNET",
        "NPA_E2E_S3_ACCESS_KEY_ID",
        "NPA_E2E_S3_SECRET_ACCESS_KEY",
    ):
        if not os.environ.get(key):
            pytest.skip(f"{key} not set")


def _output_path(test_id: str, *, bucket: str = BUCKET) -> str:
    return f"s3://{bucket}/w7groot-e2e/{test_id}/"


def _submit_command(
    *,
    project_alias: str,
    workbench_name: str,
    project_id: str,
    output_path: str,
    job_name: str,
    image: str = GROOT_IMAGE,
    gpu_type: str = GPU_TYPE,
) -> list[str]:
    return [
        "workbench",
        "groot",
        "-p",
        project_alias,
        "-n",
        workbench_name,
        "infer",
        "--runtime",
        "serverless",
        "--project-id",
        project_id,
        "--input-path",
        INPUT_PATH,
        "--dataset-path",
        DATASET_PATH,
        "--output-path",
        output_path,
        "--image",
        image,
        "--gpu-type",
        gpu_type,
        "--gpu-count",
        "1",
        "--gpu-preset",
        resolve_serverless_gpu_preset(GPU_PRESET, platform=gpu_type),
        "--model-variant",
        GROOT_MODEL_VARIANT,
        "--steps",
        str(STEPS),
        "--action-horizon",
        str(ACTION_HORIZON),
        "--job-name",
        job_name,
        "--timeout",
        str(int(MAX_WAIT)),
        "--submit-only",
        "--output",
        "json",
    ]


def _run_npa(
    args: list[str],
    *,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_npa_executable(), *args],
        cwd=Path(__file__).resolve().parents[3],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _npa_executable() -> str:
    script = Path(sys.executable).with_name("npa")
    if script.exists():
        return str(script)
    return "npa"


def _wait_for_visible_job(
    client: ServerlessClient,
    project_id: str,
    job_id: str,
):
    deadline = time.monotonic() + 60
    last: Exception | None = None
    while time.monotonic() <= deadline:
        try:
            return client.get_job(job_id, project_id)
        except Exception as exc:
            last = exc
            time.sleep(2)
    pytest.fail(f"Job {job_id} was not visible after submission: {last}")


def _poll_job(
    client: ServerlessClient,
    project_id: str,
    job_id: str,
    artifacts_dir: Path,
):
    deadline = time.monotonic() + MAX_WAIT
    started = time.monotonic()
    last = None
    tick = 0
    while time.monotonic() <= deadline:
        tick += 1
        current = client.get_job(job_id, project_id)
        last = current
        _write_job_capture(project_id, current, artifacts_dir, label=f"tick-{tick:03d}")
        if current.status in {"running", "succeeded", "failed", "cancelled"}:
            started = None
        if current.status in {"succeeded", "failed", "cancelled"}:
            return current
        if started is not None and time.monotonic() - started > STARTING_WAIT:
            pytest.fail(f"Job {job_id} did not leave queue/startup within {STARTING_WAIT}s; last={current.raw}")
        time.sleep(POLL_INTERVAL)
    pytest.fail(f"Job {job_id} did not finish within {MAX_WAIT}s; last={last}")


def _write_job_capture(
    project_id: str,
    info,
    artifacts_dir: Path,
    *,
    label: str,
) -> None:
    (artifacts_dir / f"job-detail-{label}.json").write_text(
        json.dumps(info.raw, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _capture_logs(info.id, artifacts_dir / f"job-logs-{label}.txt")
    (artifacts_dir / f"job-status-{label}.txt").write_text(
        f"project_id={project_id}\njob_id={info.id}\nname={info.name}\nstatus={info.status}\n",
        encoding="utf-8",
    )


def _capture_logs(job_id: str, path: Path) -> None:
    result = subprocess.run(
        ["nebius", "ai", "job", "logs", job_id, "--tail", "500", "--timestamps"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=180,
        check=False,
    )
    path.write_text(result.stdout, encoding="utf-8")


def _cleanup_job(project_id: str, ref: str, artifacts_dir: Path) -> None:
    client = ServerlessClient()
    try:
        info = client.cancel_job(ref, project_id)
        job_id = info.id or ref
    except EndpointNotFoundError:
        return
    except Exception as exc:
        (artifacts_dir / "cleanup-cancel.err").write_text(str(exc), encoding="utf-8")
        job_id = ref
    result = subprocess.run(
        ["nebius", "ai", "job", "delete", "--id", job_id],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=240,
        check=False,
    )
    (artifacts_dir / "cleanup-delete.log").write_text(result.stdout, encoding="utf-8")
    orphan = subprocess.run(
        ["nebius", "ai", "job", "get", "--id", job_id, "--format", "json"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
        check=False,
    )
    (artifacts_dir / "cleanup-orphan-check.log").write_text(orphan.stdout, encoding="utf-8")


def _download_s3_prefix(
    output_path: str,
    local_dir: Path,
    access_key: str,
    secret_key: str,
    endpoint_url: str,
) -> None:
    parsed = urlparse(output_path)
    prefix = parsed.path.lstrip("/")
    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    local_dir.mkdir(parents=True, exist_ok=True)
    paginator = client.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=parsed.netloc, Prefix=prefix):
        keys.extend(obj["Key"] for obj in page.get("Contents", []))
    assert keys, f"No GR00T artifacts found under {output_path}"
    for key in keys:
        rel = key.removeprefix(prefix).lstrip("/")
        target = local_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(parsed.netloc, key, str(target))


def _assert_manifest(manifest: dict[str, object], *, job_name: str) -> None:
    assert set(manifest) == EXPECTED_MANIFEST_KEYS
    assert manifest["status"] == "success"
    assert manifest["tool"] == "groot"
    assert manifest["input_path"] == INPUT_PATH
    assert manifest["dataset_path"] == DATASET_PATH
    assert manifest["embodiment_tag"] == EMBODIMENT_TAG
    assert manifest["inference_mode"] == INFERENCE_MODE
    assert manifest["steps"] == STEPS
    assert manifest["action_horizon"] == ACTION_HORIZON
    assert manifest["model_variant"] == GROOT_MODEL_VARIANT
    assert manifest["groot_import"] == "available"
    assert manifest["job"] == job_name
    assert isinstance(manifest["duration_seconds"], int | float)
    assert manifest["duration_seconds"] >= 0


def _expected_artifact_names() -> set[str]:
    return {"npa_groot_infer_results.json", "predicted_actions.json"}


def _format_result(result: subprocess.CompletedProcess[str]) -> str:
    return (
        f"command failed with rc={result.returncode}\n"
        f"stdout:\n{result.stdout[-4000:]}\n"
        f"stderr:\n{result.stderr[-4000:]}"
    )
