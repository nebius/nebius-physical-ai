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
IMAGE = "cr.eu-north1.nebius.cloud/your-registry-id/npa-genesis:0.4.6"
GPU_TYPE = "h200"
GPU_PRESET = "1gpu-16vcpu-200gb"
GPU_COUNT = 1
N_ENVS = 1
MAX_ITERATIONS = 1
ACTION_SPACE = "cartesian"
SEED = 42
JOB_PREFIX = "npa-e2e-genesis-train"
POLL_INTERVAL = float(os.environ.get("NPA_E2E_GENESIS_POLL_INTERVAL", "30"))
MAX_WAIT = float(os.environ.get("NPA_E2E_GENESIS_MAX_WAIT", "7200"))
STARTING_WAIT = float(os.environ.get("NPA_E2E_GENESIS_STARTING_WAIT", "3600"))
EXPECTED_SUMMARY_KEYS = {
    "action_space",
    "duration_seconds",
    "genesis_import",
    "job",
    "max_iterations",
    "n_envs",
    "seed",
    "status",
    "tool",
}
EXPECTED_MODEL_KEYS = EXPECTED_SUMMARY_KEYS | {"format"}


def test_genesis_smoke_helper_request_shape() -> None:
    test_id = "shape"
    output_path = _output_path(test_id)
    command = _submit_command(
        project_alias=PROJECT_ALIAS,
        workbench_name=WORKBENCH_NAME,
        project_id=PROJECT_ID,
        output_path=output_path,
        job_name=f"{JOB_PREFIX}-{test_id}",
    )

    assert IMAGE.endswith("/npa-genesis:0.4.6")
    assert "--subnet-id" not in command
    assert command[:7] == [
        "workbench",
        "genesis",
        "-p",
        PROJECT_ALIAS,
        "-n",
        WORKBENCH_NAME,
        "train-teacher",
    ]
    for flag in (
        "--runtime",
        "--project-id",
        "--n-envs",
        "--max-iterations",
        "--output-path",
        "--image",
        "--gpu-type",
        "--gpu-count",
        "--gpu-preset",
        "--seed",
        "--action-space",
        "--job-name",
        "--timeout",
        "--submit-only",
        "--output-format",
    ):
        assert flag in command
    assert _expected_artifact_names() == {"model.pt", "train_teacher_summary.json"}


@pytest.mark.e2e_serverless
def test_genesis_serverless_train_teacher(tmp_path: Path) -> None:
    _require_genesis_e2e()
    test_id = f"w7genesis-e2e-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}"
    artifacts_dir = Path("/tmp") / f"genesis-e2e-artifacts-{test_id}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    project_alias = os.environ.get("NPA_E2E_PROJECT", PROJECT_ALIAS)
    workbench_name = os.environ.get("NPA_E2E_GENESIS_WORKBENCH", WORKBENCH_NAME)
    project_id = os.environ.get("NPA_E2E_SERVERLESS_PROJECT", PROJECT_ID)
    bucket = os.environ.get("NPA_E2E_S3_BUCKET", BUCKET)
    endpoint_url = os.environ.get("NPA_E2E_S3_ENDPOINT", ENDPOINT_URL)
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
        image=resolve_image(os.environ.get("NPA_E2E_GENESIS_IMAGE", IMAGE)),
        gpu_type=resolve_serverless_gpu_type(
            os.environ.get("NPA_E2E_GENESIS_GPU_TYPE", GPU_TYPE)
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
        submitted = _run_npa(command, timeout=int(os.environ.get("NPA_E2E_GENESIS_SUBMIT_TIMEOUT", "600")))
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
        assert _submitted_subnet_id(submitted_info.raw), "submitted Job spec.subnet_id is empty"

        final = _poll_job(client, project_id, job_id, artifacts_dir)
        assert final.status == "succeeded", final.raw
        _write_job_capture(project_id, final, artifacts_dir, label="final")

        local_dir = artifacts_dir / "s3"
        _download_s3_prefix(output_path, local_dir, access_key, secret_key, endpoint_url)
        assert {path.name for path in local_dir.iterdir() if path.is_file()} >= _expected_artifact_names()

        summary = json.loads((local_dir / "train_teacher_summary.json").read_text(encoding="utf-8"))
        model = json.loads((local_dir / "model.pt").read_text(encoding="utf-8"))
        _assert_summary(summary, job_name=job_name)
        _assert_model(model, summary=summary)
    finally:
        if job_id or job_name:
            _cleanup_job(project_id, job_id or job_name, artifacts_dir)


def _require_genesis_e2e() -> None:
    if os.environ.get("NPA_INTEGRATION_E2E") != "1":
        pytest.skip("NPA_INTEGRATION_E2E not set")
    for key in (
        "NPA_E2E_SERVERLESS_PROJECT",
        "NPA_E2E_S3_ACCESS_KEY_ID",
        "NPA_E2E_S3_SECRET_ACCESS_KEY",
    ):
        if not os.environ.get(key):
            pytest.skip(f"{key} not set")


def _output_path(test_id: str, *, bucket: str = BUCKET) -> str:
    return f"s3://{bucket}/w7genesis-e2e/{test_id}/"


def _submit_command(
    *,
    project_alias: str,
    workbench_name: str,
    project_id: str,
    output_path: str,
    job_name: str,
    image: str = IMAGE,
    gpu_type: str = GPU_TYPE,
) -> list[str]:
    return [
        "workbench",
        "genesis",
        "-p",
        project_alias,
        "-n",
        workbench_name,
        "train-teacher",
        "--runtime",
        "serverless",
        "--project-id",
        project_id,
        "--n-envs",
        str(N_ENVS),
        "--max-iterations",
        str(MAX_ITERATIONS),
        "--output-path",
        output_path,
        "--image",
        image,
        "--gpu-type",
        gpu_type,
        "--gpu-count",
        str(GPU_COUNT),
        "--gpu-preset",
        resolve_serverless_gpu_preset(GPU_PRESET, platform=gpu_type),
        "--seed",
        str(SEED),
        "--action-space",
        ACTION_SPACE,
        "--job-name",
        job_name,
        "--timeout",
        str(int(MAX_WAIT)),
        "--submit-only",
        "--output-format",
        "json",
    ]


def _run_npa(args: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
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


def _wait_for_visible_job(client: ServerlessClient, project_id: str, job_id: str):
    deadline = time.monotonic() + 60
    last: Exception | None = None
    while time.monotonic() <= deadline:
        try:
            return client.get_job(job_id, project_id)
        except Exception as exc:
            last = exc
            time.sleep(2)
    pytest.fail(f"Job {job_id} was not visible after submission: {last}")


def _poll_job(client: ServerlessClient, project_id: str, job_id: str, artifacts_dir: Path):
    deadline = time.monotonic() + MAX_WAIT
    startup_deadline = time.monotonic() + STARTING_WAIT
    last = None
    tick = 0
    while time.monotonic() <= deadline:
        tick += 1
        current = client.get_job(job_id, project_id)
        last = current
        _write_job_capture(project_id, current, artifacts_dir, label=f"tick-{tick:03d}")
        if current.status in {"running", "succeeded", "failed", "cancelled"}:
            startup_deadline = 0
        if current.status in {"succeeded", "failed", "cancelled"}:
            return current
        if startup_deadline and time.monotonic() > startup_deadline:
            pytest.fail(f"Job {job_id} did not leave queue/startup within {STARTING_WAIT}s; last={current.raw}")
        time.sleep(POLL_INTERVAL)
    pytest.fail(f"Job {job_id} did not finish within {MAX_WAIT}s; last={last}")


def _write_job_capture(project_id: str, info, artifacts_dir: Path, *, label: str) -> None:
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
    assert keys, f"No Genesis artifacts found under {output_path}"
    for key in keys:
        rel = key.removeprefix(prefix).lstrip("/")
        target = local_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(parsed.netloc, key, str(target))


def _submitted_subnet_id(raw: dict[str, object]) -> str:
    spec = raw.get("spec") if isinstance(raw, dict) else None
    if isinstance(spec, dict):
        return str(spec.get("subnet_id") or "").strip()
    return ""


def _assert_summary(summary: dict[str, object], *, job_name: str) -> None:
    assert set(summary) == EXPECTED_SUMMARY_KEYS
    assert summary["status"] == "success"
    assert summary["tool"] == "genesis"
    assert summary["n_envs"] == N_ENVS
    assert summary["max_iterations"] == MAX_ITERATIONS
    assert summary["action_space"] == ACTION_SPACE
    assert summary["seed"] == SEED
    assert summary["genesis_import"] == "available"
    assert summary["job"] == job_name
    assert isinstance(summary["duration_seconds"], int | float)
    assert summary["duration_seconds"] >= 0


def _assert_model(model: dict[str, object], *, summary: dict[str, object]) -> None:
    assert set(model) == EXPECTED_MODEL_KEYS
    assert model["format"] == "npa_genesis_serverless_smoke_v1"
    for key in EXPECTED_SUMMARY_KEYS:
        assert model[key] == summary[key]


def _expected_artifact_names() -> set[str]:
    return {"model.pt", "train_teacher_summary.json"}


def _format_result(result: subprocess.CompletedProcess[str]) -> str:
    return (
        f"command failed with rc={result.returncode}\n"
        f"stdout:\n{result.stdout[-4000:]}\n"
        f"stderr:\n{result.stderr[-4000:]}"
    )
