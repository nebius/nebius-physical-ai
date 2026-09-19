from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from typing import Iterator
from urllib.parse import urlparse

import pytest

from npa.clients.project_credentials import s3_client_for_project
from npa.clients.serverless import EndpointNotFoundError, ServerlessClient, _NER_PATTERNS

from ._serverless_fallback import FallbackChain


pytestmark = pytest.mark.e2e_serverless
JOB_PREFIX = "npa-e2e-jobs"
POLL_INTERVAL = 30.0
MAX_WAIT = float(os.environ.get("NPA_E2E_JOBS_MAX_WAIT", "540"))
_SUBNET_CACHE: dict[str, str] = {}


@pytest.fixture(autouse=True)
def _require_jobs_e2e(request: pytest.FixtureRequest) -> None:
    if os.environ.get("NPA_INTEGRATION_E2E") != "1":
        pytest.skip("NPA_INTEGRATION_E2E not set")
    if not os.environ.get("NPA_E2E_SERVERLESS_PROJECT"):
        pytest.skip("NPA_E2E_SERVERLESS_PROJECT not set")
    request.getfixturevalue("s3_write_access_required")


@pytest.fixture
def jobs_to_cleanup() -> Iterator[list[tuple[str, str]]]:
    jobs: list[tuple[str, str]] = []
    yield jobs
    client = ServerlessClient()
    for project_id, ref in reversed(jobs):
        try:
            info = client.cancel_job(ref, project_id)
        except EndpointNotFoundError:
            continue
        except Exception as exc:
            print(f"!!! ORPHANED JOB cancel failed project={project_id} ref={ref}: {exc}", flush=True)
            continue
        result = subprocess.run(
            ["nebius", "ai", "job", "delete", "--id", info.id or ref],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180,
            check=False,
        )
        if result.returncode != 0:
            print(
                f"!!! ORPHANED JOB delete failed project={project_id} ref={ref} "
                f"stderr={result.stderr.strip()}",
                flush=True,
            )


def _job_name(label: str) -> str:
    return f"{JOB_PREFIX}-{label}-{uuid.uuid4().hex[:8]}"


def _run_npa(args: list[str], *, timeout: int = 520) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    repo_src = Path(__file__).resolve().parents[2] / "src"
    env["PYTHONPATH"] = str(repo_src) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c", "from npa.cli.main import app; app()", *args],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def _is_ner(result: subprocess.CompletedProcess[str]) -> bool:
    combined = f"{result.stdout}\n{result.stderr}".lower()
    return any(pattern in combined for pattern in _NER_PATTERNS)


def _subnet_id(project_id: str) -> str:
    override = os.environ.get("NPA_E2E_SERVERLESS_SUBNET_ID", "")
    if override:
        return override
    if project_id in _SUBNET_CACHE:
        return _SUBNET_CACHE[project_id]
    result = subprocess.run(
        ["nebius", "vpc", "subnet", "list", "--parent-id", project_id, "--format", "json"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Unable to list subnets for {project_id}: {result.stderr.strip()}")
    data = json.loads(result.stdout or "{}")
    items = data.get("items") if isinstance(data, dict) else data
    ready = []
    for item in items or []:
        state = str(((item.get("status") or {}).get("state") or "")).upper()
        if state in {"READY", ""}:
            ready.append(item)
    ranked = sorted(ready, key=lambda item: ("cosmos" not in str((item.get("metadata") or {}).get("name", "")).lower(), "default" not in str((item.get("metadata") or {}).get("name", "")).lower()))
    if ranked:
        subnet = str(((ranked[0].get("metadata") or {}).get("id") or ""))
        _SUBNET_CACHE[project_id] = subnet
        return subnet
    raise RuntimeError(f"No READY subnet found for {project_id}")


def _submit_train(
    name: str,
    jobs_to_cleanup: list[tuple[str, str]],
    *extra: str,
    fallback: bool = True,
    timeout: int = 3600,
) -> tuple[str, dict[str, object], subprocess.CompletedProcess[str]]:
    chain = FallbackChain.instance()
    project_id = chain.current_project()
    last_result: subprocess.CompletedProcess[str] | None = None
    while project_id:
        jobs_to_cleanup.append((project_id, name))
        project_key = chain.project_key(project_id)
        result = _run_npa(
            [
                "workbench",
                "cosmos",
                "-p",
                project_key,
                "-n",
                "cosmos",
                "train",
                "--runtime",
                "serverless",
                "--project-id",
                project_id,
                "--subnet-id",
                _subnet_id(project_id),
                "--smoke",
                "--job-name",
                name,
                "--output-format",
                "json",
                *extra,
            ],
            timeout=timeout,
        )
        if result.returncode == 0:
            return project_id, json.loads(result.stdout), result
        last_result = result
        if fallback and _is_ner(result):
            project_id = chain.mark_ner(project_id)
            continue
        return project_id, {}, result
    pytest.skip(f"All projects in fallback chain are NER-exhausted: {last_result}")


def _artifact_key(output_path: str) -> tuple[str, str]:
    parsed = urlparse(output_path)
    return parsed.netloc, f"{parsed.path.strip('/')}/checkpoint.json".strip("/")


def _artifact_exists(project_id: str, output_path: str) -> bool:
    bucket, key = _artifact_key(output_path)
    project_key = FallbackChain.instance().project_key(project_id)
    client = s3_client_for_project(project_key)
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except client.exceptions.ClientError as exc:
        if exc.response["Error"]["Code"] in {"404", "NoSuchKey"}:
            return False
        raise


def _wait_for_artifact(project_id: str, output_path: str) -> None:
    deadline = time.monotonic() + 180
    while time.monotonic() <= deadline:
        if _artifact_exists(project_id, output_path):
            return
        time.sleep(10)
    pytest.fail(f"checkpoint artifact not found at {output_path}")


def _job_logs(job_id: str) -> str:
    result = subprocess.run(
        ["nebius", "ai", "job", "logs", job_id, "--tail", "200"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=180,
        check=False,
    )
    return result.stdout + result.stderr


def _wait_for_state(project_id: str, job_id: str, states: set[str]) -> str:
    client = ServerlessClient()
    deadline = time.monotonic() + MAX_WAIT
    last = "unknown"
    while time.monotonic() <= deadline:
        info = client.get_job(job_id, project_id)
        last = info.status
        if last in states:
            return last
        if last in {"succeeded", "failed", "cancelled"}:
            pytest.fail(f"job reached terminal state {last} before {states}")
        time.sleep(POLL_INTERVAL)
    pytest.fail(f"job did not reach {states} within {MAX_WAIT}s; last={last}")


def test_e2e_cli_train_happy_path(jobs_to_cleanup: list[tuple[str, str]]) -> None:
    name = _job_name("happy")
    project_id, payload, result = _submit_train(name, jobs_to_cleanup)
    assert result.returncode == 0, result.stderr
    assert payload["status"] == "succeeded"
    _wait_for_artifact(project_id, str(payload["output_path"]))
    assert ServerlessClient().get_job(str(payload["job_id"]), project_id).status == "succeeded"


def test_e2e_cli_train_ner_handling(jobs_to_cleanup: list[tuple[str, str]]) -> None:
    """Exercise NER with env-overridable resource knobs for platform catalog drift."""
    if os.environ.get("NPA_E2E_FORCE_NER") != "1":
        pytest.skip("NPA_E2E_FORCE_NER not set")
    name = _job_name("ner")
    ner_platform = os.environ.get("NPA_E2E_NER_PLATFORM", "gpu-h200-sxm")
    ner_gpu_count = os.environ.get("NPA_E2E_NER_GPU_COUNT", "8")
    ner_preset = os.environ.get("NPA_E2E_NER_PRESET", "8gpu-128vcpu-1600gb")
    project_id, _payload, result = _submit_train(
        name,
        jobs_to_cleanup,
        "--submit-only",
        "--gpu-type",
        ner_platform,
        "--gpu-count",
        ner_gpu_count,
        "--gpu-preset",
        ner_preset,
        fallback=False,
    )
    assert result.returncode != 0
    assert _is_ner(result), result.stderr
    assert not [job for job in ServerlessClient().list_jobs(project_id, JOB_PREFIX) if job.name == name]


def test_e2e_cli_train_cancel(jobs_to_cleanup: list[tuple[str, str]]) -> None:
    name = _job_name("cancel")
    project_id, payload, result = _submit_train(
        name, jobs_to_cleanup, "--submit-only", "--smoke-seconds", "600"
    )
    assert result.returncode == 0, result.stderr
    job_id = str(payload["job_id"])
    _wait_for_state(project_id, job_id, {"running"})
    cancel = _run_npa([
        "workbench", "cosmos", "train", "--runtime", "serverless", "--project-id",
        project_id, "cancel", job_id, "--output-format", "json",
    ])
    assert cancel.returncode == 0, cancel.stderr
    assert json.loads(cancel.stdout)["status"] in {"cancelling", "cancelled"}
    assert _wait_for_state(project_id, job_id, {"cancelled"}) == "cancelled"
    assert not _artifact_exists(project_id, str(payload["output_path"]))
    ServerlessClient().cancel_job(job_id, project_id)


def test_e2e_cli_train_status_lifecycle(jobs_to_cleanup: list[tuple[str, str]]) -> None:
    name = _job_name("status")
    project_id, payload, result = _submit_train(
        name, jobs_to_cleanup, "--submit-only", "--smoke-seconds", "30"
    )
    assert result.returncode == 0, result.stderr
    observed: list[str] = []
    deadline = time.monotonic() + MAX_WAIT
    while time.monotonic() <= deadline:
        status = _run_npa([
            "workbench", "cosmos", "train", "--runtime", "serverless", "--project-id",
            project_id, "status", str(payload["job_id"]), "--output-format", "json",
        ])
        assert status.returncode == 0, status.stderr
        state = str(json.loads(status.stdout)["status"])
        if not observed or observed[-1] != state:
            observed.append(state)
        if state == "succeeded":
            break
        time.sleep(POLL_INTERVAL)
    assert observed[-1] == "succeeded"
    assert "running" in observed or "queued" in observed


def test_e2e_cli_train_hf_propagation(jobs_to_cleanup: list[tuple[str, str]]) -> None:
    name = _job_name("hf")
    project_id, payload, result = _submit_train(name, jobs_to_cleanup, "--require-hf")
    assert result.returncode == 0, result.stderr
    logs = _job_logs(str(payload["job_id"]))
    assert "HF auth missing" not in logs
    assert "401" not in logs and "403" not in logs


def test_e2e_cli_train_idempotent_submit(jobs_to_cleanup: list[tuple[str, str]]) -> None:
    name = _job_name("idempotent")
    project_id, first, first_result = _submit_train(name, jobs_to_cleanup, "--submit-only")
    assert first_result.returncode == 0, first_result.stderr
    project_id, second, second_result = _submit_train(name, jobs_to_cleanup, "--submit-only")
    assert second_result.returncode == 0, second_result.stderr
    assert second["job_id"] == first["job_id"]
    jobs = [job for job in ServerlessClient().list_jobs(project_id, JOB_PREFIX) if job.name == name]
    assert len(jobs) == 1
