from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
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
JOB_PREFIX = "npa-lerobot-e2e"
POLL_INTERVAL = 30.0
MAX_WAIT = float(os.environ.get("NPA_E2E_LEROBOT_JOBS_MAX_WAIT", "900"))
_SUBNET_CACHE: dict[str, str] = {}


@pytest.fixture(autouse=True)
def _require_lerobot_jobs_e2e(request: pytest.FixtureRequest) -> None:
    if os.environ.get("NPA_INTEGRATION_E2E") != "1":
        pytest.skip("NPA_INTEGRATION_E2E not set")
    if not os.environ.get("NPA_E2E_SERVERLESS_PROJECT"):
        pytest.skip("NPA_E2E_SERVERLESS_PROJECT not set")
    request.getfixturevalue("s3_write_access_required")


@pytest.fixture
def isolated_npa_home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    npa_home = home / ".npa"
    npa_home.mkdir(parents=True)
    for name in ("config.yaml", "credentials.yaml"):
        src = Path.home() / ".npa" / name
        if src.exists():
            shutil.copy2(src, npa_home / name)
            (npa_home / name).chmod(0o600)
    nebius_src = Path.home() / ".nebius"
    if nebius_src.exists():
        shutil.copytree(nebius_src, home / ".nebius", symlinks=True)
    return home


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
            print(f"!!! ORPHANED JOB in project {project_id} cancel failed ref={ref}: {exc}", flush=True)
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
                f"!!! ORPHANED JOB in project {project_id} delete failed ref={ref} "
                f"stderr={result.stderr.strip()}",
                flush=True,
            )
    for project_id in FallbackChain.instance().all_projects():
        orphans = [
            job.name for job in client.list_jobs(project_id, JOB_PREFIX)
            if job.name.startswith(JOB_PREFIX)
        ]
        for name in orphans:
            print(f"!!! ORPHANED JOB in project {project_id}: {name}", flush=True)


def _job_name(label: str) -> str:
    return f"{JOB_PREFIX}-{label}-{uuid.uuid4().hex[:8]}"


def _run_npa(args: list[str], *, home: Path, timeout: int = 1200) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    repo_src = Path(__file__).resolve().parents[2] / "src"
    env["HOME"] = str(home)
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
    ranked = sorted(
        ready,
        key=lambda item: (
            "lerobot" not in str((item.get("metadata") or {}).get("name", "")).lower(),
            "default" not in str((item.get("metadata") or {}).get("name", "")).lower(),
        ),
    )
    if ranked:
        subnet = str(((ranked[0].get("metadata") or {}).get("id") or ""))
        _SUBNET_CACHE[project_id] = subnet
        return subnet
    raise RuntimeError(f"No READY subnet found for {project_id}")


def _submit_train(
    name: str,
    jobs_to_cleanup: list[tuple[str, str]],
    *,
    home: Path,
    dataset: str = "lerobot/pusht",
    input_path: str = "",
    policy_type: str = "act",
    gpu_type: str = "",
    gpu_count: str = "1",
    fallback: bool = True,
    timeout: int = 1800,
    extra: tuple[str, ...] = (),
) -> tuple[str, dict[str, object], subprocess.CompletedProcess[str]]:
    from ._serverless_images import resolve_serverless_gpu_type

    # Env NPA_E2E_SERVERLESS_GPU_TYPE wins so live rtxpro can remap off H200.
    resolved_gpu = resolve_serverless_gpu_type(gpu_type or "h200")
    chain = FallbackChain.instance()
    project_id = chain.current_project()
    last_result: subprocess.CompletedProcess[str] | None = None
    attempts = 0
    while project_id and attempts < (2 if fallback else 1):
        attempts += 1
        jobs_to_cleanup.append((project_id, name))
        project_key = chain.project_key(project_id)
        dataset_args = ["--input-path", input_path] if input_path else ["--dataset", dataset]
        result = _run_npa(
            [
                "workbench",
                "lerobot",
                "-p",
                project_key,
                "-n",
                "lerobot",
                "train",
                "--runtime",
                "serverless",
                "--project-id",
                project_id,
                "--subnet-id",
                _subnet_id(project_id),
                "--policy-type",
                policy_type,
                *dataset_args,
                "--job-name",
                name,
                "--gpu-type",
                resolved_gpu,
                "--gpu-count",
                gpu_count,
                "--smoke",
                "--poll-interval",
                str(POLL_INTERVAL),
                "--wait-timeout",
                str(int(MAX_WAIT)),
                "--output",
                "json",
                *extra,
            ],
            home=home,
            timeout=timeout,
        )
        if result.returncode == 0:
            return project_id, json.loads(result.stdout), result
        last_result = result
        if fallback and _is_ner(result):
            project_id = chain.mark_ner(project_id)
            continue
        return project_id, {}, result
    pytest.skip(f"Project attempts NER-exhausted for {name}: {last_result}")


def _output_prefix(output_path: str) -> tuple[str, str]:
    parsed = urlparse(output_path)
    return parsed.netloc, parsed.path.strip("/").rstrip("/") + "/"


def _artifact_exists(project_id: str, output_path: str) -> bool:
    bucket, prefix = _output_prefix(output_path)
    project_key = FallbackChain.instance().project_key(project_id)
    client = s3_client_for_project(project_key)
    response = client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    return bool(response.get("Contents"))


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


def test_e2e_cli_lerobot_train_happy_path(
    isolated_npa_home: Path,
    jobs_to_cleanup: list[tuple[str, str]],
) -> None:
    name = _job_name("happy")
    project_id, payload, result = _submit_train(name, jobs_to_cleanup, home=isolated_npa_home)
    assert result.returncode == 0, result.stderr
    assert payload["status"] == "succeeded"
    _wait_for_artifact(project_id, str(payload["output_path"]))
    assert ServerlessClient().get_job(str(payload["job_id"]), project_id).status == "succeeded"


def test_e2e_cli_lerobot_train_ner_handling(
    isolated_npa_home: Path,
    jobs_to_cleanup: list[tuple[str, str]],
) -> None:
    """Exercise NER with env-overridable resource knobs for platform catalog drift."""
    if os.environ.get("NPA_E2E_FORCE_NER") != "1":
        pytest.skip("NPA_E2E_FORCE_NER not set")
    name = _job_name("ner")
    ner_platform = os.environ.get("NPA_E2E_NER_PLATFORM", "gpu-h200-sxm")
    ner_gpu_count = os.environ.get("NPA_E2E_NER_GPU_COUNT", "8")
    project_id, _payload, result = _submit_train(
        name,
        jobs_to_cleanup,
        home=isolated_npa_home,
        gpu_type=ner_platform,
        gpu_count=ner_gpu_count,
        fallback=False,
        extra=("--submit-only",),
    )
    assert result.returncode != 0
    assert _is_ner(result), result.stderr
    assert not [job for job in ServerlessClient().list_jobs(project_id, JOB_PREFIX) if job.name == name]


def test_e2e_cli_lerobot_train_cancel(
    isolated_npa_home: Path,
    jobs_to_cleanup: list[tuple[str, str]],
) -> None:
    name = _job_name("cancel")
    project_id, payload, result = _submit_train(
        name,
        jobs_to_cleanup,
        home=isolated_npa_home,
        extra=("--submit-only", "--steps", "100000"),
    )
    assert result.returncode == 0, result.stderr
    job_id = str(payload["job_id"])
    _wait_for_state(project_id, job_id, {"running", "queued"})
    info = ServerlessClient().cancel_job(job_id, project_id)
    assert info.status in {"cancelling", "cancelled"}
    assert _wait_for_state(project_id, job_id, {"cancelled"}) == "cancelled"


def test_e2e_cli_lerobot_train_status_lifecycle(
    isolated_npa_home: Path,
    jobs_to_cleanup: list[tuple[str, str]],
) -> None:
    name = _job_name("status")
    project_id, payload, result = _submit_train(
        name,
        jobs_to_cleanup,
        home=isolated_npa_home,
        extra=("--submit-only",),
    )
    assert result.returncode == 0, result.stderr
    observed: list[str] = []
    deadline = time.monotonic() + MAX_WAIT
    while time.monotonic() <= deadline:
        state = ServerlessClient().get_job(str(payload["job_id"]), project_id).status
        if not observed or observed[-1] != state:
            observed.append(state)
        if state == "succeeded":
            break
        time.sleep(POLL_INTERVAL)
    assert observed[-1] == "succeeded"
    assert "running" in observed or "queued" in observed


def test_e2e_cli_lerobot_train_hf_propagation(
    isolated_npa_home: Path,
    jobs_to_cleanup: list[tuple[str, str]],
) -> None:
    name = _job_name("hf")
    _project_id, payload, result = _submit_train(name, jobs_to_cleanup, home=isolated_npa_home)
    assert result.returncode == 0, result.stderr
    logs = _job_logs(str(payload["job_id"]))
    assert "HF auth missing" not in logs
    assert "401" not in logs and "403" not in logs


def test_e2e_cli_lerobot_train_idempotent_submit(
    isolated_npa_home: Path,
    jobs_to_cleanup: list[tuple[str, str]],
) -> None:
    name = _job_name("idempotent")
    project_id, first, first_result = _submit_train(
        name,
        jobs_to_cleanup,
        home=isolated_npa_home,
        extra=("--submit-only",),
    )
    assert first_result.returncode == 0, first_result.stderr
    project_id, second, second_result = _submit_train(
        name,
        jobs_to_cleanup,
        home=isolated_npa_home,
        extra=("--submit-only",),
    )
    assert second_result.returncode == 0, second_result.stderr
    assert second["job_id"] == first["job_id"]
    jobs = [job for job in ServerlessClient().list_jobs(project_id, JOB_PREFIX) if job.name == name]
    assert len(jobs) == 1


def test_e2e_cli_lerobot_train_dataset_from_hf(
    isolated_npa_home: Path,
    jobs_to_cleanup: list[tuple[str, str]],
) -> None:
    name = _job_name("hf-dataset")
    project_id, payload, result = _submit_train(
        name,
        jobs_to_cleanup,
        home=isolated_npa_home,
        dataset="lerobot/pusht",
    )
    assert result.returncode == 0, result.stderr
    assert payload["status"] == "succeeded"
    _wait_for_artifact(project_id, str(payload["output_path"]))


def test_e2e_cli_lerobot_train_dataset_from_s3(
    isolated_npa_home: Path,
    jobs_to_cleanup: list[tuple[str, str]],
) -> None:
    dataset_uri = os.environ.get("NPA_E2E_LEROBOT_S3_DATASET", "")
    if not dataset_uri:
        pytest.skip("NPA_E2E_LEROBOT_S3_DATASET not set")
    name = _job_name("s3-dataset")
    project_id, payload, result = _submit_train(
        name,
        jobs_to_cleanup,
        home=isolated_npa_home,
        input_path=dataset_uri,
    )
    assert result.returncode == 0, result.stderr
    assert payload["status"] == "succeeded"
    _wait_for_artifact(project_id, str(payload["output_path"]))


def test_e2e_cli_lerobot_train_diffusion_h200(
    isolated_npa_home: Path,
    jobs_to_cleanup: list[tuple[str, str]],
) -> None:
    name = _job_name("diffusion-h200")
    project_id, payload, result = _submit_train(
        name,
        jobs_to_cleanup,
        home=isolated_npa_home,
        policy_type="diffusion",
        gpu_type="h200",
    )
    assert result.returncode == 0, result.stderr
    assert payload["status"] == "succeeded"
    _wait_for_artifact(project_id, str(payload["output_path"]))


def test_e2e_cli_lerobot_train_submit_only(
    isolated_npa_home: Path,
    jobs_to_cleanup: list[tuple[str, str]],
) -> None:
    name = _job_name("submit-only")
    project_id, payload, result = _submit_train(
        name,
        jobs_to_cleanup,
        home=isolated_npa_home,
        extra=("--submit-only",),
    )
    assert result.returncode == 0, result.stderr
    assert payload["status"] == "submitted"
    assert _wait_for_state(project_id, str(payload["job_id"]), {"succeeded"}) == "succeeded"
