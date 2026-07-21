from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest
import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from npa.clients.config import resolve_project_storage
from npa.clients.credentials import load_credentials, storage_endpoint_url
from npa.orchestration.skypilot.workflow_state import redact_text


pytestmark = [pytest.mark.e2e, pytest.mark.e2e_skypilot]

ROOT = Path(__file__).resolve().parents[3]
SECRET_HF_MARKER = "hf_npae2eworkflowsecret1234567890"
SECRET_AWS_MARKER = "AKIAABCDEFGHIJKLMNOP"


def _default_kube_context() -> str:
    """Resolve the kube context from params/config, never a hard-coded region.

    Order: ``NPA_E2E_KUBECONTEXT`` -> ``NPA_E2E_KUBECONTEXT_FALLBACK`` ->
    ``kubectl config current-context``. Skips when none is available so a run
    is never silently pinned to a specific region's cluster.
    """

    explicit = os.environ.get("NPA_E2E_KUBECONTEXT", "").strip()
    if explicit:
        return explicit
    fallback = os.environ.get("NPA_E2E_KUBECONTEXT_FALLBACK", "").strip()
    if fallback:
        return fallback
    try:
        result = subprocess.run(
            ["kubectl", "config", "current-context"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        current = (result.stdout or "").strip()
        if current:
            return current
    except (OSError, subprocess.SubprocessError):
        pass
    pytest.skip(
        "No kube context configured; set NPA_E2E_KUBECONTEXT for the durable-S3 test"
    )


def test_workbench_workflow_durable_s3_monitor_live(
    tmp_path: Path,
    e2e_project: str | None,
) -> None:
    """Submit a tiny managed workflow and verify durable S3 state after teardown."""

    _require_live_mode()
    sky_bin = _sky_bin()
    cli_bin = _npa_bin()
    kube_context = _default_kube_context()
    kubeconfig = _kubeconfig_for_context(kube_context)
    _assert_kube_context_ready(kube_context, kubeconfig)

    endpoint = _s3_endpoint(e2e_project)
    credentials_env = _s3_credentials_env(e2e_project)
    s3_client = _s3_client(endpoint, credentials_env)
    bucket, parent_prefix, cleanup_bucket = _workflow_bucket_and_prefix()
    _ensure_bucket(s3_client, bucket)
    run_id = f"workflow-s3-e2e-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}"
    run_prefix = "/".join(part for part in (parent_prefix, run_id) if part).strip("/")
    run_uri = f"s3://{bucket}/{run_prefix}/"
    evidence_dir = Path(
        os.environ.get("NPA_E2E_WORKFLOW_S3_EVIDENCE_DIR", f"/tmp/npa-workflow-s3-e2e-{run_id}")
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)
    isolated_sky_dir = tmp_path / "sky-state"
    yaml_path = tmp_path / "durable-workflow.yaml"
    yaml_path.write_text(_workflow_yaml(), encoding="utf-8")
    env = {**os.environ, **credentials_env, "NO_COLOR": "1"}
    if kubeconfig is not None:
        env["KUBECONFIG"] = str(kubeconfig)

    submit_cmd = [
        cli_bin,
        "workbench",
        "workflow",
        "submit",
        str(yaml_path),
        "--run-id",
        run_id,
        "--durable-s3",
        "--workflow-s3-uri",
        run_uri,
        "--s3-endpoint",
        endpoint,
        "--sky-bin",
        sky_bin,
        "--isolated-config-dir",
        str(isolated_sky_dir),
        "--infra",
        f"k8s/{kube_context}",
        "--submit-timeout",
        os.environ.get("NPA_E2E_WORKFLOW_S3_SUBMIT_TIMEOUT_SECONDS", "1800"),
        "--output-format",
        "json",
    ]

    job_id = ""
    terminal = False
    teardown_done = False
    attempts: list[dict[str, Any]] = []
    try:
        submit = _run(submit_cmd, env=env, cwd=ROOT, timeout=2400)
        _write_command_evidence(evidence_dir, "submit", submit, submit_cmd, credentials_env)
        assert submit.returncode == 0, redact_text(submit.stdout + submit.stderr, credentials_env.values())
        submit_payload = json.loads(submit.stdout)
        job_id = str(submit_payload.get("job_id") or "")
        assert job_id
        assert submit_payload["log_paths"]["run_prefix_uri"] == run_uri.rstrip("/")

        status_payload = _poll_status(
            cli_bin=cli_bin,
            sky_bin=sky_bin,
            run_uri=run_uri,
            endpoint=endpoint,
            env=env,
            evidence_dir=evidence_dir,
            attempts=attempts,
        )
        terminal = status_payload["status"] == "SUCCEEDED"
        assert terminal, json.dumps(status_payload, indent=2)

        sky_env = _isolated_sky_env(env, isolated_sky_dir)
        _sky_down_and_poll(sky_bin, run_id, evidence_dir=evidence_dir, env=sky_env)
        teardown_done = True

        manifest = _get_json(s3_client, bucket, f"{run_prefix}/manifest.json")
        stage_status = _get_json(s3_client, bucket, f"{run_prefix}/logs/smoke/status.json")
        run_log = _get_text(s3_client, bucket, f"{run_prefix}/logs/smoke/run.log")
        artifact = _get_json(s3_client, bucket, f"{run_prefix}/artifacts/smoke/pod-artifact.json")

        assert manifest["last_writer"] == "pod"
        assert manifest["run_id"] == run_id
        assert manifest["stages"]["smoke"]["sky_job_id"] == job_id
        assert manifest["stages"]["smoke"]["log_uri"] == f"{run_uri}logs/smoke/run.log"
        assert manifest["stages"]["smoke"]["artifact_uri"] == f"{run_uri}artifacts/smoke/"
        assert set(stage_status) >= {
            "state",
            "tier",
            "start",
            "end",
            "sky_job_id",
            "artifact_uri",
            "log_uri",
            "error_summary",
        }
        assert stage_status["state"] == "SUCCEEDED"
        assert stage_status["tier"] == "WORKS"
        assert stage_status["sky_job_id"] == job_id
        assert stage_status["artifact_uri"] == f"{run_uri}artifacts/smoke/"
        assert stage_status["log_uri"] == f"{run_uri}logs/smoke/run.log"
        assert artifact == {"run_id": run_id, "source": "pod", "stage": "smoke"}
        assert "durable workflow smoke complete" in run_log
        assert "<redacted>" in run_log
        assert SECRET_HF_MARKER not in run_log
        assert SECRET_AWS_MARKER not in run_log

        no_sky = str(tmp_path / "sky-not-available")
        post_status = _run(
            [
                cli_bin,
                "workbench",
                "workflow",
                "status",
                run_uri,
                "--s3-endpoint",
                endpoint,
                "--sky-bin",
                no_sky,
                "--json",
            ],
            env=env,
            cwd=ROOT,
            timeout=300,
        )
        post_logs = _run(
            [
                cli_bin,
                "workbench",
                "workflow",
                "logs",
                run_uri,
                "--stage",
                "smoke",
                "--s3-endpoint",
                endpoint,
            ],
            env=env,
            cwd=ROOT,
            timeout=300,
        )
        post_artifacts = _run(
            [
                cli_bin,
                "workbench",
                "workflow",
                "artifacts",
                run_uri,
                "--stage",
                "smoke",
                "--s3-endpoint",
                endpoint,
                "--json",
            ],
            env=env,
            cwd=ROOT,
            timeout=300,
        )
        _write_command_evidence(evidence_dir, "post-status-no-sky", post_status, [], credentials_env)
        _write_command_evidence(evidence_dir, "post-logs", post_logs, [], credentials_env)
        _write_command_evidence(evidence_dir, "post-artifacts", post_artifacts, [], credentials_env)
        assert post_status.returncode == 0, redact_text(post_status.stdout + post_status.stderr)
        post_status_payload = json.loads(post_status.stdout)
        assert post_status_payload["status"] == "SUCCEEDED"
        assert post_logs.returncode == 0, redact_text(post_logs.stdout + post_logs.stderr)
        assert "durable workflow smoke complete" in post_logs.stdout
        assert post_artifacts.returncode == 0, redact_text(post_artifacts.stdout + post_artifacts.stderr)
        assert f"{run_uri}artifacts/smoke/pod-artifact.json" in json.loads(post_artifacts.stdout)["artifacts"]

        _write_evidence(
            evidence_dir,
            run_id=run_id,
            job_id=job_id,
            run_uri=run_uri,
            endpoint=endpoint,
            kube_context=kube_context,
            status=post_status_payload,
            attempts=attempts,
        )
    finally:
        if job_id and not terminal:
            cleanup = _run(
                [sky_bin, "jobs", "cancel", "--yes", job_id],
                env=_isolated_sky_env(env, isolated_sky_dir),
                cwd=ROOT,
                timeout=900,
                check=False,
            )
            _write_command_evidence(evidence_dir, "jobs-cancel", cleanup, [], credentials_env)
        if not teardown_done:
            _sky_down_and_poll(
                sky_bin,
                run_id,
                evidence_dir=evidence_dir,
                env=_isolated_sky_env(env, isolated_sky_dir),
            )
        if cleanup_bucket:
            _delete_bucket(s3_client, bucket)


def _require_live_mode() -> None:
    if os.environ.get("NPA_INTEGRATION_E2E") != "1":
        pytest.skip("NPA_INTEGRATION_E2E not set")
    if os.environ.get("NPA_DRY_RUN") in {"1", "true", "TRUE"}:
        pytest.skip("durable workflow S3 e2e requires live writes")


def _workflow_bucket_and_prefix() -> tuple[str, str, bool]:
    configured = os.environ.get("NPA_E2E_WORKFLOW_S3_BUCKET", "").strip()
    configured_prefix = os.environ.get("NPA_E2E_WORKFLOW_S3_PREFIX", "").strip("/")
    if configured:
        parsed = urlparse(configured.rstrip("/"))
        if parsed.scheme == "s3":
            prefix = configured_prefix or parsed.path.strip("/") or "npa-workflow-e2e"
            return parsed.netloc, prefix, False
        return configured, configured_prefix or "npa-workflow-e2e", False
    timestamp = time.strftime("%Y%m%dt%H%M%Sz", time.gmtime())
    bucket = f"npa-e2e-test-workflow-s3-{timestamp}"
    return bucket[:63].strip("-"), configured_prefix or "npa-workflow-e2e", True


def _s3_endpoint(project: str | None) -> str:
    storage = resolve_project_storage(project)
    credentials = load_credentials()
    endpoint = storage_endpoint_url(
        storage.endpoint_url
        or credentials.s3_endpoint
        or os.environ.get("AWS_ENDPOINT_URL", "")
        or os.environ.get("NEBIUS_S3_ENDPOINT", "")
    )
    if not endpoint:
        pytest.skip(
            "S3 endpoint is not configured; set project storage or AWS_ENDPOINT_URL"
        )
    return endpoint


def _s3_credentials_env(project: str | None) -> dict[str, str]:
    storage = resolve_project_storage(project)
    credentials = load_credentials()
    access_key = (
        storage.aws_access_key_id
        or credentials.s3_access_key_id
        or os.environ.get("AWS_ACCESS_KEY_ID", "")
    )
    secret_key = (
        storage.aws_secret_access_key
        or credentials.s3_secret_access_key
        or os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    )
    if not access_key or not secret_key:
        pytest.skip("S3 credentials are not configured")
    return {
        "AWS_ACCESS_KEY_ID": access_key,
        "AWS_SECRET_ACCESS_KEY": secret_key,
    }


def _s3_client(endpoint: str, credentials_env: dict[str, str]):
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=credentials_env["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=credentials_env["AWS_SECRET_ACCESS_KEY"],
        config=BotoConfig(signature_version="s3v4"),
    )


def _ensure_bucket(client: Any, bucket: str) -> None:
    try:
        client.head_bucket(Bucket=bucket)
    except ClientError:
        client.create_bucket(Bucket=bucket)
    key = "preflight/workflow-state-write-check.txt"
    client.put_object(Bucket=bucket, Key=key, Body=b"workflow state preflight\n")
    body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    assert body == b"workflow state preflight\n"


def _delete_bucket(client: Any, bucket: str) -> None:
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket):
        objects = [{"Key": item["Key"]} for item in page.get("Contents", [])]
        if objects:
            client.delete_objects(Bucket=bucket, Delete={"Objects": objects})
    try:
        client.delete_bucket(Bucket=bucket)
    except ClientError:
        pass


def _workflow_yaml() -> str:
    return f"""name: durable-s3-smoke
execution: serial
---
name: smoke
resources:
  cloud: kubernetes
  cpus: 1
  memory: 2
  image_id: docker:python:3.11-slim
run: |
  set -euo pipefail
  echo "pod-host=$(hostname)"
  echo "HF_TOKEN={SECRET_HF_MARKER}"
  echo "AWS_ACCESS_KEY_ID={SECRET_AWS_MARKER}"
  artifact_dir="$NPA_WORKFLOW_MOUNT_ROOT/$NPA_WORKFLOW_S3_PREFIX/artifacts/$NPA_WORKFLOW_STAGE"
  mkdir -p "$artifact_dir"
  printf '{{"run_id":"%s","source":"pod","stage":"%s"}}\\n' \\
    "$NPA_WORKFLOW_RUN_ID" "$NPA_WORKFLOW_STAGE" > "$artifact_dir/pod-artifact.json"
  test -s "$artifact_dir/pod-artifact.json"
  echo "durable workflow smoke complete"
"""


def _poll_status(
    *,
    cli_bin: str,
    sky_bin: str,
    run_uri: str,
    endpoint: str,
    env: dict[str, str],
    evidence_dir: Path,
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    deadline = time.monotonic() + int(os.environ.get("NPA_E2E_WORKFLOW_S3_TIMEOUT_SECONDS", "3600"))
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        result = _run(
            [
                cli_bin,
                "workbench",
                "workflow",
                "status",
                run_uri,
                "--s3-endpoint",
                endpoint,
                "--sky-bin",
                sky_bin,
                "--json",
            ],
            env=env,
            cwd=ROOT,
            timeout=300,
        )
        _write_command_evidence(evidence_dir, f"status-{attempt:03d}", result, [], {})
        if result.returncode == 0:
            payload = json.loads(result.stdout)
            attempts.append({"attempt": attempt, "status": payload.get("status")})
            if payload.get("status") in {"SUCCEEDED", "FAILED", "CANCELLED"}:
                return payload
        else:
            attempts.append(
                {
                    "attempt": attempt,
                    "returncode": result.returncode,
                    "stderr": redact_text(result.stderr),
                }
            )
        time.sleep(float(os.environ.get("NPA_E2E_WORKFLOW_S3_POLL_SECONDS", "20")))
    pytest.fail(f"workflow did not finish before timeout; evidence={evidence_dir}")


def _kubeconfig_for_context(context: str) -> Path | None:
    configured = os.environ.get("NPA_E2E_KUBECONFIG", "")
    if configured:
        path = Path(configured)
        if not path.exists():
            pytest.skip(f"NPA_E2E_KUBECONFIG does not exist: {path}")
        return path
    cached = Path.home() / ".npa" / "clusters" / context / "kubeconfig"
    return cached if cached.exists() else None


def _assert_kube_context_ready(context: str, kubeconfig: Path | None) -> None:
    kubectl = shutil.which("kubectl")
    if kubectl is None:
        pytest.skip("kubectl is required for durable workflow S3 e2e")
    cmd = [kubectl]
    if kubeconfig is not None:
        cmd.extend(["--kubeconfig", str(kubeconfig)])
    cmd.extend(["--context", context, "get", "nodes", "-o", "json"])
    result = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"Kubernetes context is not ready: {context}")
    payload = json.loads(result.stdout)
    schedulable_gpu_capacity = 0
    for node in payload.get("items", []):
        spec = node.get("spec", {})
        if spec.get("unschedulable"):
            continue
        capacity = node.get("status", {}).get("capacity", {})
        for key, value in capacity.items():
            if "nvidia.com/gpu" in key.lower():
                schedulable_gpu_capacity += int(value)
    required = int(os.environ.get("NPA_E2E_WORKFLOW_S3_MIN_SCHEDULABLE_GPUS", "16"))
    if schedulable_gpu_capacity < required:
        pytest.skip(
            f"{context} has {schedulable_gpu_capacity} schedulable GPUs; requires {required}"
        )


def _sky_bin() -> str:
    sky_bin = os.environ.get("NPA_SKYPILOT_BIN", "/home/ubuntu/.npa/skypilot-venv/bin/sky")
    if not Path(sky_bin).exists():
        pytest.skip(f"SkyPilot binary not found: {sky_bin}")
    return sky_bin


def _npa_bin() -> str:
    configured = os.environ.get("NPA_E2E_CLI_BIN", "")
    if configured:
        return configured
    candidate = Path(sys.executable).with_name("npa")
    if candidate.exists():
        return str(candidate)
    resolved = shutil.which("npa")
    if resolved:
        return resolved
    pytest.skip("npa CLI executable not found")


def _isolated_sky_env(env: dict[str, str], isolated_sky_dir: Path) -> dict[str, str]:
    home = isolated_sky_dir / "home"
    runtime = isolated_sky_dir / "sky-runtime"
    home.mkdir(parents=True, exist_ok=True)
    runtime.mkdir(parents=True, exist_ok=True)
    return {
        **env,
        "HOME": str(home),
        "SKY_RUNTIME_DIR": str(runtime),
        "PYTHONUNBUFFERED": "1",
    }


def _run(
    cmd: list[str],
    *,
    env: dict[str, str],
    cwd: Path,
    timeout: int,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=check,
    )


def _sky_down_and_poll(
    sky_bin: str,
    cluster: str,
    *,
    evidence_dir: Path,
    env: dict[str, str],
) -> None:
    down = subprocess.run(
        [sky_bin, "down", "--yes", cluster],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=int(os.environ.get("NPA_E2E_WORKFLOW_S3_TEARDOWN_TIMEOUT_SECONDS", "900")),
        check=False,
    )
    (evidence_dir / f"{cluster}.down.stdout.txt").write_text(
        redact_text(down.stdout),
        encoding="utf-8",
    )
    (evidence_dir / f"{cluster}.down.stderr.txt").write_text(
        redact_text(down.stderr),
        encoding="utf-8",
    )
    deadline = time.monotonic() + int(
        os.environ.get("NPA_E2E_WORKFLOW_S3_TEARDOWN_POLL_TIMEOUT_SECONDS", "1200")
    )
    while time.monotonic() < deadline:
        status = subprocess.run(
            [sky_bin, "status", "--refresh"],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300,
            check=False,
        )
        combined = status.stdout + status.stderr
        (evidence_dir / f"{cluster}.status-after-down.txt").write_text(
            redact_text(combined),
            encoding="utf-8",
        )
        if cluster not in status.stdout:
            return
        time.sleep(float(os.environ.get("NPA_E2E_WORKFLOW_S3_TEARDOWN_POLL_SECONDS", "30")))
    pytest.fail(f"SkyPilot cluster still present after teardown timeout: {cluster}")


def _get_json(client: Any, bucket: str, key: str) -> dict[str, Any]:
    text = _get_text(client, bucket, key)
    payload = json.loads(text)
    assert isinstance(payload, dict)
    return payload


def _get_text(client: Any, bucket: str, key: str) -> str:
    response = client.get_object(Bucket=bucket, Key=key)
    return response["Body"].read().decode("utf-8")


def _write_command_evidence(
    evidence_dir: Path,
    name: str,
    result: subprocess.CompletedProcess[str],
    cmd: list[str],
    secrets: dict[str, str] | Any,
) -> None:
    payload = {
        "command": _redact_cmd(cmd),
        "returncode": result.returncode,
        "stdout": redact_text(result.stdout, list(secrets.values()) if isinstance(secrets, dict) else None),
        "stderr": redact_text(result.stderr, list(secrets.values()) if isinstance(secrets, dict) else None),
    }
    (evidence_dir / f"{name}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _redact_cmd(cmd: list[str]) -> list[str]:
    redacted: list[str] = []
    for part in cmd:
        if part.endswith("/sky"):
            redacted.append("<sky>")
        else:
            redacted.append(part)
    return redacted


def _write_evidence(
    evidence_dir: Path,
    *,
    run_id: str,
    job_id: str,
    run_uri: str,
    endpoint: str,
    kube_context: str,
    status: dict[str, Any],
    attempts: list[dict[str, Any]],
) -> None:
    payload = {
        "run_id": run_id,
        "sky_job_id": job_id,
        "run_prefix_uri": run_uri,
        "s3_endpoint": endpoint,
        "kube_context": kube_context,
        "status": status,
        "attempts": attempts,
    }
    (evidence_dir / "workflow-durable-s3-evidence.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
