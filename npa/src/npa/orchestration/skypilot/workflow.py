"""Workflow submission helpers for NPA's SkyPilot orchestration layer."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import yaml

from npa.orchestration.skypilot._bin import (
    SkyBin,
    SkyPilotConfigError,
    SkyPilotNotInstalledError,
    SkyPilotVersionError,
    ensure_skypilot_version,
    resolve_config,
)
from npa.orchestration.skypilot.cleanup import sky_environment
from npa.orchestration.skypilot.controller import (
    DEFAULT_CONTROLLER_BACKEND,
    ControllerBackend,
    apply_controller_override,
)


@dataclass
class WorkflowResult:
    """Result of submitting or querying a SkyPilot managed workflow."""

    status: str
    job_id: str = ""
    log_paths: dict[str, str] = field(default_factory=dict)
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    submitted_yaml_path: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.error


class SkyPilotSubmitError(RuntimeError):
    """Raised when a SkyPilot workflow cannot be submitted."""


def submit_workflow(
    yaml_path: Path,
    run_id: str,
    *,
    isolated_config_dir: Path | None = None,
    config_path: Path | None = None,
    sky_bin: SkyBin = None,
    controller_backend: ControllerBackend = DEFAULT_CONTROLLER_BACKEND,
    secret_envs: Sequence[str] | None = None,
    timeout: int = 1800,
) -> WorkflowResult:
    """Submit a SkyPilot YAML through NPA's controller convention."""

    yaml_path = Path(yaml_path)
    submission_dir: Path | None = None
    owned_submission_dir: Path | None = None
    prepared_yaml: Path | None = None
    try:
        runtime_config = resolve_config(
            sky_bin=sky_bin,
            global_config_path=config_path,
            isolated_config_dir=isolated_config_dir,
        )
        docs = _load_yaml_documents(yaml_path)
        if not docs:
            raise ValueError("SkyPilot YAML is empty")
        submission_dir = _submission_dir(run_id, runtime_config.isolated_config_dir)
        if runtime_config.isolated_config_dir is None:
            owned_submission_dir = submission_dir
        prepared_yaml = submission_dir / "workflow.yaml"
        shutil.copy2(yaml_path, prepared_yaml)
        sky_executable = str(ensure_skypilot_version(runtime_config.sky_bin))
        global_config = apply_controller_override(
            _load_base_config(runtime_config.global_config_path),
            controller_backend=controller_backend,
        )
        generated_config_path = submission_dir / "skypilot-config.yaml"
        generated_config_path.write_text(yaml.safe_dump(global_config, sort_keys=False), encoding="utf-8")

        cmd = [
            sky_executable,
            "jobs",
            "launch",
            "--name",
            run_id,
            "--detach-run",
            "--yes",
            str(prepared_yaml),
        ]
        for secret_name in secret_envs or ():
            if os.environ.get(secret_name):
                cmd[-1:-1] = ["--secret", secret_name]
        env = sky_environment(runtime_config.isolated_config_dir)
        env["SKYPILOT_GLOBAL_CONFIG"] = str(generated_config_path)
        result = subprocess.run(
            cmd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            _cleanup_owned_submission_dir(owned_submission_dir)
            raise SkyPilotSubmitError(_format_submit_error(cmd, result))
        combined = f"{result.stdout}\n{result.stderr}"
        job_id = _parse_job_id(combined)
        return WorkflowResult(
            status="SUBMITTED",
            job_id=job_id,
            log_paths={"submission_dir": str(submission_dir), "config": str(generated_config_path)},
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            submitted_yaml_path=str(prepared_yaml),
        )
    except subprocess.TimeoutExpired as exc:
        _cleanup_owned_submission_dir(owned_submission_dir)
        raise SkyPilotSubmitError(f"sky jobs launch timed out after {timeout}s") from exc
    except (
        OSError,
        ValueError,
        yaml.YAMLError,
        SkyPilotConfigError,
        SkyPilotNotInstalledError,
        SkyPilotVersionError,
    ) as exc:
        _cleanup_owned_submission_dir(owned_submission_dir)
        raise SkyPilotSubmitError(f"SkyPilot workflow submission failed: {exc}") from exc


def workflow_status(
    job_id: str,
    *,
    isolated_config_dir: Path | None = None,
    config_path: Path | None = None,
    sky_bin: SkyBin = None,
    controller_backend: ControllerBackend = DEFAULT_CONTROLLER_BACKEND,
    timeout: int = 300,
) -> WorkflowResult:
    """Query a SkyPilot managed job status via `sky jobs queue`."""

    del controller_backend
    runtime_config = resolve_config(
        sky_bin=sky_bin,
        global_config_path=config_path,
        isolated_config_dir=isolated_config_dir,
    )
    cmd = [str(ensure_skypilot_version(runtime_config.sky_bin)), "jobs", "queue", "--all", "--output", "json"]
    if runtime_config.global_config_path is not None:
        cmd[3:3] = ["--config", str(runtime_config.global_config_path)]
    result = subprocess.run(
        cmd,
        env=sky_environment(runtime_config.isolated_config_dir),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        return WorkflowResult(
            status="UNKNOWN",
            job_id=job_id,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            error=result.stderr.strip() or result.stdout.strip(),
        )

    status = _status_from_queue_payload(result.stdout, job_id)
    return WorkflowResult(
        status=status or "UNKNOWN",
        job_id=job_id,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def _load_yaml_documents(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        docs = [doc for doc in yaml.safe_load_all(handle) if doc is not None]
    if not all(isinstance(doc, dict) for doc in docs):
        raise ValueError("SkyPilot YAML documents must be mappings")
    return docs


def _load_base_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"SkyPilot global config must be a mapping: {config_path}")
    return data


def _submission_dir(run_id: str, isolated_config_dir: Path | None) -> Path:
    if isolated_config_dir is None:
        # Successful submissions return this path for debugging; exception paths
        # remove it via _cleanup_owned_submission_dir.
        root = Path(tempfile.mkdtemp(prefix=f"npa-skypilot-{run_id}-"))
    else:
        root = Path(isolated_config_dir) / "submissions" / run_id
        root.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _parse_job_id(output: str) -> str:
    for pattern in (r"Job submitted,\s*ID:\s*([0-9]+)", r"Managed Job ID:\s*([0-9]+)"):
        match = re.search(pattern, output)
        if match:
            return match.group(1)
    return ""


def _cleanup_owned_submission_dir(path: Path | None) -> None:
    if path is not None:
        shutil.rmtree(path, ignore_errors=True)


def _format_submit_error(cmd: list[str], result: subprocess.CompletedProcess[str]) -> str:
    detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
    prefix = "SkyPilot auth failure during jobs launch" if _looks_like_auth_error(detail) else "sky jobs launch failed"
    return f"{prefix}: {' '.join(cmd)}: {detail}"


def _looks_like_auth_error(detail: str) -> bool:
    normalized = detail.lower()
    return any(token in normalized for token in ("auth", "credential", "unauthorized", "forbidden", "permission denied", "401", "403"))


def _status_from_queue_payload(output: str, job_id: str) -> str:
    try:
        payload = json.loads(output or "[]")
    except json.JSONDecodeError:
        return ""
    jobs = payload if isinstance(payload, list) else payload.get("jobs", [])
    statuses = []
    for job in jobs or []:
        current_id = str(job.get("job_id") or job.get("id") or "")
        if current_id == str(job_id):
            status = str(job.get("status", "")).upper()
            if status:
                statuses.append(status)
    if not statuses:
        return ""
    for status in statuses:
        if status.startswith("FAILED") or status == "CANCELLED":
            return status
    if all(status == "SUCCEEDED" for status in statuses):
        return "SUCCEEDED"
    for status in ("RUNNING", "RECOVERING", "STARTING", "PENDING", "CANCELLING"):
        if status in statuses:
            return status
    return statuses[0]
