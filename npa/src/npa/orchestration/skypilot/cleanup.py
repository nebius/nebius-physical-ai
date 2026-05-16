"""Explicit cleanup helpers for NPA SkyPilot workflows."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CleanupResult:
    """Result of an explicit SkyPilot cleanup operation."""

    resources_removed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    commands: list[list[str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def extend(self, other: "CleanupResult") -> None:
        self.resources_removed.extend(other.resources_removed)
        self.errors.extend(other.errors)
        self.commands.extend(other.commands)


NONTERMINAL_JOB_STATUSES = {"PENDING", "STARTING", "RUNNING", "RECOVERING", "CANCELLING"}
JOBS_CONTROLLER_PATTERN = "sky-jobs-controller-*"


def sky_down(
    cluster_name: str,
    *,
    isolated_config_dir: Path | None = None,
    config_path: Path | None = None,
    sky_bin: str = "sky",
    timeout: int = 900,
) -> CleanupResult:
    """Run `sky down --yes` for a cluster or SkyPilot glob pattern."""

    cmd = [sky_bin, "down", "--yes", cluster_name]
    result = _run(cmd, isolated_config_dir=isolated_config_dir, config_path=config_path, timeout=timeout)
    cleanup = CleanupResult(commands=[cmd])
    if result.returncode == 0:
        cleanup.resources_removed.append(cluster_name)
    else:
        cleanup.errors.append(_format_command_error(cmd, result))
    return cleanup


def cleanup_jobs_controller(
    *,
    isolated_config_dir: Path | None = None,
    config_path: Path | None = None,
    sky_bin: str = "sky",
) -> CleanupResult:
    """Tear down the managed-jobs controller VM in the active SkyPilot state."""

    return sky_down(
        JOBS_CONTROLLER_PATTERN,
        isolated_config_dir=isolated_config_dir,
        config_path=config_path,
        sky_bin=sky_bin,
    )


def cleanup_workflow(
    cluster_or_job_id: str,
    *,
    isolated_config_dir: Path | None = None,
    config_path: Path | None = None,
    sky_bin: str = "sky",
) -> CleanupResult:
    """Cancel a managed job ID or tear down a SkyPilot cluster name."""

    if cluster_or_job_id.isdigit():
        return _cancel_job(
            cluster_or_job_id,
            isolated_config_dir=isolated_config_dir,
            config_path=config_path,
            sky_bin=sky_bin,
        )
    return sky_down(
        cluster_or_job_id,
        isolated_config_dir=isolated_config_dir,
        config_path=config_path,
        sky_bin=sky_bin,
    )


def cleanup_all_for_run(
    run_id: str,
    *,
    isolated_config_dir: Path | None = None,
    config_path: Path | None = None,
    sky_bin: str = "sky",
) -> CleanupResult:
    """Cancel jobs and tear down clusters matching this run's naming pattern."""

    cleanup = CleanupResult()
    for job in _matching_jobs(
        run_id,
        isolated_config_dir=isolated_config_dir,
        config_path=config_path,
        sky_bin=sky_bin,
    ):
        if str(job.get("status", "")).upper() in NONTERMINAL_JOB_STATUSES:
            cleanup.extend(
                _cancel_job(
                    str(job.get("job_id") or job.get("id")),
                    isolated_config_dir=isolated_config_dir,
                    config_path=config_path,
                    sky_bin=sky_bin,
                )
            )

    for pattern in cluster_name_patterns_for_run(run_id):
        cleanup.extend(
            sky_down(
                pattern,
                isolated_config_dir=isolated_config_dir,
                config_path=config_path,
                sky_bin=sky_bin,
            )
        )
    cleanup.extend(
        cleanup_jobs_controller(
            isolated_config_dir=isolated_config_dir,
            config_path=config_path,
            sky_bin=sky_bin,
        )
    )
    return cleanup


def cluster_name_patterns_for_run(run_id: str) -> list[str]:
    """Return SkyPilot cluster-name globs derived from a run id."""

    tag = run_tag(run_id)
    patterns = [f"{tag}*", f"*{tag}*"]
    sanitized = _sanitize_name(run_id)
    if sanitized and sanitized != tag:
        patterns.append(f"{sanitized}*")
    return list(dict.fromkeys(patterns))


def run_tag(run_id: str, *, max_length: int = 32) -> str:
    """Return a Kubernetes/SkyPilot-safe short tag for cluster/task names."""

    sanitized = _sanitize_name(run_id)
    if len(sanitized) <= max_length:
        return sanitized
    return sanitized[-max_length:].strip("-") or sanitized[:max_length].strip("-")


@contextmanager
def skypilot_workflow(
    *,
    run_id: str,
    isolated_config_dir: Path | None = None,
    config_path: Path | None = None,
    sky_bin: str = "sky",
) -> Iterator["_SkyPilotWorkflow"]:
    """Context manager that guarantees explicit SkyPilot cleanup."""

    workflow = _SkyPilotWorkflow(
        run_id=run_id,
        isolated_config_dir=isolated_config_dir,
        config_path=config_path,
        sky_bin=sky_bin,
    )
    try:
        yield workflow
    finally:
        workflow.cleanup()


@dataclass
class _SkyPilotWorkflow:
    run_id: str
    isolated_config_dir: Path | None = None
    config_path: Path | None = None
    sky_bin: str = "sky"
    cleanup_result: CleanupResult | None = None

    def submit(self, yaml_path: Path):
        from npa.orchestration.skypilot.workflow import submit_workflow

        return submit_workflow(
            yaml_path,
            self.run_id,
            isolated_config_dir=self.isolated_config_dir,
            sky_bin=self.sky_bin,
        )

    def cleanup(self) -> CleanupResult:
        self.cleanup_result = cleanup_all_for_run(
            self.run_id,
            isolated_config_dir=self.isolated_config_dir,
            config_path=self.config_path,
            sky_bin=self.sky_bin,
        )
        return self.cleanup_result


def _matching_jobs(
    run_id: str,
    *,
    isolated_config_dir: Path | None,
    config_path: Path | None,
    sky_bin: str,
) -> list[dict[str, Any]]:
    cmd = [sky_bin, "jobs", "queue", "--all", "--output", "json"]
    result = _run(cmd, isolated_config_dir=isolated_config_dir, config_path=config_path, timeout=120)
    if result.returncode != 0:
        return []
    try:
        payload = json.loads(result.stdout or "[]")
    except json.JSONDecodeError:
        return []
    jobs = payload if isinstance(payload, list) else payload.get("jobs", [])
    patterns = {run_id, run_tag(run_id), _sanitize_name(run_id)}
    matched = []
    for job in jobs or []:
        text = " ".join(str(job.get(key, "")) for key in ("name", "job_name", "task", "job_id", "id"))
        if any(pattern and pattern in text for pattern in patterns):
            matched.append(job)
    return matched


def _cancel_job(
    job_id: str,
    *,
    isolated_config_dir: Path | None,
    config_path: Path | None,
    sky_bin: str,
) -> CleanupResult:
    cmd = [sky_bin, "jobs", "cancel", "--yes", job_id]
    result = _run(cmd, isolated_config_dir=isolated_config_dir, config_path=config_path, timeout=300)
    cleanup = CleanupResult(commands=[cmd])
    if result.returncode == 0:
        cleanup.resources_removed.append(f"job:{job_id}")
    else:
        cleanup.errors.append(_format_command_error(cmd, result))
    return cleanup


def _run(
    cmd: list[str],
    *,
    isolated_config_dir: Path | None,
    config_path: Path | None,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    effective_cmd = list(cmd)
    if config_path is not None and "--config" not in effective_cmd:
        command_name_index = 2 if len(effective_cmd) > 1 and effective_cmd[1] == "jobs" else 1
        effective_cmd[command_name_index + 1:command_name_index + 1] = ["--config", str(config_path)]
    return subprocess.run(
        effective_cmd,
        env=sky_environment(isolated_config_dir),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def sky_environment(isolated_config_dir: Path | None = None) -> dict[str, str]:
    """Return an environment that keeps SkyPilot state inside a run directory."""

    env = os.environ.copy()
    if isolated_config_dir is None:
        return env
    root = Path(isolated_config_dir)
    home = root / "home"
    runtime = root / "sky-runtime"
    home.mkdir(parents=True, exist_ok=True)
    runtime.mkdir(parents=True, exist_ok=True)
    env["HOME"] = str(home)
    env["SKY_RUNTIME_DIR"] = str(runtime)
    env["PYTHONUNBUFFERED"] = "1"
    repo_src = Path(__file__).resolve().parents[3]
    env["PYTHONPATH"] = str(repo_src) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _sanitize_name(value: str) -> str:
    sanitized = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
    sanitized = re.sub(r"-+", "-", sanitized)
    return sanitized


def _format_command_error(cmd: list[str], result: subprocess.CompletedProcess[str]) -> str:
    stderr = result.stderr.strip()
    stdout = result.stdout.strip()
    detail = stderr or stdout or f"exit {result.returncode}"
    return f"{' '.join(cmd)} failed: {detail}"
