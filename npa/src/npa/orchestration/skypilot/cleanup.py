"""Explicit cleanup helpers for NPA SkyPilot workflows."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from npa.orchestration.skypilot._bin import SkyBin, ensure_skypilot_version, resolve_config
from npa.orchestration.skypilot.controller import DEFAULT_CONTROLLER_BACKEND, ControllerBackend


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
RUN_ID_MIN_LENGTH = 12
_RUN_ID_ALLOWED_RE = re.compile(r"^[A-Za-z0-9-]+$")

# `sky jobs cancel` only *schedules* cancellation: the controller still reports the
# job as CANCELLING for a while afterwards, and `sky down` on the controller refuses
# to run while any managed job is non-terminal. Cancelling and immediately tearing
# down therefore fails with the error below even though the operator did exactly
# what the message asks for. Wait for the cancellation to land instead.
DEFAULT_JOB_DRAIN_TIMEOUT_SECONDS = 300
DEFAULT_JOB_DRAIN_INTERVAL_SECONDS = 5.0
_IN_PROGRESS_JOBS_MARKERS = ("in-progress managed jobs", "in progress managed jobs")


class InvalidRunIdError(ValueError):
    """Raised when a run id is unsafe for SkyPilot cleanup matching."""


def sky_down(
    cluster_name: str,
    *,
    isolated_config_dir: Path | None = None,
    config_path: Path | None = None,
    sky_bin: SkyBin = None,
    timeout: int = 900,
) -> CleanupResult:
    """Run `sky down --yes` for a cluster or SkyPilot glob pattern."""

    runtime_config = resolve_config(
        sky_bin=sky_bin,
        global_config_path=config_path,
        isolated_config_dir=isolated_config_dir,
    )
    cmd = [str(ensure_skypilot_version(runtime_config.sky_bin)), "down", "--yes", cluster_name]
    result = _run(
        cmd,
        isolated_config_dir=runtime_config.isolated_config_dir,
        config_path=runtime_config.global_config_path,
        timeout=timeout,
    )
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
    sky_bin: SkyBin = None,
    job_drain_timeout: int = DEFAULT_JOB_DRAIN_TIMEOUT_SECONDS,
) -> CleanupResult:
    """Tear down the managed-jobs controller in the active SkyPilot state.

    ``sky down`` refuses while any managed job is non-terminal, and a job that was
    only just cancelled still counts, so wait for the queue to drain first.
    """

    cleanup = CleanupResult()
    controller_clusters, status_error = _jobs_controller_clusters(
        isolated_config_dir=isolated_config_dir,
        config_path=config_path,
        sky_bin=sky_bin,
    )
    if status_error:
        cleanup.errors.append(status_error)
        return cleanup
    if controller_clusters:
        pending = _nonterminal_job_ids(
            isolated_config_dir=isolated_config_dir,
            config_path=config_path,
            sky_bin=sky_bin,
        )
        if pending:
            wait_for_jobs_terminal(
                pending,
                isolated_config_dir=isolated_config_dir,
                config_path=config_path,
                sky_bin=sky_bin,
                timeout=job_drain_timeout,
            )
    for controller_cluster in controller_clusters:
        controller_name = _cluster_name(controller_cluster)
        down_result = _down_jobs_controller(
            controller_name,
            isolated_config_dir=isolated_config_dir,
            config_path=config_path,
            sky_bin=sky_bin,
            job_drain_timeout=job_drain_timeout,
        )
        cleanup.extend(down_result)
        if down_result.ok and _is_kubernetes_controller(controller_cluster):
            cleanup.extend(
                _cleanup_lingering_kubernetes_controller_pods(controller_name)
            )
    return cleanup


def cleanup_workflow(
    cluster_or_job_id: str,
    *,
    isolated_config_dir: Path | None = None,
    config_path: Path | None = None,
    sky_bin: SkyBin = None,
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
    sky_bin: SkyBin = None,
    also_teardown_controller: bool = False,
    job_drain_timeout: int = DEFAULT_JOB_DRAIN_TIMEOUT_SECONDS,
) -> CleanupResult:
    """Cancel jobs and tear down this run's clusters.

    The SkyPilot managed-jobs controller is shared operator state. It is left in
    place by default; pass ``also_teardown_controller=True`` only when no other
    SkyPilot-managed work depends on it.
    """

    _validate_run_id(run_id)
    cleanup = CleanupResult()
    cancelled: list[str] = []
    for job in _matching_jobs(
        run_id,
        isolated_config_dir=isolated_config_dir,
        config_path=config_path,
        sky_bin=sky_bin,
    ):
        if str(job.get("status", "")).upper() in NONTERMINAL_JOB_STATUSES:
            job_id = str(job.get("job_id") or job.get("id"))
            cleanup.extend(
                _cancel_job(
                    job_id,
                    isolated_config_dir=isolated_config_dir,
                    config_path=config_path,
                    sky_bin=sky_bin,
                )
            )
            cancelled.append(job_id)

    # `sky jobs cancel` returns as soon as cancellation is scheduled, so tearing
    # down immediately races the controller.
    if cancelled:
        drained, still_running = wait_for_jobs_terminal(
            cancelled,
            isolated_config_dir=isolated_config_dir,
            config_path=config_path,
            sky_bin=sky_bin,
            timeout=job_drain_timeout,
        )
        if not drained:
            cleanup.errors.append(
                "managed job(s) "
                + ", ".join(still_running)
                + f" were still non-terminal {job_drain_timeout}s after cancel; "
                "teardown continued, but `sky down` of the jobs controller may refuse "
                "until they finish cancelling."
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
    if also_teardown_controller:
        cleanup.extend(
            cleanup_jobs_controller(
                isolated_config_dir=isolated_config_dir,
                config_path=config_path,
                sky_bin=sky_bin,
            )
        )
    return cleanup


def cluster_name_patterns_for_run(run_id: str) -> list[str]:
    """Return boundary-aware SkyPilot cluster-name globs for a validated run id."""

    _validate_run_id(run_id)
    tag = run_tag(run_id)
    patterns = [tag, f"{tag}-*"]
    sanitized = _sanitize_name(run_id)
    if sanitized and sanitized != tag:
        patterns.extend([sanitized, f"{sanitized}-*"])
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
    sky_bin: SkyBin = None,
    controller_backend: ControllerBackend = DEFAULT_CONTROLLER_BACKEND,
) -> Iterator["_SkyPilotWorkflow"]:
    """Context manager that guarantees run-scoped SkyPilot cleanup.

    The shared managed-jobs controller is not torn down on context exit.
    """

    workflow = _SkyPilotWorkflow(
        run_id=run_id,
        isolated_config_dir=isolated_config_dir,
        config_path=config_path,
        sky_bin=sky_bin,
        controller_backend=controller_backend,
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
    sky_bin: SkyBin = None
    controller_backend: ControllerBackend = DEFAULT_CONTROLLER_BACKEND
    cleanup_result: CleanupResult | None = None

    def submit(self, yaml_path: Path):
        from npa.orchestration.skypilot.workflow import submit_workflow

        return submit_workflow(
            yaml_path,
            self.run_id,
            isolated_config_dir=self.isolated_config_dir,
            sky_bin=self.sky_bin,
            controller_backend=self.controller_backend,
        )

    def cleanup(self) -> CleanupResult:
        self.cleanup_result = cleanup_all_for_run(
            self.run_id,
            isolated_config_dir=self.isolated_config_dir,
            config_path=self.config_path,
            sky_bin=self.sky_bin,
        )
        return self.cleanup_result


def looks_like_in_progress_jobs_error(detail: str) -> bool:
    """Whether ``sky down`` refused because managed jobs are still non-terminal."""

    lowered = str(detail or "").lower()
    return any(marker in lowered for marker in _IN_PROGRESS_JOBS_MARKERS)


def wait_for_jobs_terminal(
    job_ids: Sequence[str],
    *,
    isolated_config_dir: Path | None = None,
    config_path: Path | None = None,
    sky_bin: SkyBin = None,
    timeout: int = DEFAULT_JOB_DRAIN_TIMEOUT_SECONDS,
    interval: float = DEFAULT_JOB_DRAIN_INTERVAL_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[bool, list[str]]:
    """Block until the given managed jobs are terminal.

    Returns ``(drained, still_running)``. A queue that cannot be read is treated
    as drained: cleanup is best-effort and must not stall on a broken controller.
    """

    wanted = {str(job_id).strip() for job_id in job_ids if str(job_id).strip()}
    if not wanted:
        return True, []
    deadline = time.monotonic() + max(timeout, 0)
    still_running: list[str] = []
    while True:
        jobs, readable = _all_jobs(
            isolated_config_dir=isolated_config_dir,
            config_path=config_path,
            sky_bin=sky_bin,
        )
        if not readable:
            return True, []
        still_running = [
            job_id
            for job_id, status in _job_statuses(jobs).items()
            if job_id in wanted and status in NONTERMINAL_JOB_STATUSES
        ]
        if not still_running:
            return True, []
        if time.monotonic() >= deadline:
            return False, sorted(still_running)
        sleep(max(interval, 0.1))


def _job_statuses(jobs: Sequence[dict[str, Any]]) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for job in jobs or []:
        if not isinstance(job, dict):
            continue
        job_id = str(job.get("job_id") or job.get("id") or "").strip()
        if not job_id:
            continue
        status = str(job.get("status") or "").upper()
        # A job group reports one row per task; the job is only terminal once
        # every one of its rows is.
        if job_id in statuses and statuses[job_id] in NONTERMINAL_JOB_STATUSES:
            continue
        statuses[job_id] = status
    return statuses


def _nonterminal_job_ids(
    *,
    isolated_config_dir: Path | None,
    config_path: Path | None,
    sky_bin: SkyBin,
) -> list[str]:
    jobs, readable = _all_jobs(
        isolated_config_dir=isolated_config_dir,
        config_path=config_path,
        sky_bin=sky_bin,
    )
    if not readable:
        return []
    return sorted(
        job_id
        for job_id, status in _job_statuses(jobs).items()
        if status in NONTERMINAL_JOB_STATUSES
    )


def _all_jobs(
    *,
    isolated_config_dir: Path | None,
    config_path: Path | None,
    sky_bin: SkyBin,
) -> tuple[list[dict[str, Any]], bool]:
    runtime_config = resolve_config(
        sky_bin=sky_bin,
        global_config_path=config_path,
        isolated_config_dir=isolated_config_dir,
    )
    cmd = [
        str(ensure_skypilot_version(runtime_config.sky_bin)),
        "jobs",
        "queue",
        "--all",
        "--output",
        "json",
    ]
    result = _run(
        cmd,
        isolated_config_dir=runtime_config.isolated_config_dir,
        config_path=runtime_config.global_config_path,
        timeout=120,
    )
    if result.returncode != 0:
        return [], False
    try:
        payload = json.loads(result.stdout or "[]")
    except json.JSONDecodeError:
        return [], False
    jobs = payload if isinstance(payload, list) else payload.get("jobs", [])
    return [job for job in (jobs or []) if isinstance(job, dict)], True


def _matching_jobs(
    run_id: str,
    *,
    isolated_config_dir: Path | None,
    config_path: Path | None,
    sky_bin: SkyBin,
) -> list[dict[str, Any]]:
    runtime_config = resolve_config(
        sky_bin=sky_bin,
        global_config_path=config_path,
        isolated_config_dir=isolated_config_dir,
    )
    cmd = [str(ensure_skypilot_version(runtime_config.sky_bin)), "jobs", "queue", "--all", "--output", "json"]
    result = _run(
        cmd,
        isolated_config_dir=runtime_config.isolated_config_dir,
        config_path=runtime_config.global_config_path,
        timeout=120,
    )
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


def _validate_run_id(run_id: str) -> None:
    value = str(run_id)
    if len(value) < RUN_ID_MIN_LENGTH:
        raise InvalidRunIdError(
            f"SkyPilot run_id must be at least {RUN_ID_MIN_LENGTH} characters "
            "before cleanup can derive cluster-name patterns."
        )
    if not _RUN_ID_ALLOWED_RE.fullmatch(value):
        raise InvalidRunIdError(
            "SkyPilot run_id may contain only ASCII letters, digits, and hyphens. "
            "Use a long timestamp or UUID-style suffix and avoid glob, shell, or "
            "Kubernetes-unsafe characters."
        )


def _cancel_job(
    job_id: str,
    *,
    isolated_config_dir: Path | None,
    config_path: Path | None,
    sky_bin: SkyBin,
) -> CleanupResult:
    runtime_config = resolve_config(
        sky_bin=sky_bin,
        global_config_path=config_path,
        isolated_config_dir=isolated_config_dir,
    )
    cmd = [str(ensure_skypilot_version(runtime_config.sky_bin)), "jobs", "cancel", "--yes", job_id]
    result = _run(
        cmd,
        isolated_config_dir=runtime_config.isolated_config_dir,
        config_path=runtime_config.global_config_path,
        timeout=300,
    )
    cleanup = CleanupResult(commands=[cmd])
    if result.returncode == 0:
        cleanup.resources_removed.append(f"job:{job_id}")
    else:
        cleanup.errors.append(_format_command_error(cmd, result))
    return cleanup


def _jobs_controller_names(
    *,
    isolated_config_dir: Path | None,
    config_path: Path | None,
    sky_bin: SkyBin,
) -> tuple[list[str], str]:
    controller_clusters, status_error = _jobs_controller_clusters(
        isolated_config_dir=isolated_config_dir,
        config_path=config_path,
        sky_bin=sky_bin,
    )
    if status_error:
        return [], status_error
    return [_cluster_name(cluster) for cluster in controller_clusters], ""


def _jobs_controller_clusters(
    *,
    isolated_config_dir: Path | None,
    config_path: Path | None,
    sky_bin: SkyBin,
) -> tuple[list[dict[str, Any]], str]:
    runtime_config = resolve_config(
        sky_bin=sky_bin,
        global_config_path=config_path,
        isolated_config_dir=isolated_config_dir,
    )
    cmd = [str(ensure_skypilot_version(runtime_config.sky_bin)), "status", "--refresh", "--output", "json"]
    result = _run(
        cmd,
        isolated_config_dir=runtime_config.isolated_config_dir,
        config_path=runtime_config.global_config_path,
        timeout=300,
    )
    if result.returncode != 0:
        return [], _format_command_error(cmd, result)
    try:
        payload = json.loads(result.stdout or "[]")
    except json.JSONDecodeError:
        return [], f"{' '.join(cmd)} returned non-json output"
    clusters = payload if isinstance(payload, list) else payload.get("clusters", [])
    controllers = []
    for cluster in clusters or []:
        if not isinstance(cluster, dict):
            continue
        name = _cluster_name(cluster)
        if name.startswith("sky-jobs-controller-"):
            controllers.append(cluster)
    deduped: dict[str, dict[str, Any]] = {}
    for controller in controllers:
        deduped.setdefault(_cluster_name(controller), controller)
    return list(deduped.values()), ""


def _cluster_name(cluster: dict[str, Any]) -> str:
    return str(cluster.get("name") or cluster.get("cluster") or "")


def _down_jobs_controller(
    controller_name: str,
    *,
    isolated_config_dir: Path | None,
    config_path: Path | None,
    sky_bin: SkyBin,
    job_drain_timeout: int = DEFAULT_JOB_DRAIN_TIMEOUT_SECONDS,
) -> CleanupResult:
    runtime_config = resolve_config(
        sky_bin=sky_bin,
        global_config_path=config_path,
        isolated_config_dir=isolated_config_dir,
    )
    cmd = [str(ensure_skypilot_version(runtime_config.sky_bin)), "down", "--yes", controller_name]
    cleanup = CleanupResult(commands=[cmd])
    result = _run(
        cmd,
        isolated_config_dir=runtime_config.isolated_config_dir,
        config_path=runtime_config.global_config_path,
        timeout=900,
        input_text="delete\n",
    )
    if result.returncode != 0 and looks_like_in_progress_jobs_error(
        _combined_output(result)
    ):
        # A job that finished cancelling between our poll and this call still
        # trips the guard; drain once more and retry rather than reporting a
        # failure the operator can only fix by waiting and rerunning.
        pending = _nonterminal_job_ids(
            isolated_config_dir=isolated_config_dir,
            config_path=config_path,
            sky_bin=sky_bin,
        )
        drained, still_running = wait_for_jobs_terminal(
            pending,
            isolated_config_dir=isolated_config_dir,
            config_path=config_path,
            sky_bin=sky_bin,
            timeout=job_drain_timeout,
        )
        if drained:
            cleanup.commands.append(cmd)
            result = _run(
                cmd,
                isolated_config_dir=runtime_config.isolated_config_dir,
                config_path=runtime_config.global_config_path,
                timeout=900,
                input_text="delete\n",
            )
        elif still_running:
            cleanup.errors.append(
                f"`sky down {controller_name}` refuses while managed job(s) "
                + ", ".join(still_running)
                + " are still non-terminal, and they did not finish cancelling within "
                f"{job_drain_timeout}s. Suggested action: `sky jobs cancel -a`, wait for "
                "`sky jobs queue --all` to show them terminal, then rerun teardown."
            )
            return cleanup
    if result.returncode == 0:
        cleanup.resources_removed.append(controller_name)
    else:
        cleanup.errors.append(_format_command_error(cmd, result))
    return cleanup


def _is_kubernetes_controller(cluster: dict[str, Any]) -> bool:
    return "kubernetes" in json.dumps(cluster, sort_keys=True).lower()


def _cleanup_lingering_kubernetes_controller_pods(controller_name: str) -> CleanupResult:
    cleanup = CleanupResult()
    pods, error = _matching_kubernetes_controller_pods(controller_name)
    if error:
        cleanup.errors.append(f"NOVEL_ISSUE: unable to verify Kubernetes controller pod cleanup: {error}")
        return cleanup
    for namespace, pod_name in pods:
        cmd = ["kubectl", "delete", "pod", "-n", namespace, pod_name, "--ignore-not-found=true"]
        result = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180,
            check=False,
        )
        cleanup.commands.append(cmd)
        if result.returncode == 0:
            cleanup.resources_removed.append(f"k8s-pod:{namespace}/{pod_name}")
        else:
            cleanup.errors.append(
                "NOVEL_ISSUE: lingering Kubernetes controller pod deletion failed: "
                + _format_command_error(cmd, result)
            )
    return cleanup


def _matching_kubernetes_controller_pods(controller_name: str) -> tuple[list[tuple[str, str]], str]:
    if shutil.which("kubectl") is None:
        return [], "kubectl not found"
    cmd = ["kubectl", "get", "pods", "--all-namespaces", "-o", "json"]
    try:
        result = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return [], "kubectl get pods timed out"
    if result.returncode != 0:
        return [], _format_command_error(cmd, result)
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return [], "kubectl get pods returned non-json output"
    matches: list[tuple[str, str]] = []
    for item in payload.get("items", []):
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata") or {}
        if not isinstance(metadata, dict):
            continue
        namespace = str(metadata.get("namespace") or "default")
        pod_name = str(metadata.get("name") or "")
        labels = metadata.get("labels") if isinstance(metadata.get("labels"), dict) else {}
        searchable = " ".join(
            [
                pod_name,
                str(labels.get("skypilot-cluster", "")),
                str(labels.get("ray.io/cluster", "")),
                str(labels.get("app.kubernetes.io/name", "")),
                str(labels.get("component", "")),
            ]
        )
        if controller_name in searchable:
            matches.append((namespace, pod_name))
    return matches, ""


def _run(
    cmd: list[str],
    *,
    isolated_config_dir: Path | None,
    config_path: Path | None,
    timeout: int,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    effective_cmd = list(cmd)
    if config_path is not None and "--config" not in effective_cmd:
        command_name_index = 2 if len(effective_cmd) > 1 and effective_cmd[1] == "jobs" else 1
        effective_cmd[command_name_index + 1:command_name_index + 1] = ["--config", str(config_path)]
    return subprocess.run(
        effective_cmd,
        env=sky_environment(isolated_config_dir),
        text=True,
        input=input_text,
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
    detail = _combined_output(result) or f"exit {result.returncode}"
    return f"{' '.join(cmd)} failed: {detail}"


def _combined_output(result: subprocess.CompletedProcess[str]) -> str:
    return (result.stderr or "").strip() or (result.stdout or "").strip()
