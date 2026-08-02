"""Explain why a SkyPilot managed job is sitting in PENDING.

A managed job whose worker pod cannot start never becomes FAILED: Kubernetes
retries image pulls and rescheduling indefinitely, and SkyPilot keeps reporting
the job as PENDING. A run blocked this way burns wall-clock until somebody
notices and cancels it by hand -- 14 hours, in the case this module was written
for.

SkyPilot itself has nothing more to say, so the answer has to come from the pods
it created. They carry ``skypilot-cluster-name=<cluster>``, which the managed-job
queue already reports, so the blocked container's own waiting reason is one
kubectl call away.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import json
import subprocess

CLUSTER_LABEL = "skypilot-cluster-name"
DEFAULT_TIMEOUT_SECONDS = 60

# Container waiting reasons that Kubernetes will retry forever.
_TERMINAL_INTENT_REASONS = {
    "ImagePullBackOff": (
        "the image could not be pulled. Kubernetes retries this forever, so the job "
        "stays PENDING instead of failing. Confirm the run's identity is allowed to "
        "pull that exact image reference -- being able to list a repository's tags is "
        "a different permission from pulling it."
    ),
    "ErrImagePull": (
        "the image pull failed. Check the image reference and the credentials the run "
        "injects; Kubernetes will keep retrying rather than failing the job."
    ),
    "InvalidImageName": (
        "the image reference is malformed. Fix the image pinned by the workflow spec "
        "or the --image override."
    ),
    "CreateContainerConfigError": (
        "the container config is invalid -- usually a missing Secret or ConfigMap "
        "referenced by the pod."
    ),
    "CrashLoopBackOff": (
        "the container starts and immediately exits. Read the pod logs for the real "
        "error; the job will keep restarting."
    ),
}

_UNSCHEDULABLE_REMEDY = (
    "no node can satisfy the pod's resource request. Check the requested accelerator "
    "name and per-node GPU count against what the cluster actually advertises -- "
    "SkyPilot places all GPUs of one task on a single node."
)


@dataclass(frozen=True)
class PodBlocker:
    """One pod that cannot start, and why."""

    pod: str
    phase: str
    reason: str
    message: str = ""

    def render(self) -> str:
        detail = f"{self.pod}: {self.reason}"
        if self.phase and self.phase.lower() != "pending":
            detail = f"{detail} (phase {self.phase})"
        if self.message:
            detail = f"{detail} - {self.message}"
        return detail


@dataclass
class JobBlockerReport:
    """Why a managed job is not progressing, as seen from its pods."""

    job_id: str = ""
    cluster_name: str = ""
    blockers: list[PodBlocker] = field(default_factory=list)
    error: str = ""

    @property
    def blocked(self) -> bool:
        return bool(self.blockers)

    def remedy(self) -> str:
        for blocker in self.blockers:
            explanation = _TERMINAL_INTENT_REASONS.get(blocker.reason)
            if explanation:
                return explanation
            if blocker.reason == "Unschedulable":
                return _UNSCHEDULABLE_REMEDY
        return ""

    def render(self) -> str:
        if self.error:
            return f"blockers: unavailable ({self.error})"
        if not self.blockers:
            return "blockers: none found"
        lines = [f"blockers ({len(self.blockers)}):"]
        lines.extend(f"  {blocker.render()}" for blocker in self.blockers)
        remedy = self.remedy()
        if remedy:
            lines.append(f"  Suggested action: {remedy}")
        return "\n".join(lines)


Runner = Callable[..., subprocess.CompletedProcess[str]]


def inspect_job_blockers(
    *,
    job_id: str = "",
    cluster_name: str = "",
    namespace: str = "",
    context: str = "",
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    runner: Runner | None = None,
) -> JobBlockerReport:
    """Return the pod-level reasons a managed job is not starting.

    A cluster whose pods cannot be listed is reported as an error rather than as
    "not blocked", so a missing kubectl is never mistaken for a healthy job.
    """

    report = JobBlockerReport(job_id=str(job_id), cluster_name=str(cluster_name))
    if not cluster_name.strip():
        report.error = "managed job has no cluster yet; nothing has been scheduled"
        return report

    cmd = ["kubectl", "get", "pods", "-l", f"{CLUSTER_LABEL}={cluster_name.strip()}", "-o", "json"]
    if context.strip():
        cmd[1:1] = ["--context", context.strip()]
    if namespace.strip():
        cmd.extend(["-n", namespace.strip()])
    execute = runner or subprocess.run
    try:
        result = execute(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        report.error = f"could not run kubectl: {exc}"
        return report
    if result.returncode != 0:
        report.error = (result.stderr or result.stdout or f"exit {result.returncode}").strip()
        return report
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        report.error = "kubectl returned non-json output"
        return report

    report.blockers = _blockers_from_pods(payload.get("items") or [])
    return report


def _blockers_from_pods(items: list[object]) -> list[PodBlocker]:
    blockers: list[PodBlocker] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(((item.get("metadata") or {}) if isinstance(item.get("metadata"), dict) else {}).get("name") or "")
        status = item.get("status") if isinstance(item.get("status"), dict) else {}
        phase = str(status.get("phase") or "")
        if phase in {"Running", "Succeeded"}:
            continue
        blocker = _container_blocker(name, phase, status) or _scheduling_blocker(name, phase, status)
        if blocker is not None:
            blockers.append(blocker)
    return blockers


def _container_blocker(name: str, phase: str, status: dict) -> PodBlocker | None:
    container_lists = (
        status.get("containerStatuses") or [],
        status.get("initContainerStatuses") or [],
    )
    for containers in container_lists:
        if not isinstance(containers, list):
            continue
        for container in containers:
            if not isinstance(container, dict):
                continue
            waiting = (container.get("state") or {}).get("waiting") or {}
            reason = str(waiting.get("reason") or "")
            # ContainerCreating is normal progress, not a blocker.
            if reason and reason != "ContainerCreating":
                return PodBlocker(
                    pod=name,
                    phase=phase,
                    reason=reason,
                    message=str(waiting.get("message") or "").strip(),
                )
    return None


def _scheduling_blocker(name: str, phase: str, status: dict) -> PodBlocker | None:
    conditions = status.get("conditions") or []
    if not isinstance(conditions, list):
        return None
    for condition in conditions:
        if not isinstance(condition, dict):
            continue
        if str(condition.get("type")) != "PodScheduled":
            continue
        if str(condition.get("status")) == "True":
            continue
        return PodBlocker(
            pod=name,
            phase=phase,
            reason=str(condition.get("reason") or "Unschedulable"),
            message=str(condition.get("message") or "").strip(),
        )
    return None
