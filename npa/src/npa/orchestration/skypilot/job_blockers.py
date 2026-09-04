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
from typing import Any

from npa.verification import sanitize_reason, utc_now

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

_NODES_LOST_REMEDY = (
    "the nodes this job needs are gone, not busy. Preemptible GPU nodes are reclaimed "
    "without warning and their kubelets go NotReady, which SkyPilot reports only as a "
    "job that never leaves PENDING. Wait for the node group to reprovision, or rerun on "
    "on-demand capacity (`npa cluster up --on-demand`). CPU-only stages should not depend "
    "on a preemptible GPU pool."
)


@dataclass(frozen=True)
class PodBlocker:
    """One pod that cannot start, and why."""

    pod: str
    phase: str
    reason: str
    message: str = ""
    reason_code: str = "PENDING_UNKNOWN"
    source: str = "kubernetes_pod_condition"
    observed_at: str = ""
    live: bool = True

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
    unready_nodes: list[str] = field(default_factory=list)
    error: str = ""
    error_code: str = ""
    observed_at: str = ""

    @property
    def blocked(self) -> bool:
        return bool(self.blockers or self.unready_nodes)

    def remedy(self) -> str:
        if self.unready_nodes and not self.blockers:
            return _NODES_LOST_REMEDY
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
        if self.unready_nodes and not self.blockers:
            return (
                f"blockers: {len(self.unready_nodes)} node(s) not Ready: "
                + ", ".join(self.unready_nodes)
                + f"\n  Suggested action: {_NODES_LOST_REMEDY}"
            )
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

    ``sky jobs queue`` frequently reports ``cluster_name_on_cloud`` as null -- a
    job that never provisioned has no cluster recorded, which is exactly the case
    worth diagnosing. SkyPilot names the pod label ``<task>-<job_id>-<hash>``, so
    fall back to selecting on the job id when no cluster name is known.

    A cluster whose pods cannot be listed is reported as an error rather than as
    "not blocked", so a missing kubectl is never mistaken for a healthy job.
    """

    report = JobBlockerReport(job_id=str(job_id), cluster_name=str(cluster_name))
    report.observed_at = utc_now()
    by_job_id = not cluster_name.strip()
    if by_job_id and not str(job_id).strip():
        report.error = "no cluster name or job id to look up"
        return report

    cmd = ["kubectl", "get", "pods", "-o", "json"]
    if by_job_id:
        # Every SkyPilot pod carries the label; the value is filtered below.
        cmd[3:3] = ["-l", CLUSTER_LABEL]
    else:
        cmd[3:3] = ["-l", f"{CLUSTER_LABEL}={cluster_name.strip()}"]
    if context.strip():
        cmd[1:1] = ["--context", context.strip()]
    if namespace.strip():
        cmd.extend(["-n", namespace.strip()])
    else:
        # SkyPilot's namespace is configurable, so do not assume the context default.
        cmd.append("--all-namespaces")
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
        report.error = sanitize_reason(f"could not run kubectl: {exc}")
        report.error_code = _diagnostic_error_code(report.error)
        return report
    if result.returncode != 0:
        report.error = sanitize_reason(
            result.stderr or result.stdout or f"exit {result.returncode}"
        )
        report.error_code = _diagnostic_error_code(report.error)
        return report
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        report.error = "kubectl returned non-json output"
        report.error_code = "KUBERNETES_RESPONSE_UNPARSEABLE"
        return report

    items = payload.get("items") or []
    if by_job_id:
        items = [item for item in items if _pod_belongs_to_job(item, str(job_id))]
        if not items:
            report.error = (
                f"no pods found for managed job {job_id}; it is between tasks, or "
                "nothing has been scheduled yet"
            )
            report.error_code = "KUBERNETES_PODS_NOT_FOUND"
            return report
    report.blockers = _blockers_from_pods(items)
    pod_names = {
        str(_as_dict(item.get("metadata")).get("name") or "")
        for item in items
        if isinstance(item, dict)
    }
    report.blockers.extend(
        blocker
        for blocker in _event_blockers(
            pod_names=pod_names,
            namespace=namespace,
            context=context,
            timeout=timeout,
            runner=execute,
        )
        if not any(
            existing.pod == blocker.pod and existing.reason_code == blocker.reason_code
            for existing in report.blockers
        )
    )
    if not report.blockers:
        # A pod pending because its node vanished has no waiting reason of its own;
        # the cause is on that pod's assigned node. Do not classify an unrelated
        # NotReady node elsewhere in the cluster as evidence for this exact job.
        assigned_nodes = {
            str(_as_dict(item.get("spec")).get("nodeName") or "")
            for item in items
            if isinstance(item, dict)
            and str(_as_dict(item.get("spec")).get("nodeName") or "")
        }
        report.unready_nodes = _unready_nodes(
            context=context,
            timeout=timeout,
            runner=execute,
            assigned_nodes=assigned_nodes,
        )
    return report


def _unready_nodes(
    *,
    context: str,
    timeout: int,
    runner: Runner,
    assigned_nodes: set[str],
) -> list[str]:
    """Return assigned nodes whose kubelet is not Ready."""

    if not assigned_nodes:
        return []

    cmd = ["kubectl", "get", "nodes", "-o", "json"]
    if context.strip():
        cmd[1:1] = ["--context", context.strip()]
    try:
        result = runner(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return []
    unready: list[str] = []
    for item in payload.get("items") or []:
        if not isinstance(item, dict):
            continue
        name = str(_as_dict(item.get("metadata")).get("name") or "")
        if name not in assigned_nodes:
            continue
        conditions = _as_dict(item.get("status")).get("conditions") or []
        if not isinstance(conditions, list):
            continue
        for condition in conditions:
            if not isinstance(condition, dict):
                continue
            if (
                str(condition.get("type")) == "Ready"
                and str(condition.get("status")) != "True"
            ):
                reason = str(condition.get("reason") or "NotReady")
                unready.append(f"{name} ({reason})")
                break
    return sorted(unready)


def _as_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def classify_pending_reason(
    reason: str, message: str = "", *, source: str = "container"
) -> str:
    """Return a stable, actionable pending/failure reason code."""

    normalized = str(reason or "").lower()
    detail = str(message or "").lower()
    combined = f"{normalized} {detail}"
    if "unschedul" in combined or "failedscheduling" in combined:
        if any(item in combined for item in ("gpu", "accelerator", "nvidia.com/gpu")):
            return "ACCELERATOR_MISMATCH"
        if any(
            item in combined
            for item in ("quota", "capacity", "insufficient", "no nodes")
        ):
            return "CAPACITY_OR_QUOTA"
        if any(item in combined for item in ("persistentvolumeclaim", "pvc", "volume")):
            return "STORAGE_PENDING"
        return "UNSCHEDULABLE"
    if any(item in normalized for item in ("imagepullbackoff", "errimagepull")):
        if any(
            item in combined
            for item in ("unauthorized", "authentication required", "denied", "401")
        ):
            return "IMAGE_PULL_AUTH"
        if any(item in combined for item in ("not found", "manifest unknown", "404")):
            return "IMAGE_NOT_FOUND"
        return "IMAGE_PULL_FAILED"
    if "invalidimagename" in normalized:
        return "IMAGE_REFERENCE_INVALID"
    if "createcontainerconfigerror" in normalized:
        if "secret" in combined:
            return "MISSING_SECRET"
        if "configmap" in combined:
            return "MISSING_CONFIGMAP"
        return "CREATE_CONTAINER_CONFIG_ERROR"
    # A waiting reason of "PodInitializing" on a main container is normal
    # progress while init containers run, not an init-container failure; only a
    # genuine init-container status (source == "init") is an init failure.
    if source == "init":
        return "INIT_CONTAINER_FAILED"
    if "crashloopbackoff" in normalized or "containercannotrun" in normalized:
        return "CONTAINER_CRASH"
    if "backoff" in combined:
        return "CONTROLLER_BACKOFF"
    if any(
        item in combined
        for item in ("failedmount", "failedattachvolume", "persistentvolume")
    ):
        return "STORAGE_PENDING"
    if "notready" in combined:
        return "NODE_NOT_READY"
    return "PENDING_UNKNOWN"


def _diagnostic_error_code(message: str) -> str:
    text = str(message or "").lower()
    if any(
        item in text
        for item in ("no such host", "name or service not known", "getaddrinfo")
    ):
        return "KUBERNETES_DNS"
    if any(item in text for item in ("forbidden", "rbac", "permission denied")):
        return "KUBERNETES_RBAC"
    if any(item in text for item in ("unauthorized", "unauthenticated", "401")):
        return "KUBERNETES_AUTHENTICATION"
    if any(item in text for item in ("timed out", "timeout", "deadline exceeded")):
        return "KUBERNETES_TIMEOUT"
    return "KUBERNETES_DIAGNOSTICS_UNAVAILABLE"


def _event_blockers(
    *,
    pod_names: set[str],
    namespace: str,
    context: str,
    timeout: int,
    runner: Runner,
) -> list[PodBlocker]:
    """Collect relevant warning events without returning raw event payloads."""

    if not pod_names:
        return []
    cmd = ["kubectl", "get", "events", "-o", "json"]
    if context.strip():
        cmd[1:1] = ["--context", context.strip()]
    if namespace.strip():
        cmd.extend(["-n", namespace.strip()])
    else:
        cmd.append("--all-namespaces")
    try:
        result = runner(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    try:
        items = json.loads(result.stdout or "{}").get("items") or []
    except (AttributeError, json.JSONDecodeError):
        return []
    blockers: list[PodBlocker] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        involved = _as_dict(item.get("involvedObject") or item.get("regarding"))
        pod = str(involved.get("name") or "")
        if pod not in pod_names:
            continue
        reason = str(item.get("reason") or "")
        message = sanitize_reason(item.get("message") or item.get("note") or "")
        code = classify_pending_reason(reason, message, source="event")
        if code == "PENDING_UNKNOWN":
            continue
        blockers.append(
            PodBlocker(
                pod=pod,
                phase="Pending",
                reason=reason or "KubernetesEvent",
                message=message,
                reason_code=code,
                source="kubernetes_event",
                observed_at=utc_now(),
            )
        )
    return blockers


def _pod_belongs_to_job(item: object, job_id: str) -> bool:
    """Whether a pod's SkyPilot cluster label names this managed job.

    The label reads ``<task>-<job_id>-<user_hash>``, so the id must match a whole
    dash-separated component -- job 3 must not match ``train-333-abc``.
    """

    if not isinstance(item, dict):
        return False
    labels = _as_dict(_as_dict(item.get("metadata")).get("labels"))
    value = str(labels.get(CLUSTER_LABEL) or "")
    return bool(value) and job_id in value.split("-")


def _blockers_from_pods(items: list[object]) -> list[PodBlocker]:
    blockers: list[PodBlocker] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(_as_dict(item.get("metadata")).get("name") or "")
        status = _as_dict(item.get("status"))
        phase = str(status.get("phase") or "")
        if phase in {"Running", "Succeeded"}:
            continue
        blocker = _container_blocker(name, phase, status) or _scheduling_blocker(
            name, phase, status
        )
        if blocker is not None:
            blockers.append(blocker)
    return blockers


def _container_blocker(
    name: str, phase: str, status: dict[str, Any]
) -> PodBlocker | None:
    container_lists = (
        (status.get("containerStatuses") or [], "container"),
        (status.get("initContainerStatuses") or [], "init"),
    )
    for containers, source in container_lists:
        if not isinstance(containers, list):
            continue
        for container in containers:
            if not isinstance(container, dict):
                continue
            waiting = _as_dict(_as_dict(container.get("state")).get("waiting"))
            reason = str(waiting.get("reason") or "")
            # ContainerCreating/PodInitializing are normal progress, not blockers.
            if reason and reason not in ("ContainerCreating", "PodInitializing"):
                message = sanitize_reason(waiting.get("message") or "")
                return PodBlocker(
                    pod=name,
                    phase=phase,
                    reason=reason,
                    message=message,
                    reason_code=classify_pending_reason(reason, message, source=source),
                    source=(
                        "kubernetes_init_container_condition"
                        if source == "init"
                        else "kubernetes_container_condition"
                    ),
                    observed_at=utc_now(),
                )
    return None


def _scheduling_blocker(
    name: str, phase: str, status: dict[str, Any]
) -> PodBlocker | None:
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
        reason = str(condition.get("reason") or "Unschedulable")
        message = sanitize_reason(condition.get("message") or "")
        return PodBlocker(
            pod=name,
            phase=phase,
            reason=reason,
            message=message,
            reason_code=classify_pending_reason(reason, message, source="scheduler"),
            source="kubernetes_pod_scheduled_condition",
            observed_at=utc_now(),
        )
    return None
