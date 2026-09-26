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
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

from npa.verification import sanitize_reason, utc_now

CLUSTER_LABEL = "skypilot-cluster-name"
MANAGED_JOB_ID_ANNOTATION = "skypilot-managed-job-id"
MANAGED_JOB_NAME_ANNOTATION = "skypilot-managed-job-name"
DEFAULT_TIMEOUT_SECONDS = 60
_DNS_LABEL = r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?"
_DNS_SUBDOMAIN_RE = re.compile(rf"(?:{_DNS_LABEL}\.)*{_DNS_LABEL}")
_STORAGE_FAILURE_CODES = frozenset(
    {
        "STORAGE_QUOTA_EXCEEDED",
        "STORAGE_CAPACITY_UNAVAILABLE",
        "STORAGE_PROVISIONING_FAILED",
    }
)

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

_STORAGE_QUOTA_REMEDY = (
    "the exact rendered PersistentVolumeClaim cannot be provisioned because the "
    "storage quota is exhausted. Increase the applicable storage quota or reduce "
    "the claim request, then explicitly start or resume the workflow; automatic "
    "retry is disabled."
)

_STORAGE_CAPACITY_REMEDY = (
    "the exact rendered PersistentVolumeClaim is still Pending because storage "
    "capacity is unavailable. Keep observing this exact attempt or add capacity; "
    "do not launch a second worker against the same claim."
)

_STORAGE_PROVISIONING_REMEDY = (
    "the exact rendered PersistentVolumeClaim failed provisioning. Inspect its "
    "current UID-bound Kubernetes event, correct the storage class or claim "
    "configuration, then explicitly start or resume the workflow."
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
    namespace: str = ""
    resource_uid: str = ""
    event_timestamp: str = ""
    temporally_bound: bool = False

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
        storage_remedies = (
            ("STORAGE_QUOTA_EXCEEDED", _STORAGE_QUOTA_REMEDY),
            ("STORAGE_PROVISIONING_FAILED", _STORAGE_PROVISIONING_REMEDY),
            ("STORAGE_CAPACITY_UNAVAILABLE", _STORAGE_CAPACITY_REMEDY),
        )
        for code, remedy in storage_remedies:
            if any(blocker.reason_code == code for blocker in self.blockers):
                return remedy
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
    kubeconfig: str | os.PathLike[str] | None = None,
    environment: Mapping[str, str] | None = None,
    expected_task_names: Iterable[str] = (),
    controller_user_id: str = "",
    claim_names: Sequence[str] = (),
    controller_output: str = "",
    event_not_before: str = "",
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

    Args:
        job_id: Managed-job identity from the isolated controller queue.
        cluster_name: Exact cluster name, when the queue has recorded one.
        namespace: Kubernetes namespace to inspect, or all namespaces when empty.
        context: Explicit kubeconfig context for kubectl.
        kubeconfig: Exact controller-bound kubeconfig path.
        environment: Environment passed to kubectl.
        expected_task_names: Exact managed task names returned for ``job_id``.
            These are checked against SkyPilot's untruncated pod annotation.
        controller_user_id: Isolated controller owner appended to the pod's
            cluster label.
        claim_names: Exact PVC names declared by the rendered wave resources.
        controller_output: Bounded log text from this exact managed job's
            controller. It can explain a historical rejection before pod creation.
        event_not_before: When nonempty, accept PVC events only when their
            timestamp is at or after this exact attempt start time.
        timeout: Kubectl timeout in seconds.
        runner: Optional subprocess-compatible runner for tests.

    Returns:
        Pod blockers or a fail-closed diagnostic error.

    Raises:
        None. Process and response failures are represented in the report.
    """

    report = JobBlockerReport(job_id=str(job_id), cluster_name=str(cluster_name))
    report.observed_at = utc_now()
    by_job_id = not cluster_name.strip()
    if by_job_id and not str(job_id).strip():
        report.error = "no cluster name or job id to look up"
        return report

    cmd = ["kubectl"]
    if kubeconfig is not None and os.fspath(kubeconfig).strip():
        cmd.extend(["--kubeconfig", str(Path(kubeconfig).expanduser())])
    if context.strip():
        cmd.extend(["--context", context.strip()])
    cmd.extend(["get", "pods"])
    if by_job_id:
        # Every SkyPilot pod carries the label; the value is filtered below.
        cmd.extend(["-l", CLUSTER_LABEL])
    else:
        cmd.extend(["-l", f"{CLUSTER_LABEL}={cluster_name.strip()}"])
    cmd.extend(["-o", "json"])
    if namespace.strip():
        cmd.extend(["-n", namespace.strip()])
    execute = runner or subprocess.run
    try:
        result = execute(
            cmd,
            env=dict(environment) if environment is not None else None,
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
        task_names = tuple(
            name for raw in expected_task_names if (name := str(raw).strip())
        )
        user_id = str(controller_user_id).strip()
        if user_id:
            if not task_names:
                if not items:
                    report.blockers = _pvc_event_blockers(
                        claim_names=claim_names,
                        namespace=namespace,
                        context=context,
                        kubeconfig=kubeconfig,
                        environment=environment,
                        event_not_before=event_not_before,
                        timeout=timeout,
                        runner=execute,
                    )
                    if report.blockers:
                        return report
                report.error = (
                    "the isolated controller did not return an exact managed task name"
                )
                report.error_code = "KUBERNETES_POD_IDENTITY_INCOMPLETE"
                return report
            items = [
                item
                for item in items
                if _pod_belongs_to_owned_task(item, str(job_id), task_names, user_id)
            ]
            matching_clusters = {_pod_cluster_label(item) for item in items}
            if len(matching_clusters) > 1:
                report.error = (
                    "multiple clusters match the exact managed task and isolated "
                    "controller identity"
                )
                report.error_code = "KUBERNETES_POD_IDENTITY_AMBIGUOUS"
                return report
        else:
            items = [item for item in items if _pod_belongs_to_job(item, str(job_id))]
        if not items:
            report.blockers = _pvc_event_blockers(
                claim_names=claim_names,
                namespace=namespace,
                context=context,
                kubeconfig=kubeconfig,
                environment=environment,
                event_not_before=event_not_before,
                timeout=timeout,
                runner=execute,
            )
            if report.blockers:
                return report
            report.blockers = _controller_volume_blockers(
                controller_output, claim_names
            )
            if report.blockers:
                return report
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
            kubeconfig=kubeconfig,
            environment=environment,
            timeout=timeout,
            runner=execute,
        )
        if not any(
            existing.pod == blocker.pod and existing.reason_code == blocker.reason_code
            for existing in report.blockers
        )
    )
    if not report.blockers:
        report.blockers.extend(
            _pvc_event_blockers(
                claim_names=claim_names,
                namespace=namespace,
                context=context,
                kubeconfig=kubeconfig,
                environment=environment,
                event_not_before=event_not_before,
                timeout=timeout,
                runner=execute,
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
            kubeconfig=kubeconfig,
            environment=environment,
            timeout=timeout,
            runner=execute,
            assigned_nodes=assigned_nodes,
        )
    return report


def _controller_volume_blockers(
    controller_output: str, claim_names: Sequence[str]
) -> list[PodBlocker]:
    """Explain exact rendered claims rejected before SkyPilot created a pod."""
    names = {
        name for name in claim_names if is_valid_persistent_volume_claim_name(name)
    }
    pod_name = rf"(?:'{_DNS_LABEL}'|\"{_DNS_LABEL}\")"
    pattern = re.compile(
        rf"\bVolume (?P<claim>{_DNS_LABEL}(?:\.{_DNS_LABEL})*) "
        rf"with access mode ReadWriteOnce is already in use by Pods "
        rf"\[{pod_name}(?:, {pod_name})*\]\."
    )
    blockers = {}
    for match in pattern.finditer(controller_output[-4000:]):
        claim = match.group("claim")
        if claim not in names:
            continue
        blockers[claim] = PodBlocker(
            pod=f"persistentvolumeclaim/{claim}",
            phase="Pending",
            reason="ReadWriteOnceVolumeInUse",
            reason_code="STORAGE_VOLUME_IN_USE",
            message=(
                "This job's controller observed: "
                + match.group()
                + " Current claim ownership has not been verified."
            ),
            source="skypilot_controller_log",
            live=False,
            temporally_bound=False,
        )
    return list(blockers.values())


def _unready_nodes(
    *,
    context: str,
    kubeconfig: str | os.PathLike[str] | None,
    environment: Mapping[str, str] | None,
    timeout: int,
    runner: Runner,
    assigned_nodes: set[str],
) -> list[str]:
    """Return assigned nodes whose kubelet is not Ready."""

    if not assigned_nodes:
        return []

    cmd = ["kubectl", "get", "nodes", "-o", "json"]
    if kubeconfig is not None and os.fspath(kubeconfig).strip():
        cmd[1:1] = ["--kubeconfig", str(Path(kubeconfig).expanduser())]
    if context.strip():
        cmd[1:1] = ["--context", context.strip()]
    try:
        result = runner(
            cmd,
            env=dict(environment) if environment is not None else None,
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
    if normalized == "provisioningfailed":
        return _storage_provisioning_reason(detail)
    if "unschedul" in combined or "failedscheduling" in combined:
        # Kubernetes names GPU resources in ordinary capacity shortages too.
        if any(item in combined for item in ("quota", "capacity", "insufficient")):
            return "CAPACITY_OR_QUOTA"
        if any(item in combined for item in ("gpu", "accelerator", "nvidia.com/gpu")):
            return "ACCELERATOR_MISMATCH"
        if "no nodes" in combined:
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


def _storage_provisioning_reason(message: str) -> str:
    """Keep retryable provisioning transport errors out of cancellation paths."""

    retryable = (
        "deadlineexceeded",
        "deadline exceeded",
        "timed out",
        "timeout",
        "temporarily unavailable",
        "service unavailable",
        "code = unavailable",
        "toomanyrequests",
        "too many requests",
        "rate limit",
        "request throttled",
    )
    if any(fragment in message for fragment in retryable):
        return "STORAGE_PENDING"
    if "quota" in message or "limit reached" in message:
        return "STORAGE_QUOTA_EXCEEDED"
    if any(
        fragment in message
        for fragment in (
            "capacity",
            "no space left",
            "insufficient storage",
            "out of space",
        )
    ):
        return "STORAGE_CAPACITY_UNAVAILABLE"
    return "STORAGE_PROVISIONING_FAILED"


def persistent_volume_claim_names(
    resources_profile: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return valid PVC names declared by one rendered resource profile.

    Args:
        resources_profile: Rendered workflow resource profile to inspect.

    Returns:
        Sorted, de-duplicated Kubernetes PersistentVolumeClaim names.

    Raises:
        None.
    """

    kubernetes = _as_dict(resources_profile.get("kubernetes"))
    pod_config = _as_dict(kubernetes.get("pod_config"))
    spec = _as_dict(pod_config.get("spec"))
    volumes = spec.get("volumes") or []
    if not isinstance(volumes, list):
        return ()
    names: set[str] = set()
    for volume in volumes:
        if not isinstance(volume, dict):
            continue
        claim = _as_dict(volume.get("persistentVolumeClaim"))
        raw_name = claim.get("claimName")
        name = raw_name if isinstance(raw_name, str) else ""
        if is_valid_persistent_volume_claim_name(name):
            names.add(name)
    return tuple(sorted(names))


def is_valid_persistent_volume_claim_name(name: object) -> bool:
    """Check whether a value is an unchanged Kubernetes DNS-subdomain name.

    Args:
        name: Candidate claim name.

    Returns:
        Whether ``name`` is a valid PersistentVolumeClaim name.

    Raises:
        None.
    """

    return (
        isinstance(name, str)
        and bool(name)
        and len(name) <= 253
        and bool(_DNS_SUBDOMAIN_RE.fullmatch(name))
    )


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
    kubeconfig: str | os.PathLike[str] | None,
    environment: Mapping[str, str] | None,
    timeout: int,
    runner: Runner,
) -> list[PodBlocker]:
    """Collect relevant warning events without returning raw event payloads."""

    if not pod_names:
        return []
    cmd = ["kubectl", "get", "events", "-o", "json"]
    if kubeconfig is not None and os.fspath(kubeconfig).strip():
        cmd[1:1] = ["--kubeconfig", str(Path(kubeconfig).expanduser())]
    if context.strip():
        cmd[1:1] = ["--context", context.strip()]
    if namespace.strip():
        cmd.extend(["-n", namespace.strip()])
    try:
        result = runner(
            cmd,
            env=dict(environment) if environment is not None else None,
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


def _pvc_event_blockers(
    *,
    claim_names: Sequence[str],
    namespace: str,
    context: str,
    kubeconfig: str | os.PathLike[str] | None,
    environment: Mapping[str, str] | None,
    event_not_before: str,
    timeout: int,
    runner: Runner,
) -> list[PodBlocker]:
    """Return current UID-bound provisioning failures for exact rendered claims."""

    not_before = _parse_timestamp(event_not_before) if event_not_before else None
    if event_not_before and not_before is None:
        return []
    blockers: list[PodBlocker] = []
    valid_names = sorted(
        {name for name in claim_names if is_valid_persistent_volume_claim_name(name)}
    )
    for claim_name in valid_names:
        blockers.extend(
            _blockers_for_pending_claim(
                claim_name=claim_name,
                namespace=namespace,
                context=context,
                kubeconfig=kubeconfig,
                environment=environment,
                not_before=not_before,
                timeout=timeout,
                runner=runner,
            )
        )
    return sorted(
        blockers,
        key=lambda item: (item.namespace, item.pod, item.event_timestamp, item.reason),
    )


def _blockers_for_pending_claim(
    *,
    claim_name: str,
    namespace: str,
    context: str,
    kubeconfig: str | os.PathLike[str] | None,
    environment: Mapping[str, str] | None,
    not_before: datetime | None,
    timeout: int,
    runner: Runner,
) -> list[PodBlocker]:
    identity = _pending_claim_identity(
        claim_name=claim_name,
        namespace=namespace,
        context=context,
        kubeconfig=kubeconfig,
        environment=environment,
        timeout=timeout,
        runner=runner,
    )
    if identity is None:
        return []
    claim_namespace, claim_uid = identity
    return _events_for_pending_claim(
        claim_name=claim_name,
        claim_namespace=claim_namespace,
        claim_uid=claim_uid,
        context=context,
        kubeconfig=kubeconfig,
        environment=environment,
        not_before=not_before,
        timeout=timeout,
        runner=runner,
    )


def _pending_claim_identity(
    *,
    claim_name: str,
    namespace: str,
    context: str,
    kubeconfig: str | os.PathLike[str] | None,
    environment: Mapping[str, str] | None,
    timeout: int,
    runner: Runner,
) -> tuple[str, str] | None:
    """Resolve one exact, currently Pending, non-deleting claim."""

    payload = _kubectl_resource_json(
        resource="pvc",
        field_selector=f"metadata.name={claim_name}",
        namespace=namespace,
        context=context,
        kubeconfig=kubeconfig,
        environment=environment,
        timeout=timeout,
        runner=runner,
    )
    if payload is None:
        return None
    matches = _matching_claims(payload, claim_name=claim_name, namespace=namespace)
    return _pending_claim_coordinates(matches[0]) if len(matches) == 1 else None


def _matching_claims(
    payload: Mapping[str, Any], *, claim_name: str, namespace: str
) -> list[dict[str, Any]]:
    items = payload.get("items") or []
    if not isinstance(items, list):
        return []
    matches: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        metadata = _as_dict(item.get("metadata"))
        if str(metadata.get("name") or "") != claim_name:
            continue
        item_namespace = str(metadata.get("namespace") or "")
        if namespace.strip() and item_namespace != namespace.strip():
            continue
        matches.append(item)
    return matches


def _pending_claim_coordinates(claim: Mapping[str, Any]) -> tuple[str, str] | None:
    metadata = _as_dict(claim.get("metadata"))
    status = _as_dict(claim.get("status"))
    namespace = str(metadata.get("namespace") or "")
    uid = str(metadata.get("uid") or "")
    if (
        not namespace
        or not uid
        or metadata.get("deletionTimestamp")
        or str(status.get("phase") or "") != "Pending"
    ):
        return None
    return namespace, uid


def _events_for_pending_claim(
    *,
    claim_name: str,
    claim_namespace: str,
    claim_uid: str,
    context: str,
    kubeconfig: str | os.PathLike[str] | None,
    environment: Mapping[str, str] | None,
    not_before: datetime | None,
    timeout: int,
    runner: Runner,
) -> list[PodBlocker]:
    payload = _kubectl_resource_json(
        resource="events",
        field_selector=f"involvedObject.uid={claim_uid}",
        namespace=claim_namespace,
        context=context,
        kubeconfig=kubeconfig,
        environment=environment,
        timeout=timeout,
        runner=runner,
    )
    if payload is None or not isinstance(payload.get("items") or [], list):
        return []
    return [
        blocker
        for item in payload.get("items") or []
        if isinstance(item, dict)
        and (
            blocker := _pending_claim_event_blocker(
                item,
                claim_name=claim_name,
                claim_namespace=claim_namespace,
                claim_uid=claim_uid,
                not_before=not_before,
            )
        )
        is not None
    ]


def _pending_claim_event_blocker(
    item: Mapping[str, Any],
    *,
    claim_name: str,
    claim_namespace: str,
    claim_uid: str,
    not_before: datetime | None,
) -> PodBlocker | None:
    metadata = _as_dict(item.get("metadata"))
    involved = _as_dict(item.get("involvedObject") or item.get("regarding"))
    identity = (
        str(involved.get("kind") or ""),
        str(involved.get("name") or ""),
        str(involved.get("namespace") or metadata.get("namespace") or ""),
        str(involved.get("uid") or ""),
    )
    if str(item.get("type") or "") != "Warning" or identity != (
        "PersistentVolumeClaim",
        claim_name,
        claim_namespace,
        claim_uid,
    ):
        return None
    event_time_text, event_time = _event_timestamp(item)
    if not_before is not None and (event_time is None or event_time < not_before):
        return None
    return _classified_claim_event(
        item,
        claim_name=claim_name,
        claim_namespace=claim_namespace,
        claim_uid=claim_uid,
        event_time_text=event_time_text,
        temporally_bound=not_before is not None,
    )


def _classified_claim_event(
    item: Mapping[str, Any],
    *,
    claim_name: str,
    claim_namespace: str,
    claim_uid: str,
    event_time_text: str,
    temporally_bound: bool,
) -> PodBlocker | None:
    reason = str(item.get("reason") or "")
    message = sanitize_reason(item.get("message") or item.get("note") or "")
    code = classify_pending_reason(reason, message, source="pvc_event")
    if code not in _STORAGE_FAILURE_CODES:
        return None
    return PodBlocker(
        pod=f"pvc/{claim_name}",
        phase="Pending",
        reason=reason or "ProvisioningFailed",
        message=message,
        reason_code=code,
        source="kubernetes_pvc_event",
        observed_at=utc_now(),
        namespace=claim_namespace,
        resource_uid=claim_uid,
        event_timestamp=event_time_text,
        temporally_bound=temporally_bound,
    )


def _kubectl_resource_json(
    *,
    resource: str,
    field_selector: str,
    namespace: str,
    context: str,
    kubeconfig: str | os.PathLike[str] | None,
    environment: Mapping[str, str] | None,
    timeout: int,
    runner: Runner,
) -> dict[str, Any] | None:
    cmd = ["kubectl"]
    if kubeconfig is not None and os.fspath(kubeconfig).strip():
        cmd.extend(["--kubeconfig", str(Path(kubeconfig).expanduser())])
    if context.strip():
        cmd.extend(["--context", context.strip()])
    cmd.extend(["get", resource, "--field-selector", field_selector, "-o", "json"])
    cmd.extend(["-n", namespace.strip()] if namespace.strip() else ["--all-namespaces"])
    return _run_json_command(
        cmd, environment=environment, timeout=timeout, runner=runner
    )


def _run_json_command(
    cmd: list[str],
    *,
    environment: Mapping[str, str] | None,
    timeout: int,
    runner: Runner,
) -> dict[str, Any] | None:
    try:
        result = runner(
            cmd,
            env=dict(environment) if environment is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _event_timestamp(item: Mapping[str, Any]) -> tuple[str, datetime | None]:
    series = _as_dict(item.get("series"))
    metadata = _as_dict(item.get("metadata"))
    candidates = (
        series.get("lastObservedTime"),
        item.get("lastTimestamp"),
        item.get("eventTime"),
        metadata.get("creationTimestamp"),
        item.get("firstTimestamp"),
    )
    for value in candidates:
        text = str(value or "").strip()
        parsed = _parse_timestamp(text)
        if parsed is not None:
            return text, parsed
    return "", None


def _parse_timestamp(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


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


def _pod_belongs_to_owned_task(
    item: object,
    job_id: str,
    task_names: tuple[str, ...],
    controller_user_id: str,
) -> bool:
    """Match SkyPilot's exact managed-job annotations and controller owner."""

    if not isinstance(item, dict):
        return False
    metadata = _as_dict(item.get("metadata"))
    cluster = _pod_cluster_label(item)
    if not cluster.endswith(f"-{controller_user_id}"):
        return False
    annotations = _as_dict(metadata.get("annotations"))
    if str(annotations.get(MANAGED_JOB_ID_ANNOTATION) or "") != job_id:
        return False
    task_name = str(annotations.get(MANAGED_JOB_NAME_ANNOTATION) or "")
    return task_name in task_names


def _pod_cluster_label(item: object) -> str:
    """Return one pod's SkyPilot cluster label, or an empty string."""

    metadata = _as_dict(_as_dict(item).get("metadata"))
    labels = _as_dict(metadata.get("labels"))
    return str(labels.get(CLUSTER_LABEL) or "")


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
