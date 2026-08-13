"""Structured Kubernetes Job reconciliation for the Sim2Real controller.

Production decisions in this module come from official Python-client objects.
Human log text is retained only as diagnostics and never changes scheduling or
retry classification.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable


SOURCE_SHA_LABEL = "sim2real.npa.dev/source-sha"
RUN_ID_LABEL = "sim2real.npa.dev/run-id"
SPEC_DIGEST_ANNOTATION = "sim2real.npa.dev/job-spec-sha256"
RUNTIME_DIGEST_ANNOTATION = "sim2real.npa.dev/runtime-image"
QUEUE_LABEL = "kueue.x-k8s.io/queue-name"

_RETRYABLE_API_STATUSES = frozenset({0, 429, 500, 502, 503, 504})
_CREDENTIAL_REFRESH_STATUSES = frozenset({401})
HEARTBEAT_ANNOTATION = "sim2real.npa.dev/heartbeat"
_SERVER_METADATA = frozenset(
    {
        "creationTimestamp",
        "deletionGracePeriodSeconds",
        "deletionTimestamp",
        "generation",
        "managedFields",
        "resourceVersion",
        "selfLink",
        "uid",
    }
)


class KubernetesReconcileError(RuntimeError):
    """A structured API state cannot be safely reconciled."""


class KubernetesJobFailed(KubernetesReconcileError):
    """The Job reached a terminal failure condition."""

    def __init__(self, message: str, *, snapshot: "JobSnapshot") -> None:
        super().__init__(message)
        self.snapshot = snapshot


class KubernetesJobHang(KubernetesReconcileError):
    """An operator-configured no-progress threshold was exceeded."""

    def __init__(self, message: str, *, snapshot: "JobSnapshot") -> None:
        super().__init__(message)
        self.snapshot = snapshot


@dataclass(frozen=True)
class ApiFailure:
    operation: str
    status: int
    reason: str
    transport_failure: bool = False


@dataclass(frozen=True)
class ContainerSnapshot:
    name: str
    image: str
    image_id: str
    restart_count: int
    waiting_reason: str = ""
    terminated_reason: str = ""
    exit_code: int | None = None
    signal: int | None = None


@dataclass(frozen=True)
class PodSnapshot:
    name: str
    uid: str
    owner_uid: str
    phase: str
    node_name: str
    deletion_timestamp: str
    scheduled_status: str
    scheduled_reason: str
    resource_requests: dict[str, str]
    containers: tuple[ContainerSnapshot, ...]


@dataclass(frozen=True)
class KueueAdmission:
    workload_name: str = ""
    admitted: bool = False
    finished: bool = False
    quota_reserved: bool = False
    assigned_flavors: dict[str, str] = field(default_factory=dict)
    conditions: tuple[dict[str, str], ...] = ()


@dataclass(frozen=True)
class JobSnapshot:
    name: str
    namespace: str
    uid: str
    resource_version: str
    state: str
    active: int
    succeeded: int
    failed: int
    deleting: bool
    condition_type: str
    condition_reason: str
    condition_message: str
    pods: tuple[PodSnapshot, ...]
    kueue: KueueAdmission = field(default_factory=KueueAdmission)

    @property
    def terminal(self) -> bool:
        return self.state in {"complete", "failed", "deleting"}

    @property
    def image_digests(self) -> list[str]:
        return list(
            dict.fromkeys(
                status.image_id
                for pod in self.pods
                for status in pod.containers
                if status.image_id
            )
        )

    @property
    def selected_nodes(self) -> list[str]:
        return list(dict.fromkeys(pod.node_name for pod in self.pods if pod.node_name))

    @property
    def structured_scheduling_reason(self) -> str:
        reasons = [
            pod.scheduled_reason
            for pod in self.pods
            if pod.scheduled_status == "False" and pod.scheduled_reason
        ]
        return ",".join(dict.fromkeys(reasons))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GpuProductInventory:
    product: str
    ready_nodes: int
    allocatable: int


def _string(value: Any) -> str:
    return "" if value is None else str(value)


def _condition_true(condition: Any) -> bool:
    return _string(getattr(condition, "status", "")).lower() == "true"


def _canonical_job_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(manifest)
    payload.pop("status", None)
    metadata = payload.setdefault("metadata", {})
    for key in _SERVER_METADATA:
        metadata.pop(key, None)
    annotations = metadata.get("annotations") or {}
    annotations.pop(SPEC_DIGEST_ANNOTATION, None)
    metadata["annotations"] = annotations
    return payload


def job_spec_digest(manifest: dict[str, Any]) -> str:
    """Hash all client-owned Job fields before API defaulting."""

    encoded = json.dumps(
        _canonical_job_manifest(manifest),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class KubernetesJobClient:
    """Official-client facade with create-or-adopt and typed Job watching."""

    def __init__(
        self,
        *,
        batch_api: Any,
        core_api: Any,
        custom_api: Any | None = None,
        watch_factory: Callable[[], Any] | None = None,
        credential_refresh: Callable[[], None] | None = None,
        namespace: str = "default",
    ) -> None:
        self.batch = batch_api
        self.core = core_api
        self.custom = custom_api
        self.watch_factory = watch_factory
        self.credential_refresh = credential_refresh
        self.namespace = namespace

    @classmethod
    def from_environment(
        cls,
        *,
        namespace: str = "default",
        kubeconfig: str = "",
        context: str = "",
    ) -> "KubernetesJobClient":
        """Load in-cluster credentials or one explicitly isolated kubeconfig."""

        from kubernetes import client, config, watch

        service_account_token = Path(
            "/var/run/secrets/kubernetes.io/serviceaccount/token"
        )
        in_cluster = bool(
            os.environ.get("KUBERNETES_SERVICE_HOST")
            and service_account_token.is_file()
        )
        resolved = kubeconfig or os.environ.get("KUBECONFIG", "")
        client_configuration = client.Configuration()

        def load_credentials() -> None:
            if in_cluster:
                config.load_incluster_config(client_configuration=client_configuration)
                return
            config.load_kube_config(
                config_file=resolved or None,
                context=context or None,
                persist_config=False,
                client_configuration=client_configuration,
            )

        load_credentials()
        api_client = client.ApiClient(client_configuration)
        return cls(
            batch_api=client.BatchV1Api(api_client),
            core_api=client.CoreV1Api(api_client),
            custom_api=client.CustomObjectsApi(api_client),
            watch_factory=watch.Watch,
            credential_refresh=load_credentials,
            namespace=namespace,
        )

    @staticmethod
    def _api_error(exc: Exception, operation: str) -> ApiFailure:
        status = int(getattr(exc, "status", 0) or 0)
        reason = _string(getattr(exc, "reason", "") or type(exc).__name__)
        # This is the only compatibility fallback for an unstructured failure:
        # concrete socket/TLS/HTTP transport exception types. Message prose is
        # never inspected and arbitrary status-less application exceptions fail.
        transport_failure = isinstance(exc, (ConnectionError, TimeoutError, OSError))
        try:
            from urllib3.exceptions import HTTPError

            transport_failure = transport_failure or isinstance(exc, HTTPError)
        except ImportError:  # pragma: no cover - Kubernetes client brings urllib3
            pass
        return ApiFailure(
            operation=operation,
            status=status,
            reason=reason,
            transport_failure=transport_failure,
        )

    def _call(
        self,
        operation: str,
        function: Callable[..., Any],
        *args: Any,
        retry: bool = True,
        **kwargs: Any,
    ) -> Any:
        retry_delay = max(
            0.0, float(os.environ.get("NPA_SIM2REAL_K8S_API_RETRY_SECONDS", "2"))
        )
        while True:
            try:
                return function(*args, **kwargs)
            except Exception as exc:
                failure = self._api_error(exc, operation)
                if retry and failure.status in _CREDENTIAL_REFRESH_STATUSES:
                    if self.credential_refresh is not None:
                        self.credential_refresh()
                    time.sleep(retry_delay)
                    continue
                if retry and (
                    failure.status in (_RETRYABLE_API_STATUSES - {0})
                    or failure.transport_failure
                ):
                    time.sleep(retry_delay)
                    continue
                raise KubernetesReconcileError(
                    f"Kubernetes API {operation} failed: "
                    f"status={failure.status} reason={failure.reason}"
                ) from exc

    def list_gpu_products(
        self, *, resource: str = "nvidia.com/gpu"
    ) -> tuple[GpuProductInventory, ...]:
        nodes = self._call("list_node", self.core.list_node)
        grouped: dict[str, list[int]] = {}
        for node in list(getattr(nodes, "items", None) or []):
            labels = dict(getattr(node.metadata, "labels", None) or {})
            product = _string(labels.get("nvidia.com/gpu.product")).strip()
            if not product:
                continue
            ready = any(
                getattr(condition, "type", "") == "Ready" and _condition_true(condition)
                for condition in list(getattr(node.status, "conditions", None) or [])
            )
            allocatable = dict(getattr(node.status, "allocatable", None) or {})
            try:
                count = int(_string(allocatable.get(resource) or "0"))
            except ValueError:
                count = 0
            grouped.setdefault(product, []).append(count if ready else 0)
        return tuple(
            GpuProductInventory(
                product=product,
                ready_nodes=sum(1 for value in values if value > 0),
                allocatable=sum(values),
            )
            for product, values in grouped.items()
        )

    def create_or_adopt(
        self,
        manifest: dict[str, Any],
        *,
        run_id: str,
        source_sha: str,
        runtime_image: str,
    ) -> tuple[str, bool]:
        """Create a Job or adopt only the exact previously created object."""

        body = copy.deepcopy(manifest)
        metadata = body.setdefault("metadata", {})
        name = _string(metadata.get("name")).strip()
        namespace = _string(metadata.get("namespace") or self.namespace).strip()
        if not name:
            raise KubernetesReconcileError("Job manifest has no metadata.name")
        metadata["namespace"] = namespace
        labels = metadata.setdefault("labels", {})
        labels[RUN_ID_LABEL] = run_id[:63]
        labels[SOURCE_SHA_LABEL] = source_sha[:63]
        template_labels = (
            body.setdefault("spec", {})
            .setdefault("template", {})
            .setdefault("metadata", {})
            .setdefault("labels", {})
        )
        template_labels.update(
            {RUN_ID_LABEL: run_id[:63], SOURCE_SHA_LABEL: source_sha[:63]}
        )
        annotations = metadata.setdefault("annotations", {})
        annotations[RUNTIME_DIGEST_ANNOTATION] = runtime_image
        digest = job_spec_digest(body)
        annotations[SPEC_DIGEST_ANNOTATION] = digest

        existing = self._read_job_or_none(name, namespace)
        if existing is not None:
            self._verify_existing(
                existing, digest=digest, run_id=run_id, source_sha=source_sha
            )
            return _string(existing.metadata.uid), True
        try:
            created = self._call(
                "create_namespaced_job",
                self.batch.create_namespaced_job,
                namespace,
                body,
                retry=False,
            )
        except KubernetesReconcileError as exc:
            cause = exc.__cause__
            status = int(getattr(cause, "status", 0) or 0)
            failure = (
                self._api_error(cause, "create_namespaced_job")
                if isinstance(cause, Exception)
                else ApiFailure("create_namespaced_job", status, type(cause).__name__)
            )
            if (
                status not in (_RETRYABLE_API_STATUSES - {0}) | {409}
                and not failure.transport_failure
            ):
                raise
            # A timed-out POST or conflict is ambiguous. Read and adopt only the
            # exact fingerprint rather than issuing a replacement.
            existing = self._read_job_or_none(name, namespace)
            if existing is None:
                if status == 409:
                    raise
                return self.create_or_adopt(
                    body,
                    run_id=run_id,
                    source_sha=source_sha,
                    runtime_image=runtime_image,
                )
            self._verify_existing(
                existing, digest=digest, run_id=run_id, source_sha=source_sha
            )
            return _string(existing.metadata.uid), True
        return _string(created.metadata.uid), False

    def _read_job_or_none(self, name: str, namespace: str) -> Any | None:
        try:
            return self.batch.read_namespaced_job(name, namespace)
        except Exception as exc:
            failure = self._api_error(exc, "read_namespaced_job")
            if failure.status == 404:
                return None
            if failure.status in _RETRYABLE_API_STATUSES:
                return self._call(
                    "read_namespaced_job",
                    self.batch.read_namespaced_job,
                    name,
                    namespace,
                )
            raise KubernetesReconcileError(
                "Kubernetes API read_namespaced_job failed: "
                f"status={failure.status} reason={failure.reason}"
            ) from exc

    @staticmethod
    def _verify_existing(
        job: Any, *, digest: str, run_id: str, source_sha: str
    ) -> None:
        annotations = dict(getattr(job.metadata, "annotations", None) or {})
        labels = dict(getattr(job.metadata, "labels", None) or {})
        observed = annotations.get(SPEC_DIGEST_ANNOTATION)
        if (
            observed != digest
            or labels.get(RUN_ID_LABEL) != run_id[:63]
            or labels.get(SOURCE_SHA_LABEL) != source_sha[:63]
        ):
            raise KubernetesReconcileError(
                f"Job name collision for {job.metadata.name}: exact controller identity "
                "or spec fingerprint does not match"
            )

    def snapshot(self, name: str, *, namespace: str = "") -> JobSnapshot:
        resolved_namespace = namespace or self.namespace
        job = self._call(
            "read_namespaced_job",
            self.batch.read_namespaced_job,
            name,
            resolved_namespace,
        )
        return self._snapshot_from_job(job, namespace=resolved_namespace)

    def snapshot_if_exists(
        self, name: str, *, namespace: str = ""
    ) -> JobSnapshot | None:
        """Return typed state for an existing Job, or ``None`` on API 404."""

        resolved_namespace = namespace or self.namespace
        job = self._read_job_or_none(name, resolved_namespace)
        if job is None:
            return None
        return self._snapshot_from_job(job, namespace=resolved_namespace)

    def list_jobs(self, *, namespace: str = "") -> tuple[Any, ...]:
        """Return official-client Job objects for read-only inventory views."""

        resolved_namespace = namespace or self.namespace
        response = self._call(
            "list_namespaced_job",
            self.batch.list_namespaced_job,
            resolved_namespace,
        )
        return tuple(getattr(response, "items", None) or ())

    def _snapshot_from_job(self, job: Any, *, namespace: str) -> JobSnapshot:
        uid = _string(job.metadata.uid)
        selector = f"job-name={job.metadata.name}"
        pods_response = self._call(
            "list_namespaced_pod",
            self.core.list_namespaced_pod,
            namespace,
            label_selector=selector,
        )
        pods = tuple(
            self._pod_snapshot(pod, expected_owner_uid=uid)
            for pod in list(getattr(pods_response, "items", None) or [])
        )
        conditions = list(getattr(job.status, "conditions", None) or [])
        true_conditions = [
            condition for condition in conditions if _condition_true(condition)
        ]
        condition = true_conditions[-1] if true_conditions else None
        condition_type = _string(getattr(condition, "type", ""))
        deleting = getattr(job.metadata, "deletion_timestamp", None) is not None
        if deleting:
            state = "deleting"
        elif condition_type == "Complete":
            state = "complete"
        elif condition_type == "Failed":
            state = "failed"
        elif int(getattr(job.status, "active", 0) or 0) or any(
            pod.node_name for pod in pods
        ):
            state = "running"
        else:
            state = "pending"
        return JobSnapshot(
            name=_string(job.metadata.name),
            namespace=namespace,
            uid=uid,
            resource_version=_string(job.metadata.resource_version),
            state=state,
            active=int(getattr(job.status, "active", 0) or 0),
            succeeded=int(getattr(job.status, "succeeded", 0) or 0),
            failed=int(getattr(job.status, "failed", 0) or 0),
            deleting=deleting,
            condition_type=condition_type,
            condition_reason=_string(getattr(condition, "reason", "")),
            condition_message=_string(getattr(condition, "message", "")),
            pods=pods,
            kueue=self._kueue_admission(job, namespace=namespace),
        )

    @staticmethod
    def _pod_snapshot(pod: Any, *, expected_owner_uid: str) -> PodSnapshot:
        owners = list(getattr(pod.metadata, "owner_references", None) or [])
        owner_uid = next(
            (
                _string(owner.uid)
                for owner in owners
                if _string(getattr(owner, "kind", "")) == "Job"
                and bool(getattr(owner, "controller", False))
            ),
            "",
        )
        if owner_uid and owner_uid != expected_owner_uid:
            raise KubernetesReconcileError(
                f"Pod {pod.metadata.name} owner UID does not match Job UID"
            )
        scheduled = next(
            (
                condition
                for condition in list(getattr(pod.status, "conditions", None) or [])
                if _string(condition.type) == "PodScheduled"
            ),
            None,
        )
        requests: dict[str, str] = {}
        for container in list(getattr(pod.spec, "containers", None) or []):
            resources = getattr(container, "resources", None)
            for key, value in dict(getattr(resources, "requests", None) or {}).items():
                requests[_string(key)] = _string(value)
        statuses = tuple(
            KubernetesJobClient._container_snapshot(status)
            for status in list(getattr(pod.status, "container_statuses", None) or [])
        )
        return PodSnapshot(
            name=_string(pod.metadata.name),
            uid=_string(pod.metadata.uid),
            owner_uid=owner_uid,
            phase=_string(getattr(pod.status, "phase", "")),
            node_name=_string(getattr(pod.spec, "node_name", "")),
            deletion_timestamp=_string(getattr(pod.metadata, "deletion_timestamp", "")),
            scheduled_status=_string(getattr(scheduled, "status", "")),
            scheduled_reason=_string(getattr(scheduled, "reason", "")),
            resource_requests=requests,
            containers=statuses,
        )

    @staticmethod
    def _container_snapshot(status: Any) -> ContainerSnapshot:
        state = getattr(status, "state", None)
        waiting = getattr(state, "waiting", None)
        terminated = getattr(state, "terminated", None)
        return ContainerSnapshot(
            name=_string(status.name),
            image=_string(status.image),
            image_id=_string(status.image_id),
            restart_count=int(status.restart_count or 0),
            waiting_reason=_string(getattr(waiting, "reason", "")),
            terminated_reason=_string(getattr(terminated, "reason", "")),
            exit_code=(
                int(terminated.exit_code)
                if terminated is not None and terminated.exit_code is not None
                else None
            ),
            signal=(
                int(terminated.signal)
                if terminated is not None and terminated.signal is not None
                else None
            ),
        )

    def _kueue_admission(self, job: Any, *, namespace: str) -> KueueAdmission:
        labels = dict(getattr(job.metadata, "labels", None) or {})
        if not labels.get(QUEUE_LABEL) or self.custom is None:
            return KueueAdmission()
        # A queued Job without permission to observe its Kueue Workload is not
        # equivalent to a Job for which no Workload exists.  Preserve the
        # structured API status/reason so provisioning defects fail closed and
        # cannot be misreported as ordinary scheduling state.
        payload = self._call(
            "list_namespaced_workload",
            self.custom.list_namespaced_custom_object,
            "kueue.x-k8s.io",
            os.environ.get("NPA_SIM2REAL_KUEUE_API_VERSION", "v1beta2"),
            namespace,
            "workloads",
        )
        workload = None
        for item in list(payload.get("items") or []):
            owners = list((item.get("metadata") or {}).get("ownerReferences") or [])
            if any(
                owner.get("kind") == "Job" and owner.get("uid") == job.metadata.uid
                for owner in owners
            ):
                workload = item
                break
        if workload is None:
            return KueueAdmission()
        conditions = tuple(
            {
                "type": _string(condition.get("type")),
                "status": _string(condition.get("status")),
                "reason": _string(condition.get("reason")),
            }
            for condition in list(
                (workload.get("status") or {}).get("conditions") or []
            )
        )
        true_types = {
            item["type"] for item in conditions if item["status"].lower() == "true"
        }
        flavors: dict[str, str] = {}
        admission = (workload.get("status") or {}).get("admission") or {}
        for assignment in admission.get("podSetAssignments") or []:
            for resource, flavor in (assignment.get("flavors") or {}).items():
                flavors[_string(resource)] = _string(flavor)
        return KueueAdmission(
            workload_name=_string((workload.get("metadata") or {}).get("name")),
            admitted="Admitted" in true_types,
            finished="Finished" in true_types,
            quota_reserved="QuotaReserved" in true_types,
            assigned_flavors=flavors,
            conditions=conditions,
        )

    def watch_until_terminal(
        self,
        name: str,
        *,
        namespace: str = "",
        timeout_s: int = 0,
        hang_timeout_s: int = 0,
        heartbeat: Callable[[JobSnapshot], None] | None = None,
    ) -> JobSnapshot:
        """Watch structured state until completion, failure, or configured hang."""

        resolved_namespace = namespace or self.namespace
        started = time.monotonic()
        last_progress = started
        last_signature = ""
        while True:
            snapshot = self.snapshot(name, namespace=resolved_namespace)
            signature = json.dumps(
                {
                    "state": snapshot.state,
                    "active": snapshot.active,
                    "succeeded": snapshot.succeeded,
                    "failed": snapshot.failed,
                    "pods": [
                        {
                            "phase": pod.phase,
                            "node": pod.node_name,
                            "scheduled": pod.scheduled_status,
                            "containers": [asdict(item) for item in pod.containers],
                        }
                        for pod in snapshot.pods
                    ],
                    "admission": asdict(snapshot.kueue),
                },
                sort_keys=True,
            )
            if signature != last_signature:
                last_progress = time.monotonic()
                last_signature = signature
            if heartbeat:
                heartbeat(snapshot)
            else:
                self.record_heartbeat(snapshot)
            if snapshot.state == "complete":
                return snapshot
            if snapshot.state in {"failed", "deleting"}:
                raise KubernetesJobFailed(
                    f"Kubernetes Job {name} reached {snapshot.state}: "
                    f"condition={snapshot.condition_type} "
                    f"reason={snapshot.condition_reason}",
                    snapshot=snapshot,
                )
            now = time.monotonic()
            if timeout_s > 0 and now - started >= timeout_s:
                raise KubernetesReconcileError(
                    f"Kubernetes Job {name} exceeded the explicit {timeout_s}s wait"
                )
            if hang_timeout_s > 0 and now - last_progress >= hang_timeout_s:
                raise KubernetesJobHang(
                    f"Kubernetes Job {name} made no structured progress for "
                    f"{hang_timeout_s}s",
                    snapshot=snapshot,
                )
            self._watch_cycle(name, namespace=resolved_namespace)

    def _watch_cycle(self, name: str, *, namespace: str) -> None:
        if self.watch_factory is None:
            time.sleep(1)
            return
        watcher = self.watch_factory()
        try:
            events: Iterable[Any] = watcher.stream(
                self.batch.list_namespaced_job,
                namespace,
                field_selector=f"metadata.name={name}",
                timeout_seconds=20,
            )
            for _event in events:
                break
        except Exception as exc:
            failure = self._api_error(exc, "watch_namespaced_job")
            if (
                failure.status not in (_RETRYABLE_API_STATUSES - {0})
                and failure.status not in _CREDENTIAL_REFRESH_STATUSES
                and not failure.transport_failure
            ):
                raise KubernetesReconcileError(
                    "Kubernetes API watch_namespaced_job failed: "
                    f"status={failure.status} reason={failure.reason}"
                ) from exc
        finally:
            watcher.stop()

    def record_heartbeat(self, snapshot: JobSnapshot) -> None:
        """Persist liveness in the Job object without changing progress state."""

        from datetime import datetime, timezone

        value = json.dumps(
            {
                "at": datetime.now(timezone.utc).isoformat(),
                "state": snapshot.state,
                "resourceVersion": snapshot.resource_version,
            },
            separators=(",", ":"),
        )
        self._call(
            "patch_namespaced_job_heartbeat",
            self.batch.patch_namespaced_job,
            snapshot.name,
            snapshot.namespace,
            {"metadata": {"annotations": {HEARTBEAT_ANNOTATION: value}}},
        )

    def delete_and_wait(self, name: str, *, namespace: str = "") -> None:
        """Explicit cleanup helper for tests/operators, never contention logic."""

        resolved_namespace = namespace or self.namespace
        try:
            self.batch.delete_namespaced_job(
                name,
                resolved_namespace,
                propagation_policy="Foreground",
            )
        except Exception as exc:
            failure = self._api_error(exc, "delete_namespaced_job")
            if failure.status != 404:
                raise KubernetesReconcileError(
                    "Kubernetes API delete_namespaced_job failed: "
                    f"status={failure.status} reason={failure.reason}"
                ) from exc
        while self._read_job_or_none(name, resolved_namespace) is not None:
            time.sleep(1)

    def apply_secret(self, manifest: dict[str, Any]) -> None:
        """Create or patch one Secret through the authenticated Core API.

        This is used by the long-running controller to rotate registry pull
        credentials without requiring a ``kubectl`` binary or restarting the
        driver pod. API 401s take the same credential-reload path as Job watches.
        """

        metadata = dict(manifest.get("metadata") or {})
        name = _string(metadata.get("name")).strip()
        namespace = _string(metadata.get("namespace") or self.namespace).strip()
        if not name or not namespace:
            raise KubernetesReconcileError(
                "Secret manifest requires metadata.name and metadata.namespace"
            )
        try:
            existing = self.core.read_namespaced_secret(name, namespace)
        except Exception as exc:
            failure = self._api_error(exc, "read_namespaced_secret")
            if failure.status == 404:
                existing = None
            elif (
                failure.status in _CREDENTIAL_REFRESH_STATUSES
                or failure.status in (_RETRYABLE_API_STATUSES - {0})
                or failure.transport_failure
            ):
                if (
                    failure.status in _CREDENTIAL_REFRESH_STATUSES
                    and self.credential_refresh is not None
                ):
                    self.credential_refresh()
                existing = self._call(
                    "read_namespaced_secret",
                    self.core.read_namespaced_secret,
                    name,
                    namespace,
                )
            else:
                raise KubernetesReconcileError(
                    "Kubernetes API read_namespaced_secret failed: "
                    f"status={failure.status} reason={failure.reason}"
                ) from exc
        if existing is None:
            self._call(
                "create_namespaced_secret",
                self.core.create_namespaced_secret,
                namespace,
                manifest,
            )
            return
        self._call(
            "patch_namespaced_secret",
            self.core.patch_namespaced_secret,
            name,
            namespace,
            manifest,
        )

    def pod_logs(self, snapshot: JobSnapshot, *, tail_lines: int = 200) -> str:
        chunks: list[str] = []
        for pod in snapshot.pods:
            try:
                value = self.core.read_namespaced_pod_log(
                    pod.name,
                    snapshot.namespace,
                    tail_lines=tail_lines,
                )
            except Exception as exc:
                failure = self._api_error(exc, "read_namespaced_pod_log")
                chunks.append(
                    f"{pod.name}: log unavailable status={failure.status} "
                    f"reason={failure.reason}"
                )
            else:
                chunks.append(f"{pod.name}:\n{value}")
        return "\n".join(chunks)
