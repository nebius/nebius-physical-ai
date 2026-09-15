"""Create and retire exclusively owned Kubernetes filesystem qualification resources."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import uuid

from kubernetes import client
from kubernetes.client.exceptions import ApiException
from urllib3.exceptions import HTTPError

PROBE_IMAGE = (
    "docker.io/library/python:3.12.12-slim-bookworm@sha256:"
    "2986c55feb36e6cae00fa1fefb454283e4b33f35e75ff8bdd123b134130be301"
)
OWNER_LABEL = "npa.nebius.com/storage-verification"
STORAGE_CLASS = "csi-mounted-fs-path-sc"
STORAGE_DRIVER = "mounted-fs-path.csi.nebius.ai"


class StorageVerificationError(RuntimeError):
    """Carry a publication-safe storage failure category."""


class StorageResources:
    """Track exact resource ownership throughout one cluster qualification.

    Args:
        api: Kubernetes client bound to a verified cluster identity.
        run_id: Unique invocation ownership token.
    """

    def __init__(self, api, run_id: str) -> None:
        self.api = api
        self.core = client.CoreV1Api(api)
        self.storage = client.StorageV1Api(api)
        self.apps = client.AppsV1Api(api)
        self.run_id = run_id
        self.namespace = "default"
        self.owned = []
        self.receipts = []
        self.removed = 0

    def create(self, kind: str, body: dict):
        """Create a uniquely named object while retaining ambiguous-create ownership.

        Args:
            kind: Pod or persistent_volume_claim.
            body: Generated object with exclusive run labels.
        Returns:
            Provider-confirmed object.
        Raises:
            ApiException: Creation or ownership recovery failed.
            StorageVerificationError: Existing identity conflicts with this run.
        """
        name = body["metadata"]["name"]
        record = {"kind": kind, "name": name, "uid": None, "absent": False}
        self.owned.append(record)
        result = getattr(self.core, f"create_namespaced_{kind}")(self.namespace, body)
        self._adopt(record, result)
        return result

    def _adopt(self, record: dict, result) -> None:
        if result.metadata is None:
            raise StorageVerificationError("ownership_missing")
        if (result.metadata.labels or {}).get(OWNER_LABEL) != self.run_id:
            raise StorageVerificationError("ownership_mismatch")
        if not result.metadata.uid:
            raise StorageVerificationError("ownership_missing")
        if record["uid"] and record["uid"] != result.metadata.uid:
            raise StorageVerificationError("ownership_mismatch")
        record["uid"] = result.metadata.uid

    def _read(self, record: dict):
        try:
            return getattr(self.core, f"read_namespaced_{record['kind']}")(
                record["name"], self.namespace,
            )
        except ApiException as error:
            if error.status == 404:
                return None
            raise

    def remove(self, record: dict) -> None:
        """Delete and independently confirm absence with UID preconditions.

        Args:
            record: Exact locally tracked resource identity.
        Raises:
            ApiException: Kubernetes cannot delete or verify absence.
            StorageVerificationError: Ownership changed.
        """
        current = self._read(record)
        if current is None:
            record["absent"] = True
            return
        self._adopt(record, current)
        options = client.V1DeleteOptions(
            preconditions=client.V1Preconditions(uid=record["uid"]),
            propagation_policy="Foreground",
        )
        getattr(self.core, f"delete_namespaced_{record['kind']}")(
            record["name"], self.namespace, body=options,
        )
        while True:
            current = self._read(record)
            if current is None:
                self.removed += 1
                record["absent"] = True
                return
            self._adopt(record, current)
            time.sleep(1)

    def cleanup(self, *, claims: bool = True) -> list[str]:
        """Attempt every owned cleanup even when an earlier deletion fails.

        Args:
            claims: Include claims after deleting consumers.
        Returns:
            Publication-safe cleanup failure categories.
        """
        failures = []
        records = sorted(self.owned, key=lambda item: item["kind"] != "pod")
        for record in records:
            if not claims and record["kind"] != "pod":
                continue
            try:
                self.remove(record)
            except (ApiException, StorageVerificationError, OSError, HTTPError, ValueError):
                failures.append("resource_cleanup_failed")
        return failures

    def run_phase(self, nodes: list, cluster, configuration: dict, *, claim=None) -> list:
        """Run the same probe phase on every exact node using scheduler pinning.

        Args:
            nodes: Verified Node objects with immutable UIDs.
            cluster: Selected Fleet cluster mount declaration.
            configuration: Non-sensitive probe action and expected checksums.
            claim: Optional exclusively owned RWX claim.
        Returns:
            Exact ordered per-node probe results.
        Raises:
            StorageVerificationError: A pod or its evidence is incomplete.
            ApiException: Kubernetes operation failed.
        """
        pods = []
        for node in nodes:
            settings = dict(configuration, node_token=_node_token(node))
            body = _probe_pod(self.run_id, node, cluster, settings, claim)
            pods.append(self.create("pod", body))
        return [self._result(pod, node, configuration["action"])
                for pod, node in zip(pods, nodes, strict=True)]

    def _result(self, original, node, action) -> dict:
        while True:
            pod = self.core.read_namespaced_pod(original.metadata.name, self.namespace)
            _check_pod_identity(original, pod, node, self.run_id)
            if pod.status.phase in {"Succeeded", "Failed"}:
                break
            _check_pod_failure(pod)
            self._check_pod_events(pod)
            time.sleep(1)
        output = self._pod_output(pod)
        receipt = {"pod": self.api.sanitize_for_serialization(pod), "output": output}
        self.receipts.append(receipt)
        try:
            result = json.loads(output)
        except (ValueError, TypeError) as error:
            raise StorageVerificationError("partial_evidence") from error
        if not isinstance(result, dict) or result.get("passed") is not True:
            raise StorageVerificationError("probe_failed")
        if result.get("action") != action:
            raise StorageVerificationError("partial_evidence")
        statuses = pod.status.container_statuses or []
        _require_completed_container(pod, statuses, node)
        return result

    def _pod_output(self, pod) -> str:
        response = self.core.read_namespaced_pod_log(
            pod.metadata.name, self.namespace, _preload_content=False,
        )
        try:
            return response.read().decode("utf-8")
        finally:
            response.release_conn()

    def _check_pod_events(self, pod) -> None:
        events = self.core.list_namespaced_event(
            self.namespace, field_selector=f"involvedObject.uid={pod.metadata.uid}",
        ).items
        failures = [event for event in events if event.type == "Warning"]
        if failures:
            self.receipts.append({"pod_events": self.api.sanitize_for_serialization(failures)})
            raise StorageVerificationError("workload_start_failed")


def _node_token(node) -> str:
    return hashlib.sha256(node.metadata.uid.encode()).hexdigest()[:32]


def _check_pod_identity(original, pod, node, run_id: str) -> None:
    if pod.metadata.uid != original.metadata.uid:
        raise StorageVerificationError("ownership_mismatch")
    if (pod.metadata.labels or {}).get(OWNER_LABEL) != run_id:
        raise StorageVerificationError("ownership_mismatch")
    if pod.spec.node_name and pod.spec.node_name != node.metadata.name:
        raise StorageVerificationError("node_identity_mismatch")
    if len(pod.spec.containers) != 1 or pod.spec.containers[0].image != PROBE_IMAGE:
        raise StorageVerificationError("image_digest_mismatch")
    if pod.spec.containers[0].command != original.spec.containers[0].command:
        raise StorageVerificationError("probe_command_mismatch")


def _require_completed_container(pod, statuses, node) -> None:
    if pod.status.phase != "Succeeded" or pod.spec.node_name != node.metadata.name:
        raise StorageVerificationError("workload_failed")
    if len(statuses) != 1 or not statuses[0].state.terminated:
        raise StorageVerificationError("partial_evidence")
    if statuses[0].state.terminated.exit_code != 0 or statuses[0].restart_count != 0:
        raise StorageVerificationError("workload_failed")
    digest = PROBE_IMAGE.split("@", 1)[1]
    image_id = (statuses[0].image_id or "").rsplit("@", 1)[-1]
    if image_id != digest:
        raise StorageVerificationError("image_digest_mismatch")


def _check_pod_failure(pod) -> None:
    fatal = {"ImagePullBackOff", "ErrImagePull", "InvalidImageName",
             "CreateContainerConfigError", "CreateContainerError", "RunContainerError"}
    if pod.status.phase in {"Failed", "Unknown"}:
        raise StorageVerificationError("workload_failed")
    for status in pod.status.container_statuses or []:
        if status.state.waiting and status.state.waiting.reason in fatal:
            raise StorageVerificationError("workload_start_failed")
    for condition in pod.status.conditions or []:
        if condition.type == "PodScheduled" and condition.reason == "Unschedulable":
            raise StorageVerificationError("worker_unschedulable")


def _probe_pod(run_id: str, node, cluster, settings: dict, claim) -> dict:
    from npa.fleet import storage_probe

    source = Path(storage_probe.__file__).read_text()
    settings.update(run_id=run_id, mount_path=cluster.filestore_mount_path,
                    mount_tag=cluster.filestore_mount_tag,
                    requested_gibibytes=cluster.filestore_disk_size_gibibytes,
                    mountinfo_path="/host-proc/1/mountinfo", fstab_path="/host-fstab")
    volumes, mounts = _probe_volumes(cluster, claim)
    container = {
        "name": "probe", "image": PROBE_IMAGE, "imagePullPolicy": "IfNotPresent",
        "command": ["python3", "-c", source, json.dumps(settings)],
        "volumeMounts": mounts,
        "securityContext": {"allowPrivilegeEscalation": False,
                            "capabilities": {"drop": ["ALL"]}, "runAsUser": 0},
    }
    return {
        "apiVersion": "v1", "kind": "Pod",
        "metadata": {"name": "npa-storage-" + uuid.uuid4().hex,
                     "labels": {OWNER_LABEL: run_id}},
        "spec": {"restartPolicy": "Never", "automountServiceAccountToken": False,
                 "hostPID": True, "containers": [container], "volumes": volumes,
                 "tolerations": [{"operator": "Exists", "effect": "NoSchedule"}],
                 "affinity": _node_affinity(node.metadata.name)},
    }


def _node_affinity(name: str) -> dict:
    return {"nodeAffinity": {"requiredDuringSchedulingIgnoredDuringExecution": {
        "nodeSelectorTerms": [{"matchFields": [
            {"key": "metadata.name", "operator": "In", "values": [name]},
        ]}],
    }}}


def _probe_volumes(cluster, claim) -> tuple[list, list]:
    host_paths = [("host-storage", cluster.filestore_mount_path, "Directory", False),
                  ("host-proc", "/proc", "Directory", True),
                  ("host-fstab", "/etc/fstab", "File", True)]
    volumes = [{"name": name, "hostPath": {"path": path, "type": kind}}
               for name, path, kind, _ in host_paths]
    mounts = [{"name": name, "mountPath": "/" + name, "readOnly": readonly}
              for name, _, _, readonly in host_paths]
    mounts[0]["mountPropagation"] = "HostToContainer"
    if claim:
        volumes.append({"name": "shared", "persistentVolumeClaim": {"claimName": claim}})
        mounts.append({"name": "shared", "mountPath": "/data"})
    return volumes, mounts


def claim_manifest(run_id: str) -> dict:
    """Build an exclusively owned RWX claim that exercises default class selection.

    Args:
        run_id: Unique invocation ownership token.
    Returns:
        Kubernetes PersistentVolumeClaim manifest.
    """
    return {
        "apiVersion": "v1", "kind": "PersistentVolumeClaim",
        "metadata": {"name": "npa-storage-" + uuid.uuid4().hex,
                     "labels": {OWNER_LABEL: run_id}},
        "spec": {"accessModes": ["ReadWriteMany"],
                 "resources": {"requests": {"storage": "1Gi"}}},
    }
