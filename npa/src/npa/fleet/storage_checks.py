"""Validate fresh worker, CSI, and shared-volume evidence against a Fleet declaration."""

from __future__ import annotations

from datetime import datetime, timezone

from npa.fleet.storage_resources import (
    OWNER_LABEL, STORAGE_CLASS, STORAGE_DRIVER, StorageVerificationError,
)


def storage_nodes(resources, cluster) -> list:
    """Require every declared CPU and GPU worker to be present and Ready.

    Args:
        resources: Cluster-bound Kubernetes access.
        cluster: Selected Fleet declaration.
    Returns:
        Ordered exact Node objects.
    Raises:
        StorageVerificationError: Worker identity, count, shape, or health differs.
    """
    nodes = resources.core.list_node().items
    expected = cluster.cpu_count() + cluster.gpu_count()
    if not nodes or len(nodes) != expected:
        raise StorageVerificationError("worker_count_mismatch")
    observed = {"cpu": 0, "gpu": 0}
    for node in nodes:
        _require_ready_node(node)
        observed[_worker_role(node, cluster)] += 1
    if observed != {"cpu": cluster.cpu_count(), "gpu": cluster.gpu_count()}:
        raise StorageVerificationError("worker_shape_mismatch")
    if len({node.metadata.uid for node in nodes}) != expected:
        raise StorageVerificationError("node_identity_mismatch")
    return sorted(nodes, key=lambda node: node.metadata.uid)


def _require_ready_node(node) -> None:
    ready = any(item.type == "Ready" and item.status == "True"
                for item in node.status.conditions or [])
    if not ready or node.spec.unschedulable or not node.metadata.uid:
        raise StorageVerificationError("worker_unhealthy")
    if not node.status.node_info.boot_id:
        raise StorageVerificationError("worker_boot_identity_missing")


def _worker_role(node, cluster) -> str:
    labels = node.metadata.labels or {}
    matches = []
    for role, pool in (("cpu", cluster.cpu_nodes), ("gpu", cluster.gpu_nodes)):
        if pool and pool.count and labels.get("node.kubernetes.io/instance-type") == pool.platform:
            if labels.get("nebius.com/resource-preset") == pool.preset:
                matches.append(role)
    if len(matches) != 1:
        raise StorageVerificationError("worker_shape_mismatch")
    return matches[0]


def verify_storage_driver(resources, nodes: list) -> str:
    """Require the filesystem default StorageClass and per-worker CSI registration.

    Args:
        resources: Cluster-bound Kubernetes access.
        nodes: Every exact selected worker.
    Returns:
        Validated CSI provisioner name.
    Raises:
        StorageVerificationError: Default class or driver health is incomplete.
    """
    classes = resources.storage.list_storage_class().items
    relevant = [item for item in classes if item.metadata.name == STORAGE_CLASS or _default_class(item)]
    _retain_storage_evidence(resources, "StorageClassList", relevant)
    defaults = [item for item in classes if _default_class(item)]
    if len(defaults) != 1 or defaults[0].metadata.name != STORAGE_CLASS:
        raise StorageVerificationError("default_storage_class_mismatch")
    storage_class = defaults[0]
    if storage_class.reclaim_policy != "Delete" or storage_class.provisioner != STORAGE_DRIVER:
        raise StorageVerificationError("storage_class_policy_mismatch")
    driver = resources.storage.read_csi_driver(storage_class.provisioner)
    _retain_storage_evidence(resources, "CSIDriver", driver)
    if driver.metadata.name != storage_class.provisioner:
        raise StorageVerificationError("csi_driver_mismatch")
    for node in nodes:
        registered = resources.storage.read_csi_node(node.metadata.name)
        _retain_storage_evidence(resources, "CSINode", registered)
        matches = [item for item in registered.spec.drivers
                   if item.name == storage_class.provisioner and item.node_id]
        if len(matches) != 1:
            raise StorageVerificationError("csi_worker_registration_missing")
    _verify_driver_workloads(resources, storage_class.provisioner, nodes)
    return storage_class.provisioner


def _retain_storage_evidence(resources, kind: str, value) -> None:
    resources.receipts.append({
        "storage_component": kind,
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "object": resources.api.sanitize_for_serialization(value),
    })


def _default_class(storage_class) -> bool:
    annotations = storage_class.metadata.annotations or {}
    return any(annotations.get(name) == "true" for name in (
        "storageclass.kubernetes.io/is-default-class",
        "storageclass.beta.kubernetes.io/is-default-class",
    ))


def _verify_driver_workloads(resources, driver: str, nodes: list) -> None:
    pods = resources.core.list_pod_for_all_namespaces().items
    driver_pods = [pod for pod in pods if _driver_pod(pod, driver)]
    _retain_storage_evidence(resources, "PodList", driver_pods)
    ready_nodes = set()
    for pod in driver_pods:
        _verify_driver_owner(resources, pod)
        if pod.metadata.deletion_timestamp or pod.status.phase != "Running":
            raise StorageVerificationError("csi_workload_unhealthy")
        statuses = pod.status.container_statuses or []
        if not statuses or not all(status.ready for status in statuses):
            raise StorageVerificationError("csi_workload_unhealthy")
        ready_nodes.add(pod.spec.node_name)
    if not {node.metadata.name for node in nodes}.issubset(ready_nodes):
        raise StorageVerificationError("csi_worker_workload_missing")


def _driver_pod(pod, driver: str) -> bool:
    return any(driver in " ".join((container.args or []) + (container.command or []))
               or "csi-mounted-fs-path" in container.image
               for container in pod.spec.containers)


def _verify_driver_owner(resources, pod) -> None:
    owners = [owner for owner in pod.metadata.owner_references or [] if owner.controller]
    if len(owners) != 1 or owners[0].kind != "DaemonSet":
        raise StorageVerificationError("csi_owner_mismatch")
    owner = owners[0]
    daemon = resources.apps.read_namespaced_daemon_set(owner.name, pod.metadata.namespace)
    _retain_storage_evidence(resources, "DaemonSet", daemon)
    status = daemon.status
    valid = (daemon.metadata.uid == owner.uid and not daemon.metadata.deletion_timestamp
             and status.observed_generation == daemon.metadata.generation
             and status.desired_number_scheduled > 0
             and status.updated_number_scheduled == status.desired_number_scheduled
             and status.number_ready == status.desired_number_scheduled
             and status.number_available == status.desired_number_scheduled
             and not status.number_unavailable and not status.number_misscheduled)
    if not valid:
        raise StorageVerificationError("csi_generation_unhealthy")


def verify_bound_claim(resources, claim, driver: str):
    """Bind PVC and PV evidence to the exact owned claim and filesystem driver.

    Args:
        resources: Cluster-bound Kubernetes access.
        claim: Created RWX claim identity.
        driver: Validated filesystem provisioner.
    Returns:
        Exact dynamically provisioned PersistentVolume.
    Raises:
        StorageVerificationError: Claim binding, ownership, or driver differs.
    """
    current = resources.core.read_namespaced_persistent_volume_claim(
        claim.metadata.name, resources.namespace,
    )
    if current.metadata is None or current.status is None or current.spec is None:
        raise StorageVerificationError("pvc_binding_mismatch")
    if current.metadata.uid != claim.metadata.uid or current.status.phase != "Bound":
        raise StorageVerificationError("pvc_binding_mismatch")
    if current.spec.storage_class_name != STORAGE_CLASS:
        raise StorageVerificationError("pvc_default_class_mismatch")
    if "ReadWriteMany" not in (current.status.access_modes or []) or not current.spec.volume_name:
        raise StorageVerificationError("pvc_rwx_missing")
    volume = resources.core.read_persistent_volume(current.spec.volume_name)
    if volume.metadata is None or volume.spec is None:
        raise StorageVerificationError("volume_claim_identity_mismatch")
    reference = volume.spec.claim_ref
    if not reference or reference.uid != claim.metadata.uid:
        raise StorageVerificationError("volume_claim_identity_mismatch")
    if reference.name != claim.metadata.name or reference.namespace != resources.namespace:
        raise StorageVerificationError("volume_claim_identity_mismatch")
    if not volume.spec.csi or volume.spec.csi.driver != driver:
        raise StorageVerificationError("volume_driver_mismatch")
    _verify_claim_metadata(resources, current, volume)
    if volume.spec.persistent_volume_reclaim_policy != "Delete":
        raise StorageVerificationError("volume_cleanup_policy_mismatch")
    return volume


def _verify_claim_metadata(resources, claim, volume) -> None:
    if (claim.metadata.labels or {}).get(OWNER_LABEL) != resources.run_id:
        raise StorageVerificationError("ownership_mismatch")
    if claim.metadata.deletion_timestamp or volume.metadata.deletion_timestamp:
        raise StorageVerificationError("pvc_binding_mismatch")
    if volume.spec.storage_class_name != STORAGE_CLASS or "ReadWriteMany" not in (volume.spec.access_modes or []):
        raise StorageVerificationError("volume_driver_mismatch")
    if volume.spec.volume_mode not in {None, "Filesystem"} or not volume.spec.csi.volume_handle:
        raise StorageVerificationError("volume_driver_mismatch")


def require_unchanged_nodes(before: list, after: list) -> None:
    """Reject qualification evidence when a worker was replaced or rebooted.

    Args:
        before: Initial node inventory.
        after: Fresh node inventory after qualification and cleanup.
    Raises:
        StorageVerificationError: Immutable worker or boot identity changed.
    """
    def identities(nodes):
        return {(node.metadata.name, node.metadata.uid, node.status.node_info.boot_id)
                for node in nodes}
    if identities(before) != identities(after):
        raise StorageVerificationError("stale_worker_evidence")
