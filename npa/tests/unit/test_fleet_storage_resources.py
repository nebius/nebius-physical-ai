"""Verify storage resource ownership and evidence validation using mocked Kubernetes APIs."""

from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from kubernetes.client.exceptions import ApiException
import pytest

from npa.fleet import storage_checks, storage_resources
from npa.fleet.storage_resources import OWNER_LABEL, PROBE_IMAGE, STORAGE_CLASS

_DRIVER = "mounted-fs-path.csi.nebius.ai"
_RUN_ID = "a" * 32


def _metadata(name, uid=None, **changes):
    return SimpleNamespace(name=name, uid=uid or name + "-uid", labels={OWNER_LABEL: _RUN_ID},
                           annotations={}, deletion_timestamp=None, owner_references=[], namespace="kube-system",
                           generation=1, **changes)


def _node(role="cpu", index=0):
    metadata = _metadata(f"synthetic-{role}-worker-{index}")
    metadata.labels.update({"node.kubernetes.io/instance-type": role + "-platform",
                            "nebius.com/resource-preset": role + "-preset"})
    return SimpleNamespace(metadata=metadata, spec=SimpleNamespace(unschedulable=False),
                           status=SimpleNamespace(node_info=SimpleNamespace(boot_id="boot-id"),
                                                  conditions=[SimpleNamespace(type="Ready", status="True")]))


def _cluster():
    return SimpleNamespace(cpu_count=lambda: 1, gpu_count=lambda: 1,
                           cpu_nodes=SimpleNamespace(count=1, platform="cpu-platform", preset="cpu-preset"),
                           gpu_nodes=SimpleNamespace(count=1, platform="gpu-platform", preset="gpu-preset"),
                           filestore_mount_path="/mnt/data", filestore_mount_tag="data",
                           filestore_disk_size_gibibytes=1024)


def _pod(node=None):
    node = node or _node()
    state = SimpleNamespace(waiting=None, terminated=SimpleNamespace(exit_code=0))
    container = SimpleNamespace(args=[], command=["python3", "-c", "pass"], image=PROBE_IMAGE)
    return SimpleNamespace(
        metadata=_metadata("synthetic-probe"),
        spec=SimpleNamespace(node_name=node.metadata.name, containers=[container]),
        status=SimpleNamespace(phase="Succeeded", conditions=[],
                               container_statuses=[SimpleNamespace(ready=True, state=state,
                                                                   image_id=PROBE_IMAGE, restart_count=0)]),
    )


@pytest.fixture
def resources(monkeypatch):
    for name in ("CoreV1Api", "StorageV1Api", "AppsV1Api"):
        monkeypatch.setattr(storage_resources.client, name, MagicMock())
    monkeypatch.setattr(storage_resources.time, "sleep", lambda _: None)
    return storage_resources.StorageResources(MagicMock(), _RUN_ID)


@pytest.fixture
def driver_resources(resources):
    storage_class = SimpleNamespace(metadata=_metadata(STORAGE_CLASS), reclaim_policy="Delete",
                                    provisioner=_DRIVER)
    storage_class.metadata.annotations = {"storageclass.kubernetes.io/is-default-class": "true"}
    resources.storage.list_storage_class.return_value.items = [storage_class]
    resources.storage.read_csi_driver.return_value.metadata.name = _DRIVER
    registration = SimpleNamespace(name=_DRIVER, node_id="synthetic-node-id")
    resources.storage.read_csi_node.return_value.spec.drivers = [registration]
    pods = [_pod(_node("cpu")), _pod(_node("gpu"))]
    for pod in pods:
        pod.spec.containers[0].image = "synthetic/csi-mounted-fs-path:fixture"
        pod.spec.containers[0].args = ["--drivername=" + _DRIVER]
        pod.status.phase = "Running"
        pod.metadata.owner_references = [SimpleNamespace(controller=True, kind="DaemonSet",
                                                        name="synthetic-driver", uid="driver-uid")]
    daemon = SimpleNamespace(metadata=_metadata("synthetic-driver", uid="driver-uid"),
                             status=SimpleNamespace(observed_generation=1, desired_number_scheduled=2,
                                                    updated_number_scheduled=2, number_ready=2,
                                                    number_available=2, number_unavailable=0,
                                                    number_misscheduled=0))
    resources.apps.read_namespaced_daemon_set.return_value = daemon
    resources.core.list_pod_for_all_namespaces.return_value.items = pods
    return resources


@pytest.fixture
def binding(resources):
    claim = SimpleNamespace(metadata=_metadata("synthetic-claim"),
                            status=SimpleNamespace(phase="Bound", access_modes=["ReadWriteMany"]),
                            spec=SimpleNamespace(storage_class_name=STORAGE_CLASS,
                                                 volume_name="synthetic-volume"))
    reference = SimpleNamespace(uid=claim.metadata.uid, name=claim.metadata.name, namespace="default")
    volume = SimpleNamespace(metadata=_metadata("synthetic-volume"), spec=SimpleNamespace(
        claim_ref=reference, csi=SimpleNamespace(driver=_DRIVER, volume_handle="synthetic-volume-handle"),
        persistent_volume_reclaim_policy="Delete", storage_class_name=STORAGE_CLASS,
        volume_mode="Filesystem", access_modes=["ReadWriteMany"],
    ))
    resources.core.read_namespaced_persistent_volume_claim.return_value = claim
    resources.core.read_persistent_volume.return_value = volume
    return resources, claim, volume


def test_create_tracks_provider_confirmed_uid(resources):
    pod = _pod()
    resources.core.create_namespaced_pod.return_value = pod
    result = resources.create("pod", {"metadata": {"name": pod.metadata.name}})
    assert result is pod
    assert len(resources.owned) == 1
    assert resources.owned[0]["uid"] == pod.metadata.uid
    assert resources.owned[0]["name"] == pod.metadata.name
    assert resources.owned[0]["kind"] == "pod"


def test_ambiguous_create_is_recovered_only_with_exact_run_label(resources):
    pod = _pod()
    resources.core.create_namespaced_pod.side_effect = ApiException(status=500)
    with pytest.raises(ApiException):
        resources.create("pod", {"metadata": {"name": pod.metadata.name}})
    resources.core.read_namespaced_pod.side_effect = [pod, ApiException(status=404)]
    assert resources.cleanup() == []
    options = resources.core.delete_namespaced_pod.call_args.kwargs["body"]
    assert options.preconditions.uid == pod.metadata.uid
    assert options.propagation_policy == "Foreground"


@pytest.mark.parametrize("mismatch", ["uid", "label"])
def test_remove_refuses_replaced_or_unowned_resource(resources, mismatch):
    pod = _pod()
    record = {"kind": "pod", "name": pod.metadata.name, "uid": pod.metadata.uid}
    if mismatch == "uid":
        pod.metadata.uid = "replacement-uid"
    else:
        pod.metadata.labels[OWNER_LABEL] = "b" * 32
    resources.core.read_namespaced_pod.return_value = pod
    with pytest.raises(storage_resources.StorageVerificationError, match="ownership_mismatch"):
        resources.remove(record)
    resources.core.delete_namespaced_pod.assert_not_called()


def test_cleanup_attempts_every_pod_before_claim_when_one_fails(resources):
    resources.owned = [{"kind": "persistent_volume_claim", "name": "claim", "uid": "claim-uid"},
                       {"kind": "pod", "name": "first", "uid": "first-uid"},
                       {"kind": "pod", "name": "second", "uid": "second-uid"}]
    resources.remove = MagicMock(side_effect=[ApiException(status=403), None, None])
    assert resources.cleanup() == ["resource_cleanup_failed"]
    assert [call.args[0]["name"] for call in resources.remove.call_args_list] == ["first", "second", "claim"]


def test_cleanup_claims_false_preserves_claim_for_filesystem_recovery(resources):
    resources.owned = [{"kind": "persistent_volume_claim", "name": "claim", "uid": "claim-uid"},
                       {"kind": "pod", "name": "pod", "uid": "pod-uid"}]
    resources.remove = MagicMock()
    assert resources.cleanup(claims=False) == []
    assert resources.remove.call_count == 1
    assert resources.remove.call_args.args[0]["kind"] == "pod"


def test_remove_verifies_absence_and_counts_deletion(resources):
    pod = _pod()
    record = {"kind": "pod", "name": pod.metadata.name, "uid": pod.metadata.uid}
    resources.core.read_namespaced_pod.side_effect = [pod, pod, ApiException(status=404)]
    resources.remove(record)
    assert resources.removed == 1 and resources.core.read_namespaced_pod.call_count == 3


def test_resource_timeout_remains_failure_for_outer_cleanup(resources):
    pod = _pod()
    resources.core.read_namespaced_pod.side_effect = TimeoutError("synthetic transport timeout")
    with pytest.raises(TimeoutError):
        resources._result(pod, _node(), "write")


@pytest.mark.parametrize("reason", ["ImagePullBackOff", "ErrImagePull", "CreateContainerError"])
def test_workload_start_failure_is_detected(reason):
    pod = _pod()
    pod.status.phase = "Pending"
    pod.status.container_statuses[0].state.waiting = SimpleNamespace(reason=reason)
    with pytest.raises(storage_resources.StorageVerificationError, match="workload_start_failed"):
        storage_resources._check_pod_failure(pod)


@pytest.mark.parametrize("phase", ["Failed", "Unknown"])
def test_terminal_workload_failure_is_detected(phase):
    pod = _pod()
    pod.status.phase = phase
    with pytest.raises(storage_resources.StorageVerificationError, match="workload_failed"):
        storage_resources._check_pod_failure(pod)


def test_unschedulable_worker_is_failure():
    pod = _pod()
    pod.status.conditions = [SimpleNamespace(type="PodScheduled", reason="Unschedulable")]
    with pytest.raises(storage_resources.StorageVerificationError, match="worker_unschedulable"):
        storage_resources._check_pod_failure(pod)


@pytest.mark.parametrize("output", ["", "not JSON", "[]", '{"passed":false}'])
def test_missing_or_partial_pod_evidence_fails(resources, output):
    pod = _pod()
    resources.core.read_namespaced_pod.return_value = pod
    resources.core.read_namespaced_pod_log.return_value.read.return_value = output.encode()
    with pytest.raises(storage_resources.StorageVerificationError):
        resources._result(pod, _node(), "write")


def test_wrong_runtime_digest_fails(resources):
    pod = _pod()
    pod.status.container_statuses[0].image_id = "docker.io/library/python@sha256:" + "0" * 64
    resources.core.read_namespaced_pod.return_value = pod
    resources.core.read_namespaced_pod_log.return_value.read.return_value = json.dumps({"passed": True, "action": "write"}).encode()
    with pytest.raises(storage_resources.StorageVerificationError, match="image_digest_mismatch"):
        resources._result(pod, _node(), "write")


def test_json_logs_bypass_kubernetes_primitive_deserialization(resources):
    pod = _pod()
    resources.core.read_namespaced_pod.return_value = pod
    response = resources.core.read_namespaced_pod_log.return_value
    response.read.return_value = b'{"passed":true,"action":"write"}\n'
    assert resources._result(pod, _node(), "write") == {"passed": True, "action": "write"}
    resources.core.read_namespaced_pod_log.assert_called_once_with(
        pod.metadata.name, resources.namespace, _preload_content=False,
    )
    response.release_conn.assert_called_once_with()


def test_log_connection_is_released_after_invalid_utf8(resources):
    response = resources.core.read_namespaced_pod_log.return_value
    response.read.return_value = b'\xff'
    with pytest.raises(UnicodeDecodeError):
        resources._pod_output(_pod())
    response.release_conn.assert_called_once_with()


def test_node_pinning_default_class_and_names_are_concurrency_safe():
    node, cluster = _node(), _cluster()
    first = storage_resources._probe_pod(_RUN_ID, node, cluster, {"action": "host"}, None)
    second = storage_resources._probe_pod(_RUN_ID, node, cluster, {"action": "host"}, None)
    assert first["metadata"]["name"] != second["metadata"]["name"]
    assert first["metadata"]["labels"] == {OWNER_LABEL: _RUN_ID}
    assert "nodeName" not in first["spec"]
    term = first["spec"]["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]
    assert term["nodeSelectorTerms"][0]["matchFields"][0]["values"] == [node.metadata.name]
    claim = storage_resources.claim_manifest(_RUN_ID)
    assert "storageClassName" not in claim["spec"]
    assert claim["spec"]["accessModes"] == ["ReadWriteMany"]
    assert first["spec"]["automountServiceAccountToken"] is False
    assert first["spec"]["containers"][0]["image"] == PROBE_IMAGE


def test_all_cpu_and_gpu_nodes_must_match_declared_shape(resources):
    nodes = [_node("gpu"), _node("cpu")]
    resources.core.list_node.return_value.items = nodes
    assert len(storage_checks.storage_nodes(resources, _cluster())) == 2
    nodes[1].metadata.labels["nebius.com/resource-preset"] = "wrong-preset"
    with pytest.raises(storage_resources.StorageVerificationError, match="worker_shape_mismatch"):
        storage_checks.storage_nodes(resources, _cluster())


@pytest.mark.parametrize("failure", ["missing", "duplicate_uid", "unready", "cordoned", "boot_missing"])
def test_missing_or_unhealthy_worker_inventory_fails(resources, failure):
    nodes = [_node("gpu"), _node("cpu")]
    if failure == "missing":
        nodes.pop()
    elif failure == "duplicate_uid":
        nodes[1].metadata.uid = nodes[0].metadata.uid
    elif failure == "unready":
        nodes[1].status.conditions = []
    elif failure == "cordoned":
        nodes[1].spec.unschedulable = True
    else:
        nodes[1].status.node_info.boot_id = ""
    resources.core.list_node.return_value.items = nodes
    with pytest.raises(storage_resources.StorageVerificationError):
        storage_checks.storage_nodes(resources, _cluster())


def test_every_worker_requires_driver_registration_and_healthy_workload(driver_resources):
    nodes = [_node("cpu"), _node("gpu")]
    assert storage_checks.verify_storage_driver(driver_resources, nodes) == _DRIVER
    assert driver_resources.storage.read_csi_node.call_count == 2
    driver_resources.storage.read_csi_node.return_value.spec.drivers = []
    with pytest.raises(storage_resources.StorageVerificationError, match="csi_worker_registration_missing"):
        storage_checks.verify_storage_driver(driver_resources, nodes)


@pytest.mark.parametrize("failure", ["missing_default", "duplicate_default", "retain", "pod_unready", "pod_missing"])
def test_csi_health_failures_are_closed(driver_resources, failure):
    classes = driver_resources.storage.list_storage_class.return_value.items
    pods = driver_resources.core.list_pod_for_all_namespaces.return_value.items
    if failure == "missing_default":
        classes[0].metadata.annotations = {}
    elif failure == "duplicate_default":
        classes.append(deepcopy(classes[0]))
    elif failure == "retain":
        classes[0].reclaim_policy = "Retain"
    elif failure == "pod_unready":
        pods[0].status.container_statuses[0].ready = False
    else:
        pods.pop()
    with pytest.raises(storage_resources.StorageVerificationError):
        storage_checks.verify_storage_driver(driver_resources, [_node("cpu"), _node("gpu")])


def test_bound_claim_and_volume_require_exact_owned_link(binding):
    resources, claim, volume = binding
    assert storage_checks.verify_bound_claim(resources, claim, _DRIVER) is volume
    volume.spec.claim_ref.uid = "unrelated-uid"
    with pytest.raises(storage_resources.StorageVerificationError, match="volume_claim_identity_mismatch"):
        storage_checks.verify_bound_claim(resources, claim, _DRIVER)


@pytest.mark.parametrize("failure", ["pending", "class", "rwx", "driver", "retain", "namespace"])
def test_incomplete_or_mismatched_volume_evidence_fails(binding, failure):
    resources, claim, volume = binding
    if failure == "pending":
        claim.status.phase = "Pending"
    elif failure == "class":
        claim.spec.storage_class_name = "different-class"
    elif failure == "rwx":
        claim.status.access_modes = ["ReadWriteOnce"]
    elif failure == "driver":
        volume.spec.csi.driver = "different-driver"
    elif failure == "retain":
        volume.spec.persistent_volume_reclaim_policy = "Retain"
    else:
        volume.spec.claim_ref.namespace = "different-namespace"
    with pytest.raises(storage_resources.StorageVerificationError):
        storage_checks.verify_bound_claim(resources, claim, _DRIVER)


@pytest.mark.parametrize("identity", ["uid", "boot_id", "name"])
def test_replacement_or_reboot_makes_previous_evidence_stale(identity):
    before = [_node()]
    after = deepcopy(before)
    target = after[0].status.node_info if identity == "boot_id" else after[0].metadata
    setattr(target, identity, "replacement-identity")
    with pytest.raises(storage_resources.StorageVerificationError, match="stale_worker_evidence"):
        storage_checks.require_unchanged_nodes(before, after)


def test_missing_owner_labels_fail_closed(resources):
    pod = _pod()
    pod.metadata.labels = None
    resources.core.read_namespaced_pod.return_value = pod
    record = {"kind": "pod", "name": pod.metadata.name, "uid": pod.metadata.uid}
    with pytest.raises(storage_resources.StorageVerificationError, match="ownership_mismatch"):
        resources.remove(record)
    resources.core.delete_namespaced_pod.assert_not_called()


@pytest.mark.parametrize("failure", ["node_missing", "exit_code", "restarts", "digest_suffix", "action"])
def test_success_marker_cannot_substitute_for_complete_runtime_evidence(resources, failure):
    pod = _pod()
    output = {"passed": True, "action": "write", "checksum": "f" * 64}
    status = pod.status.container_statuses[0]
    if failure == "node_missing":
        pod.spec.node_name = None
    elif failure == "exit_code":
        status.state.terminated.exit_code = 1
    elif failure == "restarts":
        status.restart_count = 1
    elif failure == "digest_suffix":
        status.image_id += "garbage"
    else:
        output["action"] = "read"
    resources.core.read_namespaced_pod.return_value = pod
    resources.core.read_namespaced_pod_log.return_value.read.return_value = json.dumps(output).encode()
    with pytest.raises(storage_resources.StorageVerificationError):
        resources._result(pod, _node(), "write")


def test_storage_class_cannot_silently_change_driver(driver_resources):
    driver_resources.storage.list_storage_class.return_value.items[0].provisioner = "unrelated-driver"
    with pytest.raises(storage_resources.StorageVerificationError):
        storage_checks.verify_storage_driver(driver_resources, [_node("cpu"), _node("gpu")])


@pytest.mark.parametrize("failure", ["missing_owner", "changed_owner", "stale_generation", "unavailable"])
def test_csi_daemon_owner_and_generation_are_verified(driver_resources, failure):
    pod = driver_resources.core.list_pod_for_all_namespaces.return_value.items[0]
    daemon = driver_resources.apps.read_namespaced_daemon_set.return_value
    if failure == "missing_owner":
        pod.metadata.owner_references = []
    elif failure == "changed_owner":
        daemon.metadata.uid = "replacement-uid"
    elif failure == "stale_generation":
        daemon.status.observed_generation = 0
    else:
        daemon.status.number_available = 1
    with pytest.raises(storage_resources.StorageVerificationError):
        storage_checks.verify_storage_driver(driver_resources, [_node("cpu"), _node("gpu")])


@pytest.mark.parametrize("failure", ["unowned_claim", "deleting", "volume_class", "volume_rwx", "block", "handle"])
def test_claim_and_volume_ownership_and_filesystem_contract_are_required(binding, failure):
    resources, claim, volume = binding
    if failure == "unowned_claim":
        claim.metadata.labels = {}
    elif failure == "deleting":
        volume.metadata.deletion_timestamp = "synthetic-deletion-time"
    elif failure == "volume_class":
        volume.spec.storage_class_name = "wrong-class"
    elif failure == "volume_rwx":
        volume.spec.access_modes = ["ReadWriteOnce"]
    elif failure == "block":
        volume.spec.volume_mode = "Block"
    else:
        volume.spec.csi.volume_handle = ""
    with pytest.raises(storage_resources.StorageVerificationError):
        storage_checks.verify_bound_claim(resources, claim, _DRIVER)


def test_driver_health_retains_exact_private_component_snapshots(driver_resources):
    observed = []

    def snapshot(value):
        observed.append(value)
        return {"snapshot_index": len(observed)}

    driver_resources.api.sanitize_for_serialization.side_effect = snapshot
    unrelated = _pod(_node("cpu", 1))
    unrelated.spec.containers[0].image = "synthetic/application:fixture"
    unrelated.spec.containers[0].args = []
    driver_resources.core.list_pod_for_all_namespaces.return_value.items.append(unrelated)
    assert storage_checks.verify_storage_driver(driver_resources, [_node("cpu"), _node("gpu")]) == _DRIVER
    receipts = driver_resources.receipts
    assert [item["storage_component"] for item in receipts] == [
        "StorageClassList", "CSIDriver", "CSINode", "CSINode", "PodList", "DaemonSet", "DaemonSet",
    ]
    assert all(item["observed_at"] and item["object"]["snapshot_index"] == index + 1
               for index, item in enumerate(receipts))
    assert observed[0] == driver_resources.storage.list_storage_class.return_value.items
    assert observed[1] is driver_resources.storage.read_csi_driver.return_value
    assert observed[2] is driver_resources.storage.read_csi_node.return_value
    assert len(observed[4]) == 2 and unrelated not in observed[4]
    assert observed[5] is driver_resources.apps.read_namespaced_daemon_set.return_value


@pytest.mark.parametrize("failure", ["default", "registration", "daemon"])
def test_failed_driver_checks_preserve_the_failing_private_snapshot(driver_resources, failure):
    driver_resources.api.sanitize_for_serialization.side_effect = lambda value: value
    if failure == "default":
        driver_resources.storage.list_storage_class.return_value.items[0].metadata.annotations = {}
        expected_kind = "StorageClassList"
    elif failure == "registration":
        driver_resources.storage.read_csi_node.return_value.spec.drivers = []
        expected_kind = "CSINode"
    else:
        driver_resources.apps.read_namespaced_daemon_set.return_value.status.number_ready = 0
        expected_kind = "DaemonSet"
    with pytest.raises(storage_resources.StorageVerificationError):
        storage_checks.verify_storage_driver(driver_resources, [_node("cpu"), _node("gpu")])
    assert driver_resources.receipts[-1]["storage_component"] == expected_kind
    assert driver_resources.receipts[-1]["object"] is not None


def test_failed_mount_event_is_uid_scoped_retained_and_sanitized(resources):
    pod = _pod()
    event = SimpleNamespace(type="Warning", reason="FailedMount",
                            message="synthetic private filesystem diagnostic",
                            involved_object=SimpleNamespace(uid=pod.metadata.uid))
    resources.core.list_namespaced_event.return_value.items = [event]
    snapshot = [{"type": "Warning", "reason": "FailedMount"}]
    resources.api.sanitize_for_serialization.return_value = snapshot
    with pytest.raises(storage_resources.StorageVerificationError) as failure:
        resources._check_pod_events(pod)
    assert str(failure.value) == "workload_start_failed"
    assert event.message not in str(failure.value)
    resources.core.list_namespaced_event.assert_called_once_with(
        resources.namespace, field_selector=f"involvedObject.uid={pod.metadata.uid}",
    )
    resources.api.sanitize_for_serialization.assert_called_once_with([event])
    assert resources.receipts == [{"pod_events": snapshot}]


@pytest.mark.parametrize("has_event", [False, True])
def test_benign_pod_events_pass_without_failure_receipts(resources, has_event):
    pod = _pod()
    event = SimpleNamespace(type="Normal", reason="Scheduled",
                            involved_object=SimpleNamespace(uid=pod.metadata.uid))
    resources.core.list_namespaced_event.return_value.items = [event] if has_event else []
    resources._check_pod_events(pod)
    resources.core.list_namespaced_event.assert_called_once_with(
        resources.namespace, field_selector=f"involvedObject.uid={pod.metadata.uid}",
    )
    assert not resources.receipts
    resources.api.sanitize_for_serialization.assert_not_called()


@pytest.mark.parametrize(("missing_field", "category"), [
    ("metadata", "pvc_binding_mismatch"), ("access_modes", "pvc_rwx_missing"),
])
def test_typed_bound_claim_missing_evidence_raises_domain_failure(binding, missing_field, category):
    resources, claim, _ = binding
    current = storage_resources.client.V1PersistentVolumeClaim(
        metadata=storage_resources.client.V1ObjectMeta(name=claim.metadata.name, uid=claim.metadata.uid),
        spec=storage_resources.client.V1PersistentVolumeClaimSpec(
            storage_class_name=STORAGE_CLASS, volume_name="synthetic-volume",
        ),
        status=storage_resources.client.V1PersistentVolumeClaimStatus(
            phase="Bound", access_modes=["ReadWriteMany"],
        ),
    )
    if missing_field == "metadata":
        current.metadata = None
    else:
        current.status.access_modes = None
    resources.core.read_namespaced_persistent_volume_claim.return_value = current
    with pytest.raises(storage_resources.StorageVerificationError, match=category):
        storage_checks.verify_bound_claim(resources, claim, _DRIVER)
    resources.core.read_persistent_volume.assert_not_called()
