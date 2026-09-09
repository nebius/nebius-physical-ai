"""Exercise fail-closed Fleet storage orchestration without infrastructure access."""

from contextlib import nullcontext
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from kubernetes.client.exceptions import ApiException
import pytest

from npa.fleet import spec_from_mapping
from npa.fleet import storage_verification as verification
from npa.fleet.storage_resources import StorageVerificationError, _node_token


@pytest.fixture
def fleet_spec():
    return spec_from_mapping({
        "apiVersion": "npa.fleet/v0.0.1", "name": "storage-unit",
        "tenant_id": "tenant-test", "region": "us-central1",
        "defaults": {
            "cpu_nodes": {"count": 1, "platform": "cpu-d3", "preset": "4vcpu-16gb"},
            "gpu_nodes": {"count": 1, "platform": "gpu-rtx6000",
                          "preset": "1gpu-24vcpu-218gb"},
            "enable_filestore": True, "filestore_disk_size_gibibytes": 1024,
        },
        "projects": [{"name": "unit-project", "clusters": [{"name": "unit-cluster"}]}],
    })


def _node(name, platform, preset):
    return SimpleNamespace(
        metadata=SimpleNamespace(name=name, uid=name + "-uid", labels={
            "node.kubernetes.io/instance-type": platform,
            "nebius.com/resource-preset": preset,
        }),
        status=SimpleNamespace(node_info=SimpleNamespace(boot_id=name + "-boot")),
    )


def _host_evidence(cluster):
    return {"source_matches": True, "nofail": True, "read_write": True,
            "probe_deleted": True, "filesystem_type": "virtiofs",
            "requested_bytes": cluster.filestore_disk_size_gibibytes * 1024**3,
            "capacity_bytes": cluster.filestore_disk_size_gibibytes * 1024**3,
            "fragment_size": 4096,
            "checksum": hashlib.sha256(b"unit-host-probe").hexdigest()}


class FakeStorageResources:
    """Record orchestration and synthesize only explicitly owned probe results."""

    def __init__(self, nodes, cluster):
        self.nodes = nodes
        self.cluster = cluster
        self.api = SimpleNamespace(sanitize_for_serialization=lambda value: value)
        self.namespace = "default"
        self.run_id = "a" * 32
        self.owned = []
        self.receipts = []
        self.actions = []
        self.fail_action = None
        self.cleanup_failures = []
        self.resources_absent = True
        self.empty_cleanup_evidence = False
        self.cross_node_mismatch = False
        self.backing_mismatch = False
        self.backing_absent = True
        self.claim = None
        self.core = SimpleNamespace(
            read_namespaced_persistent_volume_claim=lambda *args: self.claim,
            read_persistent_volume=Mock(side_effect=ApiException(status=404)),
        )

    def create(self, kind, body):
        name = body["metadata"]["name"]
        self.claim = SimpleNamespace(metadata=SimpleNamespace(name=name, uid="claim-unit"),
                                     status=SimpleNamespace(phase="Bound"))
        self.owned.append({"kind": kind, "name": name, "uid": "claim-unit"})
        return self.claim

    def run_phase(self, nodes, cluster, configuration, *, claim=None):
        action = configuration["action"]
        self.actions.append((action, configuration["root_path"], len(nodes), claim))
        for node in nodes:
            self.owned.append({"kind": "pod", "name": str(len(self.owned)),
                               "uid": node.metadata.uid + str(len(self.owned))})
        if action == self.fail_action:
            raise StorageVerificationError("workload_failed")
        if action == "host":
            return [_host_evidence(cluster) for _ in nodes]
        backing = "csi-mounted-fs-path-data/unit-volume-handle"
        if self.backing_mismatch:
            backing = "csi-mounted-fs-path-data/unrelated-volume"
        if action == "audit_backing":
            return [{"backing_directory_absent": self.backing_absent} for _ in nodes]
        if action == "write":
            return [{"written": True, "backing_relative_path": backing,
                     "checksum": hashlib.sha256(node.metadata.uid.encode()).hexdigest()}
                    for node in nodes]
        if action == "read":
            expected = {} if self.cross_node_mismatch else configuration["expected_checksums"]
            return [{"verified_checksums": expected, "read_count": len(nodes),
                     "backing_relative_path": backing} for _ in nodes]
        if self.empty_cleanup_evidence:
            return []
        return [{"run_directory_absent": True, "backing_relative_path": backing} for _ in nodes]

    def cleanup(self, *, claims=True):
        self.actions.append(("resource_cleanup", "", 0, None))
        if not self.cleanup_failures:
            for record in self.owned:
                if claims or record["kind"] == "pod":
                    record["absent"] = self.resources_absent
        return list(self.cleanup_failures)

    def _read(self, record):
        return None if self.resources_absent else object()


@pytest.fixture
def qualification(monkeypatch, fleet_spec):
    cluster = fleet_spec.projects[0].clusters[0]
    nodes = [_node("cpu-test", "cpu-d3", "4vcpu-16gb"),
             _node("gpu-test", "gpu-rtx6000", "1gpu-24vcpu-218gb")]
    resources = FakeStorageResources(nodes, cluster)
    monkeypatch.setattr(verification, "storage_nodes", lambda *args: nodes)
    monkeypatch.setattr(verification, "verify_storage_driver", lambda *args: "unit-driver")
    monkeypatch.setattr(verification, "verify_bound_claim", lambda *args: {
        "metadata": {"name": "unit-volume", "uid": "unit-volume-uid"},
        "spec": {"csi": {"volumeHandle": "unit-volume-handle"}},
    })
    resources.api.sanitize_for_serialization = lambda value: value if isinstance(value, dict) else []
    return resources, cluster, nodes


def _qualify(qualification):
    resources, cluster, _ = qualification
    report = verification._target_report(0, cluster)
    verification._qualify_cluster(resources, cluster, report, {})
    return report


def test_host_and_shared_visibility_cover_both_worker_types(qualification):
    report = _qualify(qualification)
    resources, _, _ = qualification
    assert [node["role"] for node in report["nodes"]] == ["cpu", "gpu"]
    assert all(node["shared_visibility"] == "passed" for node in report["nodes"])
    assert all(node["cleanup"] == "passed" for node in report["nodes"])
    assert report["cleanup"]["probe_paths_absent"] is True
    assert report["cleanup"]["claims_removed"] == 1
    assert resources.actions[:3] == [
        ("host", "/host-storage", 2, None),
        ("write", "/data", 2, resources.claim.metadata.name),
        ("read", "/data", 2, resources.claim.metadata.name),
    ]
    assert ("audit", "/host-storage", 2, None) in resources.actions
    assert ("audit", "/data", 2, resources.claim.metadata.name) in resources.actions


@pytest.mark.parametrize("action", ["host", "write", "read"])
def test_probe_failure_still_cleans_every_created_resource(qualification, action):
    resources, cluster, _ = qualification
    resources.fail_action = action
    report = verification._target_report(0, cluster)
    with pytest.raises(StorageVerificationError, match="workload_failed"):
        verification._qualify_cluster(resources, cluster, report, {})
    assert resources.actions[-1][0] == "resource_cleanup"
    assert ("audit", "/host-storage", 2, None) in resources.actions
    assert report["cleanup"]["probe_paths_absent"] is True
    assert report["cleanup"]["claims_removed"] == int(action != "host")


def test_cross_node_mismatch_fails_and_cleans_owned_claim(qualification):
    resources, cluster, _ = qualification
    resources.cross_node_mismatch = True
    report = verification._target_report(0, cluster)
    with pytest.raises(StorageVerificationError, match="cross_node_evidence_mismatch"):
        verification._qualify_cluster(resources, cluster, report, {})
    assert report["nodes"] == []
    assert report["cleanup"]["claims_removed"] == 1
    assert report["cleanup"]["probe_paths_absent"] is True


@pytest.mark.parametrize("failure", ["resource_cleanup_failed", "ownership_mismatch"])
def test_cleanup_failure_marks_every_node_failed(qualification, failure):
    resources, _, _ = qualification
    resources.cleanup_failures = [failure]
    report = _qualify(qualification)
    assert failure in report["failures"]
    assert report["cleanup"]["probe_paths_absent"] is False
    assert all(node["cleanup"] == "failed" for node in report["nodes"])


def test_independent_resource_audit_rejects_surviving_resources(qualification):
    resources, _, _ = qualification
    resources.resources_absent = False
    report = _qualify(qualification)
    assert "resource_absence_unverified" in report["failures"]
    assert report["cleanup"]["probe_paths_absent"] is False


def test_cleanup_continues_after_host_probe_cleanup_failure(qualification):
    resources, _, _ = qualification
    resources.fail_action = "cleanup"
    report = _qualify(qualification)
    assert report["failures"].count("probe_cleanup_failed") == 4
    assert resources.actions[-1][0] == "resource_cleanup"
    assert sum(action[0] == "audit" for action in resources.actions) == 2


def test_empty_cleanup_evidence_cannot_prove_absence(qualification):
    resources, _, _ = qualification
    resources.empty_cleanup_evidence = True
    report = _qualify(qualification)
    assert report["cleanup"]["probe_paths_absent"] is False
    assert report["failures"]


@pytest.mark.parametrize(("field", "value", "category"), [
    ("filesystem_type", "ext4", "host_mount_mismatch"),
    ("source_matches", False, "partial_host_evidence"),
    ("nofail", False, "partial_host_evidence"),
    ("read_write", False, "partial_host_evidence"),
    ("probe_deleted", False, "partial_host_evidence"),
    ("capacity_bytes", None, "capacity_mismatch"),
    ("capacity_bytes", "1099511627776", "capacity_mismatch"),
    ("capacity_bytes", 1024 * 1000**3, "capacity_mismatch"),
    ("requested_bytes", 1024 * 1000**3, "host_mount_mismatch"),
    ("checksum", "unverified", "partial_host_evidence"),
])
def test_invalid_host_evidence_fails_closed(fleet_spec, field, value, category):
    cluster = fleet_spec.projects[0].clusters[0]
    evidence = _host_evidence(cluster)
    evidence[field] = value
    with pytest.raises(StorageVerificationError, match=category):
        verification._require_host_evidence([evidence], [object()], cluster)


def test_missing_worker_host_evidence_fails_closed(fleet_spec):
    cluster = fleet_spec.projects[0].clusters[0]
    with pytest.raises(StorageVerificationError, match="partial_host_evidence"):
        verification._require_host_evidence([_host_evidence(cluster)], [object(), object()], cluster)


def test_partial_shared_evidence_cannot_mark_nodes_successful(qualification):
    _, cluster, nodes = qualification
    with pytest.raises(StorageVerificationError, match="partial_cross_node_evidence"):
        verification._node_reports(nodes, cluster, [], [{}])


def test_disabled_filesystem_does_not_resolve_identity(fleet_spec, monkeypatch, tmp_path):
    cluster = fleet_spec.projects[0].clusters[0]
    cluster.enable_filestore = False
    identity = Mock(side_effect=AssertionError("disabled filesystem accessed infrastructure"))
    monkeypatch.setattr(verification, "resolve_storage_identity", identity)
    evidence = tmp_path / "private"
    report = verification.verify_storage(fleet_spec, evidence_dir=evidence)
    assert report["passed"] is True
    assert report["selected_clusters"] == report["skipped_clusters"] == 1
    assert report["verified_clusters"] == report["requested_gibibytes"] == 0
    identity.assert_not_called()


def test_target_failure_is_sanitized_and_receipted(fleet_spec, monkeypatch, tmp_path):
    failure = ApiException(status=403, reason="private-provider-message")
    monkeypatch.setattr(verification, "resolve_storage_identity", Mock(side_effect=failure))
    report = verification.verify_storage(fleet_spec, evidence_dir=tmp_path / "private")
    assert report["passed"] is False
    assert report["clusters"][0]["failures"] == ["verification_operation_failed"]
    assert "private-provider-message" not in str(report)
    receipts = list((tmp_path / "private").glob("*/*.json"))
    assert len(receipts) == 1
    assert receipts[0].stat().st_mode & 0o777 == 0o600
    assert hashlib.sha256(receipts[0].read_bytes()).hexdigest() == report["clusters"][0]["evidence_sha256"]


def test_target_timeout_is_structured_failure_with_cleanup(fleet_spec, qualification, monkeypatch, tmp_path):
    resources, _, _ = qualification
    resources.run_phase = Mock(side_effect=TimeoutError("private-endpoint"))
    identity = SimpleNamespace(kubeconfig=Path("unit-config"), evidence_sha256="b" * 64)
    monkeypatch.setattr(verification, "resolve_storage_identity", lambda *args, **kwargs: identity)
    monkeypatch.setattr(verification, "storage_client", lambda identity: nullcontext(resources.api))
    monkeypatch.setattr(verification, "StorageResources", lambda *args: resources)
    report = verification.verify_storage(fleet_spec, evidence_dir=tmp_path / "private")
    assert report["passed"] is False
    assert "verification_operation_failed" in report["clusters"][0]["failures"]
    assert resources.actions[-1][0] == "resource_cleanup"
    assert "private-endpoint" not in str(report)


def test_evidence_directories_are_unique_and_owner_only(tmp_path):
    parent = tmp_path / "private"
    first = verification._evidence_directory(parent)
    second = verification._evidence_directory(parent)
    assert first != second
    assert first.parent == second.parent == parent
    assert all(path.stat().st_mode & 0o777 == 0o700 for path in [parent, first, second])


def test_shared_evidence_requires_every_writer(qualification):
    resources, cluster, nodes = qualification
    resources.run_phase = Mock(return_value=[{"written": True, "checksum": "a" * 64}])
    claim = resources.create("persistent_volume_claim", {"metadata": {"name": "unit-claim"}})
    with pytest.raises((ValueError, StorageVerificationError)):
        verification._shared_visibility(resources, nodes, cluster, claim, "unit-driver", {})


def test_cross_node_payloads_are_distinct(qualification):
    resources, cluster, nodes = qualification
    written = {"written": True, "checksum": "a" * 64,
               "backing_relative_path": "csi-mounted-fs-path-data/unit-volume-handle"}
    resources.run_phase = Mock(return_value=[written] * 2)
    claim = resources.create("persistent_volume_claim", {"metadata": {"name": "unit-claim"}})
    with pytest.raises(StorageVerificationError, match="non_unique_payload_evidence"):
        verification._shared_visibility(resources, nodes, cluster, claim, "unit-driver", {})


def test_node_tokens_fit_the_probe_identity_contract(qualification):
    from npa.fleet.storage_probe import _token

    _, _, nodes = qualification
    tokens = [_node_token(node) for node in nodes]
    assert len(set(tokens)) == len(nodes)
    assert all(_token(token) == token for token in tokens)


def test_unhealthy_csi_prevents_any_probe_mutation(qualification, monkeypatch):
    resources, cluster, _ = qualification
    monkeypatch.setattr(verification, "verify_storage_driver", Mock(
        side_effect=StorageVerificationError("csi_worker_registration_missing"),
    ))
    report = verification._target_report(0, cluster)
    with pytest.raises(StorageVerificationError, match="csi_worker_registration_missing"):
        verification._qualify_cluster(resources, cluster, report, {})
    assert resources.actions == []
    assert resources.owned == []


def test_worker_reboot_after_probes_invalidates_evidence(qualification, monkeypatch):
    resources, cluster, nodes = qualification
    replacement = _node("gpu-test", "gpu-rtx6000", "1gpu-24vcpu-218gb")
    replacement.status.node_info.boot_id = "replacement-boot"
    monkeypatch.setattr(verification, "storage_nodes", Mock(side_effect=[
        nodes, [nodes[0], replacement],
    ]))
    report = verification._target_report(0, cluster)
    with pytest.raises(StorageVerificationError, match="stale_worker_evidence"):
        verification._qualify_cluster(resources, cluster, report, {})
    assert report["cleanup"]["probe_paths_absent"] is True
    assert resources.actions[-1][0] == "resource_cleanup"


def test_failed_deletions_are_not_reported_as_removed(qualification):
    resources, _, _ = qualification
    resources.cleanup_failures = ["resource_cleanup_failed"]
    resources.resources_absent = False
    report = _qualify(qualification)
    assert report["cleanup"]["pods_removed"] == 0
    assert report["cleanup"]["claims_removed"] == 0
    assert report["cleanup"]["probe_paths_absent"] is False


@pytest.mark.parametrize("evidence", [[None], [[]], ["invalid"]])
def test_malformed_host_record_fails_with_domain_category(fleet_spec, evidence):
    cluster = fleet_spec.projects[0].clusters[0]
    with pytest.raises(StorageVerificationError):
        verification._require_host_evidence(evidence, [object()], cluster)


def test_backing_identity_mismatch_never_qualifies_shared_storage(qualification):
    resources, cluster, _ = qualification
    resources.backing_mismatch = True
    report = verification._target_report(0, cluster)
    with pytest.raises(StorageVerificationError, match="shared_backing_identity_mismatch"):
        verification._qualify_cluster(resources, cluster, report, {})
    assert report["nodes"] == []
    assert resources.actions[-1][0] == "resource_cleanup"
    assert "backing_absence_unverified" in report["failures"]


def test_surviving_backing_path_fails_independent_cleanup_audit(qualification):
    resources, _, _ = qualification
    resources.backing_absent = False
    report = _qualify(qualification)
    assert "backing_absence_unverified" in report["failures"]
    assert report["cleanup"]["probe_paths_absent"] is False
    assert ("audit_backing", "/host-storage", 2, None) in resources.actions
    assert resources.actions[-1][0] == "resource_cleanup"


def test_backing_audit_failure_still_cleans_audit_pods(qualification):
    resources, _, _ = qualification
    resources.fail_action = "audit_backing"
    report = _qualify(qualification)
    assert "backing_absence_unverified" in report["failures"]
    assert report["cleanup"]["probe_paths_absent"] is False
    assert resources.actions[-1][0] == "resource_cleanup"


@pytest.mark.parametrize("fragment", [None, 0, -1, "4096", True])
def test_unknown_fragment_size_cannot_validate_capacity(fleet_spec, fragment):
    cluster = fleet_spec.projects[0].clusters[0]
    evidence = _host_evidence(cluster)
    evidence["fragment_size"] = fragment
    with pytest.raises(StorageVerificationError, match="capacity_mismatch"):
        verification._require_host_evidence([evidence], [object()], cluster)


def test_arbitrary_larger_filesystem_is_not_exact_requested_capacity(fleet_spec):
    cluster = fleet_spec.projects[0].clusters[0]
    evidence = _host_evidence(cluster)
    evidence["capacity_bytes"] *= 2
    with pytest.raises(StorageVerificationError, match="capacity_mismatch"):
        verification._require_host_evidence([evidence], [object()], cluster)
