"""Qualify every selected Fleet worker's host mount and shared CSI filesystem."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import time
import uuid

from kubernetes import config
from kubernetes.client.exceptions import ApiException
from urllib3.exceptions import HTTPError

from npa.fleet.storage_checks import (
    require_unchanged_nodes,
    storage_nodes,
    verify_bound_claim,
    verify_storage_driver,
)
from npa.fleet.storage_identity import (
    StorageIdentityError,
    resolve_storage_identity,
    resolve_storage_targets,
    storage_client,
)
from npa.fleet.storage_resources import (
    StorageResources,
    STORAGE_DRIVER,
    StorageVerificationError,
    _node_token,
    claim_manifest,
)
from npa.lifecycle_intent import OperationIntent, intent_boundary

_FAILURES = (StorageIdentityError, StorageVerificationError, ApiException, HTTPError,
             OSError, ValueError, config.ConfigException)


@intent_boundary(OperationIntent.MUTATE)
def verify_storage(spec, *, only_projects=None, only_clusters=None,
                   project_prefix=None, profile=None, evidence_dir: Path | None = None) -> dict:
    """Verify declared host mounts and all-worker RWX visibility with exact cleanup.

    Args:
        spec: Loaded Fleet declaration.
        only_projects: Existing Fleet project selectors.
        only_clusters: Existing Fleet cluster selectors within project scope.
        project_prefix: Optional Fleet project display-name override.
        profile: Optional provider authentication profile override.
        evidence_dir: Owner-only directory for exact private receipts.
    Returns:
        Publication-safe target/node categories, counts, and evidence hashes.
    Raises:
        StorageIdentityError: Selection or private evidence configuration is invalid.
        OSError: Private verification evidence cannot be persisted.
    """
    targets = resolve_storage_targets(spec, only_projects=only_projects,
                                      only_clusters=only_clusters, project_prefix=project_prefix)
    directory = _evidence_directory(evidence_dir)
    run_id = uuid.uuid4().hex
    reports = []
    for index, (project, cluster) in enumerate(targets):
        report = _target_report(index, cluster)
        if not report["skipped"]:
            _verify_target(spec, project, cluster, report, run_id, directory,
                           profile=profile, project_prefix=project_prefix)
        reports.append(report)
    return _aggregate(reports)


def _evidence_directory(directory: Path | None) -> Path:
    parent = directory or Path.home() / ".npa" / "storage-verification"
    parent = Path(parent).expanduser()
    if any((ancestor / ".git").exists() for ancestor in (parent.resolve(), *parent.resolve().parents)):
        raise StorageIdentityError("exact evidence must remain outside Git checkouts")
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if parent.is_symlink() or parent.stat().st_mode & 0o077 or parent.stat().st_uid != os.getuid():
        raise StorageIdentityError("evidence directory must be owner-private")
    child = parent / uuid.uuid4().hex
    child.mkdir(mode=0o700)
    return child


def _target_report(index: int, cluster) -> dict:
    disabled = not cluster.enable_filestore and not cluster.existing_filestore
    return {"target_index": index, "passed": disabled, "skipped": disabled,
            "requested_gibibytes": 0 if disabled else cluster.filestore_disk_size_gibibytes,
            "cpu_workers": cluster.cpu_count(), "gpu_workers": cluster.gpu_count(),
            "nodes": [], "failures": [], "evidence_sha256": "",
            "cleanup": {"pods_removed": 0, "claims_removed": 0, "probe_paths_absent": False}}


def _verify_target(spec, project, cluster, report, run_id, directory, **options) -> None:
    private = {"target_index": report["target_index"], "run_id": run_id}
    try:
        identity = resolve_storage_identity(spec, project, cluster, **options)
        private["identity_sha256"] = identity.evidence_sha256
        if getattr(identity, "evidence_json", ""):
            private["identity"] = json.loads(identity.evidence_json)
        with storage_client(identity) as api:
            resources = StorageResources(api, run_id)
            private["execution"] = resources.receipts
            _qualify_cluster(resources, cluster, report, private)
    except _FAILURES as error:
        private["failure_type"] = type(error).__name__
        if isinstance(error, (StorageIdentityError, StorageVerificationError)):
            private["failure_detail"] = str(error)
        report["failures"].append(_failure_category(error))
    except BaseException:
        report["failures"].append("verification_interrupted")
        raise
    finally:
        report["passed"] = not report["failures"] and bool(report["nodes"])
        private["report"] = dict(report)
        report["evidence_sha256"] = _write_receipt(directory, report["target_index"], private)


def _failure_category(error) -> str:
    if isinstance(error, StorageVerificationError):
        return str(error)
    if isinstance(error, StorageIdentityError):
        return "target_identity_failed"
    return "verification_operation_failed"


def _qualify_cluster(resources, cluster, report, private) -> None:
    nodes = storage_nodes(resources, cluster)
    private["nodes"] = resources.api.sanitize_for_serialization(nodes)
    driver = verify_storage_driver(resources, nodes)
    claim = None
    try:
        host = resources.run_phase(nodes, cluster, {"action": "host", "root_path": "/host-storage"})
        _require_host_evidence(host, nodes, cluster)
        claim = resources.create("persistent_volume_claim", claim_manifest(resources.run_id))
        shared = _shared_visibility(resources, nodes, cluster, claim, driver, private)
        report["nodes"] = _node_reports(nodes, cluster, host, shared)
    finally:
        _cleanup_cluster(resources, nodes, cluster, claim, report, private)
        private["execution"] = resources.receipts
        private["resources"] = resources.owned
    require_unchanged_nodes(nodes, storage_nodes(resources, cluster))
    verify_storage_driver(resources, nodes)


def _shared_visibility(resources, nodes, cluster, claim, driver, private) -> list:
    written = resources.run_phase(nodes, cluster, {"action": "write", "root_path": "/data"},
                                  claim=claim.metadata.name)
    if len(written) != len(nodes) or not all(isinstance(item, dict) for item in written):
        raise StorageVerificationError("partial_write_evidence")
    volume = verify_bound_claim(resources, claim, driver)
    private["volume"] = resources.api.sanitize_for_serialization(volume)
    _record_backing(written, private)
    checksums = {}
    for node, evidence in zip(nodes, written, strict=True):
        checksum = evidence.get("checksum", "")
        if evidence.get("written") is not True or not re.fullmatch("[0-9a-f]{64}", checksum):
            raise StorageVerificationError("partial_write_evidence")
        checksums[_node_token(node)] = checksum
    if len(set(checksums.values())) != len(nodes):
        raise StorageVerificationError("non_unique_payload_evidence")
    settings = {"action": "read", "root_path": "/data", "expected_checksums": checksums}
    results = resources.run_phase(nodes, cluster, settings, claim=claim.metadata.name)
    if len(results) != len(nodes) or not all(isinstance(item, dict) for item in results):
        raise StorageVerificationError("partial_cross_node_evidence")
    _record_backing(results, private)
    for evidence in results:
        if evidence.get("verified_checksums") != checksums or evidence.get("read_count") != len(nodes):
            raise StorageVerificationError("cross_node_evidence_mismatch")
    return results


def _require_host_evidence(evidence: list, nodes: list, cluster) -> None:
    if len(evidence) != len(nodes) or not all(isinstance(item, dict) for item in evidence):
        raise StorageVerificationError("partial_host_evidence")
    requested = cluster.filestore_disk_size_gibibytes * 1024**3
    for node in evidence:
        fields = ("source_matches", "nofail", "read_write", "probe_deleted")
        if any(node.get(field) is not True for field in fields):
            raise StorageVerificationError("partial_host_evidence")
        if node.get("filesystem_type") != "virtiofs" or node.get("requested_bytes") != requested:
            raise StorageVerificationError("host_mount_mismatch")
        if type(node.get("capacity_bytes")) is not int or node["capacity_bytes"] < requested:
            raise StorageVerificationError("capacity_mismatch")
        fragment = node.get("fragment_size")
        if type(fragment) is not int or fragment <= 0 or node["capacity_bytes"] >= requested + fragment:
            raise StorageVerificationError("capacity_mismatch")
        if not re.fullmatch("[0-9a-f]{64}", node.get("checksum", "")):
            raise StorageVerificationError("partial_host_evidence")


def _node_reports(nodes: list, cluster, host: list, shared: list) -> list:
    from npa.fleet.storage_checks import _worker_role

    if len(shared) != len(nodes):
        raise StorageVerificationError("partial_cross_node_evidence")
    return [{"node_index": index, "role": _worker_role(node, cluster),
             "host_mount": "passed", "capacity": "passed", "host_io": "passed",
             "shared_visibility": "passed", "cleanup": "pending"}
            for index, node in enumerate(nodes)]


def _cleanup_cluster(resources, nodes, cluster, claim, report, private) -> None:
    failures = resources.cleanup(claims=False)
    claim = claim or _recover_claim(resources, failures)
    try:
        _cleanup_probe_paths(resources, nodes, cluster, claim, failures, private)
    finally:
        failures.extend(resources.cleanup())
    _audit_volume_deletion(resources, private, failures)
    _audit_backing_deletion(resources, nodes, cluster, private, failures)
    failures.extend(resources.cleanup())
    failures.extend(_verify_resource_absence(resources))
    report["failures"].extend(failures)
    removed = [item for item in resources.owned if item.get("absent") and item["uid"]]
    report["cleanup"] = {
        "pods_removed": sum(item["kind"] == "pod" for item in removed),
        "claims_removed": sum(item["kind"] == "persistent_volume_claim" for item in removed),
        "probe_paths_absent": not failures,
    }
    for node in report["nodes"]:
        node["cleanup"] = "failed" if failures else "passed"


def _cleanup_probe_paths(resources, nodes, cluster, claim, failures, private) -> None:
    settings = {"action": "cleanup", "node_tokens": [_node_token(node) for node in nodes]}
    _cleanup_probe_root(resources, nodes, cluster, settings, "/host-storage", None, failures)
    if claim is not None:
        _cleanup_claim_probes(resources, nodes, cluster, claim, settings, failures, private)


def _cleanup_probe_root(resources, nodes, cluster, settings, root, claim, failures) -> list:
    observations = []
    for node in nodes:
        try:
            configuration = dict(settings, root_path=root)
            results = resources.run_phase([node], cluster, configuration, claim=claim)
            observations.extend(results)
            if len(results) != 1 or results[0].get("run_directory_absent") is not True:
                raise StorageVerificationError("probe_cleanup_failed")
        except _FAILURES:
            failures.append("probe_cleanup_failed")
    try:
        results = resources.run_phase(nodes, cluster, {"action": "audit", "root_path": root}, claim=claim)
        if len(results) != len(nodes) or not all(result.get("run_directory_absent") is True for result in results):
            raise StorageVerificationError("probe_cleanup_failed")
    except _FAILURES:
        failures.append("probe_absence_unverified")
    return observations


def _recover_claim(resources, failures):
    for record in resources.owned:
        if record["kind"] != "persistent_volume_claim":
            continue
        try:
            current = resources._read(record)
            if current is not None:
                resources._adopt(record, current)
                return current
        except _FAILURES:
            failures.append("claim_identity_unverified")
    return None


def _cleanup_claim_probes(resources, nodes, cluster, claim, settings, failures, private) -> None:
    try:
        current = resources.core.read_namespaced_persistent_volume_claim(claim.metadata.name,
                                                                        resources.namespace)
        if current.metadata.uid != claim.metadata.uid:
            raise StorageVerificationError("ownership_mismatch")
        if current.status.phase == "Bound":
            volume = verify_bound_claim(resources, claim, STORAGE_DRIVER)
            private["volume"] = resources.api.sanitize_for_serialization(volume)
            observed = _cleanup_probe_root(resources, nodes, cluster, settings, "/data",
                                           claim.metadata.name, failures)
            _record_backing(observed, private)
    except _FAILURES:
        failures.append("claim_probe_cleanup_failed")


def _record_backing(results, private) -> None:
    if not results or any(not isinstance(item.get("backing_relative_path"), str) for item in results):
        raise StorageVerificationError("shared_backing_identity_mismatch")
    paths = {item.get("backing_relative_path") for item in results}
    volume = private["volume"]["spec"]["csi"]["volumeHandle"]
    expected = "csi-mounted-fs-path-data/" + volume
    if paths != {expected}:
        raise StorageVerificationError("shared_backing_identity_mismatch")
    if private.get("backing_relative_path", expected) != expected:
        raise StorageVerificationError("shared_backing_identity_mismatch")
    private["backing_relative_path"] = expected


def _audit_backing_deletion(resources, nodes, cluster, private, failures) -> None:
    if "volume" not in private:
        return
    if "backing_relative_path" not in private:
        failures.append("backing_absence_unverified")
        return
    settings = {"action": "audit_backing", "root_path": "/host-storage",
                "backing_relative_path": private["backing_relative_path"]}
    try:
        results = resources.run_phase(nodes, cluster, settings)
        if len(results) != len(nodes) or not all(item.get("backing_directory_absent") is True
                                               for item in results):
            raise StorageVerificationError("backing_cleanup_failed")
    except _FAILURES:
        failures.append("backing_absence_unverified")


def _audit_volume_deletion(resources, private, failures) -> None:
    volume = private.get("volume")
    if not volume:
        return
    try:
        while _owned_volume_exists(resources, volume):
            time.sleep(1)
    except _FAILURES:
        failures.append("volume_cleanup_failed")


def _owned_volume_exists(resources, volume) -> bool:
    try:
        current = resources.core.read_persistent_volume(volume["metadata"]["name"])
    except ApiException as error:
        if error.status == 404:
            return False
        raise
    if current.metadata.uid != volume["metadata"]["uid"]:
        raise StorageVerificationError("volume_identity_mismatch")
    return True


def _verify_resource_absence(resources) -> list[str]:
    failures = []
    for record in resources.owned:
        try:
            if resources._read(record) is not None:
                failures.append("resource_absence_unverified")
        except _FAILURES:
            failures.append("resource_absence_unverified")
    return failures


def _write_receipt(directory: Path, index: int, receipt: dict) -> str:
    content = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(content).hexdigest()
    path = directory / f"target-{index}-{digest}.json"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)
    return digest


def _aggregate(reports: list) -> dict:
    enabled = [report for report in reports if not report["skipped"]]
    successful = [report for report in enabled if report["passed"]]
    result = {"passed": all(report["passed"] for report in reports),
              "selected_clusters": len(reports), "skipped_clusters": len(reports) - len(enabled),
              "verified_clusters": len(successful), "clusters": reports}
    for name in ("cpu_workers", "gpu_workers"):
        result[name] = sum(report[name] for report in successful)
    result["requested_gibibytes"] = sum(report["requested_gibibytes"] for report in enabled)
    result["evidence_sha256"] = hashlib.sha256(json.dumps(reports, sort_keys=True).encode()).hexdigest()
    return result
