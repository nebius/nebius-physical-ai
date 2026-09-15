"""Qualify storage on every worker of an explicitly selected existing Fleet."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from npa.fleet.spec import load_spec
from npa.fleet.storage_verification import verify_storage

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(0)]


def test_every_selected_fleet_worker_has_usable_shared_storage() -> None:
    if os.environ.get("NPA_FLEET_STORAGE_VERIFY") != "1":
        pytest.skip("operator must explicitly enable real Fleet storage qualification")
    spec_path = os.environ.get("NPA_FLEET_STORAGE_VERIFY_SPEC")
    evidence_path = os.environ.get("NPA_FLEET_STORAGE_EVIDENCE_DIR")
    if not spec_path or not evidence_path:
        pytest.fail("Fleet storage qualification requires private spec and evidence configuration", pytrace=False)
    spec = _live_spec(Path(spec_path))
    report = verify_storage(spec, evidence_dir=Path(evidence_path))
    targets = spec.cluster_targets()
    enabled = [cluster for _, cluster in targets
               if cluster.enable_filestore or cluster.existing_filestore]
    assert enabled, "the selected Fleet must contain a filesystem-enabled target"
    assert report["passed"], "Fleet storage qualification failed; inspect owner-private evidence"
    assert report["selected_clusters"] == len(targets)
    assert report["verified_clusters"] == len(enabled)
    assert report["skipped_clusters"] == len(targets) - len(enabled)
    assert report["cpu_workers"] == sum(cluster.cpu_count() for cluster in enabled)
    assert report["gpu_workers"] == sum(cluster.gpu_count() for cluster in enabled)
    assert report["requested_gibibytes"] == sum(cluster.filestore_disk_size_gibibytes for cluster in enabled)
    assert len(report["clusters"]) == len(targets)
    for index, (_, cluster) in enumerate(targets):
        _assert_target(report["clusters"][index], cluster, index)


def _live_spec(path):
    try:
        return load_spec(path)
    except (OSError, ValueError, yaml.YAMLError):
        pytest.fail("owner-private Fleet declaration is unavailable or invalid", pytrace=False)


def _assert_target(report, cluster, index):
    assert report["target_index"] == index
    if not cluster.enable_filestore and not cluster.existing_filestore:
        assert report["skipped"]
        return
    assert report["passed"]
    assert report["requested_gibibytes"] == cluster.filestore_disk_size_gibibytes
    assert len(report["nodes"]) == cluster.cpu_count() + cluster.gpu_count()
    assert {node["node_index"] for node in report["nodes"]} == set(range(len(report["nodes"])))
    assert sum(node["role"] == "cpu" for node in report["nodes"]) == cluster.cpu_count()
    assert sum(node["role"] == "gpu" for node in report["nodes"]) == cluster.gpu_count()
    for node in report["nodes"]:
        assert all(node[category] == "passed" for category in
                   ("host_mount", "capacity", "host_io", "shared_visibility", "cleanup"))
    assert report["cleanup"]["pods_removed"] >= len(report["nodes"])
    assert report["cleanup"]["claims_removed"] == 1
    assert report["cleanup"]["probe_paths_absent"] is True
    assert len(report["evidence_sha256"]) == 64
