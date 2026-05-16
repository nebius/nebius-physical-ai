from __future__ import annotations

import json

import pytest

from npa.cluster.exceptions import ClusterStateError
from npa.cluster.state import (
    ClusterState,
    delete_cluster_state,
    list_local_clusters,
    load_cluster_state,
    save_cluster_state,
)


def _state(name: str = "cluster-a") -> ClusterState:
    return ClusterState(
        name=name,
        cluster_id="mk8scluster-a",
        project_id="project-a",
        region="eu-north1",
        node_count=1,
        node_platform="cpu-e2",
        node_preset="2vcpu-8gb",
        k8s_version="1.33",
        subnet_id="vpcsubnet-a",
        created_at="2026-05-14T21:46:00Z",
        last_seen_state="READY",
        node_group_id="mk8snodegroup-a",
        endpoint="https://example.invalid",
        kubeconfig_path="/tmp/kubeconfig",
    )


def test_cluster_state_roundtrip(tmp_path) -> None:
    saved = save_cluster_state(_state(), base_dir=tmp_path, metadata={"event": "test"})

    assert saved.exists()
    loaded = load_cluster_state("cluster-a", base_dir=tmp_path)
    assert loaded == _state()
    assert json.loads((tmp_path / "cluster-a" / "metadata.json").read_text()) == {"event": "test"}


def test_missing_cluster_state_returns_none(tmp_path) -> None:
    assert load_cluster_state("missing", base_dir=tmp_path) is None


def test_list_local_clusters(tmp_path) -> None:
    save_cluster_state(_state("cluster-b"), base_dir=tmp_path)
    save_cluster_state(_state("cluster-a"), base_dir=tmp_path)

    assert [cluster.name for cluster in list_local_clusters(base_dir=tmp_path)] == ["cluster-a", "cluster-b"]


def test_delete_cluster_state_removes_directory(tmp_path) -> None:
    save_cluster_state(_state(), base_dir=tmp_path)

    delete_cluster_state("cluster-a", base_dir=tmp_path)

    assert load_cluster_state("cluster-a", base_dir=tmp_path) is None


def test_corrupted_cluster_state_raises(tmp_path) -> None:
    path = tmp_path / "cluster-a"
    path.mkdir()
    (path / "cluster.json").write_text("{bad json")

    with pytest.raises(ClusterStateError):
        load_cluster_state("cluster-a", base_dir=tmp_path)
