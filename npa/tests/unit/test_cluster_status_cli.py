from __future__ import annotations

import json

from typer.testing import CliRunner

from npa.cli.cluster import app
from npa.cli.cluster import status as status_mod
from npa.cluster.api import ClusterInfo, NodeGroupInfo
from npa.cluster.state import ClusterState


runner = CliRunner()


def _state() -> ClusterState:
    return ClusterState(
        name="cluster-a",
        cluster_id="mk8scluster-a",
        project_id="project-a",
        region="eu-north1",
        node_count=1,
        node_platform="cpu-e2",
        node_preset="2vcpu-8gb",
        k8s_version="1.33",
        subnet_id="vpcsubnet-a",
        created_at="2026-05-14T21:46:00Z",
        last_seen_state="PROVISIONING",
    )


def test_status_for_named_local_cluster(monkeypatch) -> None:
    class FakeClient:
        def __init__(self, **kwargs) -> None:
            pass

        def get_cluster(self, name, *, project_id=""):
            return ClusterInfo(
                id="mk8scluster-a",
                name="cluster-a",
                project_id=project_id,
                status="READY",
                created_at="2026-05-14T21:46:00Z",
                endpoint="https://api.example.invalid",
            )

        def list_node_groups(self, cluster_id):
            return [NodeGroupInfo(id="mk8snodegroup-a", name="cluster-a-cpu", cluster_id=cluster_id, status="READY", node_count=1)]

    saved: list[ClusterState] = []
    monkeypatch.setattr(status_mod, "MK8sClient", FakeClient)
    monkeypatch.setattr(status_mod, "load_cluster_state", lambda name: _state())
    monkeypatch.setattr(status_mod, "list_local_clusters", lambda: [_state()])
    monkeypatch.setattr(status_mod, "save_cluster_state", saved.append)

    result = runner.invoke(app, ["status", "--name", "cluster-a"])

    assert result.exit_code == 0
    assert "cluster-a" in result.output
    assert "READY" in result.output
    assert saved[0].last_seen_state == "READY"


def test_list_json_merges_remote_and_local(monkeypatch) -> None:
    class FakeClient:
        def __init__(self, **kwargs) -> None:
            pass

        def list_clusters(self, project_id):
            return [
                ClusterInfo(
                    id="mk8scluster-a",
                    name="cluster-a",
                    project_id=project_id,
                    status="READY",
                    created_at="2026-05-14T21:46:00Z",
                )
            ]

        def get_cluster(self, name, *, project_id=""):
            return self.list_clusters(project_id)[0]

        def list_node_groups(self, cluster_id):
            return [NodeGroupInfo(id="mk8snodegroup-a", name="cluster-a-cpu", cluster_id=cluster_id, status="READY", node_count=1)]

    monkeypatch.setenv("NPA_CLUSTER_PROJECT_ID", "project-a")
    monkeypatch.setattr(status_mod, "MK8sClient", FakeClient)
    monkeypatch.setattr(status_mod, "list_local_clusters", lambda: [_state()])
    monkeypatch.setattr(status_mod, "save_cluster_state", lambda state: None)

    result = runner.invoke(app, ["list", "--format", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload[0]["name"] == "cluster-a"
    assert payload[0]["state"] == "READY"
    assert payload[0]["node_count"] == 1
