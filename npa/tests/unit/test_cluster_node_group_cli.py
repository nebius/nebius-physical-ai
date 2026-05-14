from __future__ import annotations

import json

from typer.testing import CliRunner

from npa.cli.cluster import app
from npa.cli.cluster import node_group as node_group_mod
from npa.cluster.api import ClusterInfo, NodeGroupInfo
from npa.cluster.config import NodeGroupConfig
from npa.cluster.exceptions import NodeGroupNotFoundError
from npa.cluster.state import ClusterState, NodeGroupState


runner = CliRunner()


def _cluster_state() -> ClusterState:
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
        created_at="2026-05-14T22:46:30Z",
        last_seen_state="RUNNING",
    )


def _node_state() -> NodeGroupState:
    return NodeGroupState(
        cluster_name="cluster-a",
        name="cluster-a-h100-gpu",
        node_group_id="mk8snodegroup-gpu",
        gpu_type="h100",
        platform="gpu-h100-sxm",
        preset="1gpu-16vcpu-200gb",
        node_count=1,
        created_at="2026-05-14T22:46:30Z",
        last_seen_state="RUNNING",
    )


def _cluster() -> ClusterInfo:
    return ClusterInfo(
        id="mk8scluster-a",
        name="cluster-a",
        project_id="project-a",
        status="RUNNING",
        created_at="2026-05-14T22:46:30Z",
    )


def _node_group(state: str = "RUNNING") -> NodeGroupInfo:
    return NodeGroupInfo(
        id="mk8snodegroup-gpu",
        name="cluster-a-h100-gpu",
        cluster_id="mk8scluster-a",
        status=state,
        node_count=1,
        created_at="2026-05-14T22:46:30Z",
        platform="gpu-h100-sxm",
        preset="1gpu-16vcpu-200gb",
        gpu_type="h100",
    )


def test_add_node_group_saves_state(monkeypatch) -> None:
    saved: list[NodeGroupState] = []
    seen_configs: list[NodeGroupConfig] = []

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            pass

        def get_cluster(self, name, *, project_id=""):
            return _cluster()

        def create_gpu_node_group(self, config, cluster_id):
            seen_configs.append(config)
            return _node_group(state="PROVISIONING")

        def wait_for_node_group_ready(self, cluster_id, name, **kwargs):
            return _node_group(state="RUNNING")

    monkeypatch.setattr(node_group_mod, "MK8sClient", FakeClient)
    monkeypatch.setattr(node_group_mod, "load_cluster_state", lambda name: _cluster_state())
    monkeypatch.setattr(node_group_mod, "save_node_group_state", saved.append)

    result = runner.invoke(
        app,
        ["node-group", "add", "--cluster-name", "cluster-a", "--gpu-type", "h100"],
    )

    assert result.exit_code == 0
    assert seen_configs[0].public_ip is False
    assert saved[-1].last_seen_state == "RUNNING"
    assert "Node group ID: mk8snodegroup-gpu" in result.output


def test_remove_node_group_handles_local_state_remote_missing(monkeypatch) -> None:
    deleted: list[tuple[str, str]] = []

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            pass

        def get_cluster(self, name, *, project_id=""):
            return _cluster()

        def get_node_group(self, cluster_id, name):
            raise NodeGroupNotFoundError("missing")

    monkeypatch.setattr(node_group_mod, "MK8sClient", FakeClient)
    monkeypatch.setattr(node_group_mod, "load_cluster_state", lambda name: _cluster_state())
    monkeypatch.setattr(node_group_mod, "load_node_group_state", lambda cluster_name, name: _node_state())
    monkeypatch.setattr(node_group_mod, "delete_node_group_state", lambda cluster_name, name: deleted.append((cluster_name, name)))

    result = runner.invoke(
        app,
        ["node-group", "remove", "--cluster-name", "cluster-a", "--name", "cluster-a-h100-gpu", "--force"],
    )

    assert result.exit_code == 0
    assert deleted == [("cluster-a", "cluster-a-h100-gpu")]
    assert "local state removed" in result.output


def test_remove_node_group_handles_remote_only(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            pass

        def get_cluster(self, name, *, project_id=""):
            return _cluster()

        def get_node_group(self, cluster_id, name):
            calls.append(("get", name))
            return _node_group()

        def delete_node_group(self, cluster_id, name):
            calls.append(("delete", name))

        def wait_for_node_group_deleted(self, cluster_id, name, **kwargs):
            calls.append(("wait", name))

    monkeypatch.setattr(node_group_mod, "MK8sClient", FakeClient)
    monkeypatch.setattr(node_group_mod, "load_cluster_state", lambda name: _cluster_state())
    monkeypatch.setattr(node_group_mod, "load_node_group_state", lambda cluster_name, name: None)
    monkeypatch.setattr(node_group_mod, "delete_node_group_state", lambda cluster_name, name: None)

    result = runner.invoke(
        app,
        ["node-group", "remove", "--cluster-name", "cluster-a", "--name", "cluster-a-h100-gpu", "--force"],
    )

    assert result.exit_code == 0
    assert calls == [
        ("get", "cluster-a-h100-gpu"),
        ("delete", "mk8snodegroup-gpu"),
        ("wait", "mk8snodegroup-gpu"),
    ]


def test_remove_node_group_missing_everywhere_is_clean_noop(monkeypatch) -> None:
    class FakeClient:
        def __init__(self, **kwargs) -> None:
            pass

        def get_cluster(self, name, *, project_id=""):
            return _cluster()

        def get_node_group(self, cluster_id, name):
            raise NodeGroupNotFoundError("missing")

    monkeypatch.setattr(node_group_mod, "MK8sClient", FakeClient)
    monkeypatch.setattr(node_group_mod, "load_cluster_state", lambda name: _cluster_state())
    monkeypatch.setattr(node_group_mod, "load_node_group_state", lambda cluster_name, name: None)

    result = runner.invoke(
        app,
        ["node-group", "remove", "--cluster-name", "cluster-a", "--name", "missing", "--force"],
    )

    assert result.exit_code == 0
    assert "not found" in result.output


def test_status_json_merges_remote_and_local(monkeypatch) -> None:
    saved: list[NodeGroupState] = []

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            pass

        def get_cluster(self, name, *, project_id=""):
            return _cluster()

        def list_node_groups(self, cluster_id):
            return [_node_group()]

    monkeypatch.setattr(node_group_mod, "MK8sClient", FakeClient)
    monkeypatch.setattr(node_group_mod, "load_cluster_state", lambda name: _cluster_state())
    monkeypatch.setattr(node_group_mod, "list_node_group_states", lambda cluster_name: [_node_state()])
    monkeypatch.setattr(node_group_mod, "save_node_group_state", saved.append)

    result = runner.invoke(
        app,
        ["node-group", "status", "--cluster-name", "cluster-a", "--format", "json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload[0]["gpu_type"] == "h100"
    assert payload[0]["state"] == "RUNNING"
    assert saved[0].last_seen_state == "RUNNING"


def test_list_table_without_cluster_scans_local(monkeypatch) -> None:
    class FakeClient:
        def __init__(self, **kwargs) -> None:
            pass

        def get_cluster(self, name, *, project_id=""):
            return _cluster()

        def list_node_groups(self, cluster_id):
            return [_node_group()]

    monkeypatch.setattr(node_group_mod, "MK8sClient", FakeClient)
    monkeypatch.setattr(node_group_mod, "list_local_clusters", lambda: [_cluster_state()])
    monkeypatch.setattr(node_group_mod, "list_node_group_states", lambda cluster_name: [_node_state()])
    monkeypatch.setattr(node_group_mod, "resolve_project_id", lambda: "")

    result = runner.invoke(app, ["node-group", "list"])

    assert result.exit_code == 0
    assert "cluster-a-h100-gpu" in result.output
    assert "RUNNING" in result.output
