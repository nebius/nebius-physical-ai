from __future__ import annotations

from typer.testing import CliRunner

from npa.cli.cluster import app
from npa.cli.cluster import destroy as destroy_mod
from npa.cluster.api import ClusterInfo
from npa.cluster.exceptions import ClusterNotFoundError
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
    )


def test_destroy_missing_cluster_is_clean_noop(monkeypatch) -> None:
    class FakeClient:
        def __init__(self, **kwargs) -> None:
            pass

        def get_cluster(self, name, *, project_id=""):
            raise ClusterNotFoundError("missing")

    monkeypatch.setenv("NPA_CLUSTER_PROJECT_ID", "project-a")
    monkeypatch.setattr(destroy_mod, "MK8sClient", FakeClient)
    monkeypatch.setattr(destroy_mod, "load_cluster_state", lambda name: None)
    monkeypatch.setattr(destroy_mod, "list_local_clusters", lambda: [])

    result = runner.invoke(app, ["destroy", "--name", "cluster-a", "--force"])

    assert result.exit_code == 0
    assert "not found" in result.output


def test_destroy_removes_local_state_when_remote_missing(monkeypatch) -> None:
    deleted: list[str] = []

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            pass

        def get_cluster(self, name, *, project_id=""):
            raise ClusterNotFoundError("missing")

    monkeypatch.setattr(destroy_mod, "MK8sClient", FakeClient)
    monkeypatch.setattr(destroy_mod, "load_cluster_state", lambda name: _state())
    monkeypatch.setattr(destroy_mod, "delete_cluster_state", deleted.append)

    result = runner.invoke(app, ["destroy", "--name", "cluster-a", "--force"])

    assert result.exit_code == 0
    assert deleted == ["cluster-a"]
    assert "local state removed" in result.output


def test_destroy_remote_cluster_without_local_state(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            pass

        def get_cluster(self, name, *, project_id=""):
            calls.append(("get", name))
            return ClusterInfo(id="mk8scluster-a", name="cluster-a", project_id=project_id, status="READY")

        def delete_cluster(self, name, *, project_id=""):
            calls.append(("delete", name))

        def wait_for_deleted(self, name, *, project_id="", timeout_minutes=30):
            calls.append(("wait", name))

    monkeypatch.setenv("NPA_CLUSTER_PROJECT_ID", "project-a")
    monkeypatch.setattr(destroy_mod, "MK8sClient", FakeClient)
    monkeypatch.setattr(destroy_mod, "load_cluster_state", lambda name: None)
    monkeypatch.setattr(destroy_mod, "delete_cluster_state", lambda name: None)

    result = runner.invoke(app, ["destroy", "--name", "cluster-a", "--force"])

    assert result.exit_code == 0
    assert calls == [
        ("get", "cluster-a"),
        ("delete", "mk8scluster-a"),
        ("wait", "mk8scluster-a"),
    ]
    assert "destroyed" in result.output
