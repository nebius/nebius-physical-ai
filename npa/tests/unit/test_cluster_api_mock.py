from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from npa.cluster.api import MK8sClient
from npa.cluster.config import ClusterConfig
from npa.cluster.exceptions import ClusterError, ClusterNotFoundError


def _result(payload: dict | None = None, *, returncode: int = 0, stderr: str = ""):
    return subprocess.CompletedProcess(
        args=["nebius"],
        returncode=returncode,
        stdout=json.dumps(payload or {}),
        stderr=stderr,
    )


def _cluster(cluster_id: str = "mk8scluster-a", state: str = "READY") -> dict:
    return {
        "metadata": {
            "id": cluster_id,
            "name": "cluster-a",
            "parent_id": "project-a",
            "created_at": "2026-05-14T21:46:00Z",
        },
        "status": {"state": state, "endpoint": "https://api.example.invalid"},
    }


def _node_group(state: str = "READY") -> dict:
    return {
        "metadata": {"id": "mk8snodegroup-a", "name": "cluster-a-cpu"},
        "spec": {"fixed_node_count": 1},
        "status": {"state": state},
    }


def _config() -> ClusterConfig:
    return ClusterConfig(
        name="cluster-a",
        project_id="project-a",
        subnet_id="vpcsubnet-a",
    )


def test_create_cluster_creates_cluster_then_node_group() -> None:
    calls: list[list[str]] = []

    def run(args, **kwargs):
        calls.append(args)
        assert kwargs["text"] is True
        assert kwargs["check"] is False
        if args[1:4] == ["mk8s", "cluster", "create"]:
            return _result(_cluster())
        if args[1:4] == ["mk8s", "node-group", "create"]:
            return _result(_node_group())
        raise AssertionError(f"unexpected command: {args}")

    client = MK8sClient(nebius_bin="nebius", subprocess_runner=run, sleep=lambda _: None)

    info = client.create_cluster(_config())

    assert info.id == "mk8scluster-a"
    assert info.node_group_id == "mk8snodegroup-a"
    assert calls[0][1:4] == ["mk8s", "cluster", "create"]
    assert calls[1][1:4] == ["mk8s", "node-group", "create"]
    assert "--template-resources-platform" in calls[1]
    assert "--template-resources-preset" in calls[1]


def test_list_clusters_parses_items() -> None:
    def run(args, **kwargs):
        assert args[1:4] == ["mk8s", "cluster", "list"]
        return _result({"items": [_cluster()]})

    client = MK8sClient(nebius_bin="nebius", subprocess_runner=run)

    clusters = client.list_clusters("project-a")

    assert len(clusters) == 1
    assert clusters[0].name == "cluster-a"
    assert clusters[0].status == "READY"


def test_get_cluster_uses_name_fallback_after_id_miss() -> None:
    calls: list[list[str]] = []

    def run(args, **kwargs):
        calls.append(args)
        if args[1:4] == ["mk8s", "cluster", "get"]:
            return _result(returncode=1, stderr="NotFound")
        if args[1:4] == ["mk8s", "cluster", "get-by-name"]:
            return _result(_cluster())
        raise AssertionError(f"unexpected command: {args}")

    client = MK8sClient(nebius_bin="nebius", subprocess_runner=run)

    info = client.get_cluster("cluster-a", project_id="project-a")

    assert info.id == "mk8scluster-a"
    assert calls[1][1:4] == ["mk8s", "cluster", "get-by-name"]


def test_get_cluster_raises_not_found() -> None:
    def run(args, **kwargs):
        return _result(returncode=1, stderr="not found")

    client = MK8sClient(nebius_bin="nebius", subprocess_runner=run)

    with pytest.raises(ClusterNotFoundError):
        client.get_cluster("missing", project_id="project-a")


def test_transient_errors_are_retried() -> None:
    attempts = 0

    def run(args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return _result(returncode=1, stderr="503 unavailable")
        return _result({"items": [_cluster()]})

    sleeps: list[float] = []
    client = MK8sClient(nebius_bin="nebius", subprocess_runner=run, sleep=sleeps.append)

    assert client.list_clusters("project-a")[0].id == "mk8scluster-a"
    assert attempts == 2
    assert sleeps


def test_non_transient_error_raises() -> None:
    def run(args, **kwargs):
        return _result(returncode=1, stderr="permission denied")

    client = MK8sClient(nebius_bin="nebius", subprocess_runner=run)

    with pytest.raises(ClusterError, match="permission denied"):
        client.list_clusters("project-a")


def test_get_kubeconfig_invokes_get_credentials(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def run(args, **kwargs):
        calls.append(args)
        return _result({})

    client = MK8sClient(nebius_bin="nebius", subprocess_runner=run)
    target = tmp_path / "kubeconfig"

    assert client.get_kubeconfig("mk8scluster-a", target, context_name="cluster-a") == target

    assert calls[0][1:4] == ["mk8s", "cluster", "get-credentials"]
    assert "--external" in calls[0]
    assert str(target) in calls[0]
