from __future__ import annotations

import json
import subprocess

import pytest

from npa.cluster.api import MK8sClient, is_ready
from npa.cluster.config import NodeGroupConfig
from npa.cluster.exceptions import ClusterError, NodeGroupNotFoundError


def _result(payload: dict | None = None, *, returncode: int = 0, stderr: str = ""):
    return subprocess.CompletedProcess(
        args=["nebius"],
        returncode=returncode,
        stdout=json.dumps(payload or {}),
        stderr=stderr,
    )


def _node_group(name: str = "cluster-a-h100-gpu", state: str = "RUNNING") -> dict:
    return {
        "metadata": {
            "id": "mk8snodegroup-gpu",
            "name": name,
            "created_at": "2026-05-14T22:46:30Z",
        },
        "spec": {
            "fixed_node_count": 1,
            "template": {
                "resources": {
                    "platform": "gpu-h100-sxm",
                    "preset": "1gpu-16vcpu-200gb",
                },
                "network_interfaces": [{"subnet_id": "vpcsubnet-a"}],
            },
        },
        "status": {"state": state},
    }


def test_create_gpu_node_group_invokes_mk8s_create() -> None:
    calls: list[list[str]] = []

    def run(args, **kwargs):
        calls.append(args)
        if args[1:4] == ["mk8s", "node-group", "create"]:
            return _result(_node_group())
        raise AssertionError(f"unexpected command: {args}")

    client = MK8sClient(nebius_bin="nebius", subprocess_runner=run, sleep=lambda _: None)
    config = NodeGroupConfig(cluster_name="cluster-a", gpu_type="h100", subnet_id="vpcsubnet-a")

    info = client.create_gpu_node_group(config, "mk8scluster-a")

    assert info.id == "mk8snodegroup-gpu"
    assert info.status == "RUNNING"
    assert info.gpu_type == "h100"
    assert "--template-gpu-settings-drivers-preset" in calls[0]
    assert "--fixed-node-count" in calls[0]
    network_index = calls[0].index("--template-network-interfaces") + 1
    assert json.loads(calls[0][network_index]) == [{"subnet_id": "vpcsubnet-a"}]


def test_create_gpu_node_group_public_ip_opt_in() -> None:
    calls: list[list[str]] = []

    def run(args, **kwargs):
        calls.append(args)
        return _result(_node_group())

    client = MK8sClient(nebius_bin="nebius", subprocess_runner=run, sleep=lambda _: None)
    config = NodeGroupConfig(
        cluster_name="cluster-a",
        gpu_type="h100",
        subnet_id="vpcsubnet-a",
        public_ip=True,
    )

    client.create_gpu_node_group(config, "mk8scluster-a")

    network_index = calls[0].index("--template-network-interfaces") + 1
    assert json.loads(calls[0][network_index]) == [
        {"subnet_id": "vpcsubnet-a", "public_ip_address": {}}
    ]


def test_create_gpu_node_group_uses_autoscaling_flags() -> None:
    calls: list[list[str]] = []

    def run(args, **kwargs):
        calls.append(args)
        return _result(_node_group())

    client = MK8sClient(nebius_bin="nebius", subprocess_runner=run, sleep=lambda _: None)
    config = NodeGroupConfig(
        cluster_name="cluster-a",
        gpu_type="h100",
        autoscaling_min=0,
        autoscaling_max=2,
    )

    client.create_gpu_node_group(config, "mk8scluster-a")

    assert "--fixed-node-count" not in calls[0]
    assert "--autoscaling-min-node-count" in calls[0]
    assert "--autoscaling-max-node-count" in calls[0]


def test_create_gpu_node_group_uses_capacity_block_group() -> None:
    calls: list[list[str]] = []

    def run(args, **kwargs):
        calls.append(args)
        return _result(_node_group())

    client = MK8sClient(nebius_bin="nebius", subprocess_runner=run, sleep=lambda _: None)
    config = NodeGroupConfig(
        cluster_name="cluster-a",
        gpu_type="h100",
        capacity_block_group="capacityblockgroup-test",
    )

    client.create_gpu_node_group(config, "mk8scluster-a")

    assert "--template-reservation-policy-policy" in calls[0]
    policy_index = calls[0].index("--template-reservation-policy-policy") + 1
    assert calls[0][policy_index] == "strict"
    ids_index = calls[0].index("--template-reservation-policy-reservation-ids") + 1
    assert calls[0][ids_index] == "capacityblockgroup-test"


def test_get_node_group_uses_name_fallback_after_id_miss() -> None:
    calls: list[list[str]] = []

    def run(args, **kwargs):
        calls.append(args)
        if args[1:4] == ["mk8s", "node-group", "get"]:
            return _result(returncode=1, stderr="NotFound")
        if args[1:4] == ["mk8s", "node-group", "get-by-name"]:
            return _result(_node_group())
        raise AssertionError(f"unexpected command: {args}")

    client = MK8sClient(nebius_bin="nebius", subprocess_runner=run)

    info = client.get_node_group("mk8scluster-a", "cluster-a-h100-gpu")

    assert info.id == "mk8snodegroup-gpu"
    assert calls[1][1:4] == ["mk8s", "node-group", "get-by-name"]


def test_get_node_group_raises_not_found() -> None:
    def run(args, **kwargs):
        return _result(returncode=1, stderr="not found")

    client = MK8sClient(nebius_bin="nebius", subprocess_runner=run)

    with pytest.raises(NodeGroupNotFoundError):
        client.get_node_group("mk8scluster-a", "missing")


def test_delete_node_group_is_idempotent_when_missing() -> None:
    calls: list[list[str]] = []

    def run(args, **kwargs):
        calls.append(args)
        return _result(returncode=1, stderr="not found")

    client = MK8sClient(nebius_bin="nebius", subprocess_runner=run)

    client.delete_node_group("mk8scluster-a", "missing")

    assert not any(call[1:4] == ["mk8s", "node-group", "delete"] for call in calls)


@pytest.mark.parametrize("state", ["RUNNING", "READY"])
def test_wait_for_node_group_ready_accepts_running_and_ready(state: str) -> None:
    def run(args, **kwargs):
        if args[1:4] == ["mk8s", "node-group", "get"]:
            return _result(_node_group(state=state))
        raise AssertionError(f"unexpected command: {args}")

    client = MK8sClient(nebius_bin="nebius", subprocess_runner=run, sleep=lambda _: None)

    assert is_ready(state)
    assert client.wait_for_node_group_ready("mk8scluster-a", "mk8snodegroup-gpu").status == state


def test_node_group_create_error_raises_cluster_error() -> None:
    def run(args, **kwargs):
        return _result(returncode=1, stderr="capacity unavailable")

    client = MK8sClient(nebius_bin="nebius", subprocess_runner=run)
    config = NodeGroupConfig(cluster_name="cluster-a", gpu_type="h100")

    with pytest.raises(ClusterError, match="capacity unavailable"):
        client.create_gpu_node_group(config, "mk8scluster-a")
