"""Read-only acceptance of the direct RTX profile on an operator-owned cluster."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e


def _get(config: dict, resource: str, *args: str) -> list[dict]:
    response = subprocess.run(
        [
            "kubectl",
            "--kubeconfig",
            config["kubeconfig_path"],
            "--context",
            config["kube_context"],
            "--request-timeout=30s",
            "get",
            resource,
            *args,
            "-o",
            "json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert response.returncode == 0, "Cannot read the owned cluster resource"
    return json.loads(response.stdout)["items"]


def _assert_direct_release_revision(config: dict) -> None:
    state = json.loads(Path(config["terraform_state_path"]).read_text())
    releases = [
        resource
        for resource in state["resources"]
        if resource.get("module") == "module.k8s_training.module.gpu-operator[0]"
        and resource["type"] == "nebius_applications_v1alpha1_k8s_release"
        and resource["name"] == "this"
    ]
    assert len(releases) == 1, "Expected one direct-wrapper GPU Operator release"
    instances = releases[0]["instances"]
    assert len(instances) == 1, "Expected one GPU Operator release instance"
    revision = instances[0]["attributes"]["sensitive"].get("version")
    assert isinstance(revision, str) and len(revision) == 64, (
        "The direct release has no RTX values revision"
    )


def _assert_toolkit_pods(config: dict, node_names: set[str]) -> None:
    pods = _get(config, "pods", "-n", "gpu-operator")
    toolkits = [
        pod
        for pod in pods
        if pod["spec"].get("nodeName") in node_names
        and any(
            container["name"] == "nvidia-container-toolkit-ctr"
            for container in pod["spec"]["containers"]
        )
    ]
    assert len(toolkits) == len(node_names), "Expected one toolkit pod per GPU node"
    for pod in toolkits:
        container = next(
            item
            for item in pod["spec"]["containers"]
            if item["name"] == "nvidia-container-toolkit-ctr"
        )
        values = {item["name"]: item.get("value") for item in container["env"]}
        assert values.get("RUNTIME_CONFIG_SOURCE") == "file", (
            "Toolkit must read the on-disk containerd schema"
        )
        statuses = pod["status"].get("containerStatuses", [])
        assert len(statuses) == len(pod["spec"]["containers"])
        assert all(item["ready"] for item in statuses), "Toolkit pod is not Ready"


def test_direct_rtx_profile_reaches_ready_toolkit() -> None:
    filename = os.environ.get("NPA_CLUSTER_RTX_TOOLKIT_LIVE_CONFIG", "")
    if not filename:
        pytest.skip("Supply an owner-private direct-cluster verification config")
    config = json.loads(Path(filename).read_text())
    assert config["gpu_nodes"] > 0, "Verification requires an owned RTX GPU node"
    _assert_direct_release_revision(config)
    selector = {
        "node.kubernetes.io/instance-type": config["platform"],
        "nebius.com/resource-preset": config["preset"],
    }
    nodes = [
        node
        for node in _get(config, "nodes")
        if all(
            node["metadata"]["labels"].get(key) == value
            for key, value in selector.items()
        )
    ]
    assert len(nodes) == config["gpu_nodes"], "Unexpected RTX node count"
    expected_gpus = int(config["preset"].split("gpu-", 1)[0])
    for node in nodes:
        assert any(
            item["type"] == "Ready" and item["status"] == "True"
            for item in node["status"]["conditions"]
        ), "RTX node is not Ready"
        assert int(node["status"]["allocatable"]["nvidia.com/gpu"]) == expected_gpus
    drivers = [
        item
        for item in _get(config, "nvidiadrivers")
        if item["spec"].get("nodeSelector") == selector
    ]
    assert len(drivers) == 1, "Expected the exact RTX driver selector"
    assert drivers[0]["spec"]["rdma"]["enabled"] is False
    _assert_toolkit_pods(config, {node["metadata"]["name"] for node in nodes})
