"""Read-only verification of rendered RTX driver configuration on a live Fleet."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from npa.fleet.spec import load_spec

pytestmark = pytest.mark.e2e


def test_fleet_rtx_driver_config_reaches_ready_pods() -> None:
    if not os.environ.get("NPA_FLEET_RTX_VERIFY_SPEC") or not os.environ.get("NPA_FLEET_RTX_KUBECONFIGS"):
        pytest.skip("supply an owner-private Fleet spec and project-key kubeconfig mapping")
    spec = load_spec(Path(os.environ["NPA_FLEET_RTX_VERIFY_SPEC"]))
    configs = json.loads(Path(os.environ["NPA_FLEET_RTX_KUBECONFIGS"]).read_text())
    assert spec.profile
    env = dict(os.environ)
    for key in ("NEBIUS_IAM_TOKEN", "NPA_NEBIUS_IAM_TOKEN", "TF_VAR_iam_token"):
        env.pop(key, None)
    env.update(NEBIUS_PROFILE=spec.profile, NPA_NEBIUS_PROFILE=spec.profile)
    targets = list(spec.cluster_targets())
    assert targets
    for project, cluster in targets:
        assert cluster.gpu_workload_profile == "rtx-rendering"
        kubeconfig = Path(configs[project.key()][cluster.name]).expanduser()
        assert kubeconfig.is_file()

        def get(resource: str, *args: str) -> list[dict]:
            result = subprocess.run(
                ["kubectl", "--kubeconfig", str(kubeconfig), "--request-timeout=30s",
                 "get", resource, *args, "-o", "json"],
                env=env, capture_output=True, text=True, check=False,
            )
            assert result.returncode == 0, f"unreadable live {resource} response"
            return json.loads(result.stdout)["items"]

        nodes = get("nodes")
        selector = {
            "node.kubernetes.io/instance-type": cluster.gpu_nodes.platform,
            "nebius.com/resource-preset": cluster.gpu_nodes.preset,
        }
        gpu_nodes = [node for node in nodes if all(
            node["metadata"]["labels"].get(key) == value for key, value in selector.items()
        )]
        assert len(gpu_nodes) == cluster.gpu_nodes.count
        expected_gpus = int(cluster.gpu_nodes.preset.split("gpu-", 1)[0])
        assert all(int(node["status"]["allocatable"]["nvidia.com/gpu"]) == expected_gpus
                   for node in gpu_nodes)
        drivers = get("nvidiadrivers")
        matching = [driver for driver in drivers if driver["spec"].get("nodeSelector") == selector]
        assert len(matching) == 1
        driver = matching[0]
        assert driver["spec"]["rdma"]["enabled"] is False
        pods = get("pods", "-n", "gpu-operator")
        toolkit_pods = [pod for pod in pods if any(
            container["name"] == "nvidia-container-toolkit-ctr"
            for container in pod["spec"]["containers"]
        )]
        assert len(toolkit_pods) == len(gpu_nodes)
        for pod in toolkit_pods:
            container = next(container for container in pod["spec"]["containers"]
                             if container["name"] == "nvidia-container-toolkit-ctr")
            assert {item["name"]: item.get("value") for item in container["env"]}[
                "RUNTIME_CONFIG_SOURCE"
            ] == "file"
            assert all(container["ready"] for container in pod["status"]["containerStatuses"])
        driver_pods = [pod for pod in pods if any(
            container["name"] == "nvidia-driver-ctr" for container in pod["spec"]["containers"]
        )]
        assert len(driver_pods) == len(gpu_nodes)
        assert {pod["spec"]["nodeName"] for pod in driver_pods} == {
            node["metadata"]["name"] for node in gpu_nodes
        }
        for pod in driver_pods:
            statuses = pod["status"].get("containerStatuses", [])
            assert len(statuses) == len(pod["spec"]["containers"])
            assert all(container["ready"] for container in statuses)
        files = cluster.gpu_driver_package_repositories
        if files:
            name = driver["spec"]["repoConfig"]["name"]
            maps = get("configmaps", "-n", "gpu-operator")
            configmap = next(item for item in maps if item["metadata"]["name"] == name)
            assert configmap["data"] == files
            for pod in driver_pods:
                volume = next(volume for volume in pod["spec"]["volumes"]
                              if volume.get("configMap", {}).get("name") == name)
                for container in pod["spec"]["containers"]:
                    mounts = [mount for mount in container.get("volumeMounts", [])
                              if mount["name"] == volume["name"]]
                    assert {mount["subPath"] for mount in mounts} == set(files)
                    assert all(mount.get("readOnly") for mount in mounts)
