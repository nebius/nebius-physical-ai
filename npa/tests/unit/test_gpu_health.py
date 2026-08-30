"""Hermetic mk8s GPU health, stabilization, and CUDA-smoke tests."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from npa.cluster.gpu_health import (
    GpuHealthConfig,
    GpuHealthError,
    probe_gpu_health,
    validate_gpu_health,
)


def _node(
    name: str,
    *,
    platform: str,
    gpus: int,
    boot_id: str,
    ready: bool = True,
    gpu_error: bool = False,
    fabric_state: str = "",
) -> dict[str, Any]:
    conditions = [{"type": "Ready", "status": "True" if ready else "False"}]
    if gpu_error:
        conditions.append(
            {
                "type": "NebiusGPUError",
                "status": "True",
                "reason": "FabricManagerNotReady",
            }
        )
    metadata: dict[str, Any] = {
        "name": name,
        "labels": {"node.kubernetes.io/instance-type": platform},
    }
    if fabric_state:
        metadata["annotations"] = {"nebius.ai/fabric-state": fabric_state}
    return {
        "metadata": metadata,
        "status": {
            "conditions": conditions,
            "allocatable": {"nvidia.com/gpu": str(gpus)},
            "nodeInfo": {"bootID": boot_id},
        },
    }


def _component_pods() -> dict[str, Any]:
    return {
        "items": [
            {
                "metadata": {"name": "nvidia-component"},
                "status": {
                    "phase": "Running",
                    "containerStatuses": [{"ready": True}],
                },
            }
        ]
    }


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def monotonic(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


class _Kubectl:
    def __init__(
        self,
        snapshots: list[list[dict[str, Any]]],
        *,
        smoke_logs: str = "Test PASSED\n",
    ) -> None:
        self.snapshots = snapshots
        self.snapshot_index = 0
        self.smoke_logs = smoke_logs
        self.component_namespaces: list[str] = []
        self.created_nodes: list[str] = []
        self.deleted_pods: list[str] = []

    @staticmethod
    def _result(payload: object = "", returncode: int = 0):
        stdout = payload if isinstance(payload, str) else json.dumps(payload)
        return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr="")

    def __call__(self, args, **kwargs):
        command = args[1:]
        if command == ["get", "nodes", "-o", "json"]:
            index = min(self.snapshot_index, len(self.snapshots) - 1)
            self.snapshot_index += 1
            return self._result({"items": self.snapshots[index]})
        if command[:2] == ["get", "pods"]:
            namespace = command[command.index("-n") + 1]
            self.component_namespaces.append(namespace)
            return self._result(_component_pods())
        if command == ["apply", "-f", "-"]:
            manifest = json.loads(kwargs["input_text"])
            self.created_nodes.append(manifest["spec"]["nodeName"])
            return self._result("pod created\n")
        if command[:2] == ["get", "pod"]:
            return self._result(
                {
                    "status": {
                        "phase": "Succeeded",
                        "containerStatuses": [
                            {
                                "ready": False,
                                "state": {
                                    "terminated": {"exitCode": 0, "reason": "Completed"}
                                },
                            }
                        ],
                    }
                }
            )
        if command[0] == "logs":
            return self._result(self.smoke_logs)
        if command[:2] == ["delete", "pod"]:
            self.deleted_pods.append(command[2])
            return self._result("deleted\n")
        raise AssertionError(args)


def _config(**overrides) -> GpuHealthConfig:
    values = {
        "expected_nodes": 3,
        "expected_gpu_nodes": 2,
        "gpu_preset": "8gpu-160vcpu-1792gb",
        "gpu_platform": "gpu-b200-sxm",
        "driver_mode": "managed-image",
        "nvswitch": True,
        "stabilization_seconds": 0,
        "poll_seconds": 1,
        "timeout_seconds": 10,
        "cuda_smoke": False,
    }
    values.update(overrides)
    return GpuHealthConfig(**values)


def _healthy_nodes(*, boot_a: str = "boot-a") -> list[dict[str, Any]]:
    return [
        _node(
            "gpu-0",
            platform="gpu-b200-sxm",
            gpus=8,
            boot_id=boot_a,
            fabric_state="Completed",
        ),
        _node(
            "gpu-1",
            platform="gpu-b200-sxm-a",
            gpus=8,
            boot_id="boot-b",
            fabric_state="Success",
        ),
        _node("cpu-0", platform="cpu-d3", gpus=0, boot_id="boot-c"),
    ]


def test_probe_validates_generalized_topology_and_managed_components(
    tmp_path: Path,
) -> None:
    kubectl = _Kubectl([_healthy_nodes()])
    snapshot = probe_gpu_health(
        kubectl,
        kubectl_bin="kubectl",
        kubeconfig_path=tmp_path / "kubeconfig",
        config=_config(),
    )
    assert snapshot["errors"] == []
    assert snapshot["ready_nodes"] == 3
    assert snapshot["total_gpus"] == 16
    assert kubectl.component_namespaces == ["nvidia-device-plugin"]


def test_probe_accepts_current_managed_plugin_in_kube_system(tmp_path: Path) -> None:
    class CurrentManagedImage(_Kubectl):
        def __call__(self, args, **kwargs):
            command = args[1:]
            if command[:2] == ["get", "pods"]:
                namespace = command[command.index("-n") + 1]
                self.component_namespaces.append(namespace)
                if namespace == "nvidia-device-plugin":
                    return self._result({"items": []})
                return self._result(
                    {
                        "items": [
                            {
                                "metadata": {"name": "unrelated-system-pod"},
                                "spec": {"containers": [{"name": "system"}]},
                                "status": {"phase": "Pending"},
                            },
                            {
                                "metadata": {
                                    "name": "nvidia-device-plugin-daemonset-pod"
                                },
                                "spec": {
                                    "containers": [
                                        {"name": "nvidia-device-plugin-ctr"}
                                    ]
                                },
                                "status": {
                                    "phase": "Running",
                                    "containerStatuses": [{"ready": True}],
                                },
                            },
                        ]
                    }
                )
            return super().__call__(args, **kwargs)

    kubectl = CurrentManagedImage([_healthy_nodes()])
    snapshot = probe_gpu_health(
        kubectl,
        kubectl_bin="kubectl",
        kubeconfig_path=tmp_path / "kubeconfig",
        config=_config(),
    )

    assert snapshot["errors"] == []
    assert snapshot["component_namespace"] == "kube-system (nvidia-device-plugin)"
    assert kubectl.component_namespaces == ["nvidia-device-plugin", "kube-system"]


def test_probe_fails_on_nebius_gpu_error_and_incomplete_fabric(tmp_path: Path) -> None:
    nodes = _healthy_nodes()
    nodes[0] = _node(
        "gpu-0",
        platform="gpu-b200-sxm",
        gpus=8,
        boot_id="boot-a",
        gpu_error=True,
        fabric_state="In Progress",
    )
    snapshot = probe_gpu_health(
        _Kubectl([nodes]),
        kubectl_bin="kubectl",
        kubeconfig_path=tmp_path / "kubeconfig",
        config=_config(),
    )
    assert any("NebiusGPUError=True" in error for error in snapshot["errors"])
    assert any("fabric-state='In Progress'" in error for error in snapshot["errors"])


def test_operator_mode_checks_operator_namespace(tmp_path: Path) -> None:
    kubectl = _Kubectl([_healthy_nodes()])
    snapshot = probe_gpu_health(
        kubectl,
        kubectl_bin="kubectl",
        kubeconfig_path=tmp_path / "kubeconfig",
        config=_config(driver_mode="operator"),
    )
    assert snapshot["errors"] == []
    assert kubectl.component_namespaces == ["gpu-operator"]


def test_boot_id_churn_is_terminal_during_stabilization(tmp_path: Path) -> None:
    clock = _Clock()
    kubectl = _Kubectl([_healthy_nodes(), _healthy_nodes(boot_a="boot-restarted")])
    with pytest.raises(GpuHealthError, match="boot IDs changed"):
        validate_gpu_health(
            kubectl,
            kubectl_bin="kubectl",
            kubeconfig_path=tmp_path / "kubeconfig",
            config=_config(stabilization_seconds=2),
            evidence_path=tmp_path / "gpu-health.json",
            sleep_fn=clock.sleep,
            monotonic_fn=clock.monotonic,
        )
    evidence = json.loads((tmp_path / "gpu-health.json").read_text())
    assert evidence["status"] == "failed"
    assert len(evidence["observations"]) == 2


def test_timeout_fails_when_healthy_interval_is_too_short(tmp_path: Path) -> None:
    clock = _Clock()
    with pytest.raises(GpuHealthError, match="requested 2s stabilization interval"):
        validate_gpu_health(
            _Kubectl([_healthy_nodes()]),
            kubectl_bin="kubectl",
            kubeconfig_path=tmp_path / "kubeconfig",
            config=_config(stabilization_seconds=2, timeout_seconds=1),
            evidence_path=tmp_path / "gpu-health.json",
            sleep_fn=clock.sleep,
            monotonic_fn=clock.monotonic,
        )
    evidence = json.loads((tmp_path / "gpu-health.json").read_text())
    assert evidence["status"] == "failed"
    assert evidence["final_snapshot"]["errors"] == []


def test_stabilization_runs_vectoradd_and_fabric_check_on_every_gpu_node(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    logs = "Test PASSED\nFabric\n    State : Completed\n    Status : Success\n"
    kubectl = _Kubectl([_healthy_nodes()], smoke_logs=logs)
    report = validate_gpu_health(
        kubectl,
        kubectl_bin="kubectl",
        kubeconfig_path=tmp_path / "kubeconfig",
        config=_config(stabilization_seconds=2, cuda_smoke=True),
        evidence_path=tmp_path / "gpu-health.json",
        sleep_fn=clock.sleep,
        monotonic_fn=clock.monotonic,
    )
    assert report["status"] == "healthy"
    assert kubectl.created_nodes == ["gpu-0", "gpu-1"]
    assert len(kubectl.deleted_pods) == 2
    assert all(smoke["vectoradd"] == "passed" for smoke in report["cuda_smokes"])


@pytest.mark.parametrize(
    "logs",
    [
        "vector add exited zero without evidence\n",
        ("Test PASSED\nFabric\n    State : In Progress\n    Status : N/A\n"),
    ],
)
def test_cuda_smoke_fails_without_kernel_or_completed_fabric_evidence(
    tmp_path: Path, logs: str
) -> None:
    clock = _Clock()
    with pytest.raises(GpuHealthError):
        validate_gpu_health(
            _Kubectl([_healthy_nodes()], smoke_logs=logs),
            kubectl_bin="kubectl",
            kubeconfig_path=tmp_path / "kubeconfig",
            config=_config(cuda_smoke=True),
            evidence_path=tmp_path / "gpu-health.json",
            sleep_fn=clock.sleep,
            monotonic_fn=clock.monotonic,
        )
    assert json.loads((tmp_path / "gpu-health.json").read_text())["status"] == "failed"
