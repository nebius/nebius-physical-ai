"""Hermetic mk8s GPU health, stabilization, and CUDA-smoke tests."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from npa.cluster.gpu_health import (
    DEFAULT_GRAPHICS_SMOKE_IMAGE,
    GpuHealthConfig,
    GpuHealthError,
    probe_gpu_health,
    validate_gpu_health,
)


def test_graphics_smoke_uses_the_anonymously_pullable_public_catalog_path() -> None:
    assert DEFAULT_GRAPHICS_SMOKE_IMAGE.startswith(
        "ghcr.io/nebius/nebius-physical-ai/npa-sonic@sha256:"
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
        graphics_logs: str = (
            "NPA_GLX_LOADED\nNPA_EGL_LOADED\nVulkan Instance Version: 1.3.0\n"
            "GPU0:\n    deviceName = NVIDIA RTX PRO 6000 Blackwell\n"
        ),
    ) -> None:
        self.snapshots = snapshots
        self.snapshot_index = 0
        self.smoke_logs = smoke_logs
        self.graphics_logs = graphics_logs
        self.component_namespaces: list[str] = []
        self.created_nodes: list[str] = []
        self.deleted_pods: list[str] = []
        self.applied_manifests: list[dict[str, Any]] = []

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
            self.applied_manifests.append(manifest)
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
            return self._result(
                self.graphics_logs
                if command[1].startswith("npa-graphics-health-")
                else self.smoke_logs
            )
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
                                    "containers": [{"name": "nvidia-device-plugin-ctr"}]
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


def test_graphics_smoke_loads_glx_egl_and_enumerates_vulkan_device(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    nodes = [
        _node(
            "gpu-0",
            platform="gpu-rtx6000",
            gpus=1,
            boot_id="boot-a",
        )
    ]
    kubectl = _Kubectl([nodes])
    report = validate_gpu_health(
        kubectl,
        kubectl_bin="kubectl",
        kubeconfig_path=tmp_path / "kubeconfig",
        config=_config(
            expected_nodes=1,
            expected_gpu_nodes=1,
            gpu_preset="1gpu-24vcpu-218gb",
            gpu_platform="gpu-rtx6000",
            driver_mode="operator",
            nvswitch=False,
            cuda_smoke=False,
            graphics_smoke=True,
        ),
        sleep_fn=clock.sleep,
        monotonic_fn=clock.monotonic,
    )

    assert report["status"] == "healthy"
    assert report["graphics_smokes"][0]["glx"] == "loaded"
    assert report["graphics_smokes"][0]["egl"] == "loaded"
    assert report["graphics_smokes"][0]["vulkan_physical_devices"] == 1
    command = kubectl.applied_manifests[0]["spec"]["containers"][0]["args"][0]
    assert command.count("os._exit(0)") == 2
    assert command.index("NPA_GLX_LOADED") < command.index("NPA_EGL_LOADED")
    assert command.index("NPA_EGL_LOADED") < command.index("vulkaninfo --summary")


def test_graphics_smoke_fails_closed_without_vulkan_device(tmp_path: Path) -> None:
    clock = _Clock()
    nodes = [
        _node(
            "gpu-0",
            platform="gpu-rtx6000",
            gpus=1,
            boot_id="boot-a",
        )
    ]
    kubectl = _Kubectl([nodes], graphics_logs="NPA_GLX_LOADED\nNPA_EGL_LOADED\n")
    with pytest.raises(GpuHealthError, match="Vulkan instance"):
        validate_gpu_health(
            kubectl,
            kubectl_bin="kubectl",
            kubeconfig_path=tmp_path / "kubeconfig",
            config=_config(
                expected_nodes=1,
                expected_gpu_nodes=1,
                gpu_preset="1gpu-24vcpu-218gb",
                gpu_platform="gpu-rtx6000",
                driver_mode="operator",
                nvswitch=False,
                cuda_smoke=False,
                graphics_smoke=True,
            ),
            sleep_fn=clock.sleep,
            monotonic_fn=clock.monotonic,
        )


def _mixed_gpu_nodes() -> list[dict[str, Any]]:
    return [
        _node(
            f"gpu-{index}",
            platform="gpu-b200-sxm",
            gpus=count,
            boot_id=f"boot-{index}",
            fabric_state="Completed",
        )
        for index, count in enumerate((8, 1, 1, 1, 1, 1, 1, 1, 1))
    ] + [_node("cpu-0", platform="cpu-d3", gpus=0, boot_id="boot-cpu")]


def _mixed_config(**overrides) -> GpuHealthConfig:
    return _config(
        expected_nodes=10,
        expected_gpu_nodes=9,
        expected_gpu_counts=(8, 1, 1, 1, 1, 1, 1, 1, 1),
        **overrides,
    )


@pytest.mark.parametrize("counts", [(8,), (8, 0), (8, -1), (8, True), (8, 1.5)])
def test_declared_gpu_distribution_rejects_invalid_expectations(counts) -> None:
    with pytest.raises(ValueError, match="one positive integer per GPU node"):
        _config(expected_gpu_counts=counts).validate()


def test_mixed_sxm_pool_cannot_disable_fabric_checks_with_single_gpu_preset() -> None:
    with pytest.raises(ValueError, match="require NVSwitch checks"):
        _mixed_config(gpu_preset="1gpu-20vcpu-224gb", nvswitch=False).validate()


def test_mixed_pool_checks_distribution_even_when_total_is_correct(tmp_path: Path) -> None:
    nodes = _mixed_gpu_nodes()
    nodes[0]["status"]["allocatable"]["nvidia.com/gpu"] = "7"
    nodes[1]["status"]["allocatable"]["nvidia.com/gpu"] = "2"
    snapshot = probe_gpu_health(
        _Kubectl([nodes]),
        kubectl_bin="kubectl",
        kubeconfig_path=tmp_path / "kubeconfig",
        config=_mixed_config(),
    )
    assert snapshot["total_gpus"] == snapshot["expected_gpus"] == 16
    assert any("declared distribution" in error for error in snapshot["errors"])


@pytest.mark.parametrize("bad_node", [0, 1])
def test_mixed_pool_retains_gpu_and_fabric_checks_on_each_shape(
    tmp_path: Path, bad_node: int
) -> None:
    nodes = _mixed_gpu_nodes()
    nodes[bad_node]["metadata"]["annotations"]["nebius.ai/fabric-state"] = "In Progress"
    nodes[bad_node]["status"]["conditions"].append(
        {"type": "NebiusGPUError", "status": "True"}
    )
    snapshot = probe_gpu_health(
        _Kubectl([nodes]),
        kubectl_bin="kubectl",
        kubeconfig_path=tmp_path / "kubeconfig",
        config=_mixed_config(),
    )
    assert any("NebiusGPUError=True" in error for error in snapshot["errors"])
    assert any("fabric-state='In Progress'" in error for error in snapshot["errors"])


class _EveryDeviceKubectl(_Kubectl):
    def __init__(self, *, omit_last_device: bool = False) -> None:
        super().__init__([_mixed_gpu_nodes()])
        self.omit_last_device = omit_last_device

    def __call__(self, args, **kwargs):
        if args[1] == "logs":
            count = self.applied_manifests[-1]["spec"]["containers"][0]["resources"][
                "limits"
            ]["nvidia.com/gpu"]
            tested = count - 1 if self.omit_last_device else count
            return self._result(
                "Test PASSED\n" * tested
                + "".join(f"NPA_CUDA_DEVICE_{device}_PASSED\n" for device in range(tested))
                + "Fabric\n    State : Completed\n    Status : Success\n"
            )
        return super().__call__(args, **kwargs)


def test_mixed_pool_runs_cuda_on_all_sixteen_assigned_devices(tmp_path: Path) -> None:
    clock = _Clock()
    kubectl = _EveryDeviceKubectl()
    report = validate_gpu_health(
        kubectl,
        kubectl_bin="kubectl",
        kubeconfig_path=tmp_path / "kubeconfig",
        config=_mixed_config(cuda_smoke=True, stabilization_seconds=2),
        sleep_fn=clock.sleep,
        monotonic_fn=clock.monotonic,
    )
    assert report["status"] == "healthy"
    assert len(report["cuda_smokes"]) == len(kubectl.deleted_pods) == 9
    assert sum(smoke["tested_gpus"] for smoke in report["cuda_smokes"]) == 16
    assert all(smoke["fabric"] == "success" for smoke in report["cuda_smokes"])
    containers = [manifest["spec"]["containers"][0] for manifest in kubectl.applied_manifests]
    assert sorted(container["resources"]["limits"]["nvidia.com/gpu"] for container in containers) == [1] * 8 + [8]
    assert all('CUDA_VISIBLE_DEVICES="$device"' in container["args"][0] for container in containers)


def test_mixed_pool_rejects_incomplete_device_execution_and_cleans_probe(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    kubectl = _EveryDeviceKubectl(omit_last_device=True)
    with pytest.raises(GpuHealthError, match="complete per-device evidence"):
        validate_gpu_health(
            kubectl,
            kubectl_bin="kubectl",
            kubeconfig_path=tmp_path / "kubeconfig",
            config=_mixed_config(cuda_smoke=True),
            evidence_path=tmp_path / "gpu-health.json",
            sleep_fn=clock.sleep,
            monotonic_fn=clock.monotonic,
        )
    assert len(kubectl.deleted_pods) == 1
    assert json.loads((tmp_path / "gpu-health.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("counts", [(1,), (), (8, 2), (8, True)])
def test_fabric_scope_cannot_omit_required_or_add_unknown_gpu_shapes(counts) -> None:
    with pytest.raises(ValueError, match="NVSwitch subset|declared GPU-count subset"):
        _mixed_config(nvswitch_gpu_counts=counts).validate()


class _FractionalFabricKubectl(_EveryDeviceKubectl):
    def __init__(self, *, broken_eight_gpu_fabric=False, broken_single_gpu_kernel=False):
        super().__init__()
        self.broken_eight_gpu_fabric = broken_eight_gpu_fabric
        self.broken_single_gpu_kernel = broken_single_gpu_kernel

    def __call__(self, args, **kwargs):
        result = super().__call__(args, **kwargs)
        if args[1] == "logs":
            count = self.applied_manifests[-1]["spec"]["containers"][0]["resources"][
                "limits"
            ]["nvidia.com/gpu"]
            if count == 1 or self.broken_eight_gpu_fabric:
                result.stdout = result.stdout.replace("State : Completed", "State : N/A").replace("Status : Success", "Status : N/A")
            if count == 1 and self.broken_single_gpu_kernel:
                result.stdout = result.stdout.replace("Test PASSED", "Test FAILED")
        return result


def test_declared_fractional_guests_allow_na_fabric_after_all_device_cuda_passes(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    kubectl = _FractionalFabricKubectl()
    report = validate_gpu_health(
        kubectl,
        kubectl_bin="kubectl",
        kubeconfig_path=tmp_path / "kubeconfig",
        config=_mixed_config(cuda_smoke=True, nvswitch_gpu_counts=(8,)),
        sleep_fn=clock.sleep,
        monotonic_fn=clock.monotonic,
    )
    assert report["status"] == "healthy"
    assert report["final_snapshot"]["nvswitch_nodes"] == ["gpu-0"]
    assert sum(item["tested_gpus"] for item in report["cuda_smokes"]) == 16
    assert report["cuda_smokes"][0]["fabric"] == "success"
    assert all(item["fabric"] == "not-required" for item in report["cuda_smokes"][1:])


@pytest.mark.parametrize(
    "failure", [{"broken_eight_gpu_fabric": True}, {"broken_single_gpu_kernel": True}]
)
def test_scoped_fabric_preserves_full_node_fabric_and_fractional_kernel_failures(
    tmp_path: Path, failure: dict[str, bool]
) -> None:
    clock = _Clock()
    kubectl = _FractionalFabricKubectl(**failure)
    with pytest.raises(GpuHealthError):
        validate_gpu_health(
            kubectl,
            kubectl_bin="kubectl",
            kubeconfig_path=tmp_path / "kubeconfig",
            config=_mixed_config(cuda_smoke=True, nvswitch_gpu_counts=(8,)),
            sleep_fn=clock.sleep,
            monotonic_fn=clock.monotonic,
        )
    assert len(kubectl.deleted_pods) == len(kubectl.applied_manifests)


def test_fractional_fabric_exclusion_preserves_gpu_error_conditions(tmp_path: Path) -> None:
    nodes = _mixed_gpu_nodes()
    nodes[1]["metadata"]["annotations"]["nebius.ai/fabric-state"] = "N/A"
    nodes[1]["status"]["conditions"].append({"type": "NebiusGPUError", "status": "True"})
    snapshot = probe_gpu_health(
        _Kubectl([nodes]),
        kubectl_bin="kubectl",
        kubeconfig_path=tmp_path / "kubeconfig",
        config=_mixed_config(nvswitch_gpu_counts=(8,)),
    )
    assert any("NebiusGPUError=True" in error for error in snapshot["errors"])
    assert not any("fabric-state" in error for error in snapshot["errors"])


def test_explicit_fabric_attached_single_gpu_shape_rejects_na_status(tmp_path: Path) -> None:
    clock = _Clock()
    with pytest.raises(GpuHealthError, match="NVSwitch Fabric State='N/A'"):
        validate_gpu_health(
            _FractionalFabricKubectl(),
            kubectl_bin="kubectl",
            kubeconfig_path=tmp_path / "kubeconfig",
            config=_mixed_config(cuda_smoke=True, nvswitch_gpu_counts=(8, 1)),
            sleep_fn=clock.sleep,
            monotonic_fn=clock.monotonic,
        )
