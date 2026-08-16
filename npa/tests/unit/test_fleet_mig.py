"""Pure unit coverage for the declarative fleet RTX PRO hardware MIG block."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from npa.cli.main import app
from npa.fleet import MigSpec, spec_from_mapping
from npa.fleet.lifecycle import plan_fleet
from npa.fleet.mig import (
    GPU_DEVICE_PLUGIN_VERSION,
    GPU_DRIVER_VERSION,
    GPU_GFD_VERSION,
    GPU_MIG_MANAGER_VERSION,
    GPU_OPERATOR_VERSION,
    MigNodeStatus,
    MigSpecError,
    MigVerificationError,
    MigVerificationReport,
    _active_gpu_workloads,
    _collect_hardware,
    _ensure_device_plugin_mig_gate,
    _kubectl_env,
    _reconcile_ondelete_driver,
    _reconcile_stale_mig_taints,
    _replace_driver_pod,
    _restart_discovery_operands,
    _run_mig_cuda_smoke,
    inspect_mig_state,
    wait_for_mig_ready,
)
from npa.fleet.quotas import required_quotas
from npa.fleet.spec import ClusterSpec, FleetSpecError, NodePoolSpec
from npa.fleet.tfvars import render_tfvars


def _mapping(*, mig: object = None) -> dict:
    defaults: dict = {
        "gpu_nodes": {
            "count": 2,
            "platform": "gpu-rtx6000",
            "preset": "1gpu-24vcpu-218gb",
            "disk_size_gib": 128,
            "capacity_block_group": "capacityblockgroup-test",
        }
    }
    if mig is not None:
        defaults["mig"] = mig
    return {
        "apiVersion": "npa.fleet/v0.0.1",
        "name": "rtxpro-mig",
        "region": "us-central1",
        "defaults": defaults,
        "projects": [{"project_id": "project-test"}],
    }


def test_omitted_mig_preserves_whole_gpu_tfvars() -> None:
    cluster = spec_from_mapping(_mapping()).projects[0].clusters[0]
    cluster.validate()
    tfvars = render_tfvars(cluster)
    assert cluster.mig is None
    assert 'mig_strategy                 = "none"' in tfvars
    assert "mig_parted_config" not in tfvars
    assert "gpu_operator_version" not in tfvars
    assert "custom_driver                = false" in tfvars


def test_disabled_mig_is_backward_compatible() -> None:
    cluster = (
        spec_from_mapping(_mapping(mig={"enabled": False})).projects[0].clusters[0]
    )
    cluster.validate()
    assert cluster.mig == MigSpec(enabled=False, strategy="none", config="")
    assert 'mig_strategy                 = "none"' in render_tfvars(cluster)


def test_cluster_can_atomically_disable_inherited_mig_default() -> None:
    data = _mapping(mig={"enabled": True})
    data["projects"][0]["clusters"] = [{"mig": {"enabled": False}}]
    cluster = spec_from_mapping(data).projects[0].clusters[0]
    cluster.validate()
    assert cluster.mig == MigSpec(enabled=False, strategy="none", config="")


def test_enabled_mig_renders_pinned_compatibility_tuple() -> None:
    cluster = (
        spec_from_mapping(
            _mapping(
                mig={"enabled": True, "strategy": "mixed", "config": "all-balanced"}
            )
        )
        .projects[0]
        .clusters[0]
    )
    cluster.validate()
    tfvars = render_tfvars(cluster)
    assert cluster.resolved_k8s_version() == "1.34"
    assert 'k8s_version = "1.34"' in tfvars
    assert 'mig_strategy                 = "mixed"' in tfvars
    assert 'mig_parted_config            = "all-balanced"' in tfvars
    assert "gpu_mig_with_reboot          = true" in tfvars
    assert "gpu_operator_rdma_enabled    = false" in tfvars
    assert "custom_driver                = true" in tfvars
    assert "gpu_nodes_driverfull_image   = false" in tfvars
    assert 'gpu_disk_size = "128"' in tfvars
    for name, version in {
        "gpu_operator_version": GPU_OPERATOR_VERSION,
        "gpu_driver_version": GPU_DRIVER_VERSION,
        "gpu_device_plugin_version": GPU_DEVICE_PLUGIN_VERSION,
        "gpu_gfd_version": GPU_GFD_VERSION,
        "gpu_mig_manager_version": GPU_MIG_MANAGER_VERSION,
    }.items():
        assert f"{name}" in tfvars
        assert f'"{version}"' in tfvars

    plan = plan_fleet(spec_from_mapping(_mapping(mig={"enabled": True})))
    assert plan["projects"][0]["clusters"][0]["k8s_version"] == "1.34"
    assert plan["projects"][0]["clusters"][0]["gpu_driver_mode"] == "operator"


def test_enabled_mig_rejects_explicit_managed_image_driver_mode() -> None:
    data = _mapping(mig={"enabled": True})
    data["defaults"]["gpu_driver_mode"] = "managed-image"
    with pytest.raises(FleetSpecError, match="requires the pinned GPU Operator"):
        spec_from_mapping(data).projects[0].clusters[0].validate()


def test_enabled_mig_requires_deploy_cuda_smoke() -> None:
    data = _mapping(mig={"enabled": True})
    data["defaults"]["gpu_cuda_smoke"] = False
    with pytest.raises(FleetSpecError, match="requires gpu_cuda_smoke=true"):
        spec_from_mapping(data).projects[0].clusters[0].validate()


def test_mig_quota_preflight_uses_same_safe_boot_disk_as_tfvars() -> None:
    cluster = spec_from_mapping(_mapping(mig={"enabled": True})).projects[0].clusters[0]
    assert cluster.resolved_gpu_disk_size_gib() == 128
    assert required_quotas([cluster])["compute.disk.size.network-ssd"] == (
        2 * 128 * 1024**3
    )


def test_expected_resources_are_exact_and_scaled_per_node() -> None:
    mig = MigSpec()
    assert mig.expected_resources_per_gpu() == {
        "nvidia.com/gpu": 0,
        "nvidia.com/mig-1g.24gb": 2,
        "nvidia.com/mig-2g.48gb": 1,
    }
    assert mig.expected_resources_per_node(gpus_per_node=2) == {
        "nvidia.com/gpu": 0,
        "nvidia.com/mig-1g.24gb": 4,
        "nvidia.com/mig-2g.48gb": 2,
    }
    with pytest.raises(MigSpecError, match="positive"):
        mig.expected_resources_per_node(gpus_per_node=0)


@pytest.mark.parametrize(
    ("mig", "message"),
    [
        ("mixed", "must be a mapping"),
        ({"enabled": "true"}, "must be a boolean"),
        ({"enabled": False, "strategy": "mixed"}, "disabled MIG"),
        ({"enabled": True, "strategy": "single"}, "strategy must be 'mixed'"),
        ({"enabled": True, "config": "all-1g.24gb"}, "config must be 'all-balanced'"),
        ({"enabled": True, "driver_version": "latest"}, "unsupported field"),
    ],
)
def test_mig_mapping_fails_closed(mig: object, message: str) -> None:
    with pytest.raises(FleetSpecError, match=message):
        cluster = spec_from_mapping(_mapping(mig=mig)).projects[0].clusters[0]
        cluster.validate()


def test_enabled_mig_rejects_wrong_platform_missing_pool_and_kubernetes() -> None:
    with pytest.raises(FleetSpecError, match="only for RTX PRO 6000"):
        ClusterSpec(
            name="h200",
            enable_gpu_cluster=False,
            gpu_nodes=NodePoolSpec(
                count=1,
                platform="gpu-h200-sxm",
                preset="8gpu-128vcpu-1600gb",
                capacity_block_group="capacityblockgroup-test",
            ),
            mig=MigSpec(),
        ).validate()

    with pytest.raises(FleetSpecError, match="only for RTX PRO 6000"):
        ClusterSpec(
            name="cpu",
            cpu_nodes=NodePoolSpec(count=1),
            mig=MigSpec(),
        ).validate()

    with pytest.raises(MigSpecError, match="positive GPU worker count"):
        MigSpec().validate(
            platform="gpu-rtx6000",
            preset="1gpu-24vcpu-218gb",
            gpu_nodes=0,
            capacity_block_group="capacityblockgroup-test",
            disk_size_gib=128,
            k8s_version="1.34",
        )

    with pytest.raises(MigSpecError, match="tested one-GPU preset"):
        MigSpec().validate(
            platform="gpu-rtx6000",
            preset="",
            gpu_nodes=0,
            capacity_block_group="capacityblockgroup-test",
            disk_size_gib=128,
            k8s_version="1.34",
        )

    with pytest.raises(FleetSpecError, match="only for RTX PRO 6000"):
        ClusterSpec(
            name="wrong-platform-zero-workers",
            gpu_nodes=NodePoolSpec(
                count=0,
                platform="gpu-l40s",
                preset="1gpu-24vcpu-218gb",
                capacity_block_group="capacityblockgroup-test",
            ),
            mig=MigSpec(),
        ).validate()

    with pytest.raises(FleetSpecError, match="positive GPU worker count"):
        ClusterSpec(
            name="zero-workers",
            gpu_nodes=NodePoolSpec(
                count=0,
                platform="gpu-rtx6000",
                preset="1gpu-24vcpu-218gb",
                capacity_block_group="capacityblockgroup-test",
            ),
            mig=MigSpec(),
        ).validate()

    with pytest.raises(FleetSpecError, match="exactly two GPU workers"):
        ClusterSpec(
            name="one-worker",
            gpu_nodes=NodePoolSpec(
                count=1,
                platform="gpu-rtx6000",
                preset="1gpu-24vcpu-218gb",
                capacity_block_group="capacityblockgroup-test",
                disk_size_gib=128,
            ),
            mig=MigSpec(),
        ).validate()

    with pytest.raises(FleetSpecError, match="requires Kubernetes 1.34"):
        ClusterSpec(
            name="old-k8s",
            k8s_version="1.33",
            gpu_nodes=NodePoolSpec(
                count=2,
                platform="gpu-rtx6000",
                preset="1gpu-24vcpu-218gb",
                capacity_block_group="capacityblockgroup-test",
            ),
            mig=MigSpec(),
        ).validate()


def _live_payloads(*, gpu_capacity: int = 0, gpu_allocatable: int = 0):
    resources = {
        "nvidia.com/gpu": str(gpu_capacity),
        "nvidia.com/mig-1g.24gb": "2",
        "nvidia.com/mig-2g.48gb": "1",
    }
    allocatable = {**resources, "nvidia.com/gpu": str(gpu_allocatable)}
    node = {
        "metadata": {
            "name": "gpu-node-0",
            "labels": {
                "nvidia.com/gpu.present": "true",
                "nvidia.com/gpu.product": "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
                "nvidia.com/mig.strategy": "mixed",
                "nvidia.com/mig.config": "all-balanced",
                "nvidia.com/mig.config.state": "success",
            },
        },
        "spec": {},
        "status": {
            "capacity": resources,
            "allocatable": allocatable,
            "conditions": [{"type": "Ready", "status": "True"}],
        },
    }
    policy = {
        "metadata": {"labels": {"app.kubernetes.io/version": "v26.3.3"}},
        "spec": {
            "driver": {"version": "580.173.02"},
            "devicePlugin": {"version": "v0.19.3"},
            "gfd": {"version": "v0.19.3"},
            "migManager": {"version": "v0.14.2"},
            "mig": {"strategy": "mixed"},
            "cdi": {"enabled": True},
        },
        "status": {"state": "ready"},
    }
    versions = {
        "nvidia-driver-daemonset": "580.173.02",
        "nvidia-mig-manager": "v0.14.2",
        "gpu-feature-discovery": "v0.19.3",
        "nvidia-device-plugin-daemonset": "v0.19.3",
        "nvidia-container-toolkit-daemonset": "v1.19.1",
    }
    daemonsets = [
        {
            "metadata": {"name": name, "generation": 3},
            "status": {
                "observedGeneration": 3,
                "desiredNumberScheduled": 1,
                "currentNumberScheduled": 1,
                "updatedNumberScheduled": 1,
                "numberReady": 1,
                "numberAvailable": 1,
            },
            "spec": {
                "template": {
                    "spec": {"containers": [{"image": f"registry/{name}:{version}"}]}
                }
            },
        }
        for name, version in versions.items()
    ]
    deployment = {
        "metadata": {"name": "gpu-operator", "generation": 4},
        "spec": {"replicas": 1},
        "status": {
            "observedGeneration": 4,
            "updatedReplicas": 1,
            "readyReplicas": 1,
            "availableReplicas": 1,
        },
    }
    return (
        {"items": [node]},
        {"items": [policy]},
        {"items": daemonsets},
        deployment,
    )


def test_live_snapshot_requires_exact_capacity_and_allocatable() -> None:
    report = inspect_mig_state(*_live_payloads(), expected_nodes=1)
    assert report.ready
    assert report.errors == ()

    stale = inspect_mig_state(
        *_live_payloads(gpu_capacity=1, gpu_allocatable=0), expected_nodes=1
    )
    assert not stale.ready
    assert any(
        "nvidia.com/gpu capacity/allocatable=1/0" in error for error in stale.errors
    )


def test_live_snapshot_rejects_cordoned_notready_and_stale_rollouts() -> None:
    nodes, policy, daemonsets, deployment = _live_payloads()
    nodes["items"][0]["spec"]["unschedulable"] = True
    nodes["items"][0]["status"]["conditions"][0]["status"] = "False"
    daemonsets["items"][0]["status"]["updatedNumberScheduled"] = 0
    deployment["status"]["observedGeneration"] = 3
    report = inspect_mig_state(nodes, policy, daemonsets, deployment, expected_nodes=1)
    assert not report.ready
    assert any("Ready condition is not True" in error for error in report.errors)
    assert any("cordoned/unschedulable" in error for error in report.errors)
    assert any(
        "desired/current/updated/ready/available" in error for error in report.errors
    )
    assert any("generation has not been observed" in error for error in report.errors)


def test_live_snapshot_rejects_stale_mig_readiness_taint() -> None:
    nodes, policy, daemonsets, deployment = _live_payloads()
    nodes["items"][0]["spec"]["taints"] = [
        {
            "key": "nvidia.com/gpu",
            "value": "mig-not-ready",
            "effect": "NoSchedule",
        }
    ]
    report = inspect_mig_state(nodes, policy, daemonsets, deployment, expected_nodes=1)
    assert not report.ready
    assert any("stale nvidia.com/gpu=mig-not-ready" in error for error in report.errors)


def test_stale_mig_taint_reconciliation_is_exact_and_race_closed(monkeypatch) -> None:  # noqa: ANN001
    nodes, *_ = _live_payloads()
    nodes["items"][0]["spec"]["taints"] = [
        {"key": "customer", "value": "keep", "effect": "NoSchedule"},
        {
            "key": "nvidia.com/gpu",
            "value": "mig-not-ready",
            "effect": "NoSchedule",
        },
    ]
    monkeypatch.setattr("npa.fleet.mig._kubectl_json", lambda *_a, **_k: nodes)
    commands: list[list[str]] = []

    def run(command, **_kwargs):  # noqa: ANN001, ANN202
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("npa.fleet.mig.subprocess.run", run)
    assert _reconcile_stale_mig_taints("kubectl", Path("/tmp/kubeconfig")) == 1
    assert len(commands) == 1
    patch = json.loads(commands[0][-1])
    assert patch[-1] == {"op": "remove", "path": "/spec/taints/1"}
    assert {operation.get("value") for operation in patch[:-1]} == {
        "nvidia.com/gpu",
        "mig-not-ready",
        "NoSchedule",
    }


def test_kubectl_env_scrubs_stale_tokens_unless_reuse_is_explicit(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("NEBIUS_IAM_TOKEN", "stale-provider-token")
    monkeypatch.setenv("NPA_NEBIUS_IAM_TOKEN", "stale-npa-token")
    monkeypatch.setenv("NEBIUS_IAM_TOKEN_FILE", "/tmp/stale-token")
    env = _kubectl_env()
    assert "NEBIUS_IAM_TOKEN" not in env
    assert "NPA_NEBIUS_IAM_TOKEN" not in env
    assert "NEBIUS_IAM_TOKEN_FILE" not in env

    monkeypatch.setenv("NPA_REUSE_IAM_TOKEN", "true")
    reused = _kubectl_env()
    assert reused["NEBIUS_IAM_TOKEN"] == "stale-provider-token"
    assert reused["NPA_NEBIUS_IAM_TOKEN"] == "stale-npa-token"


def test_transient_driver_exec_is_a_nonready_snapshot_not_an_api_error(
    monkeypatch,
) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        "npa.fleet.mig._kubectl_json",
        lambda *_a, **_k: {
            "items": [
                {
                    "metadata": {"name": "driver-0"},
                    "spec": {"nodeName": "gpu-node-0"},
                    "status": {"phase": "Running"},
                }
            ]
        },
    )
    monkeypatch.setattr(
        "npa.fleet.mig._kubectl_text",
        lambda *_a, **_k: (_ for _ in ()).throw(
            MigVerificationError("transient pod replacement")
        ),
    )
    hardware, errors = _collect_hardware(
        "kubectl", Path("/tmp/kubeconfig"), expected_nodes=1
    )
    assert hardware == ()
    assert errors == (
        "node gpu-node-0: driver pod hardware inspection is temporarily unavailable",
    )


def test_hardware_snapshot_requires_exact_profiles_and_cuda(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        "npa.fleet.mig._kubectl_json",
        lambda *_a, **_k: {
            "items": [
                {
                    "metadata": {"name": "driver-0"},
                    "spec": {"nodeName": "gpu-node-0"},
                    "status": {"phase": "Running"},
                }
            ]
        },
    )
    outputs = iter(
        (
            "NVIDIA RTX PRO 6000 Blackwell Server Edition, 0x2BB510DE, "
            "98.02.8D.00.01, 580.173.02, 97887, Enabled, Enabled, "
            "GPU-6258f3b8-9223-0b48-2681-64d5df5ea65c\n",
            "GPU 0: NVIDIA RTX PRO 6000 Blackwell Server Edition "
            "(UUID: GPU-6258f3b8-9223-0b48-2681-64d5df5ea65c)\n"
            "  MIG 2g.48gb Device 0: (UUID: "
            "MIG-0842a98a-749e-5a49-86b2-a3deb949f2f8)\n"
            "  MIG 1g.24gb Device 1: (UUID: "
            "MIG-4e7fad71-08f6-5469-a653-1615011514ce)\n"
            "  MIG 1g.24gb Device 2: (UUID: "
            "MIG-7c13d712-950a-503e-9a9d-4bab01efc17b)\n",
            "NVIDIA-SMI 580.173.02 Driver Version: 580.173.02 CUDA Version: 13.0\n",
        )
    )
    monkeypatch.setattr("npa.fleet.mig._kubectl_text", lambda *_a, **_k: next(outputs))
    hardware, errors = _collect_hardware(
        "kubectl", Path("/tmp/kubeconfig"), expected_nodes=1
    )
    assert errors == ()
    assert hardware[0].mig_profiles == ("1g.24gb", "1g.24gb", "2g.48gb")
    assert hardware[0].cuda_version == "13.0"


def test_stale_resource_reconciliation_is_strictly_ordered(monkeypatch) -> None:  # noqa: ANN001
    commands: list[list[str]] = []

    def run(command, **_kwargs):  # noqa: ANN001, ANN202
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("npa.fleet.mig.subprocess.run", run)
    _restart_discovery_operands("kubectl", Path("/tmp/kubeconfig"))
    assert [(command[4], command[5]) for command in commands] == [
        ("restart", "daemonset/gpu-feature-discovery"),
        ("status", "daemonset/gpu-feature-discovery"),
        ("restart", "daemonset/nvidia-device-plugin-daemonset"),
        ("status", "daemonset/nvidia-device-plugin-daemonset"),
    ]


def _wait_report(*, ready: bool, errors: tuple[str, ...]) -> MigVerificationReport:
    return MigVerificationReport(
        ready=ready,
        nodes=(
            MigNodeStatus(
                name="gpu-node-0",
                product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
                config="all-balanced",
                config_state="success",
                capacity={},
                allocatable={},
                schedulable=True,
                ready=True,
            ),
        ),
        errors=errors,
        operator_state="ready",
        operator_version="v26.3.3",
    )


def test_wait_reconciles_stale_resources_once_then_requires_two_snapshots(
    monkeypatch,
) -> None:  # noqa: ANN001
    stale = _wait_report(
        ready=False,
        errors=(
            "node gpu-node-0: nvidia.com/gpu capacity/allocatable=1/0, expected 0/0",
        ),
    )
    exact = _wait_report(ready=True, errors=())
    reports = iter((stale, exact, exact))
    restarts: list[bool] = []
    monkeypatch.setattr(
        "npa.fleet.mig._ensure_device_plugin_mig_gate", lambda *_a, **_k: True
    )
    monkeypatch.setattr("npa.fleet.mig.verify_mig_cluster", lambda **_k: next(reports))
    monkeypatch.setattr(
        "npa.fleet.mig._restart_discovery_operands",
        lambda *_a, **_k: restarts.append(True),
    )
    monkeypatch.setattr("npa.fleet.mig.time.sleep", lambda _seconds: None)
    report = wait_for_mig_ready(
        kubectl_bin="kubectl",
        kubeconfig=Path("/tmp/kubeconfig"),
        expected_nodes=1,
    )
    assert report.ready
    assert restarts == [True]


def test_wait_fails_after_repeated_immutable_hardware_incompatibility(
    monkeypatch,
) -> None:  # noqa: ANN001
    incompatible = _wait_report(
        ready=False,
        errors=("node gpu-node-0: unsupported vBIOS '98.00.00.00.00'",),
    )
    monkeypatch.setattr(
        "npa.fleet.mig._ensure_device_plugin_mig_gate", lambda *_a, **_k: True
    )
    monkeypatch.setattr("npa.fleet.mig.verify_mig_cluster", lambda **_k: incompatible)
    monkeypatch.setattr("npa.fleet.mig.time.sleep", lambda _seconds: None)
    with pytest.raises(MigVerificationError, match="cannot converge.*vBIOS"):
        wait_for_mig_ready(
            kubectl_bin="kubectl",
            kubeconfig=Path("/tmp/kubeconfig"),
            expected_nodes=1,
        )


def test_wait_runs_cuda_smoke_after_two_exact_snapshots(monkeypatch) -> None:  # noqa: ANN001
    exact = _wait_report(ready=True, errors=())
    reports = iter((exact, exact))
    events: list[str] = []
    monkeypatch.setattr(
        "npa.fleet.mig._ensure_device_plugin_mig_gate", lambda *_a, **_k: True
    )

    def verify(**_kwargs):  # noqa: ANN003, ANN202
        events.append("snapshot")
        return next(reports)

    def smoke(**_kwargs):  # noqa: ANN003, ANN202
        events.append("smoke")
        return {"resource": "nvidia.com/mig-1g.24gb", "vectoradd": "passed"}

    monkeypatch.setattr("npa.fleet.mig.verify_mig_cluster", verify)
    monkeypatch.setattr("npa.fleet.mig._run_mig_cuda_smoke", smoke)
    report = wait_for_mig_ready(
        kubectl_bin="kubectl",
        kubeconfig=Path("/tmp/kubeconfig"),
        expected_nodes=1,
        timeout_seconds=60,
        cuda_smoke_image="cuda-vectoradd:test",
        sleep_fn=lambda _seconds: None,
        monotonic_fn=lambda: 100.0,
    )
    assert events == ["snapshot", "snapshot", "smoke"]
    assert report.cuda_smoke == {
        "resource": "nvidia.com/mig-1g.24gb",
        "vectoradd": "passed",
    }


def test_wait_times_out_on_recurring_nonconvergent_state(monkeypatch) -> None:  # noqa: ANN001
    pending = _wait_report(
        ready=False,
        errors=("node gpu-node-0: MIG config state is 'pending', expected 'success'",),
    )
    clock = [100.0]

    def sleep(seconds: float) -> None:
        clock[0] += seconds

    monkeypatch.setattr(
        "npa.fleet.mig._ensure_device_plugin_mig_gate", lambda *_a, **_k: True
    )
    monkeypatch.setattr("npa.fleet.mig.verify_mig_cluster", lambda **_k: pending)
    with pytest.raises(
        MigVerificationError,
        match=r"did not converge within 20s.*MIG config state is 'pending'",
    ):
        wait_for_mig_ready(
            kubectl_bin="kubectl",
            kubeconfig=Path("/tmp/kubeconfig"),
            expected_nodes=1,
            reconcile=False,
            timeout_seconds=20,
            sleep_fn=sleep,
            monotonic_fn=lambda: clock[0],
        )


def test_wait_times_out_after_one_stale_resource_reconciliation(
    monkeypatch,
) -> None:  # noqa: ANN001
    stale = _wait_report(
        ready=False,
        errors=(
            "node gpu-node-0: nvidia.com/gpu capacity/allocatable=1/0, expected 0/0",
        ),
    )
    clock = [100.0]
    restarts: list[bool] = []

    def sleep(seconds: float) -> None:
        clock[0] += seconds

    monkeypatch.setattr(
        "npa.fleet.mig._ensure_device_plugin_mig_gate", lambda *_a, **_k: True
    )
    monkeypatch.setattr("npa.fleet.mig.verify_mig_cluster", lambda **_k: stale)
    monkeypatch.setattr(
        "npa.fleet.mig._restart_discovery_operands",
        lambda *_a, **_k: restarts.append(True),
    )
    with pytest.raises(
        MigVerificationError,
        match=r"did not converge within 20s.*capacity/allocatable=1/0",
    ):
        wait_for_mig_ready(
            kubectl_bin="kubectl",
            kubeconfig=Path("/tmp/kubeconfig"),
            expected_nodes=1,
            timeout_seconds=20,
            sleep_fn=sleep,
            monotonic_fn=lambda: clock[0],
        )
    assert restarts == [True]


def test_discovery_restart_recomputes_one_shared_deadline(monkeypatch) -> None:
    clock = [100.0]
    observed_timeouts: list[float] = []

    def run(command, **kwargs):  # noqa: ANN001, ANN202
        observed_timeouts.append(kwargs["timeout"])
        clock[0] += 1.0
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("npa.fleet.mig.subprocess.run", run)
    _restart_discovery_operands(
        "kubectl",
        Path("/tmp/kubeconfig"),
        deadline=104.0,
        monotonic_fn=lambda: clock[0],
    )
    assert observed_timeouts == [4.0, 3.0, 2.0, 1.0]


def test_wait_bounds_a_hung_verification_query(monkeypatch) -> None:  # noqa: ANN001
    observed_timeouts: list[float] = []

    def run(command, **kwargs):  # noqa: ANN001, ANN202
        observed_timeouts.append(kwargs["timeout"])
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr("npa.fleet.mig.subprocess.run", run)
    with pytest.raises(MigVerificationError, match="TimeoutExpired"):
        wait_for_mig_ready(
            kubectl_bin="kubectl",
            kubeconfig=Path("/tmp/kubeconfig"),
            expected_nodes=1,
            reconcile=False,
            timeout_seconds=7,
            monotonic_fn=lambda: 10.0,
        )
    assert observed_timeouts == [7.0]


def test_driver_replacement_timeout_is_actionable(monkeypatch) -> None:  # noqa: ANN001
    clock = [10.0]
    commands: list[list[str]] = []

    def run(command, **_kwargs):  # noqa: ANN001, ANN202
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    def sleep(seconds: float) -> None:
        clock[0] += seconds

    monkeypatch.setattr("npa.fleet.mig.subprocess.run", run)
    monkeypatch.setattr(
        "npa.fleet.mig._kubectl_json",
        lambda *_a, **_k: {
            "items": [
                {
                    "metadata": {"uid": "old"},
                    "spec": {"nodeName": "gpu-node-0"},
                    "status": {"phase": "Running", "containerStatuses": []},
                }
            ]
        },
    )
    with pytest.raises(
        MigVerificationError,
        match="timed out waiting for the replacement NVIDIA driver pod.*uncordoned",
    ):
        _replace_driver_pod(
            kubectl_bin="kubectl",
            kubeconfig=Path("/tmp/kubeconfig"),
            pod_name="driver-old",
            pod_uid="old",
            node="gpu-node-0",
            deadline=30.0,
            sleep_fn=sleep,
            monotonic_fn=lambda: clock[0],
        )
    assert commands[0][-5:] == [
        "delete",
        "pod",
        "driver-old",
        "-n",
        "gpu-operator",
    ]


@pytest.mark.parametrize(
    ("already_cordoned", "expected_commands"),
    [(False, ["cordon", "uncordon"]), (True, [])],
)
def test_driver_reconciliation_restores_only_its_own_cordon(
    monkeypatch, already_cordoned: bool, expected_commands: list[str]
) -> None:  # noqa: ANN001
    driver_pods = {
        "items": [
            {
                "metadata": {
                    "namespace": "gpu-operator",
                    "name": "driver-old",
                    "uid": "old",
                    "labels": {"app": "nvidia-driver-daemonset"},
                },
                "spec": {"nodeName": "gpu-node-0"},
                "status": {"phase": "Running"},
            }
        ]
    }

    def kubectl_json(_bin, _config, args, **_kwargs):  # noqa: ANN001, ANN202
        if args[:2] == ["get", "node"]:
            return {"spec": {"unschedulable": already_cordoned}}
        return driver_pods

    commands: list[str] = []

    def run(command, **_kwargs):  # noqa: ANN001, ANN202
        commands.append(command[3])
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("npa.fleet.mig._kubectl_json", kubectl_json)
    monkeypatch.setattr("npa.fleet.mig.subprocess.run", run)
    monkeypatch.setattr(
        "npa.fleet.mig._replace_driver_pod",
        lambda **_kwargs: (_ for _ in ()).throw(
            MigVerificationError("driver replacement deadline expired")
        ),
    )
    with pytest.raises(MigVerificationError, match="deadline expired"):
        _reconcile_ondelete_driver(
            "kubectl",
            Path("/tmp/kubeconfig"),
            deadline=30.0,
            sleep_fn=lambda _seconds: None,
            monotonic_fn=lambda: 0.0,
        )
    assert commands == expected_commands


def test_driver_reconciliation_uncordons_after_ambiguous_cordon_timeout(
    monkeypatch,
) -> None:  # noqa: ANN001
    driver_pods = {
        "items": [
            {
                "metadata": {
                    "namespace": "gpu-operator",
                    "name": "driver-old",
                    "uid": "old",
                    "labels": {"app": "nvidia-driver-daemonset"},
                },
                "spec": {"nodeName": "gpu-node-0"},
                "status": {"phase": "Running"},
            }
        ]
    }

    def kubectl_json(_bin, _config, args, **_kwargs):  # noqa: ANN001, ANN202
        if args[:2] == ["get", "node"]:
            return {"spec": {"unschedulable": False}}
        return driver_pods

    commands: list[str] = []

    def run(command, **kwargs):  # noqa: ANN001, ANN202
        operation = command[3]
        commands.append(operation)
        if operation == "cordon":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("npa.fleet.mig._kubectl_json", kubectl_json)
    monkeypatch.setattr("npa.fleet.mig.subprocess.run", run)
    with pytest.raises(
        MigVerificationError,
        match="could not cordon.*TimeoutExpired",
    ):
        _reconcile_ondelete_driver(
            "kubectl",
            Path("/tmp/kubeconfig"),
            deadline=30.0,
            sleep_fn=lambda _seconds: None,
            monotonic_fn=lambda: 0.0,
        )
    assert commands == ["cordon", "uncordon"]


def test_device_plugin_gate_preserves_every_existing_selector_term(
    monkeypatch,
) -> None:  # noqa: ANN001
    daemonset = {
        "metadata": {"resourceVersion": "123"},
        "spec": {
            "template": {
                "spec": {
                    "affinity": {
                        "nodeAffinity": {
                            "requiredDuringSchedulingIgnoredDuringExecution": {
                                "nodeSelectorTerms": [
                                    {
                                        "matchExpressions": [
                                            {
                                                "key": "nvidia.com/gpu.deploy.operands",
                                                "operator": "In",
                                                "values": ["true"],
                                            }
                                        ]
                                    },
                                    {
                                        "matchFields": [
                                            {
                                                "key": "metadata.name",
                                                "operator": "NotIn",
                                                "values": ["retired-node"],
                                            }
                                        ]
                                    },
                                ]
                            }
                        }
                    }
                }
            }
        },
    }
    patches: list[dict] = []
    monkeypatch.setattr("npa.fleet.mig._kubectl_json", lambda *_a, **_k: daemonset)

    def run(command, **_kwargs):  # noqa: ANN001, ANN202
        patches.append(json.loads(command[-1]))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("npa.fleet.mig.subprocess.run", run)
    assert _ensure_device_plugin_mig_gate("kubectl", Path("/tmp/kubeconfig"))
    terms = patches[0]["spec"]["template"]["spec"]["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ]["nodeSelectorTerms"]
    assert patches[0]["metadata"]["resourceVersion"] == "123"
    assert terms[0]["matchExpressions"][0]["key"] == "nvidia.com/gpu.deploy.operands"
    assert terms[1]["matchFields"][0]["values"] == ["retired-node"]
    assert all(
        term["matchExpressions"][-1]
        == {
            "key": "nvidia.com/mig.config.state",
            "operator": "In",
            "values": ["success"],
        }
        for term in terms
    )


def test_representative_mig_cuda_smoke_requests_limits_and_cleans_up(
    monkeypatch,
) -> None:  # noqa: ANN001
    commands: list[list[str]] = []
    manifest: dict = {}

    def run(command, **kwargs):  # noqa: ANN001, ANN202
        commands.append(command)
        if "apply" in command:
            manifest.update(json.loads(kwargs["input"]))
            return subprocess.CompletedProcess(command, 0, "created", "")
        if "logs" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                "Test PASSED\nNPA_MIG_VISIBLE=MIG-01234567-89ab-cdef-0123-456789abcdef\n"
                "  MIG 1g.24gb Device 0: (UUID: "
                "MIG-01234567-89ab-cdef-0123-456789abcdef)\n"
                "0MiB / 24192MiB\n",
                "",
            )
        return subprocess.CompletedProcess(command, 0, "deleted", "")

    monkeypatch.setattr("npa.fleet.mig.subprocess.run", run)
    monkeypatch.setattr(
        "npa.fleet.mig._kubectl_json",
        lambda *_a, **_k: {
            "spec": {"nodeName": "gpu-node-0"},
            "status": {
                "phase": "Succeeded",
                "containerStatuses": [
                    {"state": {"terminated": {"exitCode": 0, "reason": "Completed"}}}
                ],
            },
        },
    )
    monkeypatch.setattr(
        "npa.fleet.mig.uuid.uuid4",
        lambda: type("UUID", (), {"hex": "0123456789abcdef"})(),
    )
    result = _run_mig_cuda_smoke(
        kubectl_bin="kubectl",
        kubeconfig=Path("/tmp/kubeconfig"),
        image="cuda-vectoradd:test",
        deadline=100.0,
        sleep_fn=lambda _seconds: None,
        monotonic_fn=lambda: 0.0,
    )
    resources = manifest["spec"]["containers"][0]["resources"]
    assert manifest["spec"]["runtimeClassName"] == "nvidia"
    assert resources == {
        "requests": {"nvidia.com/mig-1g.24gb": 1},
        "limits": {"nvidia.com/mig-1g.24gb": 1},
    }
    assert result["vectoradd"] == "passed"
    assert result["profile"] == "1g.24gb"
    assert result["mig_uuid"].startswith("MIG-")
    assert "delete" in commands[-1]
    assert "--wait=true" in commands[-1]
    assert any(argument.startswith("--timeout=") for argument in commands[-1])
    assert manifest["spec"]["activeDeadlineSeconds"] > 0


@pytest.mark.parametrize(
    ("logs", "message"),
    [
        ("MIG 1g.24gb Device 0: (UUID: MIG-0123456789abcdef)\n", "Test PASSED"),
        (
            "Test PASSED\nNPA_MIG_VISIBLE=GPU-0123456789abcdef\n"
            "MIG 1g.24gb Device 0: (UUID: MIG-0123456789abcdef)\n"
            "0MiB / 24192MiB\n",
            "inconsistent allocated-device identity",
        ),
        (
            "Test PASSED\nNPA_MIG_VISIBLE=MIG-0123456789abcdef\n"
            "MIG 2g.48gb Device 0: (UUID: MIG-0123456789abcdef)\n",
            "did not report its allocated.*requested 1g.24gb profile",
        ),
    ],
)
def test_representative_mig_cuda_smoke_fails_closed_on_missing_evidence(
    monkeypatch, logs: str, message: str
) -> None:  # noqa: ANN001
    def run(command, **_kwargs):  # noqa: ANN001, ANN202
        if "logs" in command:
            return subprocess.CompletedProcess(command, 0, logs, "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("npa.fleet.mig.subprocess.run", run)
    monkeypatch.setattr(
        "npa.fleet.mig._kubectl_json",
        lambda *_args, **_kwargs: {
            "spec": {"nodeName": "gpu-node-0"},
            "status": {"phase": "Succeeded", "containerStatuses": []},
        },
    )

    with pytest.raises(MigVerificationError, match=message):
        _run_mig_cuda_smoke(
            kubectl_bin="kubectl",
            kubeconfig=Path("/tmp/kubeconfig"),
            image="cuda-vectoradd:test",
            deadline=100.0,
            sleep_fn=lambda _seconds: None,
            monotonic_fn=lambda: 0.0,
        )


def test_representative_mig_cuda_smoke_fails_if_cleanup_fails(monkeypatch) -> None:
    def run(command, **_kwargs):  # noqa: ANN001, ANN202
        if "logs" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                "Test PASSED\nNPA_MIG_VISIBLE=MIG-0123456789abcdef\n"
                "MIG 1g.24gb Device 0: (UUID: MIG-0123456789abcdef)\n"
                "0MiB / 24192MiB\n",
                "",
            )
        if "delete" in command:
            return subprocess.CompletedProcess(command, 1, "", "cleanup failed")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("npa.fleet.mig.subprocess.run", run)
    monkeypatch.setattr(
        "npa.fleet.mig._kubectl_json",
        lambda *_args, **_kwargs: {
            "spec": {"nodeName": "gpu-node-0"},
            "status": {"phase": "Succeeded", "containerStatuses": []},
        },
    )

    with pytest.raises(MigVerificationError, match="smoke passed.*cleanup failed"):
        _run_mig_cuda_smoke(
            kubectl_bin="kubectl",
            kubeconfig=Path("/tmp/kubeconfig"),
            image="cuda-vectoradd:test",
            deadline=100.0,
            sleep_fn=lambda _seconds: None,
            monotonic_fn=lambda: 0.0,
        )


def test_representative_mig_cuda_smoke_accepts_cdi_identity(monkeypatch) -> None:
    def run(command, **_kwargs):  # noqa: ANN001, ANN202
        if "logs" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                "Test PASSED\nNPA_MIG_VISIBLE=void\n"
                "MIG 1g.24gb Device 0: (UUID: MIG-0123456789abcdef)\n"
                "0MiB / 24192MiB\n",
                "",
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("npa.fleet.mig.subprocess.run", run)
    monkeypatch.setattr(
        "npa.fleet.mig._kubectl_json",
        lambda *_args, **_kwargs: {
            "spec": {"nodeName": "gpu-node-0"},
            "status": {"phase": "Succeeded", "containerStatuses": []},
        },
    )
    result = _run_mig_cuda_smoke(
        kubectl_bin="kubectl",
        kubeconfig=Path("/tmp/kubeconfig"),
        image="cuda-vectoradd:test",
        deadline=100.0,
        sleep_fn=lambda _seconds: None,
        monotonic_fn=lambda: 0.0,
    )
    assert result["mig_uuid"] == "MIG-0123456789abcdef"
    assert result["memory_mib"] == 24192


def test_representative_mig_cuda_smoke_rejects_ambiguous_cdi_view(
    monkeypatch,
) -> None:  # noqa: ANN001
    def run(command, **_kwargs):  # noqa: ANN001, ANN202
        if "logs" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                "Test PASSED\nNPA_MIG_VISIBLE=void\n"
                "MIG 1g.24gb Device 0: (UUID: MIG-0123456789abcdef)\n"
                "MIG 1g.24gb Device 1: (UUID: MIG-fedcba9876543210)\n"
                "0MiB / 24192MiB\n",
                "",
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("npa.fleet.mig.subprocess.run", run)
    monkeypatch.setattr(
        "npa.fleet.mig._kubectl_json",
        lambda *_args, **_kwargs: {
            "spec": {"nodeName": "gpu-node-0"},
            "status": {"phase": "Succeeded", "containerStatuses": []},
        },
    )
    with pytest.raises(MigVerificationError, match="exactly one hardware MIG"):
        _run_mig_cuda_smoke(
            kubectl_bin="kubectl",
            kubeconfig=Path("/tmp/kubeconfig"),
            image="cuda-vectoradd:test",
            deadline=100.0,
            sleep_fn=lambda _seconds: None,
            monotonic_fn=lambda: 0.0,
        )


def test_representative_mig_cuda_smoke_cleans_up_after_ambiguous_create_timeout(
    monkeypatch,
) -> None:  # noqa: ANN001
    commands: list[list[str]] = []

    def run(command, **kwargs):  # noqa: ANN001, ANN202
        commands.append(command)
        if "apply" in command:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return subprocess.CompletedProcess(command, 0, "deleted", "")

    monkeypatch.setattr("npa.fleet.mig.subprocess.run", run)
    with pytest.raises(MigVerificationError, match="could not create.*TimeoutExpired"):
        _run_mig_cuda_smoke(
            kubectl_bin="kubectl",
            kubeconfig=Path("/tmp/kubeconfig"),
            image="cuda-vectoradd:test",
            deadline=100.0,
            sleep_fn=lambda _seconds: None,
            monotonic_fn=lambda: 0.0,
        )
    assert "apply" in commands[0]
    assert "delete" in commands[1]
    assert "--wait=true" in commands[1]


def test_active_gpu_workloads_block_driver_replacement() -> None:
    payload = {
        "items": [
            {
                "metadata": {"namespace": "training", "name": "cuda"},
                "status": {"phase": "Running"},
                "spec": {
                    "containers": [
                        {"resources": {"limits": {"nvidia.com/mig-1g.24gb": "1"}}}
                    ]
                },
            },
            {
                "metadata": {"namespace": "gpu-operator", "name": "validator"},
                "status": {"phase": "Running"},
                "spec": {
                    "containers": [{"resources": {"limits": {"nvidia.com/gpu": "1"}}}]
                },
            },
        ]
    }
    assert _active_gpu_workloads(payload) == ("training/cuda",)


def test_wait_fails_actionably_when_driver_update_has_active_gpu_workload(
    monkeypatch,
) -> None:  # noqa: ANN001
    pending = _wait_report(
        ready=False,
        errors=(
            "DaemonSet nvidia-driver-daemonset: "
            "desired/current/updated/ready/available=2/2/0/2/2, "
            "expected 2/2/2/2/2",
        ),
    )
    monkeypatch.setattr(
        "npa.fleet.mig._ensure_device_plugin_mig_gate", lambda *_a, **_k: True
    )
    monkeypatch.setattr("npa.fleet.mig.verify_mig_cluster", lambda **_k: pending)
    monkeypatch.setattr(
        "npa.fleet.mig._reconcile_ondelete_driver",
        lambda *_a, **_k: (_ for _ in ()).throw(
            MigVerificationError(
                "NVIDIA driver OnDelete replacement is blocked by active GPU "
                "workload(s): training/cuda; delete those workloads explicitly"
            )
        ),
    )
    with pytest.raises(MigVerificationError, match="delete those workloads explicitly"):
        wait_for_mig_ready(
            kubectl_bin="kubectl",
            kubeconfig=Path("/tmp/kubeconfig"),
            expected_nodes=2,
        )


def test_enabled_mig_requires_strict_capacity_preset_and_safe_disk() -> None:
    base = NodePoolSpec(
        count=2,
        platform="gpu-rtx6000",
        preset="1gpu-24vcpu-218gb",
        capacity_block_group="capacityblockgroup-test",
        disk_size_gib=128,
    )
    ClusterSpec(name="valid", gpu_nodes=base, mig=MigSpec()).validate()

    with pytest.raises(FleetSpecError, match="capacity_block_group"):
        ClusterSpec(
            name="payg",
            gpu_nodes=NodePoolSpec(
                count=2,
                platform="gpu-rtx6000",
                preset="1gpu-24vcpu-218gb",
                disk_size_gib=128,
            ),
            mig=MigSpec(),
        ).validate()


def test_verify_mig_cli_json(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    spec_path = tmp_path / "fleet.yaml"
    spec_path.write_text(yaml.safe_dump(_mapping(mig={"enabled": True})))
    report = MigVerificationReport(
        ready=True,
        nodes=(),
        errors=(),
        operator_state="ready",
        operator_version="v26.3.3",
    )
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/kubectl")
    monkeypatch.setattr("npa.fleet.mig.verify_mig_cluster", lambda **_kwargs: report)
    result = CliRunner().invoke(
        app,
        [
            "fleet",
            "verify-mig",
            "--spec",
            str(spec_path),
            "--kubeconfig",
            str(tmp_path / "kubeconfig"),
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["ready"] is True


def test_terraform_forces_mig_reboot_value_to_string() -> None:
    helm = (
        Path(__file__).parents[3]
        / "deploy/cluster/vendor/nebius-solutions-library/modules/"
        "gpu-operator-custom/helm.tf"
    ).read_text()
    start = helm.index('name  = "migManager.env[0].value"')
    stanza = helm[start : helm.index("},", start)]
    assert 'type  = "string"' in stanza
    with pytest.raises(FleetSpecError, match="one-GPU preset"):
        ClusterSpec(
            name="wrong-preset",
            gpu_nodes=NodePoolSpec(
                count=2,
                platform="gpu-rtx6000",
                preset="2gpu-48vcpu-436gb",
                capacity_block_group="capacityblockgroup-test",
                disk_size_gib=128,
            ),
            mig=MigSpec(),
        ).validate()
    with pytest.raises(FleetSpecError, match="128 GiB"):
        ClusterSpec(
            name="oversized-disk",
            gpu_nodes=NodePoolSpec(
                count=2,
                platform="gpu-rtx6000",
                preset="1gpu-24vcpu-218gb",
                capacity_block_group="capacityblockgroup-test",
                disk_size_gib=1023,
            ),
            mig=MigSpec(),
        ).validate()
