"""Live health gates for explicitly selected NPA mk8s GPU clusters.

The harness deliberately does not provision or select a cluster.  An operator
must provide the exact kubeconfig and the requested GPU topology. The original
gate requires a fresh reserved cluster; mixed pools require their own explicit
authorization and a separately declared GPU count for every node.
The test creates only short-lived CUDA vectorAdd pods and deletes them itself.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from npa.cli.cluster.terraform_lifecycle import _run_capture
from npa.cluster.gpu_driver import (
    DEFAULT_MANAGED_DRIVER_PRESET,
    is_nvswitch_topology,
    resolve_gpu_driver_strategy,
)
from npa.cluster.gpu_health import GpuHealthConfig, validate_gpu_health
from npa.cluster.gpu_workload_profile import resolve_gpu_workload_profile


pytestmark = pytest.mark.e2e


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        pytest.skip(f"set {name} to an exact operator-reviewed live selector")
    return value


def test_fresh_reserved_mk8s_gpu_cluster_passes_fail_closed_health_gate(
    tmp_path: Path,
) -> None:
    if os.environ.get("NPA_E2E_MK8S_GPU_HEALTH") != "1":
        pytest.skip("set NPA_E2E_MK8S_GPU_HEALTH=1 to authorize live CUDA pods")
    if os.environ.get("NPA_E2E_MK8S_FRESH_CLUSTER") != "1":
        pytest.skip("live GPU health requires an explicitly attested fresh cluster")
    if os.environ.get("NPA_E2E_MK8S_RESERVED_CAPACITY") != "1":
        pytest.skip("live GPU health requires explicitly attested reserved capacity")

    kubeconfig = Path(_required("NPA_E2E_MK8S_GPU_KUBECONFIG")).expanduser()
    assert kubeconfig.is_file(), f"missing exact kubeconfig: {kubeconfig}"
    gpu_nodes = int(_required("NPA_E2E_MK8S_GPU_NODES"))
    cpu_nodes = int(os.environ.get("NPA_E2E_MK8S_CPU_NODES", "0"))
    platform = _required("NPA_E2E_MK8S_GPU_PLATFORM")
    preset = _required("NPA_E2E_MK8S_GPU_PRESET")
    requested_mode = os.environ.get("NPA_E2E_MK8S_GPU_DRIVER_MODE", "auto")
    selection = resolve_gpu_driver_strategy(
        gpu_nodes=gpu_nodes,
        platform=platform,
        preset=preset,
        mode=requested_mode,
        managed_driver_preset=os.environ.get(
            "NPA_E2E_MK8S_MANAGED_DRIVER_PRESET", DEFAULT_MANAGED_DRIVER_PRESET
        ),
        enable_gpu_cluster=os.environ.get("NPA_E2E_MK8S_NVSWITCH") == "1",
    )

    report = validate_gpu_health(
        _run_capture,
        kubectl_bin=os.environ.get("NPA_KUBECTL_BIN", "kubectl"),
        kubeconfig_path=kubeconfig,
        config=GpuHealthConfig(
            expected_nodes=cpu_nodes + gpu_nodes,
            expected_gpu_nodes=gpu_nodes,
            gpu_preset=preset,
            gpu_platform=platform,
            driver_mode=selection.effective_mode,
            nvswitch=selection.nvswitch,
            stabilization_seconds=int(
                os.environ.get("NPA_E2E_MK8S_STABILIZATION_SECONDS", "120")
            ),
            timeout_seconds=int(
                os.environ.get("NPA_E2E_MK8S_HEALTH_TIMEOUT_SECONDS", "3600")
            ),
            cuda_smoke=True,
        ),
        evidence_path=tmp_path / "gpu-health-live.json",
    )

    assert report["status"] == "healthy"
    assert len(report["cuda_smokes"]) == gpu_nodes


def test_rtx_rendering_profile_passes_live_graphics_readiness_gate(
    tmp_path: Path,
) -> None:
    if os.environ.get("NPA_E2E_MK8S_RTX_RENDERING") != "1":
        pytest.skip("set NPA_E2E_MK8S_RTX_RENDERING=1 to authorize live graphics pods")
    kubeconfig = Path(_required("NPA_E2E_MK8S_GPU_KUBECONFIG")).expanduser()
    assert kubeconfig.is_file(), f"missing exact kubeconfig: {kubeconfig}"
    selection = resolve_gpu_workload_profile(
        profile="rtx-rendering",
        gpu_nodes=int(_required("NPA_E2E_MK8S_GPU_NODES")),
        gpu_platform=_required("NPA_E2E_MK8S_GPU_PLATFORM"),
        gpu_preset=_required("NPA_E2E_MK8S_GPU_PRESET"),
        gpu_driver_mode=_required("NPA_E2E_MK8S_GPU_DRIVER_MODE"),
    )

    report = validate_gpu_health(
        _run_capture,
        kubectl_bin=os.environ.get("NPA_KUBECTL_BIN", "kubectl"),
        kubeconfig_path=kubeconfig,
        config=GpuHealthConfig(
            expected_nodes=int(os.environ.get("NPA_E2E_MK8S_CPU_NODES", "0"))
            + selection.gpu_nodes,
            expected_gpu_nodes=selection.gpu_nodes,
            gpu_preset=selection.gpu_preset,
            gpu_platform=selection.gpu_platform,
            driver_mode=selection.gpu_driver_mode,
            stabilization_seconds=int(
                os.environ.get("NPA_E2E_MK8S_STABILIZATION_SECONDS", "120")
            ),
            timeout_seconds=int(
                os.environ.get("NPA_E2E_MK8S_HEALTH_TIMEOUT_SECONDS", "3600")
            ),
            cuda_smoke=True,
            graphics_smoke=True,
        ),
        evidence_path=tmp_path / "rtx-rendering-health-live.json",
    )

    assert report["status"] == "healthy"
    assert len(report["cuda_smokes"]) == selection.gpu_nodes
    assert len(report["graphics_smokes"]) == selection.gpu_nodes
    assert all(
        item["vulkan_physical_devices"] >= 1 for item in report["graphics_smokes"]
    )


def test_fresh_mixed_gpu_pools_pass_all_device_health_gate(tmp_path: Path) -> None:
    """Validate an explicit pool plan, including every GPU in larger workers."""
    if os.environ.get("NPA_E2E_MK8S_MIXED_GPU_HEALTH") != "1":
        pytest.skip("set NPA_E2E_MK8S_MIXED_GPU_HEALTH=1 to authorize live CUDA pods")
    if os.environ.get("NPA_E2E_MK8S_FRESH_CLUSTER") != "1":
        pytest.skip("live GPU health requires an explicitly attested fresh cluster")

    counts = tuple(int(value) for value in _required("NPA_E2E_MK8S_GPU_COUNTS").split(","))
    gpu_nodes = int(_required("NPA_E2E_MK8S_GPU_NODES"))
    total_gpus = int(_required("NPA_E2E_MK8S_TOTAL_GPUS"))
    assert len(counts) == gpu_nodes and sum(counts) == total_gpus
    kubeconfig = Path(_required("NPA_E2E_MK8S_GPU_KUBECONFIG")).expanduser()
    assert kubeconfig.is_file(), f"missing exact kubeconfig: {kubeconfig}"
    platform = _required("NPA_E2E_MK8S_GPU_PLATFORM")
    preset = _required("NPA_E2E_MK8S_GPU_PRESET")
    selection = resolve_gpu_driver_strategy(
        gpu_nodes=gpu_nodes,
        platform=platform,
        preset=preset,
        mode=_required("NPA_E2E_MK8S_GPU_DRIVER_MODE"),
        managed_driver_preset=os.environ.get(
            "NPA_E2E_MK8S_MANAGED_DRIVER_PRESET", DEFAULT_MANAGED_DRIVER_PRESET
        ),
        enable_gpu_cluster=(
            os.environ.get("NPA_E2E_MK8S_NVSWITCH") == "1"
            or is_nvswitch_topology(platform=platform, preset=f"{max(counts)}gpu-declared")
        ),
    )
    report = validate_gpu_health(
        _run_capture,
        kubectl_bin=os.environ.get("NPA_KUBECTL_BIN", "kubectl"),
        kubeconfig_path=kubeconfig,
        config=GpuHealthConfig(
            expected_nodes=gpu_nodes + int(_required("NPA_E2E_MK8S_CPU_NODES")),
            expected_gpu_nodes=gpu_nodes,
            expected_gpu_counts=counts,
            gpu_preset=preset,
            gpu_platform=platform,
            driver_mode=selection.effective_mode,
            nvswitch=selection.nvswitch,
            cuda_smoke=True,
        ),
        evidence_path=tmp_path / "mixed-gpu-health-live.json",
    )
    assert report["status"] == "healthy"
    assert report["final_snapshot"]["total_gpus"] == total_gpus
    assert len(report["cuda_smokes"]) == gpu_nodes
    assert sum(item["tested_gpus"] for item in report["cuda_smokes"]) == total_gpus
