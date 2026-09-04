"""Live health gate for a fresh, reserved-capacity NPA mk8s GPU cluster.

The harness deliberately does not provision or select a cluster.  An operator
must provide the exact kubeconfig for a separately named, freshly provisioned
NPA mk8s cluster and attest that its GPU pool is bound to reserved capacity.
The test creates only short-lived CUDA vectorAdd pods and deletes them itself.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from npa.cli.cluster.terraform_lifecycle import _run_capture
from npa.cluster.gpu_driver import (
    DEFAULT_MANAGED_DRIVER_PRESET,
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
