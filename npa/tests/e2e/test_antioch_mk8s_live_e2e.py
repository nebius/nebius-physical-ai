"""Sustained live qualification for the MK8s-native Antioch/OpenPI path.

This test intentionally retains the accepted Antioch simulator, adapter pod,
and policy Deployment. Cleanup is an explicit operator action after viewing.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from npa.sdk.workbench.antioch import live_k8s_deploy, live_k8s_status
from npa.workbench.antioch.cluster_deploy import qualify_live_metrics

pytestmark = pytest.mark.e2e_pipeline


def _enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


if not _enabled("NPA_INTEGRATION_E2E"):
    pytest.skip(
        "set NPA_INTEGRATION_E2E=1 for live infrastructure",
        allow_module_level=True,
    )

if os.environ.get("NPA_ANTIOCH_ACCEPT_TERMS") != "YES":
    pytest.skip("operator Antioch terms acceptance is absent", allow_module_level=True)
if os.environ.get("ACCEPT_EULA") != "Y":
    pytest.skip("operator NVIDIA runtime acceptance is absent", allow_module_level=True)
if os.environ.get("NPA_OPENPI_ACCEPT_GEMMA_TERMS") != "YES":
    pytest.skip("operator Gemma terms acceptance is absent", allow_module_level=True)

_RUNTIME_CONFIG_VALUE = os.environ.get("NPA_ANTIOCH_MK8S_RUNTIME_CONFIG", "").strip()
if not _RUNTIME_CONFIG_VALUE:
    pytest.skip(
        "set NPA_ANTIOCH_MK8S_RUNTIME_CONFIG to a mode-0600 private config",
        allow_module_level=True,
    )
RUNTIME_CONFIG = Path(_RUNTIME_CONFIG_VALUE)


def _accepted(metrics: dict[str, int | float]) -> bool:
    return bool(qualify_live_metrics(metrics)["accepted"])


def test_real_franka_camera_policy_loop_sustains_cluster_native_acceptance() -> None:
    deployed = live_k8s_deploy(runtime_config=RUNTIME_CONFIG)
    assert deployed["policy_service_type"] == "ClusterIP"
    assert deployed["dev_vm_in_data_path"] is False
    while True:
        status = live_k8s_status(runtime_config=RUNTIME_CONFIG)
        metrics = status.get("live_metrics") or {}
        if _accepted(metrics):
            assert status["status"] == "ready"
            assert status["daemon_liveness_ready"] is True
            assert status["relay_liveness_ready"] is True
            assert status["controller"]["scenario_run_id"]
            assert status["controller"]["heartbeat_age_seconds"] <= 30
            assert status["adapter_restarts"] == 0
            assert status["cluster_local_policy_resolved"] is True
            assert status["dev_vm_in_data_path"] is False
            return
        time.sleep(5)
