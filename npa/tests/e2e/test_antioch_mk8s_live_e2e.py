"""Live qualification and saved-record verification for Antioch/OpenPI.

The deployment test retains accepted resources for operator inspection. The
saved-record test only reads existing evidence and never dispatches a scenario.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from npa.sdk.workbench.antioch import live_k8s_deploy, live_k8s_status
from npa.workbench.antioch.cluster_deploy import load_private_config
from npa.workbench.antioch.cluster_runtime import _completed_poc_evidence
from npa.workbench.antioch.runtime import ensure_runtime
from npa.workbench.antioch.vendor_cli import AntiochCli

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


def test_real_franka_pickup_requires_persisted_task_acceptance() -> None:
    deployed = live_k8s_deploy(runtime_config=RUNTIME_CONFIG)
    assert deployed["policy_service_type"] == "ClusterIP"
    assert deployed["dev_vm_in_data_path"] is False
    while True:
        status = live_k8s_status(runtime_config=RUNTIME_CONFIG)
        controller = status.get("controller") or {}
        assert controller.get("status") not in {"failed", "cleanup_failed"}
        if controller.get("pickup_verified") is True:
            assert controller["run_phase"] == "completed"
            assert controller["run_outcome"] == "passed"
            assert controller["scenario_run_id"]
            assert status["adapter_restarts"] == 0
            assert status["dev_vm_in_data_path"] is False
            return
        time.sleep(5)


def test_completed_poc_checks_persist_after_session_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read the operator-selected saved proof without dispatching another run."""
    run_id = os.environ.get("NPA_ANTIOCH_COMPLETED_SCENARIO_ID", "").strip()
    if not run_id:
        pytest.skip("set NPA_ANTIOCH_COMPLETED_SCENARIO_ID for saved-run verification")
    config = load_private_config(RUNTIME_CONFIG)
    monkeypatch.setenv("ANTIOCH_ENV", config.antioch_deployment_profile)
    cli = AntiochCli(ensure_runtime(), config_dir=config.antioch_config_dir)
    record = cli.show(tmp_path, kind="scenario", remote_id=run_id)
    assert record["project_id"] == Path(config.antioch_project_id_file).read_text().strip()
    evidence = _completed_poc_evidence(
        record, scenario="openpi_franka_mk8s_live_v2", scenario_run_id=run_id
    )
    assert evidence["communication_verified"] is True
    results = record["results"]
    for view in ("exterior", "wrist"):
        assert results[f"{view}_luminance_mean_min"] > 5
        assert results[f"{view}_luminance_variance_min"] > 25
    assert results["latency_max_ms"] < 90_000
    assert record["artifacts"]["telemetry"]["size_bytes"] > 0
    reread = cli.show(tmp_path, kind="scenario", remote_id=run_id)
    assert reread["phase"] == "completed"
    assert reread["outcome"] == "passed"
    assert reread["results"]["checks"] == results["checks"]


def test_completed_pickup_checks_and_evidence_persist(tmp_path, monkeypatch) -> None:
    """Read an exact saved pickup without starting compute or dispatching a run."""
    run_id = os.environ.get("NPA_ANTIOCH_COMPLETED_PICKUP_SCENARIO_ID", "").strip()
    if not run_id:
        pytest.skip("set NPA_ANTIOCH_COMPLETED_PICKUP_SCENARIO_ID for pickup verification")
    config = load_private_config(RUNTIME_CONFIG)
    monkeypatch.setenv("ANTIOCH_ENV", config.antioch_deployment_profile)
    cli = AntiochCli(ensure_runtime(), config_dir=config.antioch_config_dir)
    record = cli.show(tmp_path, kind="scenario", remote_id=run_id)
    assert record["project_id"] == Path(config.antioch_project_id_file).read_text().strip()
    evidence = _completed_poc_evidence(
        record, scenario="openpi_franka_pickup_v3", scenario_run_id=run_id)
    assert evidence["pickup_verified"] is True
    reread = cli.show(tmp_path, kind="scenario", remote_id=run_id)
    assert reread["results"]["checks"] == record["results"]["checks"]
    assert reread["artifacts"]["policy-evidence.zip"] == record["artifacts"]["policy-evidence.zip"]
