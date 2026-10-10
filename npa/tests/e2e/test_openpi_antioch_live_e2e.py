"""Opt-in live OpenPI container to Antioch simulator feedback validation."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re

import pytest

from npa.workflows.byof.openpi_antioch import (
    REQUIRED_CHECKS,
    LiveLoopConfig,
    run_live_loop,
)
from npa.workflows.byof.openpi_pipeline import SOURCE_REF


pytestmark = pytest.mark.skipif(
    os.environ.get("NPA_INTEGRATION_E2E") != "1"
    or os.environ.get("NPA_OPENPI_ANTIOCH_LIVE") != "1",
    reason=(
        "Set NPA_INTEGRATION_E2E=1 and NPA_OPENPI_ANTIOCH_LIVE=1 with the "
        "operator-local Antioch/OpenPI inputs to run the connected live gate."
    ),
)


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        pytest.fail(f"live OpenPI/Antioch validation requires {name}")
    return value


def test_openpi_container_closes_real_antioch_feedback_loop() -> None:
    host_port = int(os.environ.get("NPA_OPENPI_ANTIOCH_HOST_PORT", "8000"))
    policy_port_text = os.environ.get("NPA_OPENPI_ANTIOCH_POLICY_PORT", "").strip()
    receipt = Path(_required_env("NPA_OPENPI_ANTIOCH_PRIVATE_RECEIPT"))
    if receipt.exists():
        pytest.fail("live OpenPI/Antioch receipt path must be unique and absent")
    if os.environ.get("NPA_OPENPI_ANTIOCH_SCRIPT", "").strip():
        pytest.fail("release live evidence requires the recorded scenario path")
    evidence = run_live_loop(
        LiveLoopConfig(
            project_dir=Path(_required_env("NPA_OPENPI_ANTIOCH_PROJECT_DIR")),
            cache_dir=Path(_required_env("NPA_OPENPI_ANTIOCH_CACHE_DIR")),
            image=_required_env("NPA_OPENPI_ANTIOCH_IMAGE"),
            policy_host=_required_env("NPA_OPENPI_ANTIOCH_POLICY_HOST"),
            host_port=host_port,
            policy_port=int(policy_port_text) if policy_port_text else None,
            scenario=os.environ.get("NPA_OPENPI_ANTIOCH_SCENARIO", "pi05_droid_loop"),
            chunks=int(os.environ.get("NPA_OPENPI_ANTIOCH_CHUNKS", "3")),
            docker_bin=os.environ.get("NPA_OPENPI_ANTIOCH_DOCKER", "docker"),
            antioch_bin=os.environ.get("NPA_OPENPI_ANTIOCH_BIN", "antioch"),
            rerun_from=os.environ.get("NPA_OPENPI_ANTIOCH_RERUN_FROM") or None,
            machine=os.environ.get("NPA_OPENPI_ANTIOCH_MACHINE") or None,
            container_name=_required_env("NPA_OPENPI_ANTIOCH_CONTAINER"),
            cleanup_scenario=True,
            cleanup_container=True,
            resource_owner=_required_env("NPA_OPENPI_ANTIOCH_RESOURCE_OWNER"),
            private_receipt_path=receipt,
        )
    )

    assert evidence["status"] == "passed"
    assert evidence["containerized_policy"] is True
    assert evidence["action_chunk_shape"] == [15, 8]
    assert int(evidence["chunks_run"]) >= 1
    assert set(evidence["passing_checks"]) == REQUIRED_CHECKS
    for metric in (
        "mean_inference_ms",
        "max_joint_travel_rad",
        "jaw_travel_mm",
    ):
        assert math.isfinite(float(evidence[metric]))
        assert float(evidence[metric]) > 0

    private_receipt = json.loads(receipt.read_text(encoding="utf-8"))
    assert private_receipt["status"] == "passed"
    assert re.fullmatch(r"[0-9a-f]{40}", private_receipt["harness_source_sha"])
    assert private_receipt["openpi_source_ref"] == SOURCE_REF
    assert private_receipt["image_id"] == evidence["image_id"]
    assert isinstance(private_receipt["scenario_run_id"], str)
    assert private_receipt["scenario_run_id"]
    assert private_receipt["container"]["name"] == _required_env(
        "NPA_OPENPI_ANTIOCH_CONTAINER"
    )
    assert private_receipt["container"]["owner"] == _required_env(
        "NPA_OPENPI_ANTIOCH_RESOURCE_OWNER"
    )
    assert private_receipt["acceptance"] == {
        "action_chunk_shape": [15, 8],
        "chunks_run": evidence["chunks_run"],
        "jaw_travel_mm": evidence["jaw_travel_mm"],
        "max_joint_travel_rad": evidence["max_joint_travel_rad"],
        "mean_inference_ms": evidence["mean_inference_ms"],
        "passing_checks": evidence["passing_checks"],
    }
    assert private_receipt["scenario_cleanup"]["status"] in {
        "already_terminal",
        "terminal_verified",
    }
    assert private_receipt["container_cleanup"] == {
        "absent_verified": True,
        "status": "absent_verified",
    }
