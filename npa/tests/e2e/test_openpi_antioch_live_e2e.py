"""Opt-in live OpenPI container to Antioch simulator feedback validation."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from npa.workflows.byof.openpi_antioch import LiveLoopConfig, run_live_loop


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
    evidence = run_live_loop(
        LiveLoopConfig(
            project_dir=Path(_required_env("NPA_OPENPI_ANTIOCH_PROJECT_DIR")),
            cache_dir=Path(_required_env("NPA_OPENPI_ANTIOCH_CACHE_DIR")),
            image=_required_env("NPA_OPENPI_ANTIOCH_IMAGE"),
            policy_host=_required_env("NPA_OPENPI_ANTIOCH_POLICY_HOST"),
            host_port=host_port,
            policy_port=int(policy_port_text) if policy_port_text else None,
            scenario=os.environ.get(
                "NPA_OPENPI_ANTIOCH_SCENARIO", "pi05_droid_loop"
            ),
            chunks=int(os.environ.get("NPA_OPENPI_ANTIOCH_CHUNKS", "3")),
            docker_bin=os.environ.get("NPA_OPENPI_ANTIOCH_DOCKER", "docker"),
            antioch_bin=os.environ.get("NPA_OPENPI_ANTIOCH_BIN", "antioch"),
            rerun_from=os.environ.get("NPA_OPENPI_ANTIOCH_RERUN_FROM") or None,
            machine=os.environ.get("NPA_OPENPI_ANTIOCH_MACHINE") or None,
            script=os.environ.get("NPA_OPENPI_ANTIOCH_SCRIPT") or None,
            container_name=os.environ.get(
                "NPA_OPENPI_ANTIOCH_CONTAINER", "npa-openpi-antioch-pi05"
            ),
        )
    )

    assert evidence["status"] == "passed"
    assert evidence["containerized_policy"] is True
    assert evidence["action_chunk_shape"] == [15, 8]
    assert int(evidence["chunks_run"]) >= 1
