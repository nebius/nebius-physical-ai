"""Explicitly gated live coverage for PR #218 provisioning/agent teardown.

This module is excluded from hermetic suites.  It intentionally mutates the
selected project and must be enabled only by an operator who has reviewed the
exact project, context and agent selectors.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest


pytestmark = pytest.mark.e2e_pipeline


def _live_selectors() -> tuple[str, str, str]:
    if os.environ.get("NPA_PR218_LIVE_LIFECYCLE") != "1":
        pytest.skip("set NPA_PR218_LIVE_LIFECYCLE=1 to authorize live mutation")
    values = tuple(
        os.environ.get(name, "").strip()
        for name in (
            "NPA_E2E_PROJECT",
            "NPA_E2E_CLUSTER_CONTEXT",
            "NPA_E2E_AGENT_NAME",
        )
    )
    if not all(values):
        pytest.skip(
            "NPA_E2E_PROJECT, NPA_E2E_CLUSTER_CONTEXT and NPA_E2E_AGENT_NAME "
            "must select exact existing live identities"
        )
    return values[0], values[1], values[2]


def _npa(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "npa", *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_live_provision_binds_exact_controller_owner() -> None:
    project, context, _agent = _live_selectors()

    result = _npa(
        "provision-if-absent",
        "--project",
        project,
        "--cluster-name",
        context,
        "--context",
        context,
        "--accelerator",
        "RTXPRO6000:1",
        "--output-format",
        "json",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert any(item.startswith("controller:bound ") for item in payload["actions"])


def test_live_agent_destroy_then_deploy_is_reproducible() -> None:
    project, _context, agent = _live_selectors()

    destroyed = _npa(
        "agent", "destroy", "--project", project, "--name", agent, "--yes", "--json"
    )
    assert destroyed.returncode == 0, destroyed.stderr
    assert json.loads(destroyed.stdout)["verified"] is True

    deployed = _npa("agent", "deploy", "--project", project, "--name", agent)
    assert deployed.returncode == 0, deployed.stderr
