"""Explicitly gated live coverage for PR #218 provisioning/agent teardown.

This module is excluded from hermetic suites.  It intentionally mutates the
selected project and must be enabled only by an operator who has reviewed the
exact project, context and agent selectors.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

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
    from npa.cli.invocation import internal_cli_argv

    return subprocess.run(
        internal_cli_argv(args),
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


def test_live_agent_destroy_then_deploy_is_reproducible(tmp_path: Path) -> None:
    project, _context, agent = _live_selectors()

    # A live lifecycle test must be self-contained on a genuinely fresh
    # disposable project.  Establish the baseline resource that the first
    # destroy proves instead of silently depending on an operator's old agent.
    from npa.cli.agent import _agent_record

    if not _agent_record(project, agent).get("instance_id"):
        baseline = _npa("agent", "deploy", "--project", project, "--name", agent)
        assert baseline.returncode == 0, baseline.stderr
        baseline_record = _agent_record(project, agent)
        assert baseline_record.get("setup_state") == "healthy"
        assert baseline_record.get("instance_id")

    destroyed = _npa(
        "agent", "destroy", "--project", project, "--name", agent, "--yes", "--json"
    )
    assert destroyed.returncode == 0, destroyed.stderr
    assert json.loads(destroyed.stdout)["verified"] is True

    from npa.cli.invocation import internal_cli_argv

    stdout_path = tmp_path / "interrupted-agent-deploy.stdout"
    stderr_path = tmp_path / "interrupted-agent-deploy.stderr"
    did_interrupt = False
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        interrupted = subprocess.Popen(
            internal_cli_argv(("agent", "deploy", "--project", project, "--name", agent)),
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        partial: dict = {}
        while interrupted.poll() is None:
            partial = _agent_record(project, agent)
            if partial.get("instance_id") and partial.get("setup_state") in {
                "remote_bootstrap_pending",
                "reconciliation_indeterminate",
            }:
                interrupted.kill()
                did_interrupt = True
                break
            time.sleep(1)
        interrupted.wait()

    assert did_interrupt and partial.get("instance_id"), (
        "agent deploy exited before an interruptible durable instance was recorded; "
        f"returncode={interrupted.returncode}"
    )
    interrupted_instance = str(partial["instance_id"])

    deployed = _npa("agent", "deploy", "--project", project, "--name", agent)
    assert deployed.returncode == 0, deployed.stderr
    final = _agent_record(project, agent)
    assert final.get("setup_state") == "healthy"
    assert final.get("instance_id") == interrupted_instance

    status = _npa("agent", "status", "--project", project, "--name", agent, "--json")
    assert status.returncode == 0, status.stderr
    assert json.loads(status.stdout)["health"] is True

    from npa.provisioning_journal import list_operations

    operations = [
        operation.read()
        for operation in list_operations(
            project_alias=project,
            project_id=str(final.get("project_id") or ""),
            resource_type="agent",
        )
        if operation.read().get("requested_name") == agent
    ]
    assert any(int(item.get("resume_count") or 0) >= 1 for item in operations)
