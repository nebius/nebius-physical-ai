"""Read-only live verification helpers for the NPA agent."""

from __future__ import annotations

import os
import re
import subprocess
from typing import Any, NoReturn

import httpx
import typer

from npa.cli.agent_deployment import DeploymentIdentityError


def _fail(message: str) -> NoReturn:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=1)


def artifact_only_http_probe(client: httpx.Client) -> dict[str, Any]:
    """Exercise artifact-only live APIs using GETs and prove state is unchanged."""

    def get_json(path: str) -> dict[str, Any]:
        response = client.get(path)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise DeploymentIdentityError(f"{path} returned a non-object payload")
        return payload

    before = get_json("/api/health")
    before_digest = str(before.get("state_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", before_digest):
        raise DeploymentIdentityError("artifact-only health is missing state_sha256")
    session = get_json("/api/session")
    runs = get_json("/api/artifacts/runs?prefix=&limit=100")
    tools = get_json("/api/tools")
    workflow = get_json("/api/workflows/sim2real/status")
    infra = get_json("/api/infra/k8s")
    if not isinstance(runs.get("runs"), list):
        raise DeploymentIdentityError("artifact discovery did not return a runs list")
    if not isinstance(tools.get("tool_refs"), list):
        raise DeploymentIdentityError("tool catalog did not return tool_refs")
    after = get_json("/api/health")
    after_digest = str(after.get("state_sha256") or "")
    if after_digest != before_digest:
        raise DeploymentIdentityError(
            "artifact-only live verification mutated durable session state"
        )
    return {
        "state_sha256": before_digest,
        "run_count": len(runs["runs"]),
        "tool_ref_count": len(tools["tool_refs"]),
        "session": session,
        "workflow": workflow,
        "infra": infra,
    }


def verify_artifact_only_live(
    *,
    record: dict[str, Any],
    auth_user: str,
    auth_password: str,
    tls_verify: bool,
    project: str,
    name: str,
) -> None:
    """Run the no-stock live gate without writing chat/workflow/demo state."""
    agent_base = str(record.get("agent_url", "")).rstrip("/")
    try:
        with httpx.Client(
            base_url=agent_base,
            auth=(auth_user, auth_password),
            verify=tls_verify,
            timeout=30.0,
        ) as client:
            result = artifact_only_http_probe(client)
    except (httpx.HTTPError, DeploymentIdentityError) as exc:
        _fail(f"artifact-only read-only probe failed: {exc}")

    from npa.agent_rerun_bundle_check import (
        check_rerun_bundle_load_budget,
        format_bundle_budget_report,
    )

    bundle_result = check_rerun_bundle_load_budget(
        agent_base,
        auth=(auth_user, auth_password),
        verify=tls_verify,
    )
    typer.echo(format_bundle_budget_report(bundle_result))
    if not bundle_result.ok:
        _fail("rerun bundle load budget failed: " + "; ".join(bundle_result.errors[:4]))

    test_env = {
        **dict(os.environ),
        "NPA_INTEGRATION_E2E": "1",
        "NPA_AGENT_LIVE": "1",
        "NPA_AGENT_PROJECT": project,
        "NPA_AGENT_NAME": name,
        "NPA_AGENT_VERIFY_READ_ONLY": "1",
    }
    suites = (
        (
            "smoke",
            [
                "npa/tests/smoke/test_agent_smoke.py",
                "npa/tests/smoke/test_agent_chat_smoke.py",
            ],
        ),
        (
            "unit",
            ["npa/tests/cli/test_agent.py", "npa/tests/cli/test_agent_workflow.py"],
        ),
        (
            "read-only live e2e",
            [
                "npa/tests/e2e/test_agent_live.py",
                "-k",
                (
                    "agent_ui_html_smoke or agent_health_and_session or "
                    "agent_sim_assets_and_catalog or agent_tools_catalog or "
                    "agent_workbench_actions or agent_rerun_iframe_reachable"
                ),
            ],
        ),
    )
    for label, suite_args in suites:
        proc = subprocess.run(
            ["npa/.venv/bin/python", "-m", "pytest", *suite_args, "-q"],
            check=False,
            env=test_env,
        )
        if proc.returncode != 0:
            _fail(f"artifact-only {label} verification failed")

    try:
        with httpx.Client(
            base_url=agent_base,
            auth=(auth_user, auth_password),
            verify=tls_verify,
            timeout=30.0,
        ) as client:
            final = client.get("/api/health")
            final.raise_for_status()
            final_digest = str(final.json().get("state_sha256") or "")
    except (httpx.HTTPError, ValueError) as exc:
        _fail(f"artifact-only final state probe failed: {exc}")
    if final_digest != result["state_sha256"]:
        _fail("artifact-only verification changed durable state")
    typer.echo(
        "artifact-only read-only gate: "
        f"runs={result['run_count']} tool_refs={result['tool_ref_count']} "
        f"state_sha256={result['state_sha256']}"
    )
