"""Restart-safe convergence for long-running agent remote setup."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
from typing import Any

import httpx

from npa.clients.config import ConfigError, resolve_ssh_config
from npa.clients.ssh import SSHClient, SSHError
from npa.provisioning_journal import operation_heartbeats


@dataclass(frozen=True)
class AgentSetupConvergence:
    evidence: dict[str, Any]
    primary_error: BaseException | None = None


def reconcile_agent_setup(
    *,
    host: str,
    ssh_user: str,
    ssh_key_path: str,
    project_alias: str,
    agent_name: str,
    project_id: str,
    auth_user: str,
    auth_password: str,
    agent_port: int,
    public_https: bool,
) -> dict[str, Any]:
    """Reconcile exact remote agent evidence after an uncertain transport result."""

    evidence: dict[str, Any] = {
        "state": "indeterminate",
        "endpoint": host,
        "project_id": project_id,
        "agent_name": agent_name,
        "service_fingerprint": "",
        "credential_fingerprint": "",
        "models_healthy": False,
    }
    try:
        ssh = SSHClient(
            config=resolve_ssh_config(
                ssh_host=host,
                ssh_user=ssh_user,
                ssh_key=ssh_key_path,
                project=None,
                name=None,
            ).ssh
        )
        code, stdout, _stderr = ssh.run(
            "sudo cat /opt/npa-agent/setup-state.json 2>/dev/null",
        )
        if code != 0 or not stdout.strip():
            evidence["state"] = "incomplete"
            evidence["first_incomplete_phase"] = "remote_service_deployment"
            return evidence
        payload = json.loads(stdout)
    except (ConfigError, SSHError, OSError, json.JSONDecodeError) as exc:
        evidence["error_category"] = type(exc).__name__
        return evidence
    expected = {
        "project_alias": project_alias,
        "agent_name": agent_name,
        "project_id": project_id,
        "endpoint": host,
    }
    mismatches = [
        key
        for key, value in expected.items()
        if str(payload.get(key) or "") != str(value or "")
    ]
    if mismatches:
        evidence["state"] = "identity_mismatch"
        evidence["mismatch_fields"] = mismatches
        return evidence
    evidence["service_fingerprint"] = str(payload.get("service_fingerprint") or "")
    evidence["credential_fingerprint"] = str(
        payload.get("credential_fingerprint") or ""
    )
    required_credentials = {"llm.env", "s3.env", "nebius.env"}
    credential_files = set(payload.get("credential_fingerprint_files") or [])
    if not evidence["service_fingerprint"] or not required_credentials.issubset(
        credential_files
    ):
        evidence["state"] = "incomplete"
        evidence["first_incomplete_phase"] = "credentials_staging"
        return evidence
    scheme = "https" if public_https else "http"
    port = "" if public_https else f":{agent_port}"
    try:
        response = httpx.get(
            f"{scheme}://{host}{port}/api/models",
            auth=(auth_user, auth_password),
            timeout=8.0,
            verify=not public_https,
        )
        evidence["models_healthy"] = response.status_code == 200
    except httpx.HTTPError as exc:
        evidence["error_category"] = type(exc).__name__
        return evidence
    if evidence["models_healthy"] and payload.get("phase") == "remote_health_ready":
        evidence["state"] = "healthy"
        evidence["remote_phase"] = "remote_health_ready"
    else:
        evidence["state"] = "incomplete"
        evidence["first_incomplete_phase"] = "health_verification"
    return evidence


def converge_remote_agent_setup(
    *,
    operation: Any,
    resuming: bool,
    bootstrap: Callable[..., None],
    reconcile: Callable[..., dict[str, Any]],
    bootstrap_kwargs: Mapping[str, Any],
    reconcile_kwargs: Mapping[str, Any],
    persist_pending: Callable[[str], None],
    status: Callable[[str], None],
    progress: Callable[[dict[str, Any]], None],
    transport_errors: tuple[type[BaseException], ...],
    fatal_errors: tuple[type[BaseException], ...] = (),
) -> AgentSetupConvergence:
    """Reconcile, resume if needed, then reconcile again without replacement."""

    evidence: dict[str, Any] | None = None
    primary_error: BaseException | None = None
    if resuming:
        status("reconciling the exact recorded bootstrap before remote mutation")
        if operation is not None:
            operation.checkpoint(
                "resume_reconciliation_started",
                {
                    "setup_phase": "resume_reconciliation_started",
                    "instance_id": str(bootstrap_kwargs.get("instance_id") or ""),
                    "endpoint": str(reconcile_kwargs.get("host") or ""),
                },
            )
        with operation_heartbeats(
            operation, phase="resume_reconciliation", emit=progress
        ):
            evidence = reconcile(**dict(reconcile_kwargs))
        if evidence.get("state") == "healthy":
            status("adopted the healthy exact agent; remote bootstrap is complete")

    if evidence is None or evidence.get("state") != "healthy":
        persist_pending("remote_bootstrap_pending")
        if operation is not None:
            operation.checkpoint(
                "remote_bootstrap_started",
                {
                    "setup_phase": "remote_bootstrap_started",
                    "instance_id": str(bootstrap_kwargs.get("instance_id") or ""),
                    "endpoint": str(reconcile_kwargs.get("host") or ""),
                },
            )
        remote_kwargs = dict(bootstrap_kwargs)
        remote_kwargs.pop("instance_id", None)
        try:
            with operation_heartbeats(
                operation, phase="remote_bootstrap", emit=progress
            ):
                bootstrap(**remote_kwargs)
        except fatal_errors:
            # Identity/ownership refusals happen before mutation and are not
            # uncertain transport outcomes. Never reconcile them into success
            # merely because an older deployment is still healthy.
            raise
        except transport_errors as exc:
            primary_error = exc
        evidence = reconcile(**dict(reconcile_kwargs))

    assert evidence is not None
    if operation is not None:
        operation.checkpoint(
            "remote_reconciliation",
            {
                **evidence,
                "primary_transport_error_type": (
                    type(primary_error).__name__ if primary_error else ""
                ),
            },
        )
    return AgentSetupConvergence(evidence=evidence, primary_error=primary_error)
