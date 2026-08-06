"""Terraform working-dir and variable helpers for ``npa agent``.

Extracted from the ``npa.cli.agent`` monolith (kept under a size ratchet) so the
Terraform plumbing for destroy/reclaim lives in one small module. These helpers are
re-exported from ``npa.cli.agent`` for the existing call sites and tests.
"""

from __future__ import annotations

from typing import Any

from npa.clients.config import resolve_environment, resolve_terraform_state
from npa.deploy import provisioner


def _agent_terraform_state_exists(project: str, name: str) -> bool:
    tf_dir = provisioner.working_dir_path(project, name)
    return (tf_dir / ".terraform").is_dir()


def _record_agent_destroy_event(
    project: str,
    name: str,
    *,
    terminal_state: str,
    record_present: bool | None = None,
    terraform_state_present: bool | None = None,
    purge_iam: bool | None = None,
    error: str = "",
) -> None:
    """Persist agent destroy evidence outside the removable project record."""

    from npa.teardown_receipts import record_teardown_event

    environment = resolve_environment(project)
    precheck: dict[str, object] = {
        "identity_resolved": bool(getattr(environment, "project_id", ""))
    }
    if record_present is not None:
        precheck["local_record_present"] = record_present
    if terraform_state_present is not None:
        precheck["terraform_state_present"] = terraform_state_present
    action: dict[str, object] = {"kind": "terraform_agent_destroy"}
    if purge_iam is not None:
        action["purge_iam"] = purge_iam
    record_teardown_event(
        phase="agent",
        resource=name,
        terminal_state=terminal_state,
        project_alias=project,
        project_id=str(getattr(environment, "project_id", "") or ""),
        precheck=precheck,
        action=action,
        verification={
            "remote_destroy": {
                "in_progress": "pending",
                "failed": "failed",
                "verified_deleted": "completed",
            }.get(terminal_state, terminal_state)
        },
        errors=[error] if error else [],
    )


def _resolve_destroy_tf_vars(
    project: str,
    name: str,
    record: dict[str, Any] | None,
) -> dict[str, str]:
    # Imported lazily: npa.cli.agent imports this module.
    from npa.cli.agent import (
        DEFAULT_AGENT_IMAGE_FAMILY,
        DEFAULT_AGENT_PORT,
        _resolve_agent_service_account_id,
    )
    from npa.clients.nebius import get_iam_token

    state = resolve_terraform_state(project)
    saved_env = resolve_environment(project)
    region = str((record or {}).get("region", "") or (saved_env.region if saved_env else "") or "eu-north1")
    project_id = str((record or {}).get("project_id", "") or (saved_env.project_id if saved_env else ""))
    service_account_id = str((record or {}).get("service_account_id", "")).strip()
    if not service_account_id:
        creds = (record or {}).get("credentials", {})
        if isinstance(creds, dict):
            service_account_id = str(creds.get("service_account_id", "")).strip()
    if not service_account_id:
        service_account_id = _resolve_agent_service_account_id(project, record or {})

    iam_token = get_iam_token()
    return {
        "nebius_project_id": project_id,
        "nebius_region": region,
        "service_account_id": service_account_id,
        "iam_token": iam_token,
        "instance_name": f"agent-{project}-{name}",
        "server_port": str(DEFAULT_AGENT_PORT),
        "workbench_type": "agent",
        "gpu_platform": "cpu-d3",
        "gpu_preset": "8vcpu-32gb",
        "image_family": DEFAULT_AGENT_IMAGE_FAMILY,
        "enable_preemptible": "false",
        "nebius_api_key": state.access_key,
        "nebius_secret_key": state.secret_key,
        "s3_bucket": state.bucket,
        "s3_endpoint": state.endpoint,
        "extra_ingress_ports": "[]",
    }
