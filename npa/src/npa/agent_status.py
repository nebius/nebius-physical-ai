"""Durable partial-agent status when the final config record does not exist."""

from __future__ import annotations

from typing import Any

from npa.provisioning_journal import list_operations
from npa.teardown_receipts import TERMINAL_STATES, list_teardown_receipts


def _verify_created_resources(
    summary: dict[str, Any],
) -> tuple[str, list[dict[str, str]]]:
    """Read-only verify exact immutable IDs; uncertainty is never absence."""

    project_id = str(summary.get("project_id") or "")
    tenant_id = str(summary.get("tenant_id") or "")
    resources = [
        item
        for item in summary.get("resources") or []
        if isinstance(item, dict)
        and item.get("ownership") == "created_by_this_operation"
    ]
    evidence: list[dict[str, str]] = []
    if not project_id:
        return "identity_unavailable", evidence
    from npa.clients.nebius import (
        get_compute_instance_identity,
        get_service_account_identity,
    )

    for resource in resources:
        kind = str(resource.get("resource_type") or "")
        provider_id = str(resource.get("provider_id") or "")
        if not provider_id:
            evidence.append({"resource_type": kind, "state": "unverified_no_exact_id"})
            continue
        try:
            if kind == "compute_instance":
                present = (
                    get_compute_instance_identity(provider_id, project_id=project_id)
                    is not None
                )
            elif kind == "agent_service_account":
                present = (
                    get_service_account_identity(
                        provider_id, project_id=project_id, tenant_id=tenant_id
                    )
                    is not None
                )
            else:
                evidence.append(
                    {
                        "resource_type": kind,
                        "provider_id": provider_id,
                        "state": "verification_not_supported",
                    }
                )
                continue
        except Exception as exc:  # noqa: BLE001 - provider uncertainty is reported by type only
            evidence.append(
                {
                    "resource_type": kind,
                    "provider_id": provider_id,
                    "state": "verification_unavailable",
                    "error_type": type(exc).__name__,
                }
            )
        else:
            evidence.append(
                {
                    "resource_type": kind,
                    "provider_id": provider_id,
                    "state": "present" if present else "verified_absent",
                }
            )
    if (
        resources
        and len(evidence) == len(resources)
        and all(item.get("state") == "verified_absent" for item in evidence)
    ):
        return "provider_verified_absent", evidence
    if any(item.get("state") == "present" for item in evidence):
        return "provider_verified_present", evidence
    return "provider_verification_incomplete", evidence


def partial_agent_status(project: str, name: str) -> dict[str, Any]:
    """Classify operation/receipt evidence without probing or mutating cloud state."""

    operations = list_operations(
        project_alias=project,
        resource_type="agent",
        requested_name=name,
    )
    if operations:
        summary = operations[0].recovery_summary()
        phase = str(summary.get("phase") or "")
        lifecycle = str(summary.get("lifecycle") or "unknown")
        resources = list(summary.get("resources") or [])
        current_verification, resource_evidence = _verify_created_resources(summary)
        provider_present = [
            item for item in resource_evidence if item.get("state") == "present"
        ]
        if provider_present:
            classification = "CLEANUP_REQUIRED"
        elif phase in {"destroyed", "rolled-back"}:
            classification = "VERIFIED_ABSENT"
        elif phase == "rollback-incomplete":
            classification = "ROLLBACK_REQUIRED"
        elif phase == "prepared" and not resources:
            classification = "PREFLIGHT_BLOCKED"
        elif resources:
            classification = "PARTIAL"
        else:
            classification = "RESUMABLE"
        if current_verification == "provider_verified_absent":
            classification = "VERIFIED_ABSENT"
        effective_lifecycle = (
            "partial" if classification == "CLEANUP_REQUIRED" else lifecycle
        )
        residual_service_accounts = [
            item
            for item in provider_present
            if item.get("resource_type") == "agent_service_account"
            and item.get("provider_id")
        ]
        exact_cleanup_command = ""
        if residual_service_accounts and summary.get("operation_id"):
            exact_cleanup_command = (
                f"npa agent destroy --project {project} --name {name} "
                f"--operation-id {summary['operation_id']} --yes"
            )
        return {
            "project": project,
            "name": name,
            "classification": classification,
            "operation_id": summary.get("operation_id", ""),
            "operation_journal": summary.get("journal", ""),
            "phase": phase,
            "lifecycle": effective_lifecycle,
            "recorded_lifecycle": lifecycle,
            "heartbeat_at": summary.get("heartbeat_at", ""),
            "last_error_type": summary.get("last_error_type", ""),
            "last_error": summary.get("last_error", ""),
            "resources": resources,
            "resource_evidence": resource_evidence,
            "recovery": {
                "resume_argv": summary.get("resume_argv", []),
                "resume_command": summary.get("resume_command", ""),
                "destroy_argv": summary.get("destroy_argv", []),
                "destroy_command": summary.get("destroy_command", ""),
                "exact_cleanup_command": exact_cleanup_command,
            },
            "current_verification": current_verification,
            "components": {
                "vm": [
                    item
                    for item in resource_evidence
                    if item.get("resource_type") == "compute_instance"
                ],
                "disk_network": [
                    item
                    for item in resource_evidence
                    if item.get("resource_type")
                    not in {"compute_instance", "agent_service_account"}
                ],
                "service_account": [
                    item
                    for item in resource_evidence
                    if item.get("resource_type") == "agent_service_account"
                ],
            },
        }

    # A project receipt can contain several agents. Match the exact resource,
    # and require the current immutable project ID so alias reuse cannot turn a
    # different project's old teardown into an absence claim.
    try:
        from npa.clients.config import resolve_environment

        environment = resolve_environment(project)
        project_id = (
            str(environment.project_id or "") if environment is not None else ""
        )
    except Exception:  # noqa: BLE001 - no immutable identity means no receipt claim
        project_id = ""
    if project_id:
        for receipt in list_teardown_receipts(project_id=project_id, legacy="exclude"):
            for event in reversed(receipt.get("events") or []):
                if not isinstance(event, dict):
                    continue
                if str(event.get("phase") or "") != "agent":
                    continue
                if str(event.get("resource") or "") != name:
                    continue
                terminal = str(event.get("terminal_state") or "").lower()
                if terminal in TERMINAL_STATES:
                    return {
                        "project": project,
                        "project_id": project_id,
                        "name": name,
                        "classification": "VERIFIED_ABSENT",
                        "receipt_id": str(receipt.get("receipt_id") or ""),
                        "phase": terminal,
                        "lifecycle": "succeeded",
                        "resources": [],
                        "recovery": {},
                        "current_verification": "terminal_exact_agent_receipt",
                    }
    return {
        "project": project,
        "name": name,
        "classification": "NOT_FOUND",
        "phase": "",
        "lifecycle": "unknown",
        "resources": [],
        "recovery": {},
        "current_verification": "no_local_operation_or_receipt_evidence",
    }
