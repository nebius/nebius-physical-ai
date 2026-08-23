"""Receipt-backed, identity-safe implementation of ``npa agent destroy``."""

from __future__ import annotations

import json
from typing import Any

import typer

from npa.cli._typer_defaults import resolve_typer_defaults
from npa.lifecycle_intent import OperationIntent, intent_boundary, json_stdout_contract


_AGENT_TERRAFORM_GRAPH = frozenset(
    {
        "compute_instance",
        "boot_disk",
        "network",
        "subnet",
        "security_group",
        "public_ip",
    }
)


def _emit(payload: dict[str, Any], *, output_json: bool) -> None:
    if output_json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(f"identity_source: {payload['identity_source']}")
        typer.echo(str(payload["message"]))


def _terminal_agent_graph_event(
    receipt_id: str, *, name: str, instance_id: str
) -> dict[str, Any]:
    """Return exact durable graph-absence evidence from an older destroy attempt."""

    if not receipt_id or not instance_id:
        return {}
    from npa.teardown_receipts import load_teardown_receipt

    candidates: list[dict[str, Any]] = []
    for event in load_teardown_receipt(receipt_id).get("events") or []:
        if not isinstance(event, dict):
            continue
        identity = event.get("identity")
        identity = identity if isinstance(identity, dict) else {}
        verification = event.get("verification")
        verification = verification if isinstance(verification, dict) else {}
        if (
            event.get("phase") == "agent"
            and str(event.get("resource") or "") == name
            and str(identity.get("instance_id") or "") == instance_id
            and str(event.get("terminal_state") or "").lower()
            in {"verified_absent", "verified_deleted"}
            and verification.get("exact_instance_absent") is True
            and verification.get("terraform_destroy_completed") is True
            and _AGENT_TERRAFORM_GRAPH.issubset(
                set(verification.get("terraform_dependency_graph") or [])
            )
        ):
            candidates.append(event)
    return max(
        candidates, key=lambda item: str(item.get("recorded_at") or ""), default={}
    )


@resolve_typer_defaults
@intent_boundary(OperationIntent.DESTROY)
@json_stdout_contract
def destroy_cmd(
    project: str = typer.Option("", "--project", help="NPA project alias."),
    name: str = typer.Option("agent", "--name", help="Agent deployment name."),
    receipt: str = typer.Option(
        "",
        "--receipt",
        help="Opaque teardown receipt ID from `npa cleanup --list-receipts`.",
    ),
    project_id: str = typer.Option("", "--project-id", help="Exact Nebius project ID."),
    tenant_id: str = typer.Option("", "--tenant-id", help="Exact Nebius tenant ID."),
    region: str = typer.Option("", "--region", help="Exact Nebius region."),
    instance_id: str = typer.Option("", "--instance-id", help="Exact immutable VM ID."),
    operation_id: str = typer.Option(
        "", "--operation-id", help="Exact provisioning operation journal ID."
    ),
    profile: str = typer.Option("", "--profile", help="Exact Nebius CLI profile."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
    purge_iam: bool = typer.Option(
        True,
        "--purge-iam/--keep-iam",
        help="Delete NPA-owned project agent IAM after the final agent is gone.",
    ),
    output_json: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Destroy agent VM/resources by exact identity; receipts need no project stanza."""

    from npa.cleanup_identity import CleanupIdentityError, resolve_cleanup_identity
    from npa.cli import agent as agent_module
    from npa.clients.config import resolve_environment
    from npa.clients.nebius import NebiusError, get_compute_instance_identity
    from npa.deploy.provisioner import ProvisionerError
    from npa.provisioning_journal import (
        ProvisioningOperation,
        TERMINAL_PHASES,
        list_operations,
        load_operation,
        operation_context,
    )

    selectors = any(
        (receipt, project_id, tenant_id, region, instance_id, operation_id, profile)
    )
    alias = project.strip()
    if not selectors:
        alias = agent_module._resolve_project_alias(alias)
    saved = resolve_environment(alias) if alias else None
    live = {
        "project_alias": alias,
        "project_id": str(getattr(saved, "project_id", "") or ""),
        "tenant_id": str(getattr(saved, "tenant_id", "") or ""),
        "region": str(getattr(saved, "region", "") or ""),
    }
    record = agent_module._agent_record(alias, name) if alias else {}
    if record:
        live.update(
            {
                "agent_name": name,
                "instance_id": str(record.get("instance_id") or ""),
                "project_id": str(record.get("project_id") or live["project_id"]),
                "region": str(record.get("region") or live["region"]),
                "service_account_id": str(record.get("service_account_id") or ""),
            }
        )
    try:
        identity = resolve_cleanup_identity(
            explicit={
                "project_alias": alias,
                "project_id": project_id,
                "tenant_id": tenant_id,
                "region": region,
                "agent_name": name,
                "instance_id": instance_id,
                "operation_id": operation_id,
                "profile": profile,
            },
            receipt_id=receipt,
            live=live,
            phase="agent",
            resource=name,
        )
    except (CleanupIdentityError, RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    alias = str(identity.get("project_alias") or alias)
    exact_project = str(identity.get("project_id") or "")
    exact_instance = str(identity.get("instance_id") or "")
    exact_operation = str(identity.get("operation_id") or "")
    operations = (
        [load_operation(exact_operation)]
        if exact_operation
        else list_operations(
            project_alias=alias,
            project_id=exact_project,
            resource_type="agent",
            requested_name=name,
        )
    )
    if not alias and operations:
        alias = str(operations[0].read().get("project_alias") or "")
    state_exists = bool(
        alias and agent_module._agent_terraform_state_exists(alias, name)
    )
    terminal_graph = _terminal_agent_graph_event(
        receipt, name=name, instance_id=exact_instance
    )
    newest_operation_at = max(
        (str(operation.read().get("updated_at") or "") for operation in operations),
        default="",
    )
    # A prior attempt may have destroyed and provider-verified the full exact
    # Terraform graph, then returned partial only because IAM inventory was
    # unresolved.  Retrying by receipt must reconcile IAM without reopening an
    # absent Terraform backend.  A newer same-name operation still wins so
    # alias/name reuse cannot make an old receipt affect a new deployment.
    if (
        not record
        and terminal_graph
        and (
            not newest_operation_at
            or str(terminal_graph.get("recorded_at") or "") > newest_operation_at
        )
    ):
        from npa.cli.agent_iam import AgentIAMCleanupError, report_destroyed_agent_iam
        from npa.cli.agent_terraform import AgentTerraformStateIdentityError
        from npa.cli.destructive import require_destructive_confirmation

        def fail_closed(message: str) -> None:
            agent_module._record_agent_destroy_event(
                alias,
                name,
                terminal_state="verification_failed",
                terraform_state_present=state_exists,
                error=message,
                identity=identity.values,
                project_id=exact_project,
                identity_source=identity.source,
            )
            _emit(
                {
                    **identity.to_dict(),
                    "outcome": "verification_failed",
                    "verified": False,
                    "infrastructure_absent": False,
                    "iam_cleanup_complete": False,
                    "message": message,
                },
                output_json=output_json,
            )
            raise typer.Exit(code=2)

        if state_exists:
            try:
                state_instance = agent_module._agent_terraform_instance_id(alias, name)
            except AgentTerraformStateIdentityError as exc:
                fail_closed(
                    "The terminal receipt cannot authorize IAM reconciliation because "
                    "surviving Terraform state has no single trustworthy immutable "
                    f"instance identity: {exc}. Credentials and local state were retained."
                )
            if state_instance != exact_instance:
                fail_closed(
                    "The terminal receipt names a different immutable instance than "
                    "the surviving Terraform state. The state may belong to a current "
                    "deployment; credentials and local state were retained."
                )
            if not exact_project:
                fail_closed(
                    "The terminal receipt and surviving Terraform state cannot be "
                    "provider-verified without an exact project identity. Credentials "
                    "and local state were retained."
                )
            try:
                remote = get_compute_instance_identity(
                    state_instance,
                    project_id=exact_project,
                    expected_name=(f"agent-{alias}-{name}" if alias else ""),
                    profile=str(identity.get("profile") or "") or None,
                )
            except NebiusError as exc:
                fail_closed(
                    "The terminal receipt matches surviving Terraform state, but exact "
                    f"provider absence verification is unresolved: {exc}. Credentials "
                    "and local state were retained."
                )
            if remote is not None:
                fail_closed(
                    "The exact instance in the terminal receipt and surviving Terraform "
                    "state is present at the provider. Credentials and local state were "
                    "retained."
                )

        require_destructive_confirmation(
            yes=yes,
            prompt=(
                f"Reconcile IAM for already-destroyed agent {alias}/{name} using "
                "exact terminal receipt evidence?"
            ),
            output_json=output_json,
            payload=identity.to_dict(),
        )
        try:
            iam_disposition = report_destroyed_agent_iam(
                alias, name, record=dict(identity.values), purge=purge_iam
            )
        except AgentIAMCleanupError as exc:
            agent_module._record_agent_destroy_event(
                alias,
                name,
                terminal_state="partial",
                error=str(exc),
                identity=identity.values,
                project_id=exact_project,
                identity_source=identity.source,
                terraform_graph_absent=True,
            )
            _emit(
                {
                    **identity.to_dict(),
                    "outcome": "partial_iam_cleanup",
                    "verified": False,
                    "infrastructure_absent": True,
                    "iam_cleanup_complete": False,
                    "message": str(exc),
                },
                output_json=output_json,
            )
            raise typer.Exit(code=2) from exc
        agent_module._record_agent_destroy_event(
            alias,
            name,
            terminal_state="verified_deleted",
            identity=identity.values,
            project_id=exact_project,
            identity_source=identity.source,
            terraform_graph_absent=True,
        )
        _emit(
            {
                **identity.to_dict(),
                "outcome": "verified_deleted",
                "verified": True,
                "infrastructure_absent": True,
                "iam_cleanup_complete": iam_disposition in {"absent", "deleted"},
                "shared_iam_preserved": iam_disposition == "retained_shared",
                "iam_disposition": iam_disposition,
                "no_op": iam_disposition in {"absent", "retained_shared"},
                "message": (
                    f"Agent {name!r} infrastructure was already absent; exact IAM "
                    "state was reconciled from terminal receipt evidence."
                ),
            },
            output_json=output_json,
        )
        return
    if not record and not state_exists and not operations:
        if identity.receipt_is_terminal:
            payload = {
                **identity.to_dict(),
                "outcome": "already_absent",
                "verified": True,
                "no_op": True,
                "message": f"Agent {name!r} is already absent per terminal receipt evidence.",
            }
            _emit(payload, output_json=output_json)
            return
        if not exact_project or not exact_instance:
            raise typer.BadParameter(
                "No agent state exists. Pass --receipt with immutable agent identity, "
                "or both --project-id and --instance-id; no provider or Terraform call ran."
            )
        try:
            remote = get_compute_instance_identity(
                exact_instance,
                project_id=exact_project,
                expected_name=(f"agent-{alias}-{name}" if alias else ""),
                profile=str(identity.get("profile") or "") or None,
            )
        except NebiusError as exc:
            raise typer.BadParameter(
                f"Exact provider verification is unresolved: {exc}. Nothing was deleted."
            ) from exc
        if remote is not None:
            raise typer.BadParameter(
                f"Exact instance {exact_instance} is present, but no complete Terraform "
                "ownership/state graph is available. NPA refused a VM-only deletion."
            )
        agent_module._record_agent_destroy_event(
            alias,
            name,
            terminal_state="verified_absent",
            identity=identity.values,
            project_id=exact_project,
            identity_source=identity.source,
        )
        payload = {
            **identity.to_dict(),
            "outcome": "already_absent",
            "verified": True,
            "no_op": True,
            "message": f"Provider verified exact instance {exact_instance} is absent; nothing to do.",
        }
        _emit(payload, output_json=output_json)
        return

    recovery_record = dict(record)
    for key in (
        "project_id",
        "tenant_id",
        "region",
        "instance_id",
        "service_account_id",
    ):
        if identity.get(key) and not recovery_record.get(key):
            recovery_record[key] = identity.get(key)
    from npa.cli.destructive import require_destructive_confirmation

    require_destructive_confirmation(
        yes=yes,
        prompt=f"Destroy agent {alias}/{name} (VM, network, and local config)?",
        output_json=output_json,
        payload=identity.to_dict(),
    )
    selected_operation = load_operation(exact_operation) if exact_operation else None
    if (
        selected_operation is not None
        and str(selected_operation.read().get("phase") or "") not in TERMINAL_PHASES
    ):
        # Supplying the exact nonterminal setup operation is the explicit safe
        # recovery path: teardown resumes under that operation's project lease.
        teardown_operation = selected_operation
    else:
        resume_argv = [
            "npa",
            "agent",
            "destroy",
            "--project",
            alias,
            "--name",
            name,
            "--yes",
        ]
        if exact_project:
            resume_argv.extend(["--project-id", exact_project])
        teardown_operation = ProvisioningOperation.prepare(
            command="npa agent destroy",
            project_alias=alias,
            project_id=exact_project,
            tenant_id=str(identity.get("tenant_id") or ""),
            region=str(identity.get("region") or ""),
            resource_type="agent-teardown",
            requested_name=name,
            ownership_source="agent-destroy-cli",
            resume_command="",
            resume_argv=resume_argv,
            destroy_argv=resume_argv,
        )
    agent_module._record_agent_destroy_event(
        alias,
        name,
        terminal_state="in_progress",
        record_present=bool(record),
        terraform_state_present=state_exists,
        purge_iam=purge_iam,
        identity=identity.values,
        project_id=exact_project,
        identity_source=identity.source,
    )
    with operation_context(teardown_operation):
        if str(teardown_operation.read().get("phase") or "") == "prepared":
            teardown_operation.transition("mutating")
        try:
            agent_module._destroy_agent_terraform(
                alias,
                name,
                record=recovery_record or None,
                operation_id=exact_operation,
                project_id=exact_project,
            )
        except ProvisionerError as exc:
            teardown_operation.transition(
                "recovery-required",
                error=str(exc),
                details={"error_type": type(exc).__name__},
            )
            agent_module._record_agent_destroy_event(
                alias,
                name,
                terminal_state="failed",
                error=str(exc),
                identity=identity.values,
                project_id=exact_project,
                identity_source=identity.source,
            )
            agent_module._fail(f"Terraform destroy failed: {exc}")
        phase = str(teardown_operation.read().get("phase") or "")
        if phase in {"mutating", "resource-created"}:
            teardown_operation.transition(
                "state-durable", details={"terraform_graph": "destroyed"}
            )
    terraform_noop_absence = False
    if not exact_instance:
        for operation in operations:
            for resource in operation.read().get("resources") or []:
                if (
                    isinstance(resource, dict)
                    and resource.get("resource_type") == "compute_instance"
                ):
                    exact_instance = str(resource.get("provider_id") or "").strip()
                    if exact_instance:
                        break
            if exact_instance:
                break
    if not exact_instance:
        # A successful Terraform destroy with no compute resource in any exact
        # operation is an authoritative no-op: deployment failed before VM-ID
        # persistence, so there is no VM identity to verify.  Do not invent a
        # missing identifier and turn valid infrastructure absence into failure.
        terraform_noop_absence = bool(operations) and not any(
            isinstance(resource, dict)
            and resource.get("resource_type") == "compute_instance"
            for operation in operations
            for resource in (operation.read().get("resources") or [])
        )
        if not terraform_noop_absence:
            message = (
                "Terraform accepted the destroy request, but no immutable instance ID "
                "or exact no-resource operation graph was available. Local state and "
                "IAM were preserved for recovery."
            )
            agent_module._record_agent_destroy_event(
                alias,
                name,
                terminal_state="verification_failed",
                error=message,
                identity=identity.values,
                project_id=exact_project,
                identity_source=identity.source,
            )
            agent_module._fail(message)
    else:
        try:
            remote = get_compute_instance_identity(
                exact_instance,
                project_id=exact_project,
                expected_name=(f"agent-{alias}-{name}" if alias else ""),
                profile=str(identity.get("profile") or "") or None,
            )
        except NebiusError as exc:
            message = (
                f"Terraform accepted the destroy request, but provider verification for "
                f"exact instance {exact_instance} is unresolved: {exc}. Local state and "
                "IAM were preserved for recovery."
            )
            agent_module._record_agent_destroy_event(
                alias,
                name,
                terminal_state="verification_failed",
                error=message,
                identity=identity.values,
                project_id=exact_project,
                identity_source=identity.source,
            )
            agent_module._fail(message)
        if remote is not None:
            message = (
                f"Terraform accepted the destroy request, but exact instance "
                f"{exact_instance} is still present. Local state and IAM were preserved."
            )
            agent_module._record_agent_destroy_event(
                alias,
                name,
                terminal_state="verification_failed",
                error=message,
                identity=identity.values,
                project_id=exact_project,
                identity_source=identity.source,
            )
            agent_module._fail(message)
    agent_module._record_agent_destroy_event(
        alias,
        name,
        terminal_state="verified_deleted",
        identity=identity.values,
        project_id=exact_project,
        identity_source=identity.source,
        terraform_graph_absent=True,
    )
    if record:
        agent_module._remove_agent_record(alias, name)
    agent_module._cleanup_agent_local_files(alias, name)
    iam_error = ""
    iam_disposition = "not_applicable"
    if alias:
        from npa.cli.agent_iam import AgentIAMCleanupError, report_destroyed_agent_iam

        try:
            iam_disposition = report_destroyed_agent_iam(
                alias, name, record=recovery_record or None, purge=purge_iam
            )
        except AgentIAMCleanupError as exc:
            iam_error = str(exc)
            agent_module._record_agent_destroy_event(
                alias,
                name,
                terminal_state="partial",
                error=iam_error,
                identity=identity.values,
                project_id=exact_project,
                identity_source=identity.source,
                terraform_graph_absent=True,
            )
    # The exact agent's infrastructure has already been provider-verified absent.
    # Shared project IAM may intentionally remain while other agents depend on it;
    # that degraded cleanup result must not leave the project lifecycle lease open
    # and prevent a fresh deployment of this agent name.
    if str(teardown_operation.read().get("phase") or "") not in TERMINAL_PHASES:
        teardown_operation.transition("destroyed")
    if iam_error:
        _emit(
            {
                **identity.to_dict(),
                "outcome": "partial_iam_cleanup",
                "verified": False,
                "infrastructure_absent": True,
                "iam_cleanup_complete": False,
                "message": iam_error,
            },
            output_json=output_json,
        )
        raise typer.Exit(code=2)
    _emit(
        {
            **identity.to_dict(),
            "outcome": "verified_deleted",
            "verified": True,
            "infrastructure_absent": True,
            "iam_cleanup_complete": iam_disposition in {"absent", "deleted"},
            "shared_iam_preserved": iam_disposition == "retained_shared",
            "iam_disposition": iam_disposition,
            "no_op": False,
            "message": f"destroyed: {alias}/{name}",
        },
        output_json=output_json,
    )
