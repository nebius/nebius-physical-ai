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
        action = event.get("action")
        action = action if isinstance(action, dict) else {}
        graph = verification.get("terraform_dependency_graph")
        errors = event.get("errors")
        if (
            event.get("phase") == "agent"
            and str(event.get("resource") or "") == name
            and str(identity.get("instance_id") or "") == instance_id
            and str(event.get("terminal_state") or "").lower()
            in {"verified_absent", "verified_deleted"}
            and verification.get("exact_instance_absent") is True
            and verification.get("terraform_destroy_completed") is True
            and action.get("kind") == "terraform_agent_destroy"
            and isinstance(graph, list)
            and _AGENT_TERRAFORM_GRAPH.issubset(graph)
            and isinstance(errors, list)
            and not errors
        ):
            candidates.append(event)
    return max(candidates, key=lambda item: int(item.get("sequence") or 0), default={})


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

    def read_saved_record(selected_alias: str) -> tuple[bool, dict[str, Any]]:
        if not selected_alias:
            return False, {}
        from npa.cli.agent_records import AgentRecordError, AgentRecordState

        try:
            decoded = agent_module.decode_agent_record(selected_alias, name)
        except AgentRecordError as exc:
            raise typer.BadParameter(str(exc)) from exc
        if decoded.present and decoded.state is not AgentRecordState.COMPLETE:
            raise typer.BadParameter(
                f"Saved agent record {name!r} is present but {decoded.state.value}: "
                f"{decoded.detail}. "
                "refusing receipt or IAM reconciliation until the lifecycle record "
                "is recovered or explicitly repaired."
            )
        return decoded.present, dict(decoded.record)

    def live_identity(
        selected_alias: str, saved_record: dict[str, Any]
    ) -> dict[str, str]:
        saved_environment = (
            resolve_environment(selected_alias) if selected_alias else None
        )
        values = {
            "project_alias": selected_alias,
            "project_id": str(
                getattr(saved_environment, "project_id", "") or ""
            ),
            "tenant_id": str(getattr(saved_environment, "tenant_id", "") or ""),
            "region": str(getattr(saved_environment, "region", "") or ""),
        }
        if saved_record:
            values.update(
                {
                    "agent_name": name,
                    "instance_id": str(saved_record.get("instance_id") or ""),
                    "project_id": str(
                        saved_record.get("project_id") or values["project_id"]
                    ),
                    "region": str(saved_record.get("region") or values["region"]),
                    "service_account_id": str(
                        saved_record.get("service_account_id") or ""
                    ),
                }
            )
        return values

    explicit_identity = {
        "project_alias": alias,
        "project_id": project_id,
        "tenant_id": tenant_id,
        "region": region,
        "agent_name": name,
        "instance_id": instance_id,
        "operation_id": operation_id,
        "profile": profile,
    }
    record_present, record = read_saved_record(alias)
    live = live_identity(alias, record)
    try:
        identity = resolve_cleanup_identity(
            explicit=explicit_identity,
            receipt_id=receipt,
            live=live,
            phase="agent",
            resource=name,
        )
    except (CleanupIdentityError, RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    resolved_alias = str(identity.get("project_alias") or alias)
    if resolved_alias and resolved_alias != alias:
        record_present, record = read_saved_record(resolved_alias)
        live = live_identity(resolved_alias, record)
        try:
            identity = resolve_cleanup_identity(
                explicit=explicit_identity,
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
    if exact_operation:
        operation_payload = operations[0].read()
        if (
            str(operation_payload.get("resource_type") or "") != "agent"
            or str(operation_payload.get("requested_name") or "") != name
        ):
            raise typer.BadParameter(
                "The selected operation journal does not own this exact agent name; "
                "no mutation was attempted."
            )
        operation_compute_ids = {
            str(resource.get("provider_id") or "").strip()
            for resource in operation_payload.get("resources") or []
            if isinstance(resource, dict)
            and resource.get("resource_type") == "compute_instance"
            and str(resource.get("provider_id") or "").strip()
        }
        if len(operation_compute_ids) > 1:
            raise typer.BadParameter(
                "The selected operation journal has ambiguous compute identities; "
                "no mutation was attempted."
            )
        operation_identity = {
            "project_alias": str(
                operation_payload.get("project_alias") or ""
            ).strip(),
            "project_id": str(operation_payload.get("project_id") or "").strip(),
            "tenant_id": str(operation_payload.get("tenant_id") or "").strip(),
            "region": str(operation_payload.get("region") or "").strip(),
            "instance_id": next(iter(operation_compute_ids), ""),
        }
        for field, journal_value in operation_identity.items():
            resolved_value = str(identity.get(field) or "").strip()
            if resolved_value and journal_value and resolved_value != journal_value:
                raise typer.BadParameter(
                    f"The selected operation journal conflicts with {field}; "
                    "no mutation was attempted."
                )
            if not resolved_value and journal_value:
                identity.values[field] = journal_value
                identity.field_sources[field] = f"operation:{exact_operation}"
        alias = str(identity.get("project_alias") or alias)
        exact_project = str(identity.get("project_id") or "")
        exact_instance = str(identity.get("instance_id") or "")
    if not alias and operations:
        operation_aliases = {
            str(operation.read().get("project_alias") or "").strip()
            for operation in operations
            if str(operation.read().get("project_alias") or "").strip()
        }
        if len(operation_aliases) != 1:
            raise typer.BadParameter(
                "Operation journals do not identify one exact project alias. "
                "Pass an exact receipt/alias; no mutation was attempted."
            )
        alias = next(iter(operation_aliases))
    if not exact_project:
        raise typer.BadParameter(
            "An exact immutable project ID is required before agent teardown; "
            "no mutation was attempted."
        )
    if exact_operation:
        related_operations = list_operations(
            project_alias=alias,
            project_id=exact_project,
            resource_type="agent",
            requested_name=name,
        )
        if exact_operation not in {
            operation.operation_id for operation in related_operations
        }:
            raise typer.BadParameter(
                "The selected operation journal is outside the exact agent/project "
                "inventory; no mutation was attempted."
            )
        operations = related_operations
    state_exists = bool(
        alias and agent_module._agent_terraform_state_exists(alias, name)
    )
    if state_exists:
        from npa.cli.agent_terraform import AgentTerraformStateIdentityError

        try:
            state_instance = agent_module._agent_terraform_instance_id(alias, name)
        except AgentTerraformStateIdentityError as exc:
            agent_module._record_agent_destroy_event(
                alias,
                name,
                terminal_state="verification_failed",
                error=str(exc),
                identity=identity.values,
                project_id=exact_project,
                identity_source=identity.source,
            )
            raise typer.BadParameter(
                "Saved agent state has no single trustworthy immutable instance "
                f"identity: {exc}. No mutation was attempted."
            ) from exc
        if exact_instance and state_instance != exact_instance:
            agent_module._record_agent_destroy_event(
                alias,
                name,
                terminal_state="verification_failed",
                error="saved record/receipt conflicts with Terraform instance identity",
                identity=identity.values,
                project_id=exact_project,
                identity_source=identity.source,
            )
            raise typer.BadParameter(
                "Saved agent record/receipt identity conflicts with surviving "
                "Terraform state. No mutation was attempted."
            )
        if not exact_instance:
            exact_instance = state_instance
            identity.values["instance_id"] = state_instance
    terminal_graph = _terminal_agent_graph_event(
        receipt, name=name, instance_id=exact_instance
    )

    operation_ids_snapshot = tuple(
        sorted(operation.operation_id for operation in operations)
    )

    def operation_matches_terminal_generation(operation: Any) -> bool:
        payload = operation.read()
        if (
            payload.get("audit_only") is True
            and str(payload.get("phase") or "")
            in {"committed", "destroyed", "rolled-back"}
            and not operation.state_copies()
        ):
            return True
        compute_ids = {
            str(resource.get("provider_id") or "").strip()
            for resource in payload.get("resources") or []
            if isinstance(resource, dict)
            and resource.get("resource_type") == "compute_instance"
            and str(resource.get("provider_id") or "").strip()
        }
        if compute_ids:
            return bool(exact_instance and compute_ids == {exact_instance})
        return bool(
            str(payload.get("phase") or "") in {"destroyed", "rolled-back"}
            and not operation.state_copies()
        )

    operations_match_terminal = all(
        operation_matches_terminal_generation(operation) for operation in operations
    )
    if terminal_graph and not record_present and operations and not operations_match_terminal:
        raise typer.BadParameter(
            "A current or ambiguous same-name operation generation conflicts with "
            "the historical terminal receipt. No mutation was attempted."
        )
    if exact_operation and any(
        operation.operation_id != exact_operation
        and not operation_matches_terminal_generation(operation)
        for operation in operations
    ):
        raise typer.BadParameter(
            "A sibling same-name operation generation conflicts with the selected "
            "operation. No mutation was attempted."
        )

    def prepare_teardown_operation() -> Any:
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
        return ProvisioningOperation.prepare(
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

    def retire_local_state_under_lease(
        transaction: Any, *, finalize: bool = False
    ) -> None:
        """Revalidate one generation under its project lease, then retire it."""

        try:
            with operation_context(transaction):
                phase = str(transaction.read().get("phase") or "")
                if phase == "prepared":
                    transaction.transition("mutating")
                if alias:
                    from npa.cli.agent_records import AgentRecordState

                    current_record = agent_module.decode_agent_record(alias, name)
                    if current_record.present != record_present:
                        raise agent_module.AgentLocalRetirementError(
                            "saved agent record presence changed during teardown"
                        )
                    if current_record.present and (
                        current_record.state is not AgentRecordState.COMPLETE
                        or current_record.record.get("project_id") != exact_project
                        or current_record.record.get("instance_id") != exact_instance
                    ):
                        raise agent_module.AgentLocalRetirementError(
                            "saved agent generation changed during teardown"
                        )
                current_operations = list_operations(
                    project_alias=alias,
                    project_id=exact_project,
                    resource_type="agent",
                    requested_name=name,
                )
                current_ids = tuple(
                    sorted(operation.operation_id for operation in current_operations)
                )
                if current_ids != operation_ids_snapshot:
                    raise agent_module.AgentLocalRetirementError(
                        "agent operation generation changed during teardown"
                    )
                current_state = bool(
                    alias
                    and agent_module._agent_terraform_state_exists(alias, name)
                )
                if current_state:
                    from npa.cli.agent_terraform import (
                        AgentTerraformStateIdentityError,
                    )

                    try:
                        current_state_instance = (
                            agent_module._agent_terraform_instance_id(alias, name)
                        )
                    except AgentTerraformStateIdentityError as exc:
                        raise agent_module.AgentLocalRetirementError(str(exc)) from exc
                    if not exact_instance or current_state_instance != exact_instance:
                        raise agent_module.AgentLocalRetirementError(
                            "Terraform state generation changed during teardown"
                        )
                if exact_instance:
                    try:
                        remote = get_compute_instance_identity(
                            exact_instance,
                            project_id=exact_project,
                            expected_name=(f"agent-{alias}-{name}" if alias else ""),
                            profile=str(identity.get("profile") or "") or None,
                        )
                    except NebiusError as exc:
                        raise agent_module.AgentLocalRetirementError(
                            "exact provider absence recheck is unresolved: " + str(exc)
                        ) from exc
                    if remote is not None:
                        raise agent_module.AgentLocalRetirementError(
                            "exact provider instance reappeared during teardown"
                        )
                removed_record: dict[str, Any] = {}
                if record_present:
                    removed_record = dict(current_record.record)
                    agent_module._remove_agent_record(alias, name)
                    if agent_module.decode_agent_record(alias, name).present:
                        raise agent_module.AgentLocalRetirementError(
                            "saved agent record remains after removal"
                        )
                try:
                    agent_module._cleanup_agent_local_files(
                        alias,
                        name,
                        operation_ids=operation_ids_snapshot,
                    )
                except (OSError, RuntimeError, ValueError) as exc:
                    if removed_record:
                        try:
                            agent_module._store_agent_record(
                                alias, name, removed_record
                            )
                            restored = agent_module.decode_agent_record(alias, name)
                            if (
                                not restored.present
                                or restored.record.get("project_id") != exact_project
                                or restored.record.get("instance_id") != exact_instance
                            ):
                                raise agent_module.AgentLocalRetirementError(
                                    "saved agent recovery record did not restore"
                                )
                        except (OSError, RuntimeError, ValueError) as restore_exc:
                            raise agent_module.AgentLocalRetirementError(
                                "local retirement failed and the saved agent recovery "
                                f"record could not be restored: {restore_exc}"
                            ) from exc
                    raise
                if finalize and str(transaction.read().get("phase") or "") not in {
                    "committed",
                    "destroyed",
                    "rolled-back",
                }:
                    transaction.transition("destroyed")
        except (OSError, RuntimeError, ValueError):
            phase = str(transaction.read().get("phase") or "")
            if phase not in {"committed", "destroyed", "rolled-back"}:
                transaction.transition(
                    "recovery-required",
                    error="exact local agent retirement did not converge",
                )
            raise
    # A prior attempt may have destroyed and provider-verified the full exact
    # Terraform graph, then returned partial only because IAM inventory was
    # unresolved.  Retrying by receipt must reconcile IAM without reopening an
    # absent Terraform backend.  A newer same-name operation still wins so
    # alias/name reuse cannot make an old receipt affect a new deployment.
    if (
        not record
        and terminal_graph
        and operations_match_terminal
    ):
        from npa.cli.agent_iam import AgentIAMCleanupError, report_destroyed_agent_iam
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
        retirement_operation = prepare_teardown_operation()
        with operation_context(retirement_operation):
            if str(retirement_operation.read().get("phase") or "") == "prepared":
                retirement_operation.transition("mutating")
            try:
                iam_disposition = report_destroyed_agent_iam(
                    alias, name, record=dict(identity.values), purge=purge_iam
                )
            except AgentIAMCleanupError as exc:
                retirement_operation.transition(
                    "recovery-required", error="agent IAM cleanup did not converge"
                )
                agent_module._record_agent_destroy_event(
                    alias,
                    name,
                    terminal_state="partial",
                    error=str(exc),
                    identity=identity.values,
                    project_id=exact_project,
                    identity_source=identity.source,
                    terraform_graph_absent=True,
                    iam_cleanup_complete=False,
                    iam_disposition="verification_unresolved",
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
            try:
                agent_module._record_agent_destroy_event(
                    alias,
                    name,
                    terminal_state="verified_deleted",
                    identity=identity.values,
                    project_id=exact_project,
                    identity_source=identity.source,
                    terraform_graph_absent=True,
                    purge_iam=purge_iam,
                    iam_cleanup_complete=iam_disposition in {"absent", "deleted"},
                    iam_disposition=iam_disposition,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                retirement_operation.transition(
                    "recovery-required",
                    error="terminal agent teardown receipt was not durable",
                )
                _emit(
                    {
                        **identity.to_dict(),
                        "outcome": "partial_receipt_cleanup",
                        "verified": False,
                        "infrastructure_absent": True,
                        "iam_cleanup_complete": iam_disposition
                        in {"absent", "deleted"},
                        "message": (
                            "Agent cloud and IAM convergence was verified, but the "
                            "terminal teardown receipt was not durable. Local "
                            "credentials and recovery state were retained."
                        ),
                    },
                    output_json=output_json,
                )
                raise typer.Exit(code=2) from exc
            try:
                retire_local_state_under_lease(
                    retirement_operation, finalize=True
                )
            except (OSError, RuntimeError, ValueError) as exc:
                agent_module._record_agent_destroy_event(
                    alias,
                    name,
                    terminal_state="partial",
                    error=str(exc),
                    identity=identity.values,
                    project_id=exact_project,
                    identity_source=identity.source,
                    terraform_graph_absent=True,
                    purge_iam=purge_iam,
                    iam_cleanup_complete=iam_disposition in {"absent", "deleted"},
                    iam_disposition=iam_disposition,
                )
                fail_closed(
                    "Exact cloud and IAM absence is proven, but local agent "
                    f"retirement is incomplete: {exc}. Alias and project "
                    "credentials were retained."
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
        absence_receipt_recorded = False
        if alias:
            from npa.cli.destructive import require_destructive_confirmation

            require_destructive_confirmation(
                yes=yes,
                prompt=f"Retire local state for absent agent {alias}/{name}?",
                output_json=output_json,
                payload=identity.to_dict(),
            )
            retirement_operation = prepare_teardown_operation()
            with operation_context(retirement_operation):
                if (
                    str(retirement_operation.read().get("phase") or "")
                    == "prepared"
                ):
                    retirement_operation.transition("mutating")
                try:
                    agent_module._record_agent_destroy_event(
                        alias,
                        name,
                        terminal_state="verified_absent",
                        identity=identity.values,
                        project_id=exact_project,
                        identity_source=identity.source,
                    )
                    absence_receipt_recorded = True
                except (OSError, RuntimeError, ValueError) as exc:
                    retirement_operation.transition(
                        "recovery-required",
                        error="terminal agent absence receipt was not durable",
                    )
                    raise typer.BadParameter(
                        "Provider verified the exact instance absent, but the "
                        "terminal receipt was not durable. Local credentials and "
                        "recovery state were retained."
                    ) from exc
                try:
                    retire_local_state_under_lease(
                        retirement_operation, finalize=True
                    )
                except (OSError, RuntimeError, ValueError) as exc:
                    agent_module._record_agent_destroy_event(
                        alias,
                        name,
                        terminal_state="partial",
                        error=str(exc),
                        identity=identity.values,
                        project_id=exact_project,
                        identity_source=identity.source,
                    )
                    raise typer.BadParameter(
                        "Provider verified the exact instance absent, but local agent "
                        f"retirement is incomplete: {exc}"
                    ) from exc
        if not absence_receipt_recorded:
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

    # A present current record must stand on its own. Historical receipts may
    # select a generation only when the key is absent; they never complete an
    # incomplete current schema.
    recovery_record = dict(record) if record_present else dict(identity.values)
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
        teardown_operation = prepare_teardown_operation()
    agent_module._record_agent_destroy_event(
        alias,
        name,
        terminal_state="in_progress",
        record_present=record_present,
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
    iam_disposition = "not_applicable"
    with operation_context(teardown_operation):
        if alias:
            from npa.cli.agent_iam import (
                AgentIAMCleanupError,
                report_destroyed_agent_iam,
            )

            try:
                iam_disposition = report_destroyed_agent_iam(
                    alias, name, record=recovery_record or None, purge=purge_iam
                )
            except AgentIAMCleanupError as exc:
                iam_error = str(exc)
                if (
                    str(teardown_operation.read().get("phase") or "")
                    not in TERMINAL_PHASES
                ):
                    teardown_operation.transition(
                        "recovery-required",
                        error="agent IAM cleanup did not converge",
                    )
                agent_module._record_agent_destroy_event(
                    alias,
                    name,
                    terminal_state="partial",
                    error=iam_error,
                    identity=identity.values,
                    project_id=exact_project,
                    identity_source=identity.source,
                    terraform_graph_absent=True,
                    purge_iam=purge_iam,
                    iam_cleanup_complete=False,
                    iam_disposition="verification_unresolved",
                )
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
                raise typer.Exit(code=2) from exc

        try:
            agent_module._record_agent_destroy_event(
                alias,
                name,
                terminal_state="verified_deleted",
                identity=identity.values,
                project_id=exact_project,
                identity_source=identity.source,
                terraform_graph_absent=True,
                purge_iam=purge_iam,
                iam_cleanup_complete=iam_disposition in {"absent", "deleted"},
                iam_disposition=iam_disposition,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            if (
                str(teardown_operation.read().get("phase") or "")
                not in TERMINAL_PHASES
            ):
                teardown_operation.transition(
                    "recovery-required",
                    error="terminal agent teardown receipt was not durable",
                )
            _emit(
                {
                    **identity.to_dict(),
                    "outcome": "partial_receipt_cleanup",
                    "verified": False,
                    "infrastructure_absent": True,
                    "iam_cleanup_complete": iam_disposition
                    in {"absent", "deleted"},
                    "message": (
                        "Agent cloud and IAM convergence was verified, but the "
                        "terminal teardown receipt was not durable. Local "
                        "credentials and recovery state were retained."
                    ),
                },
                output_json=output_json,
            )
            raise typer.Exit(code=2) from exc

        try:
            retire_local_state_under_lease(teardown_operation, finalize=True)
        except (OSError, RuntimeError, ValueError) as exc:
            message = (
                "Provider and IAM convergence was verified, but exact local state "
                f"retirement failed: {exc}. Credentials and recovery evidence "
                "were retained."
            )
            agent_module._record_agent_destroy_event(
                alias,
                name,
                terminal_state="partial",
                error=message,
                identity=identity.values,
                project_id=exact_project,
                identity_source=identity.source,
                terraform_graph_absent=True,
                purge_iam=purge_iam,
                iam_cleanup_complete=iam_disposition in {"absent", "deleted"},
                iam_disposition=iam_disposition,
            )
            _emit(
                {
                    **identity.to_dict(),
                    "outcome": "partial_local_cleanup",
                    "verified": False,
                    "infrastructure_absent": True,
                    "iam_cleanup_complete": iam_disposition
                    in {"absent", "deleted"},
                    "message": message,
                },
                output_json=output_json,
            )
            raise typer.Exit(code=2) from exc
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
