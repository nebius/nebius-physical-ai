"""npa network commands."""

from __future__ import annotations

import json
import typer

from npa.lifecycle_intent import OperationIntent, intent_boundary, json_stdout_contract

from npa.clients.network import (
    NetworkIngressError,
    ensure_ingress as ensure_ingress_impl,
    parse_ports,
)

app = typer.Typer(
    name="network",
    help="Network operations for Nebius resources.",
    no_args_is_help=True,
)


@app.command("delete-project-default")
@intent_boundary(OperationIntent.DESTROY)
@json_stdout_contract
def delete_project_default(
    project: str = typer.Option(..., "--project", help="Exact NPA project alias."),
    project_id: str = typer.Option(..., "--project-id", help="Exact project ID."),
    tenant_id: str = typer.Option(..., "--tenant-id", help="Exact tenant ID."),
    profile: str = typer.Option("", "--profile", help="Exact Nebius CLI profile."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Confirm exact deletion."),
    output_json: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Delete only the unique default topology of an NPA-created project."""

    from npa.clients.nebius import (
        NebiusError,
        delete_project_default_network,
        get_project_default_network_identity,
        get_project_identity,
    )
    from npa.project_destroy import _project_ownership_operation
    from npa.teardown_receipts import record_teardown_event

    try:
        if (
            get_project_identity(
                project_id, tenant_id=tenant_id, profile=profile or None
            )
            is None
        ):
            raise RuntimeError("exact project is absent")
        ownership = _project_ownership_operation(project, project_id, tenant_id)
        if ownership is None:
            raise RuntimeError(
                "default-network deletion requires unique durable NPA project-creation proof"
            )
        identity = get_project_default_network_identity(
            project_id, profile=profile or None
        )
        if identity is None:
            payload = {
                "outcome": "already_absent",
                "project_id": project_id,
                "verified": True,
            }
        else:
            payload = {
                "outcome": "planned",
                "project_id": project_id,
                "network_id": identity.network_id,
                "subnet_id": identity.subnet_id,
                "security_group_id": identity.security_group_id,
                "verified": True,
            }
            if yes:
                record_teardown_event(
                    phase="network",
                    resource=identity.network_id,
                    terminal_state="deletion_approved",
                    project_alias=project,
                    project_id=project_id,
                    identity={
                        "project_id": project_id,
                        "network_id": identity.network_id,
                        "subnet_id": identity.subnet_id,
                        "security_group_id": identity.security_group_id,
                        "ownership": "npa_disposable_project",
                        "project_operation_id": ownership.operation_id,
                    },
                    action={"kind": "delete_exact_project_default_network"},
                )
                delete_project_default_network(identity, profile=profile or None)
                if get_project_default_network_identity(
                    project_id, profile=profile or None
                ):
                    raise RuntimeError(
                        "default-network cleanup did not converge to absence"
                    )
                payload["outcome"] = "verified_deleted"
                record_teardown_event(
                    phase="network",
                    resource=identity.network_id,
                    terminal_state="verified_deleted",
                    project_alias=project,
                    project_id=project_id,
                    identity={
                        "project_id": project_id,
                        "network_id": identity.network_id,
                        "subnet_id": identity.subnet_id,
                        "security_group_id": identity.security_group_id,
                        "ownership": "npa_disposable_project",
                        "project_operation_id": ownership.operation_id,
                    },
                    verification={"exact_default_topology_absent": True},
                )
    except (NebiusError, RuntimeError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        json.dumps(payload, indent=2, sort_keys=True)
        if output_json
        else f"network: {payload['outcome']}"
    )


@app.command("ensure-ingress", help="Ensure TCP ingress to a VM security group.")
def ensure_ingress(
    vm: str | None = typer.Option(
        None,
        "--vm",
        help="Nebius compute instance ID.",
    ),
    ip: str | None = typer.Option(
        None,
        "--ip",
        help="Public IP address to resolve inside --project.",
    ),
    project: str | None = typer.Option(
        None,
        "--project",
        help="Nebius project ID used with --ip.",
    ),
    ports: str = typer.Option(
        ...,
        "--ports",
        help="Comma-separated TCP port list, for example 5151,8081,8082.",
    ),
    source: str = typer.Option(
        "0.0.0.0/0",
        "--source",
        help="Source CIDR allowed to reach the requested ports.",
    ),
    tool: str = typer.Option(
        "manual",
        "--tool",
        help="Tool name used in generated security rule names.",
    ),
) -> None:
    """Ensure the requested TCP ingress is covered by attached security group rules."""
    if vm and (ip or project):
        raise typer.BadParameter("pass exactly one of --vm or (--ip and --project)")
    if not vm and not (ip and project):
        raise typer.BadParameter("pass exactly one of --vm or (--ip and --project)")

    try:
        parsed_ports = parse_ports(ports)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--ports") from exc

    try:
        result = ensure_ingress_impl(
            vm_id=vm,
            ip=ip,
            project_id=project,
            ports=parsed_ports,
            source=source,
            tool=tool,
        )
    except NetworkIngressError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    for warning in result.warnings:
        typer.echo(f"Warning: {warning}", err=True)

    typer.echo(f"vm: {result.instance_id}")
    typer.echo(f"project: {result.project_id}")
    if result.public_ip:
        typer.echo(f"public_ip: {result.public_ip}")
    typer.echo(f"source: {result.source}")
    typer.echo("ports: " + ",".join(str(port) for port in result.ports))

    for group in result.security_groups:
        typer.echo(f"security_group: {group.security_group_id}")
        if group.security_group_name:
            typer.echo(f"security_group_name: {group.security_group_name}")
        if group.network_id:
            typer.echo(f"network: {group.network_id}")
        if group.changed:
            typer.echo(f"created_rule: {group.created_rule_id}")
            typer.echo(f"created_rule_name: {group.created_rule_name}")

    if result.changed:
        typer.echo("status: ingress rule changes applied")
    else:
        typer.echo("status: matching spec already covered, no rule changes")
