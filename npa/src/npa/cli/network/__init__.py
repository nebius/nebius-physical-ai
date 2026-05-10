"""npa network commands."""

from __future__ import annotations

import typer

from npa.clients.network import NetworkIngressError, ensure_ingress as ensure_ingress_impl, parse_ports

app = typer.Typer(
    name="network",
    help="Network operations for Nebius resources.",
    no_args_is_help=True,
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
