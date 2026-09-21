"""Expose development and operator utilities through npa tools."""

import json
import subprocess
import typer

from npa.lifecycle_intent import OperationIntent, intent_boundary, json_stdout_contract
from npa.tools.desktop import open_desktop, operate

app = typer.Typer(
    help="Development and operator tools, independent of Workbench.",
    no_args_is_help=True,
)
desktop = typer.Typer(
    help="Persistent Ubuntu desktop with VS Code and Codex on an existing VM.",
    no_args_is_help=True,
)
app.add_typer(desktop, name="desktop")


def _emit(host: str, action: str, output_json: bool, **options) -> None:
    try:
        result = operate(host, action, **options)
    except (ValueError, RuntimeError) as exc:
        typer.echo(json.dumps({"error": str(exc)}) if output_json else str(exc))
        raise typer.Exit(1) from exc
    typer.echo(
        json.dumps(result, indent=2)
        if output_json
        else "\n".join(f"{key}: {value}" for key, value in result.items())
    )


@desktop.command("setup")
@intent_boundary(OperationIntent.ENSURE_PRESENT)
@json_stdout_contract
def setup(
    ssh_host: str = typer.Option(
        ..., "--ssh-host", help="Existing SSH alias or user@hostname."
    ),
    repository_path: str = typer.Option(
        "~/nebius-physical-ai", "--repository-path", help="Existing checkout on the VM."
    ),
    dpi: int = typer.Option(192, "--dpi", min=96, max=240, help="Retina: 192."),
    geometry: str = typer.Option(
        "2880x1800", "--geometry", help="Initial framebuffer WIDTHxHEIGHT."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the plan without connecting."
    ),
    output_json: bool = typer.Option(False, "--json", help="Emit one JSON document."),
) -> None:
    """Install a private desktop, preserving existing credentials and sessions.

    Args:
        ssh_host: Selected SSH destination.
        repository_path: Existing remote checkout.
        dpi: Desktop font density.
        geometry: Initial framebuffer size.
        dry_run: Plan without mutations.
        output_json: Emit JSON instead of text.
    Returns:
        None.
    Raises:
        typer.Exit: Setup failed.
    """
    _emit(
        ssh_host,
        "setup",
        output_json,
        repository_path=repository_path,
        dpi=dpi,
        geometry=geometry,
        dry_run=dry_run,
    )


@desktop.command("display")
@intent_boundary(OperationIntent.MUTATE)
@json_stdout_contract
def display(
    ssh_host: str = typer.Option(..., "--ssh-host"),
    dpi: int = typer.Option(192, "--dpi", min=96, max=240),
    output_json: bool = typer.Option(False, "--json"),
) -> None:
    """Change desktop density without restarting applications.

    Args:
        ssh_host: Selected SSH destination.
        dpi: Desired font density.
        output_json: Emit JSON instead of text.
    Returns:
        None.
    Raises:
        typer.Exit: Display update failed.
    """
    _emit(ssh_host, "display", output_json, dpi=dpi)


@desktop.command("status")
@json_stdout_contract
def status(
    ssh_host: str | None = typer.Option(None, "--ssh-host"),
    local: bool = typer.Option(False, "--local", help="Inspect chat running on this Mac."),
    output_json: bool = typer.Option(False, "--json"),
) -> None:
    """Inspect desktop services and recovery records without disclosing secrets.

    Args:
        ssh_host: Selected SSH destination.
        local: Inspect the Mac's installed chat services.
        output_json: Emit JSON instead of text.
    Returns:
        None.
    Raises:
        typer.Exit: Inspection failed.
    """
    if local:
        _require_target(ssh_host, local)
        from npa.tools.desktop.local_runtime import local_status
        result = local_status()
        typer.echo(json.dumps(result, indent=2))
        return
    _require_target(ssh_host, local)
    _emit(ssh_host, "status", output_json)


@desktop.command("optimize")
@intent_boundary(OperationIntent.MUTATE)
@json_stdout_contract
def optimize(
    ssh_host: str = typer.Option(..., "--ssh-host"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    output_json: bool = typer.Option(False, "--json"),
) -> None:
    """Reduce desktop input latency without changing resolution or restarting apps.

    Args:
        ssh_host: Existing desktop SSH destination.
        dry_run: Show the action without connecting.
        output_json: Emit JSON instead of text.
    Returns:
        None.
    Raises:
        typer.Exit: Applying desktop preferences fails.
    """
    _emit(ssh_host, "optimize", output_json, dry_run=dry_run)


@desktop.command("public-access")
@intent_boundary(OperationIntent.ENSURE_PRESENT)
@json_stdout_contract
def public_access(
    ssh_host: str = typer.Option(..., "--ssh-host"),
    public_ip: str = typer.Option(
        ...,
        "--public-ip",
        help="Provider-verified static public IPv4 address of this VM.",
    ),
    https_port: int = typer.Option(8443, "--https-port", min=1024, max=65535),
    dry_run: bool = typer.Option(False, "--dry-run"),
    output_json: bool = typer.Option(False, "--json"),
) -> None:
    """Enable HTTPS with a trusted IP certificate and a separate strong login.

    Args:
        ssh_host: Selected SSH destination.
        public_ip: VM's public IPv4 address, already assigned by its operator.
        https_port: Dedicated HTTPS port; existing port 443 services are preserved.
        dry_run: Plan without mutations.
        output_json: Emit JSON instead of text.
    Returns:
        None.
    Raises:
        typer.Exit: Gateway setup or certificate issuance failed.
    """
    _emit(
        ssh_host,
        "public-access",
        output_json,
        public_ip=public_ip,
        https_port=https_port,
        dry_run=dry_run,
    )


@desktop.command("open")
def open_cmd(
    ssh_host: str | None = typer.Option(None, "--ssh-host"),
    local: bool = typer.Option(False, "--local", help="Open chat attached to this Mac."),
    local_port: int = typer.Option(16080, "--local-port", min=1024, max=65535),
    chat: bool = typer.Option(
        False, "--chat", help="Open the mobile Codex chat interface."
    ),
) -> None:
    """Open the desktop using public HTTPS or a private SSH tunnel.

    Args:
        ssh_host: Selected SSH destination.
        local: Open the saved Mac chat URL.
        local_port: Local tunnel port when public access is not configured.
        chat: Open shared mobile chat instead of the desktop viewer.
    Returns:
        None.
    Raises:
        typer.Exit: Opening or connecting failed.
    """
    try:
        _require_target(ssh_host, local)
        if local:
            from npa.tools.desktop.local_runtime import open_local
            typer.echo(open_local())
            return
        typer.echo(open_desktop(ssh_host, local_port=local_port, chat=chat))
    except (ValueError, RuntimeError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc


@desktop.command("chat-setup")
@intent_boundary(OperationIntent.ENSURE_PRESENT)
@json_stdout_contract
def chat_setup(
    ssh_host: str | None = typer.Option(None, "--ssh-host"),
    local: bool = typer.Option(False, "--local", help="Use Codex and open VS Code chats on this Mac."),
    gateway_ssh_host: str | None = typer.Option(None, "--gateway-ssh-host", help="Existing managed HTTPS gateway for local chat."),
    local_port: int = typer.Option(6091, "--local-port", min=1024, max=65535),
    gateway_port: int = typer.Option(6091, "--gateway-port", min=1024, max=65535),
    connect_vscode: bool = typer.Option(
        False,
        "--connect-vscode",
        help="Configure VS Code to share sessions after its next window reload.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run"),
    output_json: bool = typer.Option(False, "--json"),
) -> None:
    """Install authenticated mobile chat on the existing desktop HTTPS gateway.

    Args:
        ssh_host: Existing desktop SSH alias.
        local: Attach to existing Codex sessions on this Mac.
        gateway_ssh_host: Existing managed HTTPS gateway for the Mac.
        local_port: Mac loopback port on first setup.
        gateway_port: Gateway loopback port on first setup.
        connect_vscode: Configure the VDI IDE's shared runtime without restarting it.
        dry_run: Show the plan without connecting.
        output_json: Emit JSON instead of text.
    Returns:
        None.
    Raises:
        typer.Exit: Setup fails.
    """
    _require_target(ssh_host, local)
    if local:
        if connect_vscode:
            raise typer.BadParameter("Local mode follows VS Code automatically; omit --connect-vscode.")
        _local_setup(gateway_ssh_host, local_port, gateway_port, dry_run, output_json)
        return
    if gateway_ssh_host:
        raise typer.BadParameter("--gateway-ssh-host requires --local")
    _emit(
        ssh_host,
        "chat-setup",
        output_json,
        connect_vscode=connect_vscode,
        dry_run=dry_run,
    )


def _require_target(host, local):
    if bool(host) == local:
        raise typer.BadParameter("Choose exactly one of --local or --ssh-host.")


def _local_setup(host, port, gateway_port, dry_run, output_json):
    from npa.tools.desktop.local_runtime import setup_local
    try:
        result = setup_local(gateway_host=host, port=port, gateway_port=gateway_port, dry_run=dry_run)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        typer.echo(json.dumps({"error": str(error)}) if output_json else str(error))
        raise typer.Exit(1) from error
    typer.echo(json.dumps(result, indent=2))
