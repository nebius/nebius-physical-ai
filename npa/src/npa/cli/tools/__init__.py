"""Expose development and operator utilities through npa tools."""

import json
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
    ssh_host: str = typer.Option(..., "--ssh-host"),
    output_json: bool = typer.Option(False, "--json"),
) -> None:
    """Inspect desktop services and recovery records without disclosing secrets.

    Args:
        ssh_host: Selected SSH destination.
        output_json: Emit JSON instead of text.
    Returns:
        None.
    Raises:
        typer.Exit: Inspection failed.
    """
    _emit(ssh_host, "status", output_json)


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
    ssh_host: str = typer.Option(..., "--ssh-host"),
    local_port: int = typer.Option(16080, "--local-port", min=1024, max=65535),
    chat: bool = typer.Option(
        False, "--chat", help="Open the mobile Codex chat interface."
    ),
) -> None:
    """Open the desktop using public HTTPS or a private SSH tunnel.

    Args:
        ssh_host: Selected SSH destination.
        local_port: Local tunnel port when public access is not configured.
        chat: Open shared mobile chat instead of the desktop viewer.
    Returns:
        None.
    Raises:
        typer.Exit: Opening or connecting failed.
    """
    try:
        typer.echo(open_desktop(ssh_host, local_port=local_port, chat=chat))
    except (ValueError, RuntimeError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc


@desktop.command("chat-setup")
@intent_boundary(OperationIntent.ENSURE_PRESENT)
@json_stdout_contract
def chat_setup(
    ssh_host: str = typer.Option(..., "--ssh-host"),
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
        connect_vscode: Configure the VDI IDE's shared runtime without restarting it.
        dry_run: Show the plan without connecting.
        output_json: Emit JSON instead of text.
    Returns:
        None.
    Raises:
        typer.Exit: Setup fails.
    """
    _emit(
        ssh_host,
        "chat-setup",
        output_json,
        connect_vscode=connect_vscode,
        dry_run=dry_run,
    )
