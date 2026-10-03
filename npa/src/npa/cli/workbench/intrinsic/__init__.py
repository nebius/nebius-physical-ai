"""Intrinsic Core workbench toolRef: read-only deployment validation.

This module intentionally exposes only what is real and read-only:

- ``preflight``: host checks (Ubuntu >= 22.04, ``ROS_DISTRO=lyrical``,
  k3s, ``inctl`` on PATH, GPU warning) plus runtime checks (k3s pods,
  TCP ingress, ``inctl service state list`` with no errored services).
  Fails fast with remediation when unusable.
- ``icon-status``: read-only ICON real-time control status.
- ``world-probe``: read-only digital-twin reachability (TCP ingress +
  world/ObjectWorld entry in ``inctl service state list``).

Mutating operations (``inctl world reset``, ``asset install``,
``icon clear-faults``, ...) are not implemented and are not exposed.

The CLI is a thin client: probing logic lives in
:mod:`npa.workbench.intrinsic`.
"""

from __future__ import annotations

from typing import Any

import typer
from rich.console import Console

from npa.workbench import intrinsic as intrinsic_workbench
from npa.workbench.intrinsic import ADDRESS_ENV, DEFAULT_ADDRESS

console = Console()

app = typer.Typer(
    name="intrinsic",
    help=(
        "Intrinsic Core read-only validation: preflight, ICON status, "
        "and digital-twin reachability (mutating operations not exposed)."
    ),
    no_args_is_help=True,
)

_ADDRESS_HELP = "Intrinsic ingress address (host:port)."


def _print_checks(payload: dict[str, Any]) -> None:
    for check in payload.get("checks", []):
        if check.get("skipped"):
            status = "[yellow]SKIP[/yellow]"
        elif check["ok"]:
            status = "[green]ok[/green]"
        else:
            status = "[red]FAIL[/red]"
        console.print(f"  [{status}] {check['name']}: {check['detail']}")
    for warning in payload.get("warnings", []):
        console.print(f"[yellow]warning:[/yellow] {warning}")


@app.command("preflight")
def preflight_cmd(
    address: str = typer.Option(
        DEFAULT_ADDRESS, "--address", envvar=ADDRESS_ENV, help=_ADDRESS_HELP
    ),
) -> dict[str, Any]:
    """Check Intrinsic Core prerequisites; fail fast with remediation if unusable."""
    payload = intrinsic_workbench.preflight(address=address)
    if payload["ok"]:
        console.print(f"[green]Intrinsic preflight OK:[/green] {payload['detail']}")
        _print_checks(payload)
        return payload
    console.print("[red]Intrinsic preflight FAILED.[/red]")
    _print_checks(payload)
    console.print(
        "Install Intrinsic Core per the getting-started guide "
        "(https://github.com/intrinsic-ai/intrinsic-core): Ubuntu 24.04/26.04, "
        "ROS 2 Lyrical Luth (`source /opt/ros/lyrical/setup.bash`), k3s, and "
        "the inctl CLI."
    )
    raise typer.Exit(code=3)


@app.command("icon-status")
def icon_status_cmd(
    address: str = typer.Option(
        DEFAULT_ADDRESS, "--address", envvar=ADDRESS_ENV, help=_ADDRESS_HELP
    ),
    instance_name: str = typer.Option(
        "icon", "--instance-name", help="ICON instance name."
    ),
) -> dict[str, Any]:
    """Report read-only ICON real-time control status."""
    payload = intrinsic_workbench.icon_status(
        address=address, instance_name=instance_name
    )
    if payload["ok"]:
        console.print(f"[green]ICON status OK:[/green] {payload['detail']}")
        return payload
    console.print(
        "[red]ICON status FAILED.[/red]\n"
        f"{payload['detail']}\n"
        "Ensure the Intrinsic Core runtime is up and the instance name is "
        "correct (`inctl asset instance list`)."
    )
    raise typer.Exit(code=3)


@app.command("world-probe")
def world_probe_cmd(
    address: str = typer.Option(
        DEFAULT_ADDRESS, "--address", envvar=ADDRESS_ENV, help=_ADDRESS_HELP
    ),
) -> dict[str, Any]:
    """Probe digital-twin (world) reachability without mutating state."""
    payload = intrinsic_workbench.world_probe(address=address)
    if payload["ok"]:
        console.print(f"[green]World probe OK:[/green] {payload['detail']}")
        return payload
    console.print(
        "[red]World probe FAILED.[/red]\n"
        f"{payload['detail']}\n"
        "Ensure the Intrinsic Core runtime is up (`inctl service state list`)."
    )
    raise typer.Exit(code=3)
