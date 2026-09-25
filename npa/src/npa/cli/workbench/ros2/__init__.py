"""ROS 2 workbench toolRef: prerequisite detection and deployment planning.

This module intentionally exposes only what is real:

- ``preflight``: detect whether a usable ROS 2 Jazzy environment is present
  (``ros2`` CLI on PATH, ``ROS_DISTRO`` check, ``rclpy`` importable) and fail
  fast with a remediation message when it is not.

Bridge (bag/MCAP <-> topics), bag conversion, and the Open-RMF fleet adapter
are not implemented; the corresponding commands were removed rather than
shipped as stubs. Deployment planning (stage specs, Jetson Thor target) lives
in :mod:`npa.workflows.byof.ros2_pipeline`, which emits specs without
executing ROS 2 code.

The CLI is a thin client: prerequisite detection lives in
:mod:`npa.workbench.ros2`.
"""

from __future__ import annotations

from typing import Any

import typer
from rich.console import Console

from npa.workbench import ros2 as ros2_workbench

console = Console()

app = typer.Typer(
    name="ros2",
    help=(
        "ROS 2 Jazzy prerequisite detection and deployment planning "
        "(bridge/bag-conversion/fleet execution not implemented)."
    ),
    no_args_is_help=True,
)


@app.command("preflight")
def preflight_cmd() -> dict[str, Any]:
    """Check ROS 2 Jazzy prerequisites; fail fast with remediation if unusable."""
    payload = ros2_workbench.preflight()
    if payload["ok"]:
        console.print(f"[green]ROS 2 preflight OK:[/green] {payload['detail']}")
        return payload
    console.print(
        "[red]ROS 2 preflight FAILED.[/red]\n"
        f"{payload['detail']}\n"
        "Install ROS 2 Jazzy (https://docs.ros.org/en/jazzy/Installation.html) "
        "and source it (`source /opt/ros/jazzy/setup.bash`) before retrying."
    )
    raise typer.Exit(code=3)
