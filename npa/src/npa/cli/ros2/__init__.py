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
"""

from __future__ import annotations

import os
import shutil
from typing import Any

import typer
from rich.console import Console

#: ROS 2 distribution this tooling targets.
SUPPORTED_ROS_DISTRO = "jazzy"

console = Console()

app = typer.Typer(
    name="ros2",
    help=(
        "ROS 2 Jazzy prerequisite detection and deployment planning "
        "(bridge/bag-conversion/fleet execution not implemented)."
    ),
    no_args_is_help=True,
)


def _ros2_status() -> tuple[bool, str]:
    """Return ``(ok, detail)`` for a usable ROS 2 Jazzy environment."""
    ros_distro = (os.environ.get("ROS_DISTRO") or "").strip()
    if not shutil.which("ros2"):
        return (
            False,
            "the `ros2` CLI is not on PATH (ROS 2 is not installed or its "
            "environment is not sourced)",
        )
    if ros_distro and ros_distro != SUPPORTED_ROS_DISTRO:
        return (
            False,
            f"ROS_DISTRO={ros_distro!r}; this tooling targets ROS 2 "
            f"{SUPPORTED_ROS_DISTRO.upper()} LTS",
        )
    try:
        import rclpy  # noqa: F401
    except Exception:
        return (
            False,
            "rclpy is not importable; source the ROS 2 environment first, e.g. "
            "`source /opt/ros/jazzy/setup.bash`",
        )
    return (
        True,
        f"ROS 2 {SUPPORTED_ROS_DISTRO.upper()} "
        f"(ROS_DISTRO={ros_distro or SUPPORTED_ROS_DISTRO})",
    )


@app.command("preflight")
def preflight_cmd() -> dict[str, Any]:
    """Check ROS 2 Jazzy prerequisites; fail fast with remediation if unusable."""
    ok, detail = _ros2_status()
    payload = {
        "ok": ok,
        "detail": detail,
        "supported_distro": SUPPORTED_ROS_DISTRO,
    }
    if ok:
        console.print(f"[green]ROS 2 preflight OK:[/green] {detail}")
        return payload
    console.print(
        "[red]ROS 2 preflight FAILED.[/red]\n"
        f"{detail}\n"
        "Install ROS 2 Jazzy (https://docs.ros.org/en/jazzy/Installation.html) "
        "and source it (`source /opt/ros/jazzy/setup.bash`) before retrying."
    )
    raise typer.Exit(code=3)
