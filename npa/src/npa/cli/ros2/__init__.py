"""``npa workbench ros2`` -- ROS 2 bridge, bag/MCAP conversion, Open-RMF fleet tooling.

Two-way middleware bridge between Physical AI workbench artifacts and ROS 2
(Jazzy LTS): bag/MCAP files <-> live ROS 2 topics, plus an Open-RMF fleet
adapter stub.  Unlike the foxglove/lichtblick viz bridges (one-way,
artifacts -> viewer), the ros2 bridge is bidirectional: workbench outputs can
be *published* onto robot topics, and robot telemetry can be *recorded* back
into artifacts.

Every command requires ROS 2 Jazzy installed and sourced.  When ROS 2 is
absent the commands fail fast with an operator-facing remediation message
instead of a traceback.
"""

from __future__ import annotations

import os
import shutil
from enum import Enum
from typing import Any

import typer
from rich.console import Console

console = Console()

app = typer.Typer(
    name="ros2",
    help="ROS 2 bridge: bag/MCAP <-> topics (bidirectional), bag conversion, Open-RMF fleet.",
    no_args_is_help=True,
)

#: ROS 2 distribution this tooling targets.
SUPPORTED_ROS_DISTRO = "jazzy"


class OutputFormat(str, Enum):
    """``--output-format`` selects a rendering, not a destination path."""

    text = "text"
    json = "json"


class BridgeDirection(str, Enum):
    """Direction of a ``bridge`` run."""

    to_ros = "to-ros"
    from_ros = "from-ros"
    bidir = "bidir"


class BagFormat(str, Enum):
    """On-disk recording formats understood by ``bag-convert``."""

    ros2bag = "ros2bag"
    mcap = "mcap"


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
        f"ROS 2 {SUPPORTED_ROS_DISTRO.upper()} (ROS_DISTRO={ros_distro or SUPPORTED_ROS_DISTRO})",
    )


def _require_ros2() -> None:
    """Abort with a remediation message unless ROS 2 Jazzy is usable."""
    ok, detail = _ros2_status()
    if ok:
        return
    console.print(
        "[red]npa workbench ros2 requires ROS 2 Jazzy.[/red]\n"
        f"{detail}\n"
        "Install ROS 2 Jazzy "
        "(https://docs.ros.org/en/jazzy/Installation.html) and source it "
        "(`source /opt/ros/jazzy/setup.bash`) before retrying.",
        style=None,
    )
    raise typer.Exit(code=3)


def _emit(payload: dict[str, Any], fmt: OutputFormat) -> None:
    if fmt is OutputFormat.json:
        import json

        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    for key, value in payload.items():
        typer.echo(f"{key}: {value}")


def _split_topics(topics: list[str]) -> list[str]:
    out: list[str] = []
    for item in topics:
        out.extend(part.strip() for part in str(item).split(",") if part.strip())
    # Preserve order, drop duplicates.
    return list(dict.fromkeys(out))


@app.command("bridge")
def bridge_cmd(
    direction: BridgeDirection = typer.Option(
        BridgeDirection.bidir,
        "--direction",
        help="Bridge direction: to-ros publishes artifacts onto topics, "
        "from-ros records topics into artifacts, bidir does both.",
    ),
    topics: list[str] = typer.Option(
        [],
        "--topics",
        help="ROS 2 topics to bridge. Repeat the flag or comma-separate.",
    ),
    input_path: str = typer.Option(
        "",
        "--input",
        "-i",
        help="Input bag/MCAP file to publish onto ROS 2 topics (to-ros/bidir).",
    ),
    output_path: str = typer.Option(
        "",
        "--output",
        "-o",
        help="Output bag/MCAP file for recording ROS 2 topics (from-ros/bidir).",
    ),
    domain_id: int = typer.Option(0, "--domain-id", help="ROS_DOMAIN_ID."),
    rmw: str = typer.Option(
        "", "--rmw", help="RMW implementation override, e.g. rmw_zenoh_cpp."
    ),
    rate: float = typer.Option(
        1.0, "--rate", help="Replay rate multiplier for to-ros publishing."
    ),
    duration: float = typer.Option(
        0.0,
        "--duration",
        help="Bridge run duration in seconds; 0 runs until interrupted.",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.text, "--output-format", help="Output format."
    ),
) -> None:
    """Bidirectional bridge between bag/MCAP artifacts and ROS 2 topics."""
    resolved_topics = _split_topics(topics)
    if direction in (BridgeDirection.to_ros, BridgeDirection.bidir) and not input_path:
        console.print("[red]--input is required for direction 'to-ros'/'bidir'.[/red]")
        raise typer.Exit(code=2)
    if (
        direction in (BridgeDirection.from_ros, BridgeDirection.bidir)
        and not output_path
    ):
        console.print(
            "[red]--output is required for direction 'from-ros'/'bidir'.[/red]"
        )
        raise typer.Exit(code=2)
    if duration < 0:
        console.print("[red]--duration must be >= 0.[/red]")
        raise typer.Exit(code=2)
    _require_ros2()
    _emit(
        {
            "status": "not-implemented",
            "direction": direction.value,
            "topics": resolved_topics,
            "input": input_path or None,
            "output": output_path or None,
            "domain_id": domain_id,
            "rmw": rmw or None,
            "rate": rate,
            "duration": duration,
            "detail": (
                "The live rosbag2/MCAP <-> ROS 2 bridge backend is not "
                "implemented yet. It requires ROS 2 Jazzy with rosbag2 and "
                "the rosbag2 mcap storage plugin installed."
            ),
        },
        output_format,
    )
    raise typer.Exit(code=4)


@app.command("bag-convert")
def bag_convert_cmd(
    input_path: str = typer.Argument(
        ..., help="Input ROS 2 bag directory or MCAP file."
    ),
    output_path: str = typer.Argument(
        ..., help="Output ROS 2 bag directory or MCAP file."
    ),
    from_format: BagFormat = typer.Option(
        BagFormat.ros2bag,
        "--from-format",
        help="Format of the input recording.",
    ),
    to_format: BagFormat = typer.Option(
        BagFormat.mcap,
        "--to-format",
        help="Format of the output recording.",
    ),
    topics: list[str] = typer.Option(
        [],
        "--topics",
        help="Only convert these topics. Repeat the flag or comma-separate.",
    ),
    compression: str = typer.Option(
        "", "--compression", help="Output compression, e.g. zstd."
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.text, "--output-format", help="Output format."
    ),
) -> None:
    """Convert recordings between ROS 2 bags and MCAP files."""
    _require_ros2()
    _emit(
        {
            "status": "not-implemented",
            "input": input_path,
            "output": output_path,
            "from_format": from_format.value,
            "to_format": to_format.value,
            "topics": _split_topics(topics),
            "compression": compression or None,
            "detail": (
                "bag/MCAP conversion is not implemented yet. It requires "
                "ROS 2 Jazzy with rosbag2 and the rosbag2 mcap storage "
                "plugin installed."
            ),
        },
        output_format,
    )
    raise typer.Exit(code=4)


@app.command("fleet-status")
def fleet_status_cmd(
    fleet_config: str = typer.Option(
        "", "--fleet-config", help="Open-RMF fleet adapter config YAML."
    ),
    rmf_server: str = typer.Option(
        "http://localhost:8000",
        "--rmf-server",
        help="Open-RMF API server base URL.",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.text, "--output-format", help="Output format."
    ),
) -> None:
    """Show Open-RMF fleet adapter status (stub)."""
    _require_ros2()
    _emit(
        {
            "status": "not-implemented",
            "fleet_config": fleet_config or None,
            "rmf_server": rmf_server,
            "detail": (
                "The Open-RMF fleet adapter is a stub. A full adapter "
                "requires ROS 2 Jazzy plus the rmf_fleet_adapter packages "
                "and a reachable Open-RMF API server."
            ),
        },
        output_format,
    )
    raise typer.Exit(code=4)
