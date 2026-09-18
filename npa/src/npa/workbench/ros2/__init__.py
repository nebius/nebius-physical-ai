"""npa.workbench.ros2 - ROS 2 bridge workbench toolRef.

``workbench.ros2.*`` is the two-way middleware bridge between Physical AI
workbench artifacts and ROS 2 (Jazzy LTS): bag/MCAP files <-> live ROS 2
topics, plus an Open-RMF fleet adapter stub and a Jetson Thor deploy target
(see :mod:`npa.workflows.byof.ros2_pipeline`).

Unlike the foxglove/lichtblick viz bridges (one-way, artifacts -> viewer),
the ros2 bridge is bidirectional: workbench outputs can be *published* onto
robot topics, and robot telemetry can be *recorded* back into artifacts,
closing the sim-to-real loop "train on Nebius -> run on robot".
"""

from __future__ import annotations

from npa._sdk import make_cli_wrapper

#: toolRef identity for this workbench module.
TOOLREF = "workbench.ros2"

#: ROS 2 distribution this tooling targets.
SUPPORTED_ROS_DISTRO = "jazzy"

bridge = make_cli_wrapper(
    "npa.cli.ros2",
    "bridge_cmd",
    "Bidirectional bridge between bag/MCAP artifacts and ROS 2 topics.",
)
bag_convert = make_cli_wrapper(
    "npa.cli.ros2",
    "bag_convert_cmd",
    "Convert recordings between ROS 2 bags and MCAP files.",
)
fleet = make_cli_wrapper(
    "npa.cli.ros2",
    "fleet_status_cmd",
    "Open-RMF fleet adapter status (stub).",
)

__all__ = [
    "TOOLREF",
    "SUPPORTED_ROS_DISTRO",
    "bridge",
    "bag_convert",
    "fleet",
]
