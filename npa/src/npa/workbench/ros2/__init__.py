"""npa.workbench.ros2 - ROS 2 workbench toolRef.

Currently exposes only honest, implemented functionality:

- ``preflight``: detect whether a usable ROS 2 Jazzy environment is present.

Bridge (bag/MCAP <-> topics), bag conversion, and the Open-RMF fleet adapter
are not implemented and are not exposed. Deployment planning specs live in
:mod:`npa.workflows.byof.ros2_pipeline`.
"""

from __future__ import annotations

from npa._sdk import make_cli_wrapper

#: toolRef identity for this workbench module.
TOOLREF = "workbench.ros2"

#: ROS 2 distribution this tooling targets.
SUPPORTED_ROS_DISTRO = "jazzy"

preflight = make_cli_wrapper(
    "npa.cli.ros2",
    "preflight_cmd",
    "Check ROS 2 Jazzy prerequisites (ros2 CLI, ROS_DISTRO, rclpy).",
)

__all__ = [
    "TOOLREF",
    "SUPPORTED_ROS_DISTRO",
    "preflight",
]
