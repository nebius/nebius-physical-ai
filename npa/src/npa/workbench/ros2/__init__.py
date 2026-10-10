"""npa.workbench.ros2 - ROS 2 workbench toolRef.

Currently exposes only honest, implemented functionality:

- ``preflight``: detect whether a usable ROS 2 Jazzy environment is present.

Bridge (bag/MCAP <-> topics), bag conversion, and the Open-RMF fleet adapter
are not implemented and are not exposed. Deployment planning specs live in
:mod:`npa.workflows.byof.ros2_pipeline`.
"""

from __future__ import annotations

import os
import shutil
from typing import Any

#: toolRef identity for this workbench module.
TOOLREF = "workbench.ros2"

#: ROS 2 distribution this tooling targets.
SUPPORTED_ROS_DISTRO = "jazzy"


def _failed_preflight(detail: str) -> dict[str, Any]:
    """Build a failed ROS 2 preflight payload.

    Args:
        detail: The concrete prerequisite failure for the caller.

    Returns:
        A failed preflight payload with the supported distribution.

    Raises:
        None.
    """
    return {
        "ok": False,
        "detail": detail,
        "supported_distro": SUPPORTED_ROS_DISTRO,
    }


def preflight() -> dict[str, Any]:
    """Check ROS 2 Jazzy prerequisites; return a status payload.

    Args:
        None.

    Returns:
        A status payload with ``ok``, ``detail``, and ``supported_distro``.

    Raises:
        None. Missing prerequisites return a failed status for callers to
        surface (the CLI exits 3 and the pipeline prints JSON).
    """
    ros_distro = (os.environ.get("ROS_DISTRO") or "").strip()
    if not shutil.which("ros2"):
        return _failed_preflight(
            "the `ros2` CLI is not on PATH (ROS 2 is not installed or its "
            "environment is not sourced)"
        )
    if ros_distro != SUPPORTED_ROS_DISTRO:
        detail = (
            "ROS_DISTRO is unset; source the ROS 2 Jazzy environment first"
            if not ros_distro
            else f"ROS_DISTRO={ros_distro!r}"
        )
        return _failed_preflight(
            f"{detail}; this tooling targets ROS 2 {SUPPORTED_ROS_DISTRO.upper()} LTS"
        )
    try:
        import rclpy  # noqa: F401
    except ImportError:
        return _failed_preflight(
            "rclpy is not importable; source the ROS 2 environment first, e.g. "
            "`source /opt/ros/jazzy/setup.bash`"
        )
    return {
        "ok": True,
        "detail": f"ROS 2 {SUPPORTED_ROS_DISTRO.upper()} (ROS_DISTRO={ros_distro})",
        "supported_distro": SUPPORTED_ROS_DISTRO,
    }


__all__ = [
    "TOOLREF",
    "SUPPORTED_ROS_DISTRO",
    "preflight",
]
