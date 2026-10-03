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


def preflight() -> dict[str, Any]:
    """Check ROS 2 Jazzy prerequisites; return a status payload.

    Returns ``{"ok": bool, "detail": str, "supported_distro": str}``. Never
    raises for a missing environment — callers decide how to surface the
    failure (the CLI exits 3 with remediation; the pipeline prints JSON).
    """
    ros_distro = (os.environ.get("ROS_DISTRO") or "").strip()
    if not shutil.which("ros2"):
        ok, detail = (
            False,
            (
                "the `ros2` CLI is not on PATH (ROS 2 is not installed or its "
                "environment is not sourced)"
            ),
        )
    elif ros_distro and ros_distro != SUPPORTED_ROS_DISTRO:
        ok, detail = (
            False,
            (
                f"ROS_DISTRO={ros_distro!r}; this tooling targets ROS 2 "
                f"{SUPPORTED_ROS_DISTRO.upper()} LTS"
            ),
        )
    else:
        try:
            import rclpy  # noqa: F401
        except Exception:
            ok, detail = (
                False,
                (
                    "rclpy is not importable; source the ROS 2 environment first, e.g. "
                    "`source /opt/ros/jazzy/setup.bash`"
                ),
            )
        else:
            ok, detail = (
                True,
                (
                    f"ROS 2 {SUPPORTED_ROS_DISTRO.upper()} "
                    f"(ROS_DISTRO={ros_distro or SUPPORTED_ROS_DISTRO})"
                ),
            )
    return {
        "ok": ok,
        "detail": detail,
        "supported_distro": SUPPORTED_ROS_DISTRO,
    }


__all__ = [
    "TOOLREF",
    "SUPPORTED_ROS_DISTRO",
    "preflight",
]
