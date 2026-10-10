"""Native, opt-in Jazzy preflight checks; no robot, cloud, or bridge execution."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        os.environ.get("NPA_ROS2_PREFLIGHT_LIVE") != "1",
        reason="requires an explicitly selected, sourced ROS 2 Jazzy runtime",
    ),
]


def _module_preflight(env: dict[str, str] | None = None):
    return subprocess.run(
        [sys.executable, "-m", "npa.workflows.byof.ros2_pipeline", "--preflight"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_native_jazzy_sdk_preflight() -> None:
    from npa.workbench.ros2 import preflight

    assert os.environ.get("ROS_DISTRO") == "jazzy"
    payload = preflight()
    assert payload["ok"] is True
    assert payload["supported_distro"] == "jazzy"


def test_native_jazzy_toolref_module() -> None:
    result = _module_preflight()
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["ok"] is True


def test_native_module_refuses_wrong_distribution() -> None:
    result = _module_preflight({**os.environ, "ROS_DISTRO": "unsupported-test"})
    assert result.returncode == 3, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert "unsupported-test" in payload["detail"]
