"""Tests for the ROS 2 bridge workbench / CLI / workflow scaffolding (issue #501)."""

from __future__ import annotations

import importlib
import json


import pytest
import typer
from typer.testing import CliRunner

import npa.workbench
from npa.cli.ros2 import (
    SUPPORTED_ROS_DISTRO,
    BridgeDirection,
    OutputFormat,
    app,
    bag_convert_cmd,
    bridge_cmd,
    fleet_status_cmd,
)
from npa.workflows.byof import ros2_pipeline
from npa.workflows.byof.ros2_pipeline import (
    ISAAC_ROS_RELEASE,
    OPEN_RMF_VERSION,
    ROS_DISTRO,
    JetsonThorTarget,
    Ros2PipelineConfig,
    build_plan,
)

runner = CliRunner()


def _bridge_kwargs(**overrides):
    """Explicit kwargs so direct calls do not see typer OptionInfo defaults."""
    kwargs = dict(
        direction=BridgeDirection.bidir,
        topics=[],
        input_path="in.mcap",
        output_path="out.mcap",
        domain_id=0,
        rmw="",
        rate=1.0,
        duration=0.0,
        output_format=OutputFormat.text,
    )
    kwargs.update(overrides)
    return kwargs


def test_cli_app_registered_as_ros2():
    assert isinstance(app, typer.Typer)
    command_names = {c.name for c in app.registered_commands}
    assert {"bridge", "bag-convert", "fleet-status"} <= command_names


def test_supported_distro_is_jazzy():
    assert SUPPORTED_ROS_DISTRO == "jazzy"
    assert ROS_DISTRO == "jazzy"


def test_app_help_works():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "bridge" in result.output
    assert "bag-convert" in result.output
    assert "fleet-status" in result.output


@pytest.mark.parametrize("command", ["bridge", "bag-convert", "fleet-status"])
def test_command_help_works(command: str):
    result = runner.invoke(app, [command, "--help"])
    assert result.exit_code == 0, result.output


def test_workbench_wrappers_resolve():
    module = importlib.import_module("npa.workbench.ros2")
    assert module.bridge.__npa_cli_module__ == "npa.cli.ros2"
    assert module.bridge.__npa_cli_callback__ == "bridge_cmd"
    assert module.bag_convert.__npa_cli_module__ == "npa.cli.ros2"
    assert module.bag_convert.__npa_cli_callback__ == "bag_convert_cmd"
    assert module.fleet.__npa_cli_module__ == "npa.cli.ros2"
    assert module.fleet.__npa_cli_callback__ == "fleet_status_cmd"


def test_workbench_lazy_namespace():
    module = npa.workbench.ros2
    assert callable(module.bridge)
    assert callable(module.bag_convert)
    assert callable(module.fleet)
    assert module.TOOLREF == "workbench.ros2"


def test_bridge_rejects_missing_input_for_to_ros(monkeypatch):
    monkeypatch.setattr(
        "npa.cli.ros2._ros2_status", lambda: (True, "ROS 2 Jazzy (stubbed)")
    )
    with pytest.raises(typer.Exit) as exc_info:
        bridge_cmd(**_bridge_kwargs(direction=BridgeDirection.to_ros, input_path=""))
    assert exc_info.value.exit_code == 2


def test_bridge_rejects_negative_duration(monkeypatch):
    monkeypatch.setattr(
        "npa.cli.ros2._ros2_status", lambda: (True, "ROS 2 Jazzy (stubbed)")
    )
    with pytest.raises(typer.Exit) as exc_info:
        bridge_cmd(**_bridge_kwargs(duration=-1.0))
    assert exc_info.value.exit_code == 2


def test_commands_fail_fast_without_ros2(monkeypatch):
    monkeypatch.setattr(
        "npa.cli.ros2._ros2_status",
        lambda: (False, "the `ros2` CLI is not on PATH"),
    )
    for call in (
        lambda: bridge_cmd(**_bridge_kwargs()),
        lambda: bag_convert_cmd(input_path="in/", output_path="out.mcap"),
        lambda: fleet_status_cmd(),
    ):
        with pytest.raises(typer.Exit) as exc_info:
            call()
        assert exc_info.value.exit_code == 3


def test_bridge_cli_reports_missing_ros2(monkeypatch):
    monkeypatch.setattr("npa.cli.ros2.shutil.which", lambda name: None)
    result = runner.invoke(
        app,
        ["bridge", "--input", "in.mcap", "--output", "out.mcap"],
    )
    assert result.exit_code == 3
    assert "requires ROS 2 Jazzy" in result.output


def test_jetson_thor_target_spec():
    target = JetsonThorTarget()
    assert target.arch == "aarch64"
    assert target.ros_distro == "jazzy"
    assert target.isaac_ros_release == ISAAC_ROS_RELEASE
    image = target.isaac_ros_image()
    assert image == f"nvcr.io/nvidia/isaac/ros:{ISAAC_ROS_RELEASE}-aarch64"
    steps = target.deploy_steps("registry.example/wb-ros2:aarch64")
    assert [s["step"] for s in steps] == ["preflight", "eula-preflight", "pull", "run"]


def test_build_plan_stages():
    config = Ros2PipelineConfig(
        direction="bidir",
        topics=["/camera/image_raw"],
        input_path="in.mcap",
        output_path="out.mcap",
    )
    plan = build_plan(config, image="registry.example/wb-ros2:aarch64")
    names = [stage["name"] for stage in plan]
    assert names == [
        "ros2-bridge",
        "bag-convert",
        "open-rmf-fleet-adapter",
        "jetson-thor-deploy",
    ]
    bridge_argv = plan[0]["argv"]
    assert "--direction" in bridge_argv and "bidir" in bridge_argv
    assert "--topics" in bridge_argv and "/camera/image_raw" in bridge_argv
    fleet_stage = plan[2]
    assert fleet_stage["open_rmf_version"] == OPEN_RMF_VERSION
    assert fleet_stage["status"] == "stub"


def test_pipeline_main_jetson_target(capsys):
    assert ros2_pipeline.main(["--jetson-target"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "jetson-thor"
    assert payload["arch"] == "aarch64"


def test_pipeline_main_plan(capsys):
    assert ros2_pipeline.main(["--plan", "--topics", "/a,/b"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert isinstance(plan, list)
    assert plan[0]["name"] == "ros2-bridge"


def test_pipeline_main_isaac_ros_image(capsys):
    assert ros2_pipeline.main(["--isaac-ros-image"]) == 0
    out = capsys.readouterr().out.strip()
    assert out == f"nvcr.io/nvidia/isaac/ros:{ISAAC_ROS_RELEASE}-aarch64"
