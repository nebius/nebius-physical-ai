"""Tests for the ROS 2 workbench preflight / pipeline planning (issue #501).

Honest surface: ``preflight`` prerequisite detection is real and tested.
Bridge, bag conversion, and Open-RMF fleet execution are not implemented and
are not exposed. Pipeline stage specs are planning metadata; nothing executes
ROS 2 code.
"""

from __future__ import annotations

import importlib
import json

import pytest
import typer
from typer.testing import CliRunner

import npa.workbench
from npa.cli.ros2 import SUPPORTED_ROS_DISTRO, app, preflight_cmd
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


def _nested_app():
    """Minimal npa > workbench nesting mirroring the real CLI dispatch."""
    import typer as _typer

    top = _typer.Typer(name="npa", no_args_is_help=True)
    wb = _typer.Typer(name="workbench", help="wb", no_args_is_help=True)
    top.add_typer(wb, name="workbench")
    wb.add_typer(app, name="ros2")
    return top


def test_cli_app_exposes_only_preflight():
    """Stub commands were removed; only the real preflight remains."""
    assert isinstance(app, typer.Typer)
    command_names = {c.name for c in app.registered_commands}
    assert command_names == {"preflight"}


def test_supported_distro_is_jazzy():
    assert SUPPORTED_ROS_DISTRO == "jazzy"
    assert ROS_DISTRO == "jazzy"


def test_app_help_is_honest():
    result = runner.invoke(_nested_app(), ["workbench", "ros2", "--help"])
    assert result.exit_code == 0, result.output
    assert "preflight" in result.output
    assert "not implemented" in result.output


def test_preflight_help_works():
    result = runner.invoke(_nested_app(), ["workbench", "ros2", "preflight", "--help"])
    assert result.exit_code == 0, result.output


def test_workbench_wrappers_resolve():
    module = importlib.import_module("npa.workbench.ros2")
    assert module.preflight.__npa_cli_module__ == "npa.cli.ros2"
    assert module.preflight.__npa_cli_callback__ == "preflight_cmd"
    assert module.TOOLREF == "workbench.ros2"


def test_workbench_lazy_namespace():
    module = npa.workbench.ros2
    assert callable(module.preflight)
    assert not hasattr(module, "bridge")


def test_preflight_ok_when_ros2_usable(monkeypatch):
    monkeypatch.setattr(
        "npa.cli.ros2._ros2_status", lambda: (True, "ROS 2 Jazzy (stubbed)")
    )
    payload = preflight_cmd()
    assert payload["ok"] is True
    assert payload["supported_distro"] == "jazzy"


def test_preflight_cli_ok(monkeypatch):
    monkeypatch.setattr(
        "npa.cli.ros2._ros2_status", lambda: (True, "ROS 2 Jazzy (stubbed)")
    )
    result = runner.invoke(app, [])
    assert result.exit_code == 0, result.output
    assert "preflight OK" in result.output


def test_preflight_fails_fast_without_ros2(monkeypatch):
    monkeypatch.setattr(
        "npa.cli.ros2._ros2_status",
        lambda: (False, "the `ros2` CLI is not on PATH"),
    )
    with pytest.raises(typer.Exit) as exc_info:
        preflight_cmd()
    assert exc_info.value.exit_code == 3


def test_preflight_cli_reports_missing_ros2(monkeypatch):
    monkeypatch.setattr("npa.cli.ros2.shutil.which", lambda name: None)
    result = runner.invoke(app, [])
    assert result.exit_code == 3
    assert "preflight FAILED" in result.output
    assert "source /opt/ros/jazzy/setup.bash" in result.output


def test_preflight_detects_wrong_distro(monkeypatch):
    monkeypatch.setattr("npa.cli.ros2.shutil.which", lambda name: "/usr/bin/ros2")
    monkeypatch.setenv("ROS_DISTRO", "humble")
    from npa.cli import ros2 as ros2_cli

    ok, detail = ros2_cli._ros2_status()
    assert ok is False
    assert "humble" in detail


def test_toolref_argv_is_exact():
    from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG

    entry = TOOL_CATALOG["workbench.ros2.preflight"]
    assert "not implemented" in entry.description
    assert entry.argv_template == [
        "python3",
        "-m",
        "npa.workflows.byof.ros2_pipeline",
        "--preflight",
    ]


def test_pipeline_main_preflight_ok(monkeypatch, capsys):
    monkeypatch.setattr(
        "npa.cli.ros2._ros2_status", lambda: (True, "ROS 2 Jazzy (stubbed)")
    )
    assert ros2_pipeline.main(["--preflight"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"ok": True, "detail": "ROS 2 Jazzy (stubbed)"}


def test_pipeline_main_preflight_fails(monkeypatch, capsys):
    monkeypatch.setattr(
        "npa.cli.ros2._ros2_status", lambda: (False, "the `ros2` CLI is not on PATH")
    )
    assert ros2_pipeline.main(["--preflight"]) == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False


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
