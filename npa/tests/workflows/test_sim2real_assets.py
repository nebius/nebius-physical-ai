"""Tests for Stage 2 sim asset materialization."""

from __future__ import annotations

import json
from pathlib import Path

from npa.workflows.sim2real_assets import run_assets_stage
from npa.workflows.sim2real_loop import Sim2RealLoopConfig


def test_run_assets_stage_stock_writes_scene_and_robot_specs(tmp_path: Path) -> None:
    config = Sim2RealLoopConfig(
        run_id="assets-stock",
        output_dir=tmp_path,
        sim_backend="isaac",
        robot_preset="franka",
    )
    result = run_assets_stage(config, tmp_path)
    scene = json.loads(Path(result.consumed_scene_path).read_text())
    robot = json.loads(Path(result.consumed_robot_path).read_text())
    assert scene["status"] == "stock_tabletop"
    assert scene["sim_backend"] == "isaac"
    assert "workspace" in scene["cameras"]
    assert robot["robot_spec"]["preset"] == "franka"
    assert robot["robot_spec"]["robot_source"] == "stock_franka"
    assert result.component["tier"] == "WORKS"


def test_run_assets_stage_ur_preset_marks_pending_urdf(tmp_path: Path) -> None:
    config = Sim2RealLoopConfig(
        run_id="assets-ur",
        output_dir=tmp_path,
        robot_preset="ur5e",
    )
    result = run_assets_stage(config, tmp_path)
    robot = json.loads(Path(result.consumed_robot_path).read_text())
    assert robot["status"] == "preset_pending_urdf"
    assert robot["robot_spec"]["preset"] == "ur5e"
