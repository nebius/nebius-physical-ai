"""Tests for Stage 2 sim asset materialization."""

from __future__ import annotations

import json
from pathlib import Path

from npa.workflows.sim2real_assets import (
    run_assets_stage,
    robot_spec_doc_from_consumed,
    scene_spec_doc_from_consumed,
)
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


def test_scene_spec_doc_from_consumed_unwraps_stage_two_envelope() -> None:
    stock = {
        "schema": "npa.sim2real.consumed_scene_spec.v1",
        "scene_spec": {"objects": [{"name": "cube", "role": "manipuland", "asset_source": "isaac_stock", "builtin_path": "lift_cube", "pos": [0, 0, 0.04], "euler": [0, 0, 0], "color": [1, 0, 0], "fixed": False, "friction": 0.5, "mass": 0.1}]},
    }
    doc = scene_spec_doc_from_consumed(stock)
    assert isinstance(doc.get("objects"), list)


def test_robot_spec_doc_from_consumed_stock_returns_none() -> None:
    stock = {
        "schema": "npa.sim2real.consumed_robot_spec.v1",
        "status": "stock_franka",
        "robot_spec": {"preset": "franka"},
    }
    assert robot_spec_doc_from_consumed(stock) is None
