"""Tests for Stage 2 sim asset materialization."""

from __future__ import annotations

import json
from pathlib import Path

from npa.workflows.sim2real_assets import (
    cameras_from_consumed_uri,
    resolve_stage_cameras,
    run_assets_stage,
    resolve_robot_spec_from_consumed_doc,
    robot_spec_doc_from_consumed,
    scene_spec_doc_from_consumed,
    DEFAULT_CAMERA_STOCK,
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


def test_resolve_robot_spec_from_consumed_franka_envelope() -> None:
    from npa.genesis import robot_assets as ra

    consumed = {
        "schema": "npa.sim2real.consumed_robot_spec.v1",
        "status": "stock_franka",
        "robot_preset": "franka",
        "robot_spec": {"preset": "franka", "robot_source": "stock_franka"},
    }
    spec = resolve_robot_spec_from_consumed_doc(consumed)
    assert spec is not None
    assert spec.robot_source == ra.ROBOT_SOURCE_STOCK_FRANKA


def test_resolve_robot_spec_from_consumed_ur5e_envelope() -> None:
    from npa.genesis import robot_assets as ra

    consumed = {
        "schema": "npa.sim2real.consumed_robot_spec.v1",
        "status": "preset_pending_urdf",
        "robot_preset": "ur5e",
        "robot_spec": {
            "schema": ra.ROBOT_SPEC_SCHEMA,
            "preset": "ur5e",
            "robot_source": ra.ROBOT_SOURCE_BYO_URDF,
            "name": "ur5e",
            "ee_link": "tool0",
            "n_arm_joints": 6,
            "n_gripper_joints": 0,
            "isaac_robot_hint": "ur5e",
            "robot_uri": "",
        },
    }
    spec = resolve_robot_spec_from_consumed_doc(consumed)
    assert spec is not None
    assert spec.name == "ur5e"
    assert spec.ee_link == "tool0"


def test_resolve_robot_spec_from_consumed_byo_usd_path_not_leaked_preset() -> None:
    # Regression: Stage-2 consumption injects a default top-level robot_preset
    # ("franka") even for a BYO robot. A minimal inner spec keyed on the usd_path
    # alias (no robot_uri) must still resolve to the BYO robot — NOT short-circuit
    # to the leaked Franka preset. Mirrors the real consumed_byo envelope.
    from npa.genesis import robot_assets as ra

    consumed = {
        "schema": "npa.sim2real.consumed_robot_spec.v1",
        "status": "consumed_byo",
        "robot_preset": "franka",
        "robot_source": "",
        "robot_spec": {
            "name": "lite6_parallel",
            "robot_source": ra.ROBOT_SOURCE_BYO_USD,
            "usd_path": "s3://bucket/robots/lite6_combined.usda",
            "base_link": "world",
            "ee_link": "tcp_ee",
            "n_arm_joints": 6,
            "n_gripper_joints": 2,
            "joint_names": [
                "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
                "finger_joint1", "finger_joint2",
            ],
            "gripper_joint_names": ["finger_joint1", "finger_joint2"],
            "finger_links": ["uflite_finger1", "uflite_finger2"],
        },
    }
    spec = resolve_robot_spec_from_consumed_doc(consumed)
    assert spec is not None
    assert spec.robot_source == ra.ROBOT_SOURCE_BYO_USD  # not stock_franka
    assert spec.name == "lite6_parallel"
    assert spec.robot_uri.endswith("lite6_combined.usda")
    assert spec.dof_count == 8


def test_resolve_stage_cameras_from_scene_spec_file(tmp_path: Path) -> None:
    stage_dir = tmp_path / "stage_02_assets"
    stage_dir.mkdir()
    custom = {
        "workspace": {"placement": "custom", "resolution": [1280, 720], "dtype": "uint8"},
    }
    (stage_dir / "scene-spec.json").write_text(
        json.dumps({"objects": [], "cameras": custom}),
        encoding="utf-8",
    )
    config = Sim2RealLoopConfig(run_id="cam-test", output_dir=tmp_path)
    resolved = resolve_stage_cameras(config, stage_dir)
    assert resolved == custom


def test_cameras_from_consumed_uri_reads_envelope(tmp_path: Path) -> None:
    consumed_path = tmp_path / "consumed_scene_spec.json"
    custom = {"wrist": {"placement": "custom", "resolution": [640, 480], "dtype": "uint8"}}
    consumed_path.write_text(
        json.dumps({"schema": "npa.sim2real.consumed_scene_spec.v1", "cameras": custom}),
        encoding="utf-8",
    )
    assert cameras_from_consumed_uri(str(consumed_path)) == custom
    assert cameras_from_consumed_uri("") == DEFAULT_CAMERA_STOCK
