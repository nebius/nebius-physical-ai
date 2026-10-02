"""Prove routed robot SDG with real hosted inference, MuJoCo cameras, and native LeRobot."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from npa.sdk.workbench.token_factory import RobotSdgRequest, robot_sdg
from npa.workbench.token_factory import robot_sim
from npa.workbench.token_factory.sdg_protocol import FAST_MODEL, REASONING_MODEL

pytestmark = pytest.mark.token_factory_e2e


def _gate():
    if os.environ.get("NPA_TOKEN_FACTORY_ROBOT_SDG_LIVE") != "1":
        pytest.skip("Set NPA_TOKEN_FACTORY_ROBOT_SDG_LIVE=1 for real robot SDG")


def test_live_robot_sdg_native_dataset_and_action_replay(tmp_path):
    _gate()
    native_python = os.environ.get("NPA_LEROBOT_PROOF_PYTHON")
    assert native_python and Path(native_python).is_file(), (
        "Set NPA_LEROBOT_PROOF_PYTHON to an isolated LeRobot 0.5.1 interpreter"
    )
    root = Path(__file__).resolve().parents[3]
    output = Path(
        os.environ.get("NPA_ROBOT_SDG_OUTPUT_DIR", str(tmp_path / "robot-data"))
    )
    report = robot_sdg(
        RobotSdgRequest(
            input_path=str(root / "npa/examples/token-factory-robot-sdg/seeds.jsonl"),
            output_path=str(output),
            seed=42,
        )
    )
    assert report["status"] == "completed"
    assert report["seed_count"] == report["accepted_count"] == 6
    assert report["error_count"] == report["rejected_count"] == 0
    assert report["generation_model_counts"] == {FAST_MODEL: 3, REASONING_MODEL: 3}
    records = [
        json.loads(line)
        for line in (output / "provenance.jsonl").read_text().splitlines()
    ]
    assert all(record["simulation"]["accepted"] for record in records)
    assert all(
        call["status"] == "completed" for record in records for call in record["calls"]
    )
    subprocess.run(
        [
            native_python,
            str(root / "npa/scripts/validate_robot_sdg_lerobot.py"),
            "--input-path",
            str(output),
            "--output-path",
            str(output / "native-lerobot-validation.json"),
        ],
        check=True,
    )
    _replay_actions(output, records[0])


def _replay_actions(output, record):
    trace = np.load(output / record["episode_path"] / "physics.npz", allow_pickle=False)
    maximum_error = 0.0
    with robot_sim._world(
        record["scene"], record["simulation_seed"], 480, 360
    ) as world:
        env = world[0]
        for index, action in enumerate(trace["actions"]):
            np.testing.assert_allclose(
                robot_sim._robot_state(env), trace["state"][index], atol=1e-6
            )
            env.step(action)
            state = robot_sim._robot_state(env)
            maximum_error = max(
                maximum_error, float(np.abs(state - trace["next_state"][index]).max())
            )
            np.testing.assert_allclose(state, trace["next_state"][index], atol=1e-6)
    (output / "action-replay-validation.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "replayed_frames": len(trace["actions"]),
                "maximum_joint_state_error": maximum_error,
            },
            indent=2,
        )
        + "\n"
    )


def test_live_open_gripper_control_is_rejected(tmp_path, monkeypatch):
    _gate()
    phases = robot_sim._phases
    monkeypatch.setattr(
        robot_sim,
        "_phases",
        lambda obj, goal: [
            (name, target, 1.0, steps) for name, target, _, steps in phases(obj, goal)
        ],
    )
    scene = {
        "object_x": -0.1,
        "object_y": -0.08,
        "goal_x": 0.1,
        "goal_y": 0.08,
        "object_color": "red",
        "target_color": "green",
        "lighting": 1.0,
    }
    result = robot_sim.simulate_robot_episode(
        scene, seed=42, output=tmp_path / "failed-grasp"
    )
    (tmp_path / "negative-physics-validation.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    assert not result["accepted"]
    assert not result["checks"]["bilateral_grasp_contact"]
    assert not result["checks"]["lifted"]
