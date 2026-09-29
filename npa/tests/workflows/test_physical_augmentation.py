"""Reject false physical labels and broken action/image transition alignment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from npa.workflows.physical_augmentation_contract import (
    LiftController,
    accepted_steps,
    longest_hold,
    make_recipe,
    read_recipe,
)
from npa.workflows.physical_augmentation_report import verify_episode


def test_workflow_argv_matches_stage_parser_and_gpu_requirement():
    from npa.orchestration.npa_workflow.interpreter import build_plan
    from npa.orchestration.npa_workflow.spec import load_spec
    from npa.workflows.physical_augmentation import build_parser

    path = (
        Path(__file__).resolve().parents[3]
        / "workflows/testing/physical-augmentation.yaml"
    )
    plan = build_plan(load_spec(path), run_id="test-physical")
    assert [step.state for step in plan.steps] == ["prepare", "collect", "report"]
    for step in plan.steps:
        args = build_parser().parse_args(step.argv[3:])
        assert args.stage == step.state
        assert all(
            output["uri"].startswith(args.output_path) for output in step.outputs
        )
    assert plan.steps[1].resources_profile["accelerators"] == "RTXPRO6000:1"


def test_controller_recomputes_actions_for_displaced_object_and_limits_speed():
    tcp = np.array([0.5, 0.0, 0.3])
    cube = np.array([0.5, 0.0, 0.025])
    controller = LiftController(np.array([1.0, 0.0, 0.0, 0.0]), 0.02)
    nominal = controller.action(tcp, cube)
    shifted = controller.action(tcp, cube + [0.04, 0.08, 0])
    assert not np.array_equal(nominal, shifted)
    assert np.linalg.norm(shifted[:3] - tcp) <= 0.003001
    assert shifted[-1] == 1.0
    assert controller.phase == 0  # Elapsed calls alone cannot certify reaching.


def test_hold_requires_consecutive_contact_geometry_and_low_velocity():
    recipe = make_recipe("test", 0, 1, 600)
    arrays = {
        "next_object": np.tile([0.5, 0.0, 0.2], (60, 1)),
        "next_tcp": np.tile([0.5, 0.0, 0.2], (60, 1)),
        "next_velocity": np.zeros((60, 3)),
        "actions": np.tile([0.5, 0.0, 0.2, 1, 0, 0, 0, -1], (60, 1)),
    }
    arrays["next_velocity"][29] = [0, 0, 1]  # A launched cube breaks the streak.
    arrays["actions"][59, -1] = 1  # Opening the gripper breaks it again.
    mask = accepted_steps(arrays, np.array([0.5, 0.0, 0.025]), recipe["success"])
    assert longest_hold(mask) == 29
    arrays["next_tcp"][:] += 0.2  # A distant, unsupported object cannot qualify.
    assert not accepted_steps(
        arrays, np.array([0.5, 0, 0.025]), recipe["success"]
    ).any()


@pytest.fixture
def recording(tmp_path):
    count = 32
    state = np.arange((count + 1) * 9, dtype=np.float32).reshape(count + 1, 9) / 1000
    rgb = np.zeros((count, 480, 640, 3), dtype=np.uint8)
    rgb[:, :20, :20] = 200
    rgb[:, 100:120, 100:120, 0] = np.arange(count)[:, None, None]
    arrays = {
        "state": state[:-1],
        "next_state": state[1:],
        "rgb": rgb,
        "actions": np.tile([0.5, 0.0, 0.2, 1, 0, 0, 0, -1], (count, 1)),
        "next_object": np.tile([0.5, 0.0, 0.2], (count, 1)),
        "object": np.tile([0.5, 0.0, 0.2], (count, 1)),
        "tcp": np.tile([0.5, 0.0, 0.2], (count, 1)),
        "next_tcp": np.tile([0.5, 0.0, 0.2], (count, 1)),
        "next_velocity": np.zeros((count, 3)),
        "timestamp": np.arange(count) * 0.02,
    }
    arrays["object"][0, 2] = 0.025
    for name, value in arrays.items():
        np.save(tmp_path / f"{name}.npy", value)
    result = {
        "condition": "nominal",
        "attempt": 0,
        "seed": 0,
        "length": count,
        "terminated": False,
        "initial_object_m": [0.5, 0, 0.025],
        "success": True,
        "longest_hold_steps": count,
    }
    (tmp_path / "result.json").write_text(json.dumps(result))
    return tmp_path, make_recipe("test", 0, 1, 600)


def test_verified_recording_retains_measured_outcome_and_hashes(recording):
    root, recipe = recording
    result = verify_episode(root, recipe, 0.02)
    assert result["success"]
    assert set(result["files"]) == {
        "rgb.npy",
        "state.npy",
        "actions.npy",
        "next_state.npy",
        "object.npy",
        "tcp.npy",
        "next_object.npy",
        "next_tcp.npy",
        "next_velocity.npy",
        "timestamp.npy",
    }
    assert all(len(value) == 64 for value in result["files"].values())


@pytest.mark.parametrize(
    "channel,mutation,expected",
    [
        ("state", "reset", "discontinuous"),
        ("object", "reset", "discontinuous"),
        ("timestamp", "offset", "Timestamps"),
        ("next_velocity", "nan", "Nonfinite"),
        ("actions", "short", "misaligned"),
        ("actions", "quaternion", "quaternion"),
        ("rgb", "frozen", "frozen"),
        ("next_velocity", "flying", "outcome"),
    ],
)
def test_tampered_capture_fails_closed(recording, channel, mutation, expected):
    root, recipe = recording
    path = root / f"{channel}.npy"
    data = np.load(path)
    if mutation == "short":
        data = data[:-1]
    elif mutation == "frozen":
        data[:] = data[0]
    elif mutation == "quaternion":
        data[:, 3:7] = 0
    elif mutation == "nan":
        data[0, 0] = np.nan
    elif mutation == "flying":
        data[:, 2] = 1
    else:
        data[1] += 1
    np.save(path, data)
    with pytest.raises(ValueError, match=expected):
        verify_episode(root, recipe, 0.02)


def test_terminated_attempt_cannot_be_exported_as_success(recording):
    root, recipe = recording
    path = root / "result.json"
    result = json.loads(path.read_text())
    result["terminated"] = True
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="outcome"):
        verify_episode(root, recipe, 0.02)


@pytest.mark.parametrize(
    "field,value",
    [
        ("conditions", {"../../outside": {}}),
        ("task", "unqualified-task"),
        ("success", {"hold_steps": 0}),
        ("episodes_per_condition", True),
    ],
)
def test_recipe_cannot_change_paths_task_or_acceptance(tmp_path, field, value):
    recipe = make_recipe("test", 0, 1, 600)
    recipe[field] = value
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError):
        read_recipe(path)
