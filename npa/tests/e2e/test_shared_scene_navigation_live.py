"""Opt-in native Isaac task acceptance inside the operator's exact BYOF runtime."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from npa.workflows.navigation.artifacts import file_sha256, materialize
from npa.workflows.navigation.contract import read_recipe
from npa.workflows.navigation.stages import prepare, run_stage

pytestmark = [pytest.mark.e2e, pytest.mark.gpu]


def test_native_shared_scene_navigation_checkpoint_and_isolation(tmp_path: Path):
    if os.environ.get("NPA_NAVIGATION_LIVE") != "1":
        pytest.skip(
            "set NPA_NAVIGATION_LIVE=1 inside the authorized Isaac BYOF runtime"
        )
    required = (
        "NPA_NAVIGATION_INPUT_URI",
        "NPA_NAVIGATION_OUTPUT_URI",
        "NPA_TASK_IMAGE",
    )
    missing = [key for key in required if not os.environ.get(key)]
    assert not missing, f"native navigation acceptance requires {', '.join(missing)}"
    destination = os.environ["NPA_NAVIGATION_OUTPUT_URI"].rstrip("/")
    assert destination.startswith("s3://"), (
        "live acceptance must verify durable S3 artifacts"
    )
    prepare(
        os.environ["NPA_NAVIGATION_INPUT_URI"],
        destination + "/prepared",
        os.environ["NPA_TASK_IMAGE"],
    )
    training = run_stage("train", destination + "/prepared", destination + "/training")
    evaluation = run_stage(
        "evaluate", destination + "/training", destination + "/evaluation"
    )
    result = materialize(destination + "/evaluation", tmp_path / "readback")
    recipe = read_recipe(result)
    assert evaluation["policy_loaded"] is True and evaluation["passed"] is True
    assert (
        evaluation["checkpoint_sha256"]
        == training["checkpoint_sha256"]
        == file_sha256(result / "policy.pt")
    )
    assert training["policy_parameter_delta_l2"] > 0
    assert len(evaluation["episodes"]) == recipe.num_envs
    assert all(
        row["steps"] > 0 and row["peer_collision_steps"] == 0
        for row in evaluation["episodes"]
    )
    assert json.loads((result / "trajectory.json").read_text())
    for record in (training, evaluation):
        assert record["isolation"]["passed"] is True
        assert record["isolation"]["obstacle_contact"] > 0
        assert record["runtime"]["scene_instances"] == 1
        if recipe.sensor_mode == "rgbd":
            assert record["isolation"]["camera_isolation_verified"] is True
            assert record["isolation"]["camera"]["fresh_render_steps"] == 3
    if recipe.sensor_mode == "rgbd":
        assert (result / "camera-probes.npz").is_file()
