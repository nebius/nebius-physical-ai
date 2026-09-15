"""Task qualification uses current simulator state and upstream outcomes."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from npa.workbench.isaac_arena.errors import IsaacArenaError
from npa.workbench.isaac_arena.ground_truth import simulator_ground_truth


def _write_episode(root: Path, *, trace, success=True, episode_length=None):
    with h5py.File(root / "simulator_ground_truth_rank0.hdf5", "w") as dataset:
        episode = dataset.create_group("data/demo_0")
        episode.create_dataset("success", data=[success])
        episode.create_dataset(
            "revolute_joint_state", data=np.asarray(trace).reshape(-1, 1)
        )
    record = {
        "success": success,
        "episode_length": len(trace) - 1 if episode_length is None else episode_length,
    }
    (root / "episode_results_rank0.jsonl").write_text(json.dumps(record) + "\n")


def _validate(root, *, successes=1, steps=80, required=True):
    return simulator_ground_truth(
        root,
        environment="gr1_open_microwave",
        policy_type="replay",
        expected_episodes=1,
        expected_successes=successes,
        expected_action_steps=steps,
        require_task_success=required,
    )


def test_opened_door_requires_real_progress_and_success_in_same_episode(tmp_path):
    _write_episode(tmp_path, trace=[0.2, 0.2, 0.3, 0.55, 0.81])
    result = _validate(tmp_path)
    motion = result["task_motion"]
    assert result["task_progress_adapter"] == "arena.open-door.revolute-joint.v1"
    assert motion["adapter"] == result["task_progress_adapter"]
    assert motion["environment"] == "gr1_open_microwave"
    assert motion["initial_openness"] == 0.2
    assert motion["final_openness"] == 0.81
    assert motion["openness_delta"] == pytest.approx(0.61)
    assert motion["progress_interval"] == {
        "start_action_step": 1,
        "end_action_step": 4,
        "total_action_steps": 4,
    }
    assert motion["visual_interval_strategy"] == "task_progress_with_leading_context"
    assert motion["visual_context_steps"] == 30
    assert motion["visual_progress_signal"] == "monotonic_structural_change"
    assert motion["visual_progress_region"]["name"] == "microwave_door_workspace"
    assert motion["visual_association_radius_fraction"] == 0.03
    assert motion["visual_interval"] == {
        "start_action_step": 0,
        "end_action_step": 4,
        "total_action_steps": 4,
    }
    assert motion["visual_progress_qualified"] is True
    assert len(result["files"][0]["sha256"]) == 64


@pytest.mark.parametrize(
    "trace,reason",
    [
        ([0.2, 0.2, 0.2], "final simulator door openness"),
        ([0.2, 0.8, 0.8], "final simulator door openness"),
        ([0.2, 0.9, 0.2], "final simulator door openness"),
        ([0.85, 0.86, 0.9], "below the acceptance delta"),
        ([0.2, float("nan"), 0.9], "invalid simulator ground-truth signal"),
    ],
)
def test_success_flag_cannot_qualify_static_preopened_or_corrupt_state(
    tmp_path, trace, reason
):
    _write_episode(tmp_path, trace=trace)
    with pytest.raises(IsaacArenaError, match=reason):
        _validate(tmp_path)


def test_ordinary_preopened_success_remains_truthful_without_visual_qualification(
    tmp_path,
):
    _write_episode(tmp_path, trace=[0.75, 0.81])
    motion = _validate(tmp_path, required=False)["task_motion"]
    assert motion["task_success"] is True
    assert motion["visual_progress_qualified"] is False
    assert motion["progress_interval"] is None
    assert "visual_interval" not in motion


@pytest.mark.parametrize("length", [True, 2.7, "2", 0, -1])
def test_episode_length_cannot_be_coerced_to_match_trace(tmp_path, length):
    _write_episode(tmp_path, trace=[0.2, 0.4, 0.81], episode_length=length)
    with pytest.raises(IsaacArenaError, match="positive integer"):
        _validate(tmp_path)


def test_large_door_motion_does_not_override_upstream_failure(tmp_path):
    _write_episode(tmp_path, trace=[0.2, 0.5, 0.9], success=False)
    with pytest.raises(IsaacArenaError, match="no upstream-defined successful"):
        _validate(tmp_path, successes=0)
    result = _validate(tmp_path, successes=0, required=False)
    assert result["successes"] == 0
    assert result["task_motion"] is None


def test_trace_cannot_exceed_source_actions_or_disagree_with_episode_length(tmp_path):
    _write_episode(tmp_path, trace=[0.2, 0.3, 0.5, 0.81])
    with pytest.raises(IsaacArenaError, match="exact replay action horizon"):
        _validate(tmp_path, steps=2)
    _write_episode(tmp_path, trace=[0.2, 0.3, 0.81], episode_length=250)
    with pytest.raises(IsaacArenaError, match="trace length disagrees"):
        _validate(tmp_path)


def test_matching_aggregate_success_count_cannot_hide_swapped_episodes(tmp_path):
    _write_episode(tmp_path, trace=[0.2, 0.3, 0.81])
    path = tmp_path / "simulator_ground_truth_rank0.hdf5"
    with h5py.File(path, "a") as dataset:
        episode = dataset.create_group("data/demo_1")
        episode.create_dataset("success", data=[False])
        episode.create_dataset("revolute_joint_state", data=[[0.2], [0.3], [0.4]])
    entries = [
        {"success": False, "episode_length": 2},
        {"success": True, "episode_length": 2},
    ]
    (tmp_path / "episode_results_rank0.jsonl").write_text(
        "\n".join(map(json.dumps, entries))
    )
    with pytest.raises(IsaacArenaError, match="success flags disagree"):
        simulator_ground_truth(
            tmp_path,
            environment="gr1_open_microwave",
            expected_episodes=2,
            policy_type="replay",
            expected_successes=1,
            expected_action_steps=80,
            require_task_success=True,
        )


@pytest.mark.parametrize("steps", [[1, 1], [2, 3], [1, 3]])
def test_video_step_trace_rejects_repeated_skipped_or_shifted_actions(tmp_path, steps):
    _write_episode(tmp_path, trace=[0.2, 0.5, 0.81])
    with h5py.File(tmp_path / "simulator_ground_truth_rank0.hdf5", "a") as dataset:
        episode = dataset["data/demo_0"]
        episode.create_dataset("npa_video/initial_action_step", data=[[0]])
        episode.create_dataset(
            "npa_video/action_step", data=np.asarray(steps).reshape(-1, 1)
        )
        episode.create_dataset("npa_video/terminal_action_step", data=[[steps[-1]]])
    with pytest.raises(IsaacArenaError, match="action steps are not contiguous"):
        _validate(tmp_path)


def test_unregistered_environment_cannot_inherit_microwave_semantics(tmp_path):
    _write_episode(tmp_path, trace=[0.2, 0.5, 0.81])
    with pytest.raises(IsaacArenaError, match="no registered progress adapter"):
        simulator_ground_truth(
            tmp_path,
            environment="gr1_turn_stand_mixer_knob",
            policy_type="replay",
            expected_episodes=1,
            expected_successes=1,
            expected_action_steps=2,
            require_task_success=True,
        )
