"""Keep incomplete task state available without changing upstream outcomes."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from npa.workbench.isaac_arena.simulator_diagnostics import retain_unscored_episode


def _environment(buffer):
    episode = SimpleNamespace(data=buffer, success=None)
    state = {"articulation": {"microwave": {"joint_position": np.array([[0.2]])}}}
    return SimpleNamespace(
        num_envs=1,
        recorder_manager=SimpleNamespace(get_episode=lambda env_id: episode),
        scene=SimpleNamespace(get_state=lambda *, is_relative: state),
        _npa_video_initial_physics_state={
            "physics_time": 0.0,
            "state": {"door": [0.1]},
        },
    ), episode


def test_incomplete_trace_survives_without_creating_a_success_or_scored_episode(
    tmp_path,
):
    buffer = {"revolute_joint_state": [np.array([0.1]), np.array([0.2])]}
    env, episode = _environment(buffer)
    upstream = tmp_path / "episode_results_rank0.jsonl"
    upstream.write_bytes(b"")
    result = retain_unscored_episode(env, tmp_path, 1)
    path = tmp_path / result["path"]
    payload = json.loads(path.read_text())
    assert payload["remaining_recorder_buffer"]["revolute_joint_state"] == [
        [0.1],
        [0.2],
    ]
    assert payload["final_scene_state"]["articulation"]["microwave"][
        "joint_position"
    ] == [[0.2]]
    assert payload["captured_action_steps"] == 1
    assert payload["scored"] is False and payload["task_success_claimed"] is False
    assert result["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert path.stat().st_mode & 0o777 == 0o600
    assert episode.data is buffer and episode.success is None
    assert upstream.read_bytes() == b""
    assert not list(tmp_path.glob("*.hdf5"))


def test_empty_buffer_does_not_fabricate_an_episode(tmp_path):
    env, _ = _environment({})
    assert retain_unscored_episode(env, tmp_path, 0) is None
    assert not list(tmp_path.iterdir())


def test_nonfinite_state_cannot_be_serialized_as_factual_diagnostics(tmp_path):
    env, _ = _environment({"revolute_joint_state": [np.array([float("nan")])]})
    with pytest.raises(ValueError, match="Out of range"):
        retain_unscored_episode(env, tmp_path, 1)
    assert not list(tmp_path.iterdir())
