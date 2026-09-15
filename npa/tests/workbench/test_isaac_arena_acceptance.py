"""The shared visual contract rejects independently plausible partial proofs."""

from __future__ import annotations

import copy

import pytest

from npa.workbench.isaac_arena.acceptance import (
    ACCEPTANCE_SCHEMA,
    qualify_visual_acceptance,
)
from npa.workbench.isaac_arena.errors import IsaacArenaError
from npa.workbench.isaac_arena.task_progress import (
    task_progress_adapter,
    task_progress_capabilities,
)


def _proof(length: int = 20) -> dict:
    interval = {
        "start_action_step": 2,
        "end_action_step": length,
        "total_action_steps": length,
    }
    episode = {
        "episode": "demo_0",
        "episode_length": length,
        "success": True,
    }
    motion = {
        "adapter": "arena.open-door.revolute-joint.v1",
        "environment": "gr1_open_microwave",
        "episode": "demo_0",
        "episode_length": length,
        "task_success": True,
        "visual_progress_qualified": True,
        "progress_interval": interval,
    }
    return {
        "environment": "gr1_open_microwave",
        "policy_type": "replay",
        "evidence": {
            "trajectory": {"nonzero_actions": True},
            "execution": {
                "strategy": "actions_initial_state_exact_replay",
                "source_steps": length,
                "prepared_steps": length,
                "action_padding_steps": 0,
            },
        },
        "summary": {
            "episodes": 1,
            "successes": 1,
            "success_rate": 1.0,
            "metrics": {"success_rate": 1.0},
        },
        "ground_truth": {
            "successes": 1,
            "episodes": [episode],
            "task_progress_adapter": motion["adapter"],
            "task_motion": motion,
        },
        "capture": {
            "captured_action_steps": length,
            "terminal": {"action_step": length},
            "terminal_frame_comparison": {"matched": True},
            "source_mp4_sha256": "a" * 64,
        },
        "video": {
            "frame_count": length,
            "motion": {
                "meaningful": True,
                "analysis_interval": {
                    "source": "simulator_ground_truth",
                    "action_steps": {
                        "start": interval["start_action_step"],
                        "end": interval["end_action_step"],
                        "total": interval["total_action_steps"],
                    },
                },
            },
            "frame_evidence": {"samples": {"first": {}, "last": {}}},
        },
    }


def _set_path(payload: dict, path: str, value: object) -> None:
    target = payload
    fields = path.split(".")
    for field in fields[:-1]:
        target = target[int(field)] if isinstance(target, list) else target[field]
    if isinstance(target, list):
        target[int(fields[-1])] = value
    else:
        target[fields[-1]] = value


def test_shared_contract_binds_all_evidence_to_one_episode() -> None:
    result = qualify_visual_acceptance(**_proof())
    assert result["schema"] == ACCEPTANCE_SCHEMA
    assert result["qualified"] is True
    assert result["actions"] == {
        "source": "replay_input",
        "source_steps": 20,
        "executed_steps": 20,
        "padding_steps": 0,
    }
    assert result["episode"]["native_success"] is True
    assert result["capture"]["captured_action_steps"] == 20
    assert result["video"]["progress_action_steps"]["total"] == 20


@pytest.mark.parametrize(
    "path,value,reason",
    [
        ("evidence.trajectory.nonzero_actions", False, "exact nonzero actions"),
        ("evidence.execution.source_steps", 19, "exact nonzero actions"),
        ("evidence.execution.prepared_steps", 21, "exact nonzero actions"),
        ("evidence.execution.action_padding_steps", 1, "without padding"),
        ("summary.successes", 0, "successful native scored episode"),
        ("summary.metrics.success_rate", 0.0, "successful native scored episode"),
        ("ground_truth.successes", 0, "successful native scored episode"),
        ("ground_truth.episodes.0.success", False, "successful native scored episode"),
        ("ground_truth.episodes.0.episode", "demo_1", "registered native task progress"),
        ("ground_truth.task_progress_adapter", None, "registered native task progress"),
        ("ground_truth.task_motion.environment", "other", "registered native task progress"),
        ("ground_truth.task_motion.visual_progress_qualified", False, "registered native task progress"),
        ("ground_truth.task_motion.progress_interval.total_action_steps", 19, "span the scored episode"),
        ("capture.captured_action_steps", 19, "span the scored episode"),
        ("capture.terminal.action_step", 19, "span the scored episode"),
        ("video.frame_count", 19, "bound to native task progress"),
        ("video.motion.meaningful", False, "bound to native task progress"),
        ("video.motion.analysis_interval.action_steps.total", 19, "bound to native task progress"),
    ],
)
def test_partial_or_mismatched_proofs_cannot_qualify(path, value, reason) -> None:
    proof = copy.deepcopy(_proof())
    _set_path(proof, path, value)
    with pytest.raises(IsaacArenaError, match=reason):
        qualify_visual_acceptance(**proof)


def test_native_policy_actions_use_the_same_scored_horizon() -> None:
    proof = _proof()
    proof["policy_type"] = "rsl_rl"
    proof["evidence"] = None
    result = qualify_visual_acceptance(**proof)
    assert result["actions"] == {
        "source": "native_policy_actions",
        "executed_steps": 20,
        "padding_steps": 0,
    }


def test_task_specific_progress_is_explicitly_registered() -> None:
    adapter = task_progress_adapter("gr1_open_microwave")
    assert adapter is not None
    assert adapter.name == "arena.open-door.revolute-joint.v1"
    assert adapter.signal_names == ("revolute_joint_state",)
    assert task_progress_adapter("gr1_turn_stand_mixer_knob") is None
    assert task_progress_capabilities() == [
        {
            "name": "arena.open-door.revolute-joint.v1",
            "environment": "gr1_open_microwave",
            "signal_names": ["revolute_joint_state"],
            "thresholds": {
                "final_openness_greater_than": 0.8,
                "minimum_peak_minus_initial_openness": 0.5,
            },
        }
    ]
