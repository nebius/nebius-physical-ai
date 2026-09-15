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
        "video_capture": {
            "initial_action_step": 0,
            "action_steps": list(range(1, length + 1)),
            "terminal_action_step": length,
        },
    }
    return {
        "environment": "gr1_open_microwave",
        "policy_type": "replay",
        "evidence": {
            "sha256": "5" * 64,
            "trajectory": {"nonzero_actions": True},
            "execution": {
                "strategy": "actions_initial_state_exact_replay",
                "source_steps": length,
                "prepared_steps": length,
                "action_padding_steps": 0,
                "prepared_sha256": "6" * 64,
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
            "files": [{"sha256": "7" * 64}],
            "task_progress_adapter": motion["adapter"],
            "task_motion": motion,
        },
        "capture": {
            "captured_action_steps": length,
            "initial": {"action_step": 0, "sha256": "b" * 64},
            "terminal": {"action_step": length, "sha256": "c" * 64},
            "sidecar": {"sha256": "d" * 64},
            "physics_freeze": {
                "verified_capture_count": length + 1,
                "simulation_advanced_during_render": False,
                "physics_state_changed_during_render": False,
                "rendering": {
                    "mode": "RaytracedLighting",
                    "antialiasing": "FXAA",
                    "stochastic_accumulation": False,
                    "accumulation_renders_per_frame": 0,
                },
            },
            "terminal_frame_comparison": {"source_frame_index": length - 1},
            "source_mp4_sha256": "a" * 64,
        },
        "video": {
            "frame_count": length,
            "sha256": "e" * 64,
            "derivation": {
                "source_sha256": "a" * 64,
                "changes_simulator_outcome": False,
            },
            "binding": {
                "run_id": "arena-proof-run",
                "upstream_run_directory": "arena-proof-run",
                "environment": "gr1_open_microwave",
                "policy_type": "replay",
                "episode": "demo_0",
                "action_steps": length,
                "input_sha256": "5" * 64,
                "execution_input_sha256": "6" * 64,
                "simulator_ground_truth_sha256": ["7" * 64],
            },
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
            "frame_evidence": {
                "samples": {
                    role: {"decoded_luma_sha256": character * 64}
                    for role, character in zip(
                        (
                            "first",
                            "last",
                            "motion_previous",
                            "motion_current",
                            "motion_continuation",
                        ),
                        "f1234",
                        strict=True,
                    )
                }
            },
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
        (
            "ground_truth.episodes.0.episode",
            "demo_1",
            "registered native task progress",
        ),
        ("ground_truth.task_progress_adapter", None, "registered native task progress"),
        (
            "ground_truth.task_motion.environment",
            "other",
            "registered native task progress",
        ),
        (
            "ground_truth.task_motion.visual_progress_qualified",
            False,
            "registered native task progress",
        ),
        (
            "ground_truth.task_motion.video_capture.terminal_action_step",
            19,
            "registered native task progress",
        ),
        (
            "ground_truth.task_motion.progress_interval.total_action_steps",
            19,
            "span the scored episode",
        ),
        ("capture.captured_action_steps", 19, "span the scored episode"),
        ("capture.terminal.action_step", 19, "span the scored episode"),
        (
            "capture.physics_freeze.verified_capture_count",
            20,
            "span the scored episode",
        ),
        (
            "capture.physics_freeze.rendering.stochastic_accumulation",
            True,
            "span the scored episode",
        ),
        (
            "capture.physics_freeze.rendering.accumulation_renders_per_frame",
            1,
            "span the scored episode",
        ),
        ("capture.source_mp4_sha256", "invalid", "span the scored episode"),
        ("video.frame_count", 19, "bound to native task progress"),
        ("video.derivation.source_sha256", "f" * 64, "bound to native task progress"),
        ("video.binding.episode", "demo_1", "bound to native task progress"),
        ("video.binding.input_sha256", "8" * 64, "bound to native task progress"),
        (
            "video.binding.execution_input_sha256",
            "8" * 64,
            "bound to native task progress",
        ),
        (
            "video.binding.simulator_ground_truth_sha256",
            ["8" * 64],
            "bound to native task progress",
        ),
        ("video.sha256", "invalid", "bound to native task progress"),
        (
            "video.frame_evidence.samples.last.decoded_luma_sha256",
            "",
            "bound to native task progress",
        ),
        ("video.motion.meaningful", False, "bound to native task progress"),
        (
            "video.motion.analysis_interval.action_steps.total",
            19,
            "bound to native task progress",
        ),
    ],
)
def test_partial_or_mismatched_proofs_cannot_qualify(path, value, reason) -> None:
    proof = copy.deepcopy(_proof())
    _set_path(proof, path, value)
    with pytest.raises(IsaacArenaError, match=reason):
        qualify_visual_acceptance(**proof)


def test_malformed_extra_ground_truth_file_cannot_be_omitted_from_binding() -> None:
    proof = _proof()
    proof["ground_truth"]["files"].append("not-a-file-record")
    with pytest.raises(IsaacArenaError, match="bound to native task progress"):
        qualify_visual_acceptance(**proof)


def test_native_policy_actions_use_the_same_scored_horizon() -> None:
    proof = _proof()
    proof["policy_type"] = "rsl_rl"
    proof["evidence"] = None
    proof["video"]["binding"]["policy_type"] = "rsl_rl"
    proof["video"]["binding"]["input_sha256"] = ""
    proof["video"]["binding"]["execution_input_sha256"] = ""
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
