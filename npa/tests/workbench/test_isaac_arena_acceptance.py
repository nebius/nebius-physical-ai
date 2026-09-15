"""The shared visual contract rejects independently plausible partial proofs."""

from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest

from npa.workbench.isaac_arena.acceptance import (
    ACCEPTANCE_SCHEMA,
    qualify_visual_acceptance,
)
from npa.workbench.isaac_arena.action_evidence import (
    ACTION_EVIDENCE_SCHEMA,
    action_sequence_evidence,
)
from npa.workbench.isaac_arena.errors import IsaacArenaError
from npa.workbench.isaac_arena.task_progress import (
    task_progress_adapter,
    task_progress_capabilities,
)
from npa.workbench.isaac_arena.video_evidence import (
    EVIDENCE_FILTER,
    EVIDENCE_PLAYBACK_RATE,
)


def _proof(length: int = 20, source_length: int | None = None) -> dict:
    source_length = source_length or length
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
        "visual_interval_strategy": "task_progress_with_leading_context",
        "visual_context_steps": 30,
        "visual_progress_signal": "monotonic_structural_change",
        "visual_progress_region": {
            "name": "microwave_door_workspace",
            "left": 0.25,
            "top": 0.3,
            "right": 0.7,
            "bottom": 0.95,
        },
        "visual_association_radius_fraction": 0.03,
        "visual_interval": {
            "start_action_step": 0,
            "end_action_step": length,
            "total_action_steps": length,
        },
        "video_capture": {
            "initial_action_step": 0,
            "action_steps": list(range(1, length + 1)),
            "terminal_action_step": length,
        },
    }
    prepared_actions = np.arange(source_length * 3, dtype=np.float32).reshape(
        source_length, 3
    )
    prepared_sequence = action_sequence_evidence(prepared_actions)
    executed_sequence = action_sequence_evidence(prepared_actions[:length])
    executed_actions = {
        **executed_sequence,
        "schema": ACTION_EVIDENCE_SCHEMA,
        "source": "upstream policy.get_action to env.step boundary",
        "recording_phase": "after_native_env_step_return",
        "policy_type": "replay",
        "executed_steps": length,
        "action_steps": list(range(1, length + 1)),
        "synthetic_padding_steps": 0,
        "raw_actions_retained": False,
        "file": {
            "path": "simulator-action-evidence.json",
            "bytes": 4096,
            "sha256": "9" * 64,
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
                "source_steps": source_length,
                "prepared_steps": source_length,
                "action_padding_steps": 0,
                "prepared_sha256": "6" * 64,
                "prepared_action_sequence": prepared_sequence,
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
            "action_evidence": executed_actions,
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
                    "legacy_mode_enabled": True,
                    "rt2_enabled": False,
                    "path_tracing_enabled": False,
                    "antialiasing": "TAA",
                    "dlss_execution_mode": "quality",
                    "dl_denoiser_enabled": True,
                    "frame_generation_enabled": False,
                    "minimum_settling_renders": 8,
                    "stochastic_accumulation": False,
                    "readback_phase": "after_final_accepted_render",
                },
            },
            "terminal_frame_comparison": {"source_frame_index": length - 1},
            "source_mp4_sha256": "a" * 64,
        },
        "video": {
            "frame_count": length,
            "sha256": "e" * 64,
            "derivation": {
                "kind": "ffmpeg_spatiotemporal_denoise",
                "filter": EVIDENCE_FILTER,
                "playback_rate": EVIDENCE_PLAYBACK_RATE,
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
                "executed_action_evidence_sha256": "9" * 64,
                "executed_action_sequence_sha256": executed_sequence["sequence_sha256"],
                "simulator_ground_truth_sha256": ["7" * 64],
            },
            "motion": {
                "meaningful": True,
                "required_progress_overlap": True,
                "progress_overlapping_frame_pairs": 1,
                "required_progress_spatial_binding": True,
                "progress_spatially_bound_frame_pairs": 1,
                "progress_association_radius_pixels": 5,
                "analysis_interval": {
                    "source": "simulator_ground_truth",
                    "action_steps": {
                        "start": 0,
                        "end": length,
                        "total": length,
                    },
                },
            },
            "frame_evidence": {
                "motion_pair": {
                    "source_frame_support": {"start": 0, "end_exclusive": 16},
                    "coherent_blocks": 2,
                    "coherent_block_origins": [
                        {"row": 40, "column": 40},
                        {"row": 40, "column": 48},
                    ],
                },
                "samples": {
                    role: {
                        "decoded_luma_sha256": character * 64,
                        "source_frame_index": frame,
                        "timestamp_seconds": frame / 15,
                    }
                    for role, character, frame in zip(
                        (
                            "first",
                            "last",
                            "motion_previous",
                            "motion_current",
                            "motion_continuation",
                        ),
                        "f1234",
                        (0, length - 1, 1, 6, 11),
                        strict=True,
                    )
                },
            },
            "progress_change": {
                "signal": "monotonic_structural_change",
                "meaningful": True,
                "decoded_samples": length - 1,
                "association_radius_fraction": 0.03,
                "association_radius_pixels": 5,
                "trending_pixels": 12,
                "largest_connected_trending_region_pixels": 8,
                "largest_connected_region_bounds": {
                    "top": 40,
                    "left": 52,
                    "bottom_exclusive": 44,
                    "right_exclusive": 56,
                },
                "task_region": {
                    "name": "microwave_door_workspace",
                    "left": 0.25,
                    "top": 0.3,
                    "right": 0.7,
                    "bottom": 0.95,
                    "sample_bounds": {
                        "top": 27,
                        "left": 40,
                        "bottom_exclusive": 86,
                        "right_exclusive": 112,
                    },
                },
                "thresholds": {
                    "minimum_projected_luma_delta": 6.0,
                    "minimum_linear_r_squared": 0.94,
                    "minimum_connected_pixels": 8,
                },
                "analysis_interval": {
                    "source": "simulator_ground_truth",
                    "signal_horizon": "exact_task_progress",
                    "action_steps": {
                        "start": 2,
                        "end": length,
                        "total": length,
                    },
                    "source_frame_indices": {
                        "start": 1,
                        "end_exclusive": length,
                    },
                },
                "frame_evidence": {
                    "first": {
                        "source_frame_index": 1,
                        "timestamp_seconds": 1 / 15,
                        "decoded_luma_sha256": "2" * 64,
                    },
                    "last": {
                        "source_frame_index": length - 1,
                        "timestamp_seconds": (length - 1) / 15,
                        "decoded_luma_sha256": "1" * 64,
                    },
                },
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
    proof = _proof()
    result = qualify_visual_acceptance(**proof)
    assert result["schema"] == ACCEPTANCE_SCHEMA
    assert result["qualified"] is True
    assert result["actions"] == {
        "source": "replay_input",
        "source_steps": 20,
        "executed_steps": 20,
        "unexecuted_source_steps": 0,
        "execution_stop": "native_scored_episode_terminal",
        "padding_steps": 0,
        "sequence_sha256": proof["ground_truth"]["action_evidence"]["sequence_sha256"],
        "evidence_sha256": "9" * 64,
        "trailing_held_action_fraction": 0.05,
        "maximum_trailing_held_action_fraction": 0.25,
    }
    assert result["episode"]["native_success"] is True
    assert result["capture"]["captured_action_steps"] == 20
    assert result["video"]["visual_action_steps"]["total"] == 20


def test_native_success_may_end_before_varied_replay_source_horizon() -> None:
    proof = _proof(length=20, source_length=30)
    result = qualify_visual_acceptance(**proof)
    assert result["actions"]["source_steps"] == 30
    assert result["actions"]["executed_steps"] == 20
    assert result["actions"]["unexecuted_source_steps"] == 10
    assert result["actions"]["execution_stop"] == "native_scored_episode_terminal"
    assert (
        result["actions"]["sequence_sha256"]
        == proof["ground_truth"]["action_evidence"]["sequence_sha256"]
    )


def test_progress_interval_may_end_before_full_scored_capture_horizon() -> None:
    proof = _proof()
    proof["ground_truth"]["task_motion"]["progress_interval"]["end_action_step"] = 10
    proof["ground_truth"]["task_motion"]["visual_interval"]["end_action_step"] = 10
    proof["video"]["motion"]["analysis_interval"]["action_steps"]["end"] = 10
    proof["video"]["frame_evidence"]["samples"]["last"]["source_frame_index"] = 9
    proof["video"]["frame_evidence"]["motion_pair"]["source_frame_support"][
        "end_exclusive"
    ] = 10
    for role, frame in zip(
        ("motion_previous", "motion_current", "motion_continuation"),
        (1, 4, 7),
        strict=True,
    ):
        proof["video"]["frame_evidence"]["samples"][role]["source_frame_index"] = frame
    proof["video"]["progress_change"]["analysis_interval"]["action_steps"]["end"] = 10
    proof["video"]["progress_change"]["analysis_interval"]["source_frame_indices"][
        "end_exclusive"
    ] = 10
    proof["video"]["progress_change"]["decoded_samples"] = 9
    proof["video"]["progress_change"]["frame_evidence"]["last"][
        "source_frame_index"
    ] = 9
    result = qualify_visual_acceptance(**proof)
    assert result["task_progress"]["progress_interval"] == {
        "start_action_step": 2,
        "end_action_step": 10,
        "total_action_steps": 20,
    }
    assert result["video"]["visual_action_steps"] == {
        "start": 0,
        "end": 10,
        "total": 20,
    }
    assert result["video"]["progress_action_steps"] == {
        "start": 2,
        "end": 10,
        "total": 20,
    }
    assert result["capture"]["terminal_action_step"] == 20


def test_coherent_motion_before_native_progress_cannot_qualify_video() -> None:
    proof = _proof(length=45)
    task_motion = proof["ground_truth"]["task_motion"]
    task_motion["progress_interval"]["start_action_step"] = 30
    progress_change = proof["video"]["progress_change"]
    progress_change["analysis_interval"]["action_steps"]["start"] = 30
    progress_change["analysis_interval"]["source_frame_indices"] = {
        "start": 29,
        "end_exclusive": 45,
    }
    progress_change["decoded_samples"] = 16
    progress_change["frame_evidence"]["first"]["source_frame_index"] = 29
    frame_evidence = proof["video"]["frame_evidence"]
    frame_evidence["motion_pair"]["source_frame_support"] = {
        "start": 3,
        "end_exclusive": 20,
    }
    for role, frame in zip(
        ("motion_previous", "motion_current", "motion_continuation"),
        (5, 10, 15),
        strict=True,
    ):
        frame_evidence["samples"][role]["source_frame_index"] = frame
    with pytest.raises(IsaacArenaError, match="bound to native task progress"):
        qualify_visual_acceptance(**proof)


def test_spatially_disjoint_motion_and_progress_cannot_qualify_video() -> None:
    proof = _proof(length=45)
    task_motion = proof["ground_truth"]["task_motion"]
    task_motion["progress_interval"]["start_action_step"] = 30
    progress_change = proof["video"]["progress_change"]
    progress_change["analysis_interval"]["action_steps"]["start"] = 30
    progress_change["analysis_interval"]["source_frame_indices"] = {
        "start": 29,
        "end_exclusive": 45,
    }
    progress_change["decoded_samples"] = 16
    progress_change["frame_evidence"]["first"].update(
        source_frame_index=29,
        timestamp_seconds=29 / 15,
        decoded_luma_sha256="8" * 64,
    )
    progress_change["largest_connected_region_bounds"] = {
        "top": 70,
        "left": 100,
        "bottom_exclusive": 74,
        "right_exclusive": 104,
    }
    frame_evidence = proof["video"]["frame_evidence"]
    frame_evidence["motion_pair"].update(
        source_frame_support={"start": 28, "end_exclusive": 45},
        coherent_block_origins=[
            {"row": 30, "column": 40},
            {"row": 30, "column": 48},
        ],
    )
    for role, frame in zip(
        ("motion_previous", "motion_current", "motion_continuation"),
        (30, 35, 40),
        strict=True,
    ):
        frame_evidence["samples"][role].update(
            source_frame_index=frame,
            timestamp_seconds=frame / 15,
        )
    with pytest.raises(IsaacArenaError, match="bound to native task progress"):
        qualify_visual_acceptance(**proof)


def test_duplicate_frame_index_with_different_hash_cannot_be_spliced() -> None:
    proof = _proof()
    proof["video"]["progress_change"]["frame_evidence"]["last"][
        "decoded_luma_sha256"
    ] = "9" * 64
    with pytest.raises(IsaacArenaError, match="bound to native task progress"):
        qualify_visual_acceptance(**proof)


def test_duplicate_coherent_block_origins_cannot_be_spliced() -> None:
    proof = _proof()
    origins = proof["video"]["frame_evidence"]["motion_pair"]["coherent_block_origins"]
    origins[1] = dict(origins[0])
    with pytest.raises(IsaacArenaError, match="bound to native task progress"):
        qualify_visual_acceptance(**proof)


def test_nonadjacent_coherent_block_origins_cannot_be_spliced() -> None:
    proof = _proof()
    proof["video"]["frame_evidence"]["motion_pair"]["coherent_block_origins"][1] = {
        "row": 64,
        "column": 96,
    }
    with pytest.raises(IsaacArenaError, match="bound to native task progress"):
        qualify_visual_acceptance(**proof)


def test_connected_pixel_count_must_fit_component_bounds() -> None:
    proof = _proof()
    proof["video"]["progress_change"]["largest_connected_region_bounds"] = {
        "top": 40,
        "left": 52,
        "bottom_exclusive": 41,
        "right_exclusive": 53,
    }
    with pytest.raises(IsaacArenaError, match="bound to native task progress"):
        qualify_visual_acceptance(**proof)


@pytest.mark.parametrize(
    "path,value,reason",
    [
        ("evidence.trajectory.nonzero_actions", False, "exact nonzero actions"),
        ("evidence.execution.source_steps", 19, "exact nonzero actions"),
        ("evidence.execution.prepared_steps", 21, "exact nonzero actions"),
        ("evidence.execution.action_padding_steps", 1, "without padding"),
        (
            "ground_truth.action_evidence.executed_steps",
            19,
            "measured nonzero varied actions",
        ),
        (
            "ground_truth.action_evidence.synthetic_padding_steps",
            1,
            "without padding",
        ),
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
            "invalid visual interval",
        ),
        (
            "ground_truth.task_motion.visual_interval_strategy",
            "task_progress",
            "registered native task progress",
        ),
        (
            "ground_truth.task_motion.visual_interval.start_action_step",
            2,
            "registered native task progress",
        ),
        ("capture.captured_action_steps", 19, "span the scored episode"),
        ("capture.terminal.action_step", 19, "span the scored episode"),
        (
            "capture.physics_freeze.verified_capture_count",
            20,
            "span the scored episode",
        ),
        (
            "capture.physics_freeze.rendering.legacy_mode_enabled",
            False,
            "span the scored episode",
        ),
        (
            "capture.physics_freeze.rendering.rt2_enabled",
            True,
            "span the scored episode",
        ),
        (
            "capture.physics_freeze.rendering.path_tracing_enabled",
            True,
            "span the scored episode",
        ),
        (
            "capture.physics_freeze.rendering.readback_phase",
            "before_render",
            "span the scored episode",
        ),
        (
            "capture.physics_freeze.rendering.stochastic_accumulation",
            True,
            "span the scored episode",
        ),
        (
            "capture.physics_freeze.rendering.minimum_settling_renders",
            1,
            "span the scored episode",
        ),
        ("capture.source_mp4_sha256", "invalid", "span the scored episode"),
        ("video.frame_count", 19, "bound to native task progress"),
        ("video.derivation.playback_rate", 1.0, "bound to native task progress"),
        ("video.derivation.source_sha256", "f" * 64, "bound to native task progress"),
        ("video.binding.episode", "demo_1", "bound to native task progress"),
        ("video.binding.input_sha256", "8" * 64, "bound to native task progress"),
        (
            "video.binding.execution_input_sha256",
            "8" * 64,
            "bound to native task progress",
        ),
        (
            "video.binding.executed_action_evidence_sha256",
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
        (
            "video.motion.progress_overlapping_frame_pairs",
            0,
            "bound to native task progress",
        ),
        (
            "video.progress_change.meaningful",
            False,
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
    proof["evidence"] = {"kind": "rsl_rl_checkpoint", "sha256": "8" * 64}
    proof["ground_truth"]["action_evidence"]["policy_type"] = "rsl_rl"
    proof["video"]["binding"]["policy_type"] = "rsl_rl"
    proof["video"]["binding"]["input_sha256"] = "8" * 64
    proof["video"]["binding"]["execution_input_sha256"] = "8" * 64
    result = qualify_visual_acceptance(**proof)
    assert result["actions"] == {
        "source": "native_policy_actions",
        "executed_steps": 20,
        "padding_steps": 0,
        "sequence_sha256": proof["ground_truth"]["action_evidence"]["sequence_sha256"],
        "evidence_sha256": "9" * 64,
        "trailing_held_action_fraction": 0.05,
        "maximum_trailing_held_action_fraction": 0.25,
    }


def test_replay_executed_actions_must_match_exact_prepared_prefix() -> None:
    proof = _proof(length=20, source_length=30)
    replacement = np.arange(60, dtype=np.float32).reshape(20, 3) + 1000
    observed = action_sequence_evidence(replacement)
    action_evidence = proof["ground_truth"]["action_evidence"]
    action_evidence.update(observed)
    action_evidence["schema"] = ACTION_EVIDENCE_SCHEMA
    proof["video"]["binding"]["executed_action_sequence_sha256"] = observed[
        "sequence_sha256"
    ]
    with pytest.raises(IsaacArenaError, match="exact nonzero actions"):
        qualify_visual_acceptance(**proof)


def test_unexecuted_dominant_held_source_tail_cannot_hide_after_success() -> None:
    proof = _proof(length=80, source_length=250)
    values = np.arange(250 * 3, dtype=np.float32).reshape(250, 3)
    values[80:] = values[79]
    proof["evidence"]["execution"]["prepared_action_sequence"] = (
        action_sequence_evidence(values)
    )
    with pytest.raises(IsaacArenaError, match="replay prepared actions"):
        qualify_visual_acceptance(**proof)


def test_near_identical_jitter_cannot_hide_a_dominant_held_source_tail() -> None:
    proof = _proof(length=80, source_length=250)
    values = np.arange(250 * 3, dtype=np.float64).reshape(250, 3)
    values[80:] = values[79] + (np.arange(170, dtype=np.float64)[:, None] + 1) * 1e-9
    prepared = action_sequence_evidence(values)
    assert prepared["distinct_action_steps"] == 250
    assert prepared["trailing_held_action_steps"] == 171
    proof["evidence"]["execution"]["prepared_action_sequence"] = prepared
    with pytest.raises(IsaacArenaError, match="replay prepared actions"):
        qualify_visual_acceptance(**proof)


def test_malformed_delta_summary_cannot_hide_exact_repeated_source_tail() -> None:
    proof = _proof(length=80, source_length=250)
    values = np.arange(250 * 3, dtype=np.float64).reshape(250, 3)
    values[80:] = values[79]
    prepared = action_sequence_evidence(values)
    prepared["interstep_delta_abs_max"] = [3.0] * 249
    prepared["action_step_delta_abs_max"] = 3.0
    prepared["trailing_held_action_steps"] = 1
    prepared["trailing_held_action_fraction"] = 1 / 250
    proof["evidence"]["execution"]["prepared_action_sequence"] = prepared
    with pytest.raises(IsaacArenaError, match="replay prepared actions"):
        qualify_visual_acceptance(**proof)


def test_rsl_actions_with_near_identical_jitter_tail_cannot_qualify() -> None:
    proof = _proof(length=250)
    values = np.arange(250 * 3, dtype=np.float64).reshape(250, 3)
    values[80:] = values[79] + (np.arange(170, dtype=np.float64)[:, None] + 1) * 1e-9
    observed = action_sequence_evidence(values)
    assert observed["distinct_action_steps"] == 250
    assert observed["trailing_held_action_steps"] == 171
    action_evidence = proof["ground_truth"]["action_evidence"]
    action_evidence.update(observed)
    action_evidence.update(
        schema=ACTION_EVIDENCE_SCHEMA,
        source="upstream policy.get_action to env.step boundary",
        recording_phase="after_native_env_step_return",
        policy_type="rsl_rl",
        executed_steps=250,
        action_steps=list(range(1, 251)),
        synthetic_padding_steps=0,
        raw_actions_retained=False,
    )
    proof["policy_type"] = "rsl_rl"
    proof["evidence"] = {"kind": "rsl_rl_checkpoint", "sha256": "8" * 64}
    proof["video"]["binding"].update(
        policy_type="rsl_rl",
        input_sha256="8" * 64,
        execution_input_sha256="8" * 64,
        executed_action_sequence_sha256=observed["sequence_sha256"],
    )
    with pytest.raises(IsaacArenaError, match="nonzero varied actions"):
        qualify_visual_acceptance(**proof)


@pytest.mark.parametrize("kind", ["zero", "constant", "mutation", "count"])
def test_zero_constant_mutated_or_miscounted_actions_cannot_qualify(kind) -> None:
    proof = _proof()
    length = proof["ground_truth"]["episodes"][0]["episode_length"]
    if kind == "zero":
        values = np.zeros((length, 3), dtype=np.float32)
    elif kind == "constant":
        values = np.ones((length, 3), dtype=np.float32)
    else:
        values = np.arange(length * 3, dtype=np.float32).reshape(length, 3)
    observed = action_sequence_evidence(values)
    action_evidence = proof["ground_truth"]["action_evidence"]
    action_evidence.update(observed)
    action_evidence["schema"] = ACTION_EVIDENCE_SCHEMA
    if kind == "mutation":
        action_evidence["sequence_sha256"] = "1" * 64
    elif kind == "count":
        action_evidence["executed_steps"] = length - 1
    elif kind in {"zero", "constant"}:
        proof["evidence"]["execution"]["prepared_action_sequence"] = observed
    proof["video"]["binding"]["executed_action_sequence_sha256"] = action_evidence[
        "sequence_sha256"
    ]
    with pytest.raises(IsaacArenaError, match="actions|replay"):
        qualify_visual_acceptance(**proof)


@pytest.mark.parametrize("policy_type", ["replay", "rsl_rl"])
def test_prior_shaped_varying_prefix_with_dominant_held_tail_cannot_qualify(
    policy_type,
) -> None:
    proof = _proof(length=250)
    values = np.arange(250 * 3, dtype=np.float32).reshape(250, 3)
    values[80:] = values[79]
    observed = action_sequence_evidence(values)
    assert observed["trailing_held_action_steps"] == 171
    action_evidence = proof["ground_truth"]["action_evidence"]
    action_evidence.update(observed)
    action_evidence["schema"] = ACTION_EVIDENCE_SCHEMA
    proof["video"]["binding"]["executed_action_sequence_sha256"] = observed[
        "sequence_sha256"
    ]
    if policy_type == "replay":
        proof["evidence"]["execution"]["prepared_action_sequence"] = observed
    else:
        proof["policy_type"] = "rsl_rl"
        proof["evidence"] = {"kind": "rsl_rl_checkpoint", "sha256": "8" * 64}
        action_evidence["policy_type"] = "rsl_rl"
        proof["video"]["binding"].update(
            policy_type="rsl_rl",
            input_sha256="8" * 64,
            execution_input_sha256="8" * 64,
        )
    with pytest.raises(IsaacArenaError, match="actions|replay"):
        qualify_visual_acceptance(**proof)


def test_task_specific_progress_is_explicitly_registered() -> None:
    adapter = task_progress_adapter("gr1_open_microwave")
    assert adapter is not None
    assert adapter.name == "arena.open-door.revolute-joint.v1"
    assert adapter.supported_policy_types == ("replay", "rsl_rl")
    assert adapter.signal_names == ("revolute_joint_state",)
    assert task_progress_adapter("gr1_turn_stand_mixer_knob") is None
    assert task_progress_capabilities() == [
        {
            "name": "arena.open-door.revolute-joint.v1",
            "environment": "gr1_open_microwave",
            "supported_policy_types": ["replay", "rsl_rl"],
            "maximum_trailing_held_action_fraction": 0.25,
            "visual_interval_strategy": "task_progress_with_leading_context",
            "visual_context_steps": 30,
            "visual_progress_signal": "monotonic_structural_change",
            "visual_progress_region": {
                "name": "microwave_door_workspace",
                "left": 0.25,
                "top": 0.3,
                "right": 0.7,
                "bottom": 0.95,
            },
            "visual_association_radius_fraction": 0.03,
            "signal_names": ["revolute_joint_state"],
            "thresholds": {
                "final_openness_greater_than": 0.8,
                "minimum_peak_minus_initial_openness": 0.5,
            },
        }
    ]


def test_task_registration_rejects_unregistered_policy_pairing(monkeypatch) -> None:
    adapter = task_progress_adapter("gr1_open_microwave")
    assert adapter is not None
    replay_only = replace(adapter, supported_policy_types=("replay",))
    monkeypatch.setattr(
        "npa.workbench.isaac_arena.acceptance.task_progress_adapter",
        lambda _environment: replay_only,
    )
    proof = _proof()
    proof["policy_type"] = "rsl_rl"
    proof["evidence"] = {"kind": "rsl_rl_checkpoint", "sha256": "8" * 64}
    proof["ground_truth"]["action_evidence"]["policy_type"] = "rsl_rl"
    proof["video"]["binding"].update(
        policy_type="rsl_rl",
        input_sha256="8" * 64,
        execution_input_sha256="8" * 64,
    )
    with pytest.raises(IsaacArenaError, match="registered native task progress"):
        qualify_visual_acceptance(**proof)
