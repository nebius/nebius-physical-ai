"""Bind Arena actions, native outcomes, task progress, and video evidence."""

from __future__ import annotations

import math
from typing import Any

from .action_evidence import (
    validate_action_evidence,
    validate_prepared_action_sequence,
)
from .errors import IsaacArenaError
from .task_progress import task_progress_adapter

ACCEPTANCE_SCHEMA = "npa.isaac-arena.visual-acceptance.v1"


def _positive_integer(value: Any) -> bool:
    return type(value) is int and value > 0


def _sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _native_episode(summary: dict, ground_truth: dict) -> tuple[dict, int]:
    metric_rate = (summary.get("metrics") or {}).get("success_rate")
    valid_summary = (
        summary.get("episodes") == 1
        and summary.get("successes") == 1
        and summary.get("success_rate") == 1.0
        and isinstance(metric_rate, (int, float))
        and not isinstance(metric_rate, bool)
        and math.isfinite(float(metric_rate))
        and float(metric_rate) > 0
        and ground_truth.get("successes") == 1
    )
    episodes = ground_truth.get("episodes")
    if not valid_summary or not isinstance(episodes, list) or len(episodes) != 1:
        raise IsaacArenaError(
            "visual acceptance requires one successful native scored episode"
        )
    episode = episodes[0]
    if not isinstance(episode, dict):
        raise IsaacArenaError(
            "visual acceptance requires one successful native scored episode"
        )
    length = episode.get("episode_length")
    if episode.get("success") is not True or not _positive_integer(length):
        raise IsaacArenaError(
            "visual acceptance requires one successful native scored episode"
        )
    return episode, length


def _task_progress(
    environment: str, policy_type: str, ground_truth: dict, episode: dict
) -> dict:
    adapter = ground_truth.get("task_progress_adapter")
    motion = ground_truth.get("task_motion")
    registered = task_progress_adapter(environment)
    length = episode.get("episode_length")
    capture = motion.get("video_capture") if isinstance(motion, dict) else None
    capture_bound = (
        isinstance(capture, dict)
        and capture.get("initial_action_step") == 0
        and capture.get("terminal_action_step") == length
        and capture.get("action_steps") == list(range(1, length + 1))
    )
    valid = (
        registered is not None
        and policy_type in registered.supported_policy_types
        and adapter == registered.name
        and isinstance(motion, dict)
        and motion.get("adapter") == adapter
        and motion.get("environment") == environment
        and motion.get("episode") == episode.get("episode")
        and motion.get("episode_length") == episode.get("episode_length")
        and motion.get("task_success") is True
        and motion.get("visual_progress_qualified") is True
        and capture_bound
    )
    if not valid:
        raise IsaacArenaError(
            "visual acceptance requires registered native task progress"
        )
    return motion


def _action_contract(
    environment: str,
    policy_type: str,
    evidence: dict | None,
    ground_truth: dict,
    episode_length: int,
) -> dict:
    adapter = task_progress_adapter(environment)
    if adapter is None or policy_type not in adapter.supported_policy_types:
        raise IsaacArenaError(
            "visual acceptance requires registered native task progress"
        )
    maximum_held_fraction = adapter.maximum_trailing_identical_action_fraction
    observed = validate_action_evidence(
        ground_truth.get("action_evidence"),
        policy_type=policy_type,
        expected_steps=episode_length,
        require_nonzero_varied=True,
        maximum_trailing_identical_fraction=maximum_held_fraction,
    )
    action_file = observed.get("file")
    if not isinstance(action_file, dict) or not _sha256(action_file.get("sha256")):
        raise IsaacArenaError("visual acceptance requires measured executed actions")
    if policy_type == "rsl_rl":
        if (
            not isinstance(evidence, dict)
            or evidence.get("kind") != "rsl_rl_checkpoint"
            or not _sha256(evidence.get("sha256"))
        ):
            raise IsaacArenaError(
                "visual acceptance requires a hash-bound native policy checkpoint"
            )
        return {
            "source": "native_policy_actions",
            "executed_steps": episode_length,
            "padding_steps": 0,
            "sequence_sha256": observed["sequence_sha256"],
            "evidence_sha256": action_file["sha256"],
            "trailing_identical_action_fraction": observed[
                "trailing_identical_action_fraction"
            ],
            "maximum_trailing_identical_action_fraction": maximum_held_fraction,
        }
    execution = (evidence or {}).get("execution") or {}
    source_steps = execution.get("source_steps")
    prepared_steps = execution.get("prepared_steps")
    prepared = validate_prepared_action_sequence(
        execution.get("prepared_action_sequence"),
        expected_steps=episode_length,
        maximum_trailing_identical_fraction=maximum_held_fraction,
    )
    valid = (
        policy_type == "replay"
        and ((evidence or {}).get("trajectory") or {}).get("nonzero_actions") is True
        and _positive_integer(source_steps)
        and source_steps == prepared_steps == episode_length
        and execution.get("action_padding_steps") == 0
        and execution.get("strategy") == "actions_initial_state_exact_replay"
        and observed.get("action_shape") == prepared.get("action_shape")
        and observed.get("sequence_sha256") == prepared.get("sequence_sha256")
    )
    if not valid:
        raise IsaacArenaError(
            "visual acceptance requires exact nonzero actions without padding"
        )
    return {
        "source": "replay_input",
        "source_steps": source_steps,
        "executed_steps": prepared_steps,
        "padding_steps": 0,
        "sequence_sha256": observed["sequence_sha256"],
        "evidence_sha256": action_file["sha256"],
        "trailing_identical_action_fraction": observed[
            "trailing_identical_action_fraction"
        ],
        "maximum_trailing_identical_action_fraction": maximum_held_fraction,
    }


def _capture_contract(capture: dict, episode_length: int) -> dict:
    if not isinstance(capture, dict):
        raise IsaacArenaError(
            "visual acceptance capture does not span the scored episode"
        )
    terminal = capture.get("terminal")
    initial = capture.get("initial")
    sidecar = capture.get("sidecar")
    freeze = capture.get("physics_freeze")
    rendering = freeze.get("rendering") if isinstance(freeze, dict) else None
    comparison = capture.get("terminal_frame_comparison")
    valid = (
        capture.get("captured_action_steps") == episode_length
        and isinstance(initial, dict)
        and initial.get("action_step") == 0
        and _sha256(initial.get("sha256"))
        and isinstance(terminal, dict)
        and terminal.get("action_step") == episode_length
        and _sha256(terminal.get("sha256"))
        and isinstance(sidecar, dict)
        and _sha256(sidecar.get("sha256"))
        and isinstance(freeze, dict)
        and freeze.get("verified_capture_count") == episode_length + 1
        and freeze.get("simulation_advanced_during_render") is False
        and freeze.get("physics_state_changed_during_render") is False
        and isinstance(rendering, dict)
        and rendering.get("mode") == "RaytracedLighting"
        and rendering.get("antialiasing") == "FXAA"
        and rendering.get("stochastic_accumulation") is False
        and rendering.get("accumulation_renders_per_frame") == 0
        and isinstance(comparison, dict)
        and comparison.get("source_frame_index") == episode_length - 1
        and _sha256(capture.get("source_mp4_sha256"))
    )
    if not valid:
        raise IsaacArenaError(
            "visual acceptance capture does not span the scored episode"
        )
    return {
        "captured_action_steps": episode_length,
        "terminal_action_step": terminal["action_step"],
        "source_mp4_sha256": capture["source_mp4_sha256"],
        "initial_png_sha256": initial["sha256"],
        "terminal_png_sha256": terminal["sha256"],
        "sidecar_sha256": sidecar["sha256"],
        "verified_capture_count": freeze["verified_capture_count"],
        "rendering": rendering,
    }


def _video_contract(
    video: dict,
    interval: dict,
    episode: dict,
    environment: str,
    policy_type: str,
    evidence: dict | None,
    ground_truth: dict,
    source_mp4_sha256: str,
) -> dict:
    if not isinstance(video, dict):
        raise IsaacArenaError(
            "visual acceptance video is not bound to native task progress"
        )
    motion = video.get("motion")
    analysis = motion.get("analysis_interval") if isinstance(motion, dict) else None
    derivation = video.get("derivation")
    binding = video.get("binding")
    frame_evidence = video.get("frame_evidence")
    samples = (
        frame_evidence.get("samples") if isinstance(frame_evidence, dict) else None
    )
    episode_length = episode["episode_length"]
    source_hash = str((evidence or {}).get("sha256") or "")
    execution_hash = str(
        ((evidence or {}).get("execution") or {}).get("prepared_sha256") or source_hash
    )
    ground_truth_files = ground_truth.get("files")
    action_evidence = ground_truth.get("action_evidence") or {}
    action_file = action_evidence.get("file") or {}
    ground_truth_hashes = (
        [item.get("sha256") for item in ground_truth_files]
        if isinstance(ground_truth_files, list)
        and ground_truth_files
        and all(isinstance(item, dict) for item in ground_truth_files)
        else []
    )
    expected = {
        "start": interval.get("start_action_step"),
        "end": interval.get("end_action_step"),
        "total": interval.get("total_action_steps"),
    }
    valid = (
        isinstance(motion, dict)
        and video.get("frame_count") == episode_length
        and motion.get("meaningful") is True
        and isinstance(analysis, dict)
        and analysis.get("source") == "simulator_ground_truth"
        and analysis.get("action_steps") == expected
        and isinstance(derivation, dict)
        and derivation.get("source_sha256") == source_mp4_sha256
        and derivation.get("changes_simulator_outcome") is False
        and _sha256(video.get("sha256"))
        and isinstance(binding, dict)
        and binding.get("environment") == environment
        and binding.get("policy_type") == policy_type
        and binding.get("episode") == episode.get("episode")
        and binding.get("action_steps") == episode_length
        and isinstance(binding.get("run_id"), str)
        and bool(binding["run_id"])
        and isinstance(binding.get("upstream_run_directory"), str)
        and bool(binding["upstream_run_directory"])
        and binding.get("input_sha256") == source_hash
        and binding.get("execution_input_sha256") == execution_hash
        and binding.get("executed_action_evidence_sha256") == action_file.get("sha256")
        and _sha256(binding.get("executed_action_evidence_sha256"))
        and binding.get("executed_action_sequence_sha256")
        == action_evidence.get("sequence_sha256")
        and _sha256(binding.get("executed_action_sequence_sha256"))
        and ground_truth_hashes
        and all(_sha256(item) for item in ground_truth_hashes)
        and binding.get("simulator_ground_truth_sha256") == ground_truth_hashes
        and isinstance(samples, dict)
        and all(
            isinstance(samples.get(role), dict)
            and _sha256(samples[role].get("decoded_luma_sha256"))
            for role in (
                "first",
                "last",
                "motion_previous",
                "motion_current",
                "motion_continuation",
            )
        )
    )
    if not valid:
        raise IsaacArenaError(
            "visual acceptance video is not bound to native task progress"
        )
    return {
        "frame_count": video["frame_count"],
        "motion": "noise_resistant_coherent_scene_motion",
        "progress_action_steps": expected,
        "source_mp4_sha256": source_mp4_sha256,
        "evidence_mp4_sha256": video["sha256"],
        "frame_evidence": frame_evidence,
    }


def qualify_visual_acceptance(
    *,
    environment: str,
    policy_type: str,
    evidence: dict | None,
    summary: dict,
    ground_truth: dict,
    capture: dict,
    video: dict,
) -> dict[str, Any]:
    """Require every proof dimension to describe one successful episode."""

    if policy_type not in {"replay", "rsl_rl"}:
        raise IsaacArenaError(
            "task-qualified visual acceptance requires a nonzero policy adapter"
        )
    episode, length = _native_episode(summary, ground_truth)
    progress = _task_progress(environment, policy_type, ground_truth, episode)
    interval = progress.get("progress_interval")
    if not isinstance(interval, dict) or interval.get("total_action_steps") != length:
        raise IsaacArenaError(
            "visual acceptance progress does not span the scored episode"
        )
    capture_contract = _capture_contract(capture, length)
    return {
        "schema": ACCEPTANCE_SCHEMA,
        "qualified": True,
        "environment": environment,
        "actions": _action_contract(
            environment, policy_type, evidence, ground_truth, length
        ),
        "episode": {
            "name": episode["episode"],
            "length": length,
            "native_success": True,
            "success_rate": summary["success_rate"],
        },
        "task_progress": progress,
        "capture": capture_contract,
        "video": _video_contract(
            video,
            interval,
            episode,
            environment,
            policy_type,
            evidence,
            ground_truth,
            capture_contract["source_mp4_sha256"],
        ),
    }
