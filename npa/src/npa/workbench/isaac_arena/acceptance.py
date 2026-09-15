"""Bind Arena actions, native outcomes, task progress, and video evidence."""

from __future__ import annotations

import math
from typing import Any

from .errors import IsaacArenaError
from .task_progress import task_progress_adapter

ACCEPTANCE_SCHEMA = "npa.isaac-arena.visual-acceptance.v1"


def _positive_integer(value: Any) -> bool:
    return type(value) is int and value > 0


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


def _task_progress(environment: str, ground_truth: dict, episode: dict) -> dict:
    adapter = ground_truth.get("task_progress_adapter")
    motion = ground_truth.get("task_motion")
    registered = task_progress_adapter(environment)
    valid = (
        registered is not None
        and adapter == registered.name
        and isinstance(motion, dict)
        and motion.get("adapter") == adapter
        and motion.get("environment") == environment
        and motion.get("episode") == episode.get("episode")
        and motion.get("episode_length") == episode.get("episode_length")
        and motion.get("task_success") is True
        and motion.get("visual_progress_qualified") is True
    )
    if not valid:
        raise IsaacArenaError(
            "visual acceptance requires registered native task progress"
        )
    return motion


def _action_contract(
    policy_type: str,
    evidence: dict | None,
    episode_length: int,
) -> dict:
    if policy_type == "rsl_rl":
        return {
            "source": "native_policy_actions",
            "executed_steps": episode_length,
            "padding_steps": 0,
        }
    execution = (evidence or {}).get("execution") or {}
    source_steps = execution.get("source_steps")
    prepared_steps = execution.get("prepared_steps")
    valid = (
        policy_type == "replay"
        and ((evidence or {}).get("trajectory") or {}).get("nonzero_actions") is True
        and _positive_integer(source_steps)
        and source_steps == prepared_steps == episode_length
        and execution.get("action_padding_steps") == 0
        and execution.get("strategy") == "actions_initial_state_exact_replay"
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
    }


def _capture_contract(capture: dict, episode_length: int) -> dict:
    if not isinstance(capture, dict):
        raise IsaacArenaError(
            "visual acceptance capture does not span the scored episode"
        )
    terminal = capture.get("terminal")
    valid = (
        capture.get("captured_action_steps") == episode_length
        and isinstance(terminal, dict)
        and terminal.get("action_step") == episode_length
        and isinstance(capture.get("terminal_frame_comparison"), dict)
    )
    if not valid:
        raise IsaacArenaError(
            "visual acceptance capture does not span the scored episode"
        )
    return {
        "captured_action_steps": episode_length,
        "terminal_action_step": terminal["action_step"],
        "source_mp4_sha256": capture.get("source_mp4_sha256"),
    }


def _video_contract(video: dict, interval: dict, episode_length: int) -> dict:
    if not isinstance(video, dict):
        raise IsaacArenaError(
            "visual acceptance video is not bound to native task progress"
        )
    motion = video.get("motion")
    analysis = motion.get("analysis_interval") if isinstance(motion, dict) else None
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
    )
    if not valid:
        raise IsaacArenaError(
            "visual acceptance video is not bound to native task progress"
        )
    return {
        "frame_count": video["frame_count"],
        "motion": "noise_resistant_coherent_scene_motion",
        "progress_action_steps": expected,
        "frame_evidence": video.get("frame_evidence"),
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
    progress = _task_progress(environment, ground_truth, episode)
    interval = progress.get("progress_interval")
    if not isinstance(interval, dict) or interval.get("total_action_steps") != length:
        raise IsaacArenaError(
            "visual acceptance progress does not span the scored episode"
        )
    return {
        "schema": ACCEPTANCE_SCHEMA,
        "qualified": True,
        "environment": environment,
        "actions": _action_contract(policy_type, evidence, length),
        "episode": {
            "name": episode["episode"],
            "length": length,
            "native_success": True,
            "success_rate": summary["success_rate"],
        },
        "task_progress": progress,
        "capture": _capture_contract(capture, length),
        "video": _video_contract(video, interval, length),
    }
