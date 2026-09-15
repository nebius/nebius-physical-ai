"""Bind Arena actions, native outcomes, task progress, and video evidence."""

from __future__ import annotations

import math
from typing import Any

from .action_evidence import (
    validate_action_evidence,
    validate_prepared_action_sequence,
)
from .errors import IsaacArenaError
from .task_progress import task_progress_adapter, task_visual_interval
from .video_evidence import (
    EVIDENCE_FILTER,
    EVIDENCE_PLAYBACK_RATE,
    video_acceptance_thresholds,
)

ACCEPTANCE_SCHEMA = "npa.isaac-arena.visual-acceptance.v1"


def _positive_integer(value: Any) -> bool:
    return type(value) is int and value > 0


def _sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _frame_records_consistent(records: list[Any]) -> bool:
    observed: dict[int, tuple[str, float]] = {}
    for record in records:
        if not isinstance(record, dict):
            return False
        index, digest, timestamp = (
            record.get("source_frame_index"),
            record.get("decoded_luma_sha256"),
            record.get("timestamp_seconds"),
        )
        if (
            type(index) is not int
            or index < 0
            or not _sha256(digest)
            or not isinstance(timestamp, (int, float))
            or isinstance(timestamp, bool)
            or not math.isfinite(float(timestamp))
            or timestamp < 0
        ):
            return False
        identity = (digest, float(timestamp))
        if index in observed and observed[index] != identity:
            return False
        observed[index] = identity
    return True


def _spatial_evidence_consistent(
    motion_pair: Any,
    progress_change: Any,
    region_bounds: dict[str, int],
    radius: int,
    block_size: int,
    block_stride: int,
    dimensions: tuple[int, int],
) -> bool:
    if not isinstance(motion_pair, dict) or not isinstance(progress_change, dict):
        return False
    component = progress_change.get("largest_connected_region_bounds")
    origins = motion_pair.get("coherent_block_origins")
    coherent_count = motion_pair.get("coherent_blocks")
    component_count = progress_change.get("largest_connected_trending_region_pixels")
    if (
        not isinstance(component, dict)
        or set(component) != {"top", "left", "bottom_exclusive", "right_exclusive"}
        or any(type(value) is not int for value in component.values())
        or not isinstance(origins, list)
        or len(origins) < 2
        or type(coherent_count) is not int
        or coherent_count != len(origins)
        or type(component_count) is not int
        or component_count < 1
    ):
        return False
    width, height = dimensions
    if not (
        region_bounds["top"]
        <= component["top"]
        < component["bottom_exclusive"]
        <= region_bounds["bottom_exclusive"]
        and region_bounds["left"]
        <= component["left"]
        < component["right_exclusive"]
        <= region_bounds["right_exclusive"]
        and component_count
        <= (component["bottom_exclusive"] - component["top"])
        * (component["right_exclusive"] - component["left"])
    ):
        return False
    associated = False
    coordinates: set[tuple[int, int]] = set()
    for origin in origins:
        if (
            not isinstance(origin, dict)
            or set(origin) != {"row", "column"}
            or type(origin["row"]) is not int
            or type(origin["column"]) is not int
            or not 0 <= origin["row"] <= height - block_size
            or not 0 <= origin["column"] <= width - block_size
            or origin["row"] % block_stride
            or origin["column"] % block_stride
        ):
            return False
        coordinates.add((origin["row"], origin["column"]))
        expanded = {
            "top": max(0, origin["row"] - radius),
            "left": max(0, origin["column"] - radius),
            "bottom_exclusive": min(height, origin["row"] + block_size + radius),
            "right_exclusive": min(width, origin["column"] + block_size + radius),
        }
        associated |= (
            expanded["top"] < component["bottom_exclusive"]
            and component["top"] < expanded["bottom_exclusive"]
            and expanded["left"] < component["right_exclusive"]
            and component["left"] < expanded["right_exclusive"]
        )
    adjacent = any(
        (row + block_stride, column) in coordinates
        or (row, column + block_stride) in coordinates
        for row, column in coordinates
    )
    return len(coordinates) == len(origins) and adjacent and associated


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
    progress_interval = (
        motion.get("progress_interval") if isinstance(motion, dict) else None
    )
    expected_visual_interval = (
        task_visual_interval(registered, progress_interval, length)
        if registered is not None
        and isinstance(progress_interval, dict)
        and _positive_integer(length)
        else None
    )
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
        and motion.get("visual_interval_strategy")
        == registered.visual_interval_strategy
        and motion.get("visual_context_steps") == registered.visual_context_steps
        and motion.get("visual_progress_signal") == registered.visual_progress_signal
        and motion.get("visual_progress_region")
        == dict(registered.visual_progress_region)
        and motion.get("visual_association_radius_fraction")
        == registered.visual_association_radius_fraction
        and motion.get("visual_interval") == expected_visual_interval
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
    maximum_held_fraction = adapter.maximum_trailing_held_action_fraction
    observed = validate_action_evidence(
        ground_truth.get("action_evidence"),
        policy_type=policy_type,
        expected_steps=episode_length,
        require_nonzero_varied=True,
        maximum_trailing_held_fraction=maximum_held_fraction,
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
            "trailing_held_action_fraction": observed["trailing_held_action_fraction"],
            "maximum_trailing_held_action_fraction": maximum_held_fraction,
        }
    execution = (evidence or {}).get("execution") or {}
    source_steps = execution.get("source_steps")
    prepared_steps = execution.get("prepared_steps")
    if not _positive_integer(source_steps) or source_steps != prepared_steps:
        raise IsaacArenaError(
            "visual acceptance requires exact nonzero actions without padding"
        )
    prepared = validate_prepared_action_sequence(
        execution.get("prepared_action_sequence"),
        expected_steps=source_steps,
        maximum_trailing_held_fraction=maximum_held_fraction,
    )
    prepared_hashes = prepared.get("action_step_sha256")
    observed_hashes = observed.get("action_step_sha256")
    valid = (
        policy_type == "replay"
        and ((evidence or {}).get("trajectory") or {}).get("nonzero_actions") is True
        and source_steps >= episode_length
        and execution.get("action_padding_steps") == 0
        and execution.get("strategy") == "actions_initial_state_exact_replay"
        and observed.get("action_shape") == prepared.get("action_shape")
        and isinstance(prepared_hashes, list)
        and isinstance(observed_hashes, list)
        and observed_hashes == prepared_hashes[:episode_length]
    )
    if not valid:
        raise IsaacArenaError(
            "visual acceptance requires exact nonzero actions without padding"
        )
    return {
        "source": "replay_input",
        "source_steps": source_steps,
        "executed_steps": episode_length,
        "unexecuted_source_steps": source_steps - episode_length,
        "execution_stop": "native_scored_episode_terminal",
        "padding_steps": 0,
        "sequence_sha256": observed["sequence_sha256"],
        "evidence_sha256": action_file["sha256"],
        "trailing_held_action_fraction": observed["trailing_held_action_fraction"],
        "maximum_trailing_held_action_fraction": maximum_held_fraction,
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
        and rendering.get("legacy_mode_enabled") is True
        and rendering.get("rt2_enabled") is False
        and rendering.get("path_tracing_enabled") is False
        and rendering.get("antialiasing") == "TAA"
        and rendering.get("dlss_execution_mode") == "quality"
        and rendering.get("dl_denoiser_enabled") is True
        and rendering.get("frame_generation_enabled") is False
        and rendering.get("minimum_settling_renders") == 8
        and rendering.get("stochastic_accumulation") is False
        and rendering.get("readback_phase") == "after_final_accepted_render"
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
    visual_interval: dict,
    progress_interval: dict,
    progress_signal: str,
    progress_region: dict,
    association_radius_fraction: float,
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
    motion_pair = (
        frame_evidence.get("motion_pair") if isinstance(frame_evidence, dict) else None
    )
    motion_support = (
        motion_pair.get("source_frame_support")
        if isinstance(motion_pair, dict)
        else None
    )
    progress_change = video.get("progress_change")
    progress_analysis = (
        progress_change.get("analysis_interval")
        if isinstance(progress_change, dict)
        else None
    )
    progress_frames = (
        progress_change.get("frame_evidence")
        if isinstance(progress_change, dict)
        else None
    )
    progress_first = (
        progress_frames.get("first") if isinstance(progress_frames, dict) else None
    )
    progress_last = (
        progress_frames.get("last") if isinstance(progress_frames, dict) else None
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
    expected_visual = {
        "start": visual_interval.get("start_action_step"),
        "end": visual_interval.get("end_action_step"),
        "total": visual_interval.get("total_action_steps"),
    }
    expected_progress = {
        "start": progress_interval.get("start_action_step"),
        "end": progress_interval.get("end_action_step"),
        "total": progress_interval.get("total_action_steps"),
    }
    visual_frame_start = max(expected_visual["start"], 1) - 1
    visual_frame_end = expected_visual["end"]
    progress_frame_start = max(expected_progress["start"], 1) - 1
    progress_frame_end = expected_progress["end"]
    thresholds = video_acceptance_thresholds()
    sample_width, sample_height = thresholds["sample_dimensions"]
    expected_region_bounds = {
        "top": math.floor(float(progress_region["top"]) * sample_height),
        "left": math.floor(float(progress_region["left"]) * sample_width),
        "bottom_exclusive": math.ceil(float(progress_region["bottom"]) * sample_height),
        "right_exclusive": math.ceil(float(progress_region["right"]) * sample_width),
    }
    expected_radius_pixels = math.ceil(
        max(sample_width, sample_height) * association_radius_fraction
    )
    motion_indices = (
        [
            samples[role].get("source_frame_index")
            for role in (
                "motion_previous",
                "motion_current",
                "motion_continuation",
            )
        ]
        if isinstance(samples, dict)
        and all(
            isinstance(samples.get(role), dict)
            for role in (
                "motion_previous",
                "motion_current",
                "motion_continuation",
            )
        )
        else []
    )
    progress_thresholds = (
        progress_change.get("thresholds") if isinstance(progress_change, dict) else None
    )
    valid = (
        isinstance(motion, dict)
        and video.get("frame_count") == episode_length
        and motion.get("meaningful") is True
        and isinstance(analysis, dict)
        and analysis.get("source") == "simulator_ground_truth"
        and analysis.get("action_steps") == expected_visual
        and motion.get("required_progress_overlap") is True
        and _positive_integer(motion.get("progress_overlapping_frame_pairs"))
        and motion.get("required_progress_spatial_binding") is True
        and _positive_integer(motion.get("progress_spatially_bound_frame_pairs"))
        and motion.get("progress_association_radius_pixels") == expected_radius_pixels
        and isinstance(derivation, dict)
        and derivation.get("kind") == "ffmpeg_spatiotemporal_denoise"
        and derivation.get("filter") == EVIDENCE_FILTER
        and derivation.get("playback_rate") == EVIDENCE_PLAYBACK_RATE
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
        and samples["first"].get("source_frame_index") == visual_frame_start
        and samples["last"].get("source_frame_index") == visual_frame_end - 1
        and isinstance(motion_support, dict)
        and type(motion_support.get("start")) is int
        and type(motion_support.get("end_exclusive")) is int
        and motion_support["start"] < progress_frame_end
        and progress_frame_start < motion_support["end_exclusive"]
        and visual_frame_start
        <= motion_support["start"]
        < motion_support["end_exclusive"]
        <= visual_frame_end
        and len(motion_indices) == 3
        and all(type(index) is int for index in motion_indices)
        and motion_support["start"]
        <= motion_indices[0]
        < motion_indices[1]
        < motion_indices[2]
        < motion_support["end_exclusive"]
        and isinstance(progress_change, dict)
        and progress_change.get("signal") == progress_signal
        and progress_change.get("meaningful") is True
        and isinstance(progress_analysis, dict)
        and progress_analysis.get("source") == "simulator_ground_truth"
        and progress_analysis.get("signal_horizon") == "exact_task_progress"
        and progress_analysis.get("action_steps") == expected_progress
        and progress_analysis.get("source_frame_indices")
        == {"start": progress_frame_start, "end_exclusive": progress_frame_end}
        and progress_change.get("decoded_samples")
        == progress_frame_end - progress_frame_start
        and progress_change.get("association_radius_fraction")
        == association_radius_fraction
        and progress_change.get("association_radius_pixels") == expected_radius_pixels
        and progress_change.get("task_region")
        == {**progress_region, "sample_bounds": expected_region_bounds}
        and _spatial_evidence_consistent(
            motion_pair,
            progress_change,
            expected_region_bounds,
            expected_radius_pixels,
            thresholds["tracking_block_size"],
            thresholds["tracking_block_stride"],
            (sample_width, sample_height),
        )
        and progress_thresholds
        == {
            "minimum_projected_luma_delta": thresholds[
                "minimum_projected_progress_luma_delta"
            ],
            "minimum_linear_r_squared": thresholds["minimum_progress_linear_r_squared"],
            "minimum_connected_pixels": thresholds["minimum_connected_progress_pixels"],
        }
        and _positive_integer(progress_change.get("trending_pixels"))
        and type(progress_change.get("largest_connected_trending_region_pixels")) is int
        and progress_change["largest_connected_trending_region_pixels"]
        >= thresholds["minimum_connected_progress_pixels"]
        and progress_change["trending_pixels"]
        >= progress_change["largest_connected_trending_region_pixels"]
        and isinstance(progress_first, dict)
        and isinstance(progress_last, dict)
        and progress_first.get("source_frame_index") == progress_frame_start
        and progress_last.get("source_frame_index") == progress_frame_end - 1
        and _sha256(progress_first.get("decoded_luma_sha256"))
        and _sha256(progress_last.get("decoded_luma_sha256"))
        and _frame_records_consistent(
            [
                *(samples[role] for role in samples),
                progress_first,
                progress_last,
            ]
        )
    )
    if not valid:
        raise IsaacArenaError(
            "visual acceptance video is not bound to native task progress"
        )
    return {
        "frame_count": video["frame_count"],
        "motion": "noise_resistant_coherent_scene_motion",
        "visual_action_steps": expected_visual,
        "progress_action_steps": expected_progress,
        "progress_signal": progress_signal,
        "progress_region": progress_region,
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
    progress_interval = progress.get("progress_interval")
    visual_interval = progress.get("visual_interval")
    if (
        not isinstance(progress_interval, dict)
        or progress_interval.get("total_action_steps") != length
        or not isinstance(visual_interval, dict)
        or visual_interval.get("total_action_steps") != length
    ):
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
            visual_interval,
            progress_interval,
            progress["visual_progress_signal"],
            progress["visual_progress_region"],
            progress["visual_association_radius_fraction"],
            episode,
            environment,
            policy_type,
            evidence,
            ground_truth,
            capture_contract["source_mp4_sha256"],
        ),
    }
