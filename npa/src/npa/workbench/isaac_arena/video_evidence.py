"""Validate Arena viewport geometry within its simulator-bound action interval."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from .errors import IsaacArenaError
from .hashing import file_sha256 as _file_sha256
from .simulator_video import _MINIMUM_SETTLING_RENDERS, _RENDER_SETTINGS

_SAMPLE_WIDTH = 160
_SAMPLE_HEIGHT = 90
_MEDIAN_WINDOW = 5
_COMPARISON_LAG = 5
_MINIMUM_PAIRS = 2
_MINIMUM_PROGRESS_SAMPLES = 10
_MINIMUM_PROJECTED_PROGRESS_LUMA_DELTA = 6.0
_MINIMUM_PROGRESS_LINEAR_R_SQUARED = 0.94
_MINIMUM_CONNECTED_PROGRESS_PIXELS = 8
_BLOCK_SIZE = 16
_BLOCK_STRIDE = 8
_SEARCH_RADIUS = 8
DENOISE_FILTER = "hqdn3d=20:16:30:24,gblur=sigma=3.5"
EVIDENCE_PLAYBACK_RATE = 0.5
EVIDENCE_FILTER = f"{DENOISE_FILTER},setpts={1 / EVIDENCE_PLAYBACK_RATE:g}*PTS"


def video_acceptance_thresholds() -> dict[str, Any]:
    """Return the reproducible visual gate advertised by the capability manifest.

    Args:
        None.
    Returns:
        Codec, decoding, noise rejection, and geometric tracking requirements.
    Raises:
        None.
    """
    return {
        "codec": "h264",
        "pixel_format": "yuv420p",
        "minimum_dimensions": [320, 240],
        "minimum_duration_seconds": 1.0,
        "sample_dimensions": [_SAMPLE_WIDTH, _SAMPLE_HEIGHT],
        "sampling": "every native frame; area resampling; no time or frame cap",
        "temporal_median_window": _MEDIAN_WINDOW,
        "comparison_lag_samples": _COMPARISON_LAG,
        "minimum_changed_frame_pairs": _MINIMUM_PAIRS,
        "minimum_changed_pixel_ratio": 0.01,
        "minimum_frame_mean_abs_luma_delta": 1.0,
        "changed_pixel_luma_delta": 6,
        "minimum_coherent_blocks": 2,
        "minimum_block_texture_standard_deviation": 8.0,
        "maximum_normalized_tracking_error": 0.35,
        "minimum_tracking_error_improvement": 0.6,
        "minimum_tracking_displacement_pixels": 1.0,
        "maximum_neighbor_displacement_difference_pixels": 1.5,
        "minimum_tracking_uniqueness_margin": 0.02,
        "tracking_block_size": _BLOCK_SIZE,
        "tracking_block_stride": _BLOCK_STRIDE,
        "tracking_search_radius": _SEARCH_RADIUS,
        "tracking": "brightness-centered patches; adjacent agreeing displacements over three disjoint temporal windows",
        "consecutive_tracking_intervals": 2,
        "progress_overlap": "at least one accepted coherent-motion track spans the exact simulator progress interval",
        "progress_signal": "adapter-selected per-pixel linear luma trend within the exact simulator progress interval",
        "progress_spatial_binding": "largest connected trend must be inside the adapter-declared task region and within its declared radius of an overlapping coherent track",
        "minimum_progress_samples": _MINIMUM_PROGRESS_SAMPLES,
        "minimum_projected_progress_luma_delta": (
            _MINIMUM_PROJECTED_PROGRESS_LUMA_DELTA
        ),
        "minimum_progress_linear_r_squared": _MINIMUM_PROGRESS_LINEAR_R_SQUARED,
        "minimum_connected_progress_pixels": _MINIMUM_CONNECTED_PROGRESS_PIXELS,
        "evidence_filter": EVIDENCE_FILTER,
        "evidence_playback_rate": EVIDENCE_PLAYBACK_RATE,
    }


def _run(arguments: list[str], failure: str) -> subprocess.CompletedProcess:
    try:
        completed = subprocess.run(
            arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
        )
    except OSError as exc:
        raise IsaacArenaError(failure) from exc
    if completed.returncode:
        raise IsaacArenaError(failure)
    return completed


def _video_metadata(
    path: Path, *, minimum_duration_seconds: float = 1.0
) -> tuple[dict[str, Any], list[float]]:
    completed = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height,nb_read_frames,avg_frame_rate:"
            "format=duration:frame=best_effort_timestamp_time",
            "-of",
            "json",
            str(path),
        ],
        f"invalid viewport MP4: {path.name}",
    )
    try:
        payload = json.loads(completed.stdout)
        stream = payload["streams"][0]
        metadata = {
            "codec": stream["codec_name"],
            "pixel_format": stream["pix_fmt"],
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "duration_seconds": float(payload["format"]["duration"]),
            "frame_count": int(stream["nb_read_frames"]),
        }
        timestamps = [
            float(frame["best_effort_timestamp_time"]) for frame in payload["frames"]
        ]
    except (KeyError, IndexError, ValueError, TypeError) as exc:
        raise IsaacArenaError(f"invalid viewport MP4: {path.name}") from exc
    _validate_metadata(
        metadata,
        timestamps,
        path,
        minimum_duration_seconds=minimum_duration_seconds,
    )
    return metadata, timestamps


def _validate_metadata(
    metadata: dict[str, Any],
    timestamps: list[float],
    path: Path,
    *,
    minimum_duration_seconds: float,
) -> None:
    if (
        metadata["codec"] != "h264"
        or metadata["pixel_format"] != "yuv420p"
        or metadata["width"] < 320
        or metadata["height"] < 240
        or not math.isfinite(metadata["duration_seconds"])
        or metadata["duration_seconds"] <= 0
        or metadata["duration_seconds"] < minimum_duration_seconds
        or len(timestamps) != metadata["frame_count"]
        or not all(math.isfinite(stamp) for stamp in timestamps)
        or any(later <= earlier for earlier, later in zip(timestamps, timestamps[1:]))
    ):
        raise IsaacArenaError(f"invalid viewport MP4: {path.name}")


def _analysis_interval(
    frame_count: int, interval: dict[str, int] | None
) -> dict[str, Any]:
    start, stop = 0, frame_count
    metadata: dict[str, Any] = {"source": "full_video"}
    if interval is not None:
        keys = ("start_action_step", "end_action_step", "total_action_steps")
        if any(type(interval.get(key)) is not int for key in keys):
            raise IsaacArenaError("invalid simulator-ground-truth video interval")
        first, last, total = (interval[key] for key in keys)
        if not 0 <= first < last <= total:
            raise IsaacArenaError("invalid simulator-ground-truth video interval")
        if frame_count != total:
            raise IsaacArenaError(
                "viewport frame count does not match simulator action steps"
            )
        start, stop = max(first, 1) - 1, last
        metadata = {
            "source": "simulator_ground_truth",
            "action_steps": {"start": first, "end": last, "total": total},
            "first_video_frame_action_step": 1,
        }
    if stop - start < _MEDIAN_WINDOW + 2 * _COMPARISON_LAG + _MINIMUM_PAIRS - 1:
        raise IsaacArenaError("video interval is too short for motion validation")
    metadata["source_frame_indices"] = {"start": start, "end_exclusive": stop}
    metadata["decoded_sample_indices"] = {"start": start, "end_exclusive": stop}
    return metadata


def _progress_analysis_interval(
    frame_count: int, interval: dict[str, int]
) -> dict[str, Any]:
    keys = ("start_action_step", "end_action_step", "total_action_steps")
    if any(type(interval.get(key)) is not int for key in keys):
        raise IsaacArenaError("invalid simulator-ground-truth progress interval")
    first, last, total = (interval[key] for key in keys)
    if not 0 <= first < last <= total or frame_count != total:
        raise IsaacArenaError("invalid simulator-ground-truth progress interval")
    start, stop = max(first, 1) - 1, last
    if stop - start < _MINIMUM_PROGRESS_SAMPLES:
        raise IsaacArenaError("video progress interval is too short for validation")
    return {
        "source": "simulator_ground_truth",
        "signal_horizon": "exact_task_progress",
        "action_steps": {"start": first, "end": last, "total": total},
        "first_video_frame_action_step": 1,
        "source_frame_indices": {"start": start, "end_exclusive": stop},
        "decoded_sample_indices": {"start": start, "end_exclusive": stop},
    }


def _decode_interval(path: Path, interval: dict[str, Any]) -> np.ndarray:
    bounds = interval["source_frame_indices"]
    filters = (
        f"trim=start_frame={bounds['start']}:end_frame={bounds['end_exclusive']},"
        f"scale={_SAMPLE_WIDTH}:{_SAMPLE_HEIGHT}:flags=area,format=gray"
    )
    completed = _run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            filters,
            "-vsync",
            "0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "pipe:1",
        ],
        f"viewport MP4 frame decode failed: {path.name}",
    )
    count = bounds["end_exclusive"] - bounds["start"]
    if len(completed.stdout) != count * _SAMPLE_WIDTH * _SAMPLE_HEIGHT:
        raise IsaacArenaError(
            "viewport decoded frame count differs from source interval"
        )
    return np.frombuffer(completed.stdout, dtype=np.uint8).reshape(
        count, _SAMPLE_HEIGHT, _SAMPLE_WIDTH
    )


def _temporal_medians(frames: np.ndarray) -> np.ndarray:
    windows = np.lib.stride_tricks.sliding_window_view(frames, _MEDIAN_WINDOW, axis=0)
    return np.median(windows, axis=-1).astype(np.float32)


def _center_patch(patches: np.ndarray) -> np.ndarray:
    return patches - np.mean(patches, axis=(-2, -1), keepdims=True)


def _track_block(
    previous: np.ndarray, current: np.ndarray, row: int, column: int
) -> tuple[float, float] | None:
    patch = _center_patch(
        previous[row : row + _BLOCK_SIZE, column : column + _BLOCK_SIZE]
    )
    variance = float(np.mean(patch**2))
    if variance < 8.0**2:
        return None
    top, left = max(0, row - _SEARCH_RADIUS), max(0, column - _SEARCH_RADIUS)
    bottom = min(_SAMPLE_HEIGHT, row + _BLOCK_SIZE + _SEARCH_RADIUS)
    right = min(_SAMPLE_WIDTH, column + _BLOCK_SIZE + _SEARCH_RADIUS)
    windows = np.lib.stride_tricks.sliding_window_view(
        current[top:bottom, left:right], (_BLOCK_SIZE, _BLOCK_SIZE)
    )
    errors = np.mean((_center_patch(windows) - patch) ** 2, axis=(-2, -1)) / variance
    best_row, best_column = np.unravel_index(np.argmin(errors), errors.shape)
    best_error = float(errors[best_row, best_column])
    stationary_error = float(errors[row - top, column - left])
    displacement = (float(best_row + top - row), float(best_column + left - column))
    if (
        math.hypot(*displacement) < 1
        or best_error > 0.35
        or best_error > stationary_error * 0.4
    ):
        return None
    alternatives = errors.copy()
    alternatives[
        max(0, best_row - 1) : best_row + 2, max(0, best_column - 1) : best_column + 2
    ] = np.inf
    if float(alternatives.min()) - best_error < 0.02:
        return None
    return displacement


def _persistent_track(
    previous: np.ndarray,
    current: np.ndarray,
    following: np.ndarray,
    row: int,
    column: int,
) -> tuple[float, float] | None:
    displacement = _track_block(previous, current, row, column)
    if displacement is None:
        return None
    target_row, target_column = (
        int(row + displacement[0]),
        int(column + displacement[1]),
    )
    continuation = _track_block(current, following, target_row, target_column)
    if continuation is None or math.dist(displacement, continuation) > 1.5:
        return None
    return displacement


def _coherent_tracks(
    previous: np.ndarray, current: np.ndarray, following: np.ndarray
) -> set[tuple[int, int]]:
    tracks = {}
    for row in range(0, _SAMPLE_HEIGHT - _BLOCK_SIZE + 1, _BLOCK_STRIDE):
        for column in range(0, _SAMPLE_WIDTH - _BLOCK_SIZE + 1, _BLOCK_STRIDE):
            displacement = _persistent_track(previous, current, following, row, column)
            if displacement is not None:
                tracks[(row, column)] = displacement
    coherent = set()
    for (row, column), displacement in tracks.items():
        for neighbor in ((row + _BLOCK_STRIDE, column), (row, column + _BLOCK_STRIDE)):
            other = tracks.get(neighbor)
            if other is not None and math.dist(displacement, other) <= 1.5:
                coherent.update(((row, column), neighbor))
    return coherent


def _pair_statistics(
    previous: np.ndarray, current: np.ndarray, following: np.ndarray, index: int
) -> dict[str, Any]:
    signed = current - previous
    differences = np.abs(signed - np.median(signed))
    mean_delta = float(np.mean(differences))
    changed_ratio = float(np.mean(differences >= 6))
    coherent = (
        _coherent_tracks(previous, current, following)
        if mean_delta >= 1 and changed_ratio >= 0.01
        else set()
    )
    return {
        "previous_sample_index": index,
        "current_sample_index": index + _COMPARISON_LAG,
        "mean_abs_luma_delta": mean_delta,
        "changed_pixel_ratio": changed_ratio,
        "coherent_blocks": len(coherent),
        "coherent_block_origins": [
            {"row": row, "column": column} for row, column in sorted(coherent)
        ],
        "continuation_sample_index": index + 2 * _COMPARISON_LAG,
        "meaningful": (
            mean_delta >= 1 and changed_ratio >= 0.01 and len(coherent) >= 2
        ),
    }


def _pair_source_support(pair: dict[str, Any], source_offset: int) -> dict[str, int]:
    return {
        "start": source_offset + pair["previous_sample_index"],
        "end_exclusive": source_offset
        + pair["continuation_sample_index"]
        + _MEDIAN_WINDOW,
    }


def _intervals_overlap(first: dict[str, int], second: dict[str, int]) -> bool:
    return (
        first["start"] < second["end_exclusive"]
        and second["start"] < first["end_exclusive"]
    )


def _motion_metadata(
    frames: np.ndarray,
    interval: dict[str, Any],
    required_progress_interval: dict[str, Any] | None,
    required_progress_component: np.ndarray | None,
    association_radius: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    denoised = _temporal_medians(frames)
    pairs = [
        _pair_statistics(previous, current, following, index)
        for index, (previous, current, following) in enumerate(
            zip(
                denoised[: -2 * _COMPARISON_LAG],
                denoised[_COMPARISON_LAG:-_COMPARISON_LAG],
                denoised[2 * _COMPARISON_LAG :],
            )
        )
    ]
    source_offset = interval["source_frame_indices"]["start"]
    for pair in pairs:
        pair["source_frame_support"] = _pair_source_support(pair, source_offset)
    accepted = [pair for pair in pairs if pair["meaningful"]]
    if len(accepted) < _MINIMUM_PAIRS:
        raise IsaacArenaError(
            "viewport MP4 is decodable but visually static or lacks "
            "noise-resistant coherent scene motion"
        )
    overlapping = accepted
    if required_progress_interval is not None:
        progress_bounds = required_progress_interval["source_frame_indices"]
        overlapping = [
            pair
            for pair in accepted
            if _intervals_overlap(pair["source_frame_support"], progress_bounds)
        ]
        if not overlapping:
            raise IsaacArenaError(
                "noise-resistant coherent scene motion does not overlap native task progress"
            )
    spatially_bound = overlapping
    if required_progress_component is not None:
        if association_radius is None:
            raise IsaacArenaError("missing task-progress spatial association radius")
        spatially_bound = [
            pair
            for pair in overlapping
            if _spatially_associated(
                pair, required_progress_component, association_radius
            )
        ]
        if not spatially_bound:
            raise IsaacArenaError(
                "coherent scene motion is spatially disjoint from task-progress change"
            )
    best = max(
        spatially_bound,
        key=lambda pair: (
            pair["coherent_blocks"],
            pair["continuation_sample_index"],
        ),
    )
    return {
        "sample_width": _SAMPLE_WIDTH,
        "sample_height": _SAMPLE_HEIGHT,
        "sampling": "native frames",
        "decoded_samples": len(frames),
        "temporal_median_samples": len(denoised),
        "comparison_lag_samples": _COMPARISON_LAG,
        "changed_frame_pairs": len(accepted),
        "max_frame_mean_abs_luma_delta": max(
            pair["mean_abs_luma_delta"] for pair in pairs
        ),
        "max_changed_pixel_ratio": max(pair["changed_pixel_ratio"] for pair in pairs),
        "max_coherent_blocks": max(pair["coherent_blocks"] for pair in pairs),
        "progress_overlapping_frame_pairs": (
            len(overlapping) if required_progress_interval is not None else None
        ),
        "progress_spatially_bound_frame_pairs": (
            len(spatially_bound) if required_progress_component is not None else None
        ),
        "progress_association_radius_pixels": association_radius,
        "required_progress_overlap": required_progress_interval is not None,
        "required_progress_spatial_binding": required_progress_component is not None,
        "analysis_interval": interval,
        "thresholds": video_acceptance_thresholds(),
        "meaningful": True,
    }, best


def _largest_connected_region(
    mask: np.ndarray,
) -> tuple[int, np.ndarray, dict[str, int] | None]:
    visited = np.zeros(mask.shape, dtype=bool)
    largest_points: list[tuple[int, int]] = []
    for row, column in np.argwhere(mask):
        if visited[row, column]:
            continue
        visited[row, column] = True
        pending = [(int(row), int(column))]
        points: list[tuple[int, int]] = []
        while pending:
            current_row, current_column = pending.pop()
            points.append((current_row, current_column))
            for next_row, next_column in (
                (current_row - 1, current_column),
                (current_row + 1, current_column),
                (current_row, current_column - 1),
                (current_row, current_column + 1),
            ):
                if (
                    0 <= next_row < mask.shape[0]
                    and 0 <= next_column < mask.shape[1]
                    and mask[next_row, next_column]
                    and not visited[next_row, next_column]
                ):
                    visited[next_row, next_column] = True
                    pending.append((next_row, next_column))
        if len(points) > len(largest_points):
            largest_points = points
    component = np.zeros(mask.shape, dtype=bool)
    if not largest_points:
        return 0, component, None
    rows, columns = zip(*largest_points, strict=True)
    component[rows, columns] = True
    return (
        len(largest_points),
        component,
        {
            "top": min(rows),
            "left": min(columns),
            "bottom_exclusive": max(rows) + 1,
            "right_exclusive": max(columns) + 1,
        },
    )


def _progress_region_bounds(region: dict[str, Any]) -> dict[str, int]:
    required = {"name", "left", "top", "right", "bottom"}
    values = [region.get(key) for key in ("left", "top", "right", "bottom")]
    if (
        set(region) != required
        or not isinstance(region.get("name"), str)
        or not region["name"]
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in values
        )
        or not 0 <= float(values[0]) < float(values[2]) <= 1
        or not 0 <= float(values[1]) < float(values[3]) <= 1
    ):
        raise IsaacArenaError("invalid adapter-declared task visual region")
    return {
        "top": math.floor(float(values[1]) * _SAMPLE_HEIGHT),
        "left": math.floor(float(values[0]) * _SAMPLE_WIDTH),
        "bottom_exclusive": math.ceil(float(values[3]) * _SAMPLE_HEIGHT),
        "right_exclusive": math.ceil(float(values[2]) * _SAMPLE_WIDTH),
    }


def _spatially_associated(
    pair: dict[str, Any], component: np.ndarray, radius: int
) -> bool:
    for origin in pair["coherent_block_origins"]:
        row, column = origin["row"], origin["column"]
        top, left = max(0, row - radius), max(0, column - radius)
        bottom = min(_SAMPLE_HEIGHT, row + _BLOCK_SIZE + radius)
        right = min(_SAMPLE_WIDTH, column + _BLOCK_SIZE + radius)
        if np.any(component[top:bottom, left:right]):
            return True
    return False


def _progress_change_metadata(
    frames: np.ndarray,
    interval: dict[str, Any],
    timestamps: list[float],
    signal: str,
    task_region: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    if signal != "monotonic_structural_change":
        raise IsaacArenaError("unsupported task-progress visual signal")
    values = frames.astype(np.float32)
    values -= np.median(values, axis=(1, 2), keepdims=True)
    time = np.arange(len(values), dtype=np.float32)
    centered_time = time - np.mean(time)
    centered_values = values - np.mean(values, axis=0)
    covariance = np.sum(
        centered_time[:, np.newaxis, np.newaxis] * centered_values, axis=0
    )
    time_variance = float(np.sum(centered_time**2))
    value_variance = np.sum(centered_values**2, axis=0)
    r_squared = np.divide(
        covariance**2,
        time_variance * value_variance,
        out=np.zeros_like(covariance),
        where=value_variance > 0,
    )
    projected_delta = covariance / time_variance * (len(values) - 1)
    trending = (np.abs(projected_delta) >= _MINIMUM_PROJECTED_PROGRESS_LUMA_DELTA) & (
        r_squared >= _MINIMUM_PROGRESS_LINEAR_R_SQUARED
    )
    region_bounds = _progress_region_bounds(task_region)
    region_mask = np.zeros(trending.shape, dtype=bool)
    region_mask[
        region_bounds["top"] : region_bounds["bottom_exclusive"],
        region_bounds["left"] : region_bounds["right_exclusive"],
    ] = True
    trending &= region_mask
    largest, largest_component, component_bounds = _largest_connected_region(trending)
    if largest < _MINIMUM_CONNECTED_PROGRESS_PIXELS:
        raise IsaacArenaError(
            "viewport MP4 has no noise-resistant structural change during native task progress"
        )
    bounds = interval["source_frame_indices"]
    first_index, last_index = bounds["start"], bounds["end_exclusive"] - 1
    return {
        "signal": signal,
        "meaningful": True,
        "analysis_interval": interval,
        "decoded_samples": len(frames),
        "brightness_compensation": "per-frame median",
        "linear_fit": "ordinary least squares over every native progress frame",
        "trending_pixels": int(np.count_nonzero(trending)),
        "largest_connected_trending_region_pixels": largest,
        "task_region": {**task_region, "sample_bounds": region_bounds},
        "largest_connected_region_bounds": component_bounds,
        "thresholds": {
            "minimum_projected_luma_delta": (_MINIMUM_PROJECTED_PROGRESS_LUMA_DELTA),
            "minimum_linear_r_squared": _MINIMUM_PROGRESS_LINEAR_R_SQUARED,
            "minimum_connected_pixels": _MINIMUM_CONNECTED_PROGRESS_PIXELS,
        },
        "frame_evidence": {
            "first": {
                "source_frame_index": first_index,
                "timestamp_seconds": timestamps[first_index],
                "decoded_luma_sha256": hashlib.sha256(frames[0].tobytes()).hexdigest(),
            },
            "last": {
                "source_frame_index": last_index,
                "timestamp_seconds": timestamps[last_index],
                "decoded_luma_sha256": hashlib.sha256(frames[-1].tobytes()).hexdigest(),
            },
        },
    }, largest_component


def _frame_evidence(
    frames: np.ndarray,
    interval: dict[str, Any],
    timestamps: list[float],
    pair: dict[str, Any],
) -> dict[str, Any]:
    offset = interval["source_frame_indices"]["start"]
    centers = (
        pair["previous_sample_index"] + _MEDIAN_WINDOW // 2,
        pair["current_sample_index"] + _MEDIAN_WINDOW // 2,
    )
    roles = {
        "first": 0,
        "last": len(frames) - 1,
        "motion_previous": centers[0],
        "motion_current": centers[1],
        "motion_continuation": pair["continuation_sample_index"] + _MEDIAN_WINDOW // 2,
    }
    samples = {
        role: {
            "source_frame_index": offset + index,
            "timestamp_seconds": timestamps[offset + index],
            "decoded_luma_sha256": hashlib.sha256(frames[index].tobytes()).hexdigest(),
        }
        for role, index in roles.items()
    }
    return {
        "hash_representation": "decoded gray8; area resampled to 160x90; before temporal median",
        "samples": samples,
        "motion_pair": pair,
        "median_context_radius_frames": _MEDIAN_WINDOW // 2,
    }


def probe_mp4(
    path: Path,
    *,
    evidence_interval: dict[str, int] | None = None,
    progress_interval: dict[str, int] | None = None,
    progress_signal: str | None = None,
    progress_region: dict[str, Any] | None = None,
    progress_association_radius_fraction: float | None = None,
) -> dict[str, Any]:
    """Require browser video and geometric motion inside the exact task interval.

    Area resampling and five-frame medians suppress grain. Brightness-centered
    patch tracks must agree spatially and persist through three disjoint median
    windows. Pixel differences alone never qualify; the calling runtime remains
    responsible for validating simulator state and upstream task success.

    Args:
        path: Original viewport MP4, with one frame after each simulator action.
        evidence_interval: Inclusive action-step bounds and total action count;
            action one maps to frame zero. None analyzes the complete video.
        progress_interval: Exact native progress bounds which accepted coherent
            motion must overlap when task qualification is requested.
        progress_signal: Adapter-selected structural progress validator.
        progress_region: Adapter-declared normalized task-object workspace.
        progress_association_radius_fraction: Maximum normalized image distance
            between progress structure and an overlapping coherent track.
    Returns:
        Codec metadata, robust motion statistics, and reproducible frame hashes.
    Raises:
        IsaacArenaError: Invalid video, mismatched action mapping, short interval,
            or insufficient noise-resistant geometric motion.
    """
    metadata, timestamps = _video_metadata(path)
    interval = _analysis_interval(metadata["frame_count"], evidence_interval)
    frames = _decode_interval(path, interval)
    progress_contract = (
        progress_interval,
        progress_signal,
        progress_region,
        progress_association_radius_fraction,
    )
    if any(value is None for value in progress_contract) and not all(
        value is None for value in progress_contract
    ):
        raise IsaacArenaError("incomplete simulator-ground-truth progress binding")
    progress_analysis = (
        _progress_analysis_interval(metadata["frame_count"], progress_interval)
        if progress_interval is not None
        else None
    )
    if progress_analysis is not None and not (
        interval["source_frame_indices"]["start"]
        <= progress_analysis["source_frame_indices"]["start"]
        < progress_analysis["source_frame_indices"]["end_exclusive"]
        <= interval["source_frame_indices"]["end_exclusive"]
    ):
        raise IsaacArenaError("task progress is outside the visual evidence interval")
    progress_change = None
    progress_component = None
    association_radius = None
    if progress_analysis is not None:
        if (
            not isinstance(progress_signal, str)
            or not isinstance(progress_region, dict)
            or not isinstance(progress_association_radius_fraction, (int, float))
            or isinstance(progress_association_radius_fraction, bool)
            or not math.isfinite(float(progress_association_radius_fraction))
            or not 0 < float(progress_association_radius_fraction) <= 0.25
        ):
            raise IsaacArenaError("invalid task-progress visual contract")
        offset = interval["source_frame_indices"]["start"]
        progress_bounds = progress_analysis["source_frame_indices"]
        progress_frames = frames[
            progress_bounds["start"] - offset : progress_bounds["end_exclusive"]
            - offset
        ]
        progress_change, progress_component = _progress_change_metadata(
            progress_frames,
            progress_analysis,
            timestamps,
            progress_signal,
            progress_region,
        )
        association_radius = math.ceil(
            max(_SAMPLE_WIDTH, _SAMPLE_HEIGHT)
            * float(progress_association_radius_fraction)
        )
        progress_change["association_radius_fraction"] = float(
            progress_association_radius_fraction
        )
        progress_change["association_radius_pixels"] = association_radius
    motion, best = _motion_metadata(
        frames,
        interval,
        progress_analysis,
        progress_component,
        association_radius,
    )
    motion["sample_fps"] = (len(timestamps) - 1) / (timestamps[-1] - timestamps[0])
    metadata["motion"] = motion
    metadata["frame_evidence"] = _frame_evidence(frames, interval, timestamps, best)
    if progress_change is not None:
        metadata["progress_change"] = progress_change
    return metadata


def _render_denoised_video(source: Path, target: Path) -> None:
    arguments = ["ffmpeg", "-v", "error", "-i", str(source)]
    arguments += ["-vf", EVIDENCE_FILTER, "-an", "-vsync", "0"]
    arguments += ["-c:v", "libx264", "-preset", "medium", "-crf", "18"]
    arguments += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", "-y", str(target)]
    try:
        _run(arguments, f"viewport MP4 denoising failed: {source.name}")
    except IsaacArenaError:
        target.unlink(missing_ok=True)
        raise
    if not target.is_file() or target.stat().st_size == 0:
        target.unlink(missing_ok=True)
        raise IsaacArenaError(f"viewport MP4 denoising failed: {source.name}")


def denoise_mp4(source: Path) -> tuple[Path, dict[str, Any]]:
    """Create a declared denoised derivative while retaining the upstream MP4.

    Args:
        source: Original upstream MP4, preserved unchanged.
    Returns:
        Derivative path and source hash with the exact FFmpeg filter declaration.
    Raises:
        IsaacArenaError: FFmpeg cannot produce a nonempty derivative.
    """
    target = source.with_name(f"{source.stem}-evidence-denoised.mp4")
    _render_denoised_video(source, target)
    return target, {
        "kind": "ffmpeg_spatiotemporal_denoise",
        "filter": EVIDENCE_FILTER,
        "playback_rate": EVIDENCE_PLAYBACK_RATE,
        "source_path": source.name,
        "source_sha256": _file_sha256(source),
        "changes_simulator_outcome": False,
    }


def _capture_sidecar(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = run_dir / "simulator-video-evidence.json"
    try:
        if path.is_symlink():
            raise IsaacArenaError("simulator capture sidecar must be a run-owned file")
        content = path.read_bytes()
        payload = json.loads(content)
    except (OSError, ValueError) as exc:
        raise IsaacArenaError(
            "missing or invalid simulator video capture sidecar"
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "npa.isaac-arena.video-capture.v2"
        or payload.get("capture_phase") != "recorder_post_step_before_autoreset"
        or type(payload.get("captured_action_steps")) is not int
        or payload["captured_action_steps"] < 1
        or not isinstance(payload.get("terminals"), list)
        or len(payload["terminals"]) != 1
    ):
        raise IsaacArenaError("invalid simulator video capture schema or phase")
    return payload, {"path": path.name, "sha256": hashlib.sha256(content).hexdigest()}


def _capture_step_mapping(task_motion: dict[str, Any], total_steps: int) -> int:
    try:
        interval = task_motion["progress_interval"]
        capture = task_motion["video_capture"]
        progress_end = interval["end_action_step"]
        terminal = capture["terminal_action_step"]
        steps = capture["action_steps"]
        valid = (
            type(progress_end) is int
            and 0 < progress_end <= total_steps
            and type(terminal) is int
            and terminal == total_steps
            and type(interval["total_action_steps"]) is int
            and interval["total_action_steps"] == total_steps
            and type(capture["initial_action_step"]) is int
            and capture["initial_action_step"] == 0
            and isinstance(steps, list)
            and all(type(step) is int for step in steps)
            and steps == list(range(1, terminal + 1))
        )
    except (KeyError, TypeError) as exc:
        raise IsaacArenaError(
            "simulator HDF5 has no exact video action-step binding"
        ) from exc
    if not valid:
        raise IsaacArenaError(
            "simulator HDF5 video action-step binding disagrees with task motion"
        )
    return terminal


def _capture_image_path(run_dir: Path, record: Any, step: int) -> Path:
    if not isinstance(record, dict):
        raise IsaacArenaError("missing simulator capture frame evidence")
    name = record.get("path")
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise IsaacArenaError("invalid simulator capture frame path")
    path = run_dir / name
    if path.is_symlink() or not path.is_file():
        raise IsaacArenaError("missing simulator capture PNG")
    expected_index = step - 1 if step else None
    if (
        type(record.get("action_step")) is not int
        or record["action_step"] != step
        or record.get("source_frame_index") != expected_index
        or (step > 0 and type(record.get("source_frame_index")) is not int)
    ):
        raise IsaacArenaError(
            "simulator capture frame action step disagrees with task motion"
        )
    return path


def _verify_capture_png(
    run_dir: Path, record: Any, step: int, dimensions: tuple[int, int]
) -> dict[str, Any]:
    from PIL import Image

    path = _capture_image_path(run_dir, record, step)
    digest = _file_sha256(path)
    try:
        with Image.open(path) as picture:
            valid = (
                picture.format == "PNG"
                and picture.mode == "RGB"
                and picture.size == dimensions
            )
            pixels = np.asarray(picture).copy()
    except (OSError, ValueError) as exc:
        raise IsaacArenaError("invalid simulator capture PNG") from exc
    pixel_digest = hashlib.sha256(pixels.tobytes()).hexdigest()
    if (
        not valid
        or record.get("width") != dimensions[0]
        or record.get("height") != dimensions[1]
        or record.get("sha256") != digest
        or record.get("decoded_rgb_sha256") != pixel_digest
    ):
        raise IsaacArenaError("simulator capture PNG hash or dimensions disagree")
    if not np.any(pixels):
        raise IsaacArenaError("simulator capture PNG is entirely black")
    return {**record, "sha256": digest, "decoded_rgb_sha256": pixel_digest}


def _terminal_frame_comparison(
    raw_mp4: Path, png: Path, terminal: int
) -> dict[str, Any]:
    source_interval = {
        "source_frame_indices": {"start": terminal - 1, "end_exclusive": terminal}
    }
    image_interval = {"source_frame_indices": {"start": 0, "end_exclusive": 1}}
    actual = _decode_interval(raw_mp4, source_interval)[0].astype(np.float32)
    expected = _decode_interval(png, image_interval)[0].astype(np.float32)
    delta = np.abs(actual - expected)
    block_means = delta.reshape(9, 10, 16, 10).mean(axis=(1, 3))
    mean_delta, block_delta = float(delta.mean()), float(block_means.max())
    if mean_delta > 4.0 or block_delta > 8.0:
        raise IsaacArenaError(
            "raw MP4 terminal frame disagrees with captured pre-reset PNG"
        )
    return {
        "source_frame_index": terminal - 1,
        "comparison": "area-resampled gray8 at 160x90; no brightness compensation",
        "mean_absolute_luma_difference": mean_delta,
        "maximum_block_mean_absolute_luma_difference": block_delta,
        "maximum_mean_luma_difference": 4.0,
        "maximum_block_mean_luma_difference": 8.0,
    }


def _capture_action_count(
    capture: dict[str, Any], metadata: dict[str, Any], expected_steps: int | None
) -> int:
    total = capture["captured_action_steps"]
    if metadata["frame_count"] != total or (
        expected_steps is not None and total != expected_steps
    ):
        raise IsaacArenaError(
            "raw MP4 frame count disagrees with captured or replay action steps"
        )
    return total


def _frozen_field_valid(row: dict[str, Any], prefix: str) -> bool:
    before, after = row.get(f"{prefix}_before"), row.get(f"{prefix}_after")
    if type(before) is not type(after) or before != after:
        return False
    if prefix == "physics_time":
        return type(before) in (float, int) and math.isfinite(before) and before >= 0
    if prefix in ("physics_step", "native_physics_step"):
        return type(before) is int and before >= 0
    return (
        isinstance(before, str)
        and len(before) == 64
        and all(c in "0123456789abcdef" for c in before)
    )


def _freeze_row_valid(row: Any, step: int) -> bool:
    if (
        not isinstance(row, dict)
        or type(row.get("action_step")) is not int
        or row["action_step"] != step
    ):
        return False
    if not all(
        _frozen_field_valid(row, prefix)
        for prefix in (
            "physics_time",
            "physics_step",
            "native_physics_step",
            "state_sha256",
        )
    ):
        return False
    return (
        type(row.get("render_calls")) is int
        and type(row.get("pre_settling_render_calls")) is int
        and row["pre_settling_render_calls"] >= 1
        and type(row.get("settling_render_calls")) is int
        and row["settling_render_calls"] >= _MINIMUM_SETTLING_RENDERS
        and row["render_calls"]
        == row["pre_settling_render_calls"] + row["settling_render_calls"]
        and type(row.get("consecutive_ready_render_calls")) is int
        and row["consecutive_ready_render_calls"] == row["settling_render_calls"] + 1
        and row["render_calls"] >= row["consecutive_ready_render_calls"]
        and all(
            row.get(field) is True
            for field in ("stage_streaming_idle", "stage_assets_loaded", "nonblack_rgb")
        )
    )


def _rendering_proof(capture: dict[str, Any]) -> dict[str, Any]:
    rendering = capture.get("rendering")
    expected = {
        "mode": "RaytracedLighting",
        "legacy_mode_enabled": True,
        "rt2_enabled": False,
        "path_tracing_enabled": False,
        "antialiasing": "TAA",
        "dlss_execution_mode": "quality",
        "dl_denoiser_enabled": True,
        "frame_generation_enabled": False,
        "minimum_settling_renders": _MINIMUM_SETTLING_RENDERS,
        "stochastic_accumulation": False,
        "readback_phase": "after_final_accepted_render",
    }
    if not isinstance(rendering, dict) or any(
        type(rendering.get(key)) is not type(value) or rendering[key] != value
        for key, value in expected.items()
    ):
        raise IsaacArenaError("invalid simulator capture rendering configuration")
    settings = rendering.get("settings")
    if (
        not isinstance(settings, dict)
        or set(settings) != set(_RENDER_SETTINGS)
        or any(
            type(settings[key]) is not type(value) or settings[key] != value
            for key, value in _RENDER_SETTINGS.items()
        )
    ):
        raise IsaacArenaError(
            "simulator renderer readback disagrees with capture settings"
        )
    return rendering


def _physics_freeze_proof(capture: dict[str, Any], total: int) -> dict[str, Any]:
    rows = capture.get("physics_freeze_checks")
    if (
        capture.get("physics_clock") != "native_physx_step_events_since_capture_setup"
        or not isinstance(rows, list)
        or len(rows) != total + 1
        or not all(_freeze_row_valid(row, step) for step, row in enumerate(rows))
    ):
        raise IsaacArenaError(
            "missing or inconsistent render-only physics freeze evidence"
        )
    for before, after in zip(rows, rows[1:]):
        if any(
            after[f"{field}_before"] <= before[f"{field}_after"]
            for field in ("physics_time", "physics_step", "native_physics_step")
        ):
            raise IsaacArenaError(
                "physics observation did not advance between real actions"
            )
    return {
        "verified_capture_count": len(rows),
        "simulation_advanced_during_render": False,
        "physics_state_changed_during_render": False,
        "rendering": _rendering_proof(capture),
    }


def verify_capture_evidence(
    run_dir: Path,
    raw_mp4: Path,
    *,
    task_motion: dict[str, Any],
    expected_steps: int | None = None,
) -> dict[str, Any]:
    """Verify native frame capture against HDF action steps and the raw MP4.

    Args:
        run_dir: Current upstream run directory containing the capture sidecar.
        raw_mp4: Untouched upstream MP4; derived videos are not source evidence.
        task_motion: Verified task interval and video_capture fields from HDF5.
        expected_steps: Exact native scored-episode action count when applicable.
    Returns:
        Verified sidecar/PNG hashes, action counts, and terminal pixel comparison.
    Raises:
        IsaacArenaError: Missing, inconsistent, or mismatched capture evidence.
    """
    capture, sidecar = _capture_sidecar(run_dir)
    # A native task may succeed in under one second; the raw source still has
    # to contain every action frame. The declared evidence derivative slows
    # playback without adding frames and retains the public one-second gate.
    metadata, _ = _video_metadata(raw_mp4, minimum_duration_seconds=0.0)
    total = _capture_action_count(capture, metadata, expected_steps)
    terminal = _capture_step_mapping(task_motion, total)
    physics_freeze = _physics_freeze_proof(capture, total)
    dimensions = (metadata["width"], metadata["height"])
    initial = _verify_capture_png(run_dir, capture.get("initial"), 0, dimensions)
    final = _verify_capture_png(run_dir, capture["terminals"][0], terminal, dimensions)
    comparison = _terminal_frame_comparison(raw_mp4, run_dir / final["path"], terminal)
    return {
        "sidecar": sidecar,
        "initial": initial,
        "terminal": final,
        "captured_action_steps": total,
        "physics_freeze": physics_freeze,
        "source_mp4_sha256": _file_sha256(raw_mp4),
        "terminal_frame_comparison": comparison,
    }
