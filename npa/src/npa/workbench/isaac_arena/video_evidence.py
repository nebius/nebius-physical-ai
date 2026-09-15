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
from .simulator_video import _RENDER_SETTINGS

_SAMPLE_WIDTH = 160
_SAMPLE_HEIGHT = 90
_MEDIAN_WINDOW = 5
_COMPARISON_LAG = 5
_MINIMUM_PAIRS = 2
_BLOCK_SIZE = 16
_BLOCK_STRIDE = 8
_SEARCH_RADIUS = 8
_DENOISE_FILTER = "hqdn3d=8:6:12:9"


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


def _video_metadata(path: Path) -> tuple[dict[str, Any], list[float]]:
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
    _validate_metadata(metadata, timestamps, path)
    return metadata, timestamps


def _validate_metadata(
    metadata: dict[str, Any], timestamps: list[float], path: Path
) -> None:
    if (
        metadata["codec"] != "h264"
        or metadata["pixel_format"] != "yuv420p"
        or metadata["width"] < 320
        or metadata["height"] < 240
        or not math.isfinite(metadata["duration_seconds"])
        or metadata["duration_seconds"] < 1.0
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
) -> int:
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
    return len(coherent)


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
        else 0
    )
    return {
        "previous_sample_index": index,
        "current_sample_index": index + _COMPARISON_LAG,
        "mean_abs_luma_delta": mean_delta,
        "changed_pixel_ratio": changed_ratio,
        "coherent_blocks": coherent,
        "continuation_sample_index": index + 2 * _COMPARISON_LAG,
        "meaningful": mean_delta >= 1 and changed_ratio >= 0.01 and coherent >= 2,
    }


def _motion_metadata(
    frames: np.ndarray, interval: dict[str, Any]
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
    accepted = [pair for pair in pairs if pair["meaningful"]]
    if len(accepted) < _MINIMUM_PAIRS:
        raise IsaacArenaError(
            "viewport MP4 is decodable but visually static or lacks "
            "noise-resistant coherent scene motion"
        )
    best = max(accepted, key=lambda pair: pair["coherent_blocks"])
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
        "analysis_interval": interval,
        "thresholds": video_acceptance_thresholds(),
        "meaningful": True,
    }, best


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
    path: Path, *, evidence_interval: dict[str, int] | None = None
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
    Returns:
        Codec metadata, robust motion statistics, and reproducible frame hashes.
    Raises:
        IsaacArenaError: Invalid video, mismatched action mapping, short interval,
            or insufficient noise-resistant geometric motion.
    """
    metadata, timestamps = _video_metadata(path)
    interval = _analysis_interval(metadata["frame_count"], evidence_interval)
    frames = _decode_interval(path, interval)
    motion, best = _motion_metadata(frames, interval)
    motion["sample_fps"] = (len(timestamps) - 1) / (timestamps[-1] - timestamps[0])
    metadata["motion"] = motion
    metadata["frame_evidence"] = _frame_evidence(frames, interval, timestamps, best)
    return metadata


def _render_denoised_video(source: Path, target: Path) -> None:
    arguments = ["ffmpeg", "-v", "error", "-i", str(source)]
    arguments += ["-vf", _DENOISE_FILTER, "-an", "-vsync", "0"]
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
        "filter": _DENOISE_FILTER,
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
        and row["render_calls"] >= 1
        and type(row.get("accumulation_render_calls")) is int
        and row["accumulation_render_calls"] == 0
        and all(
            row.get(field) is True
            for field in ("stage_streaming_idle", "stage_assets_loaded", "nonblack_rgb")
        )
    )


def _rendering_proof(capture: dict[str, Any]) -> dict[str, Any]:
    rendering = capture.get("rendering")
    expected = {
        "mode": "RaytracedLighting",
        "antialiasing": "FXAA",
        "stochastic_accumulation": False,
        "accumulation_renders_per_frame": 0,
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
        expected_steps: Exact source replay action count when applicable.
    Returns:
        Verified sidecar/PNG hashes, action counts, and terminal pixel comparison.
    Raises:
        IsaacArenaError: Missing, inconsistent, or mismatched capture evidence.
    """
    capture, sidecar = _capture_sidecar(run_dir)
    metadata, _ = _video_metadata(raw_mp4)
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
