"""Prepare and verify the source timeline used by PAIDF Cosmos 3 transfer."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

WIDTH = 832
HEIGHT = 480
DEFAULT_FPS = 24


class VideoAlignmentError(RuntimeError):
    """Report failed video decoding or correspondence with the prepared source.

    Args:
        args: Standard exception details.
    Returns:
        Exception describing an invalid media contract.
    Raises:
        None.
    """


def _command(argv: list[str]) -> bytes:
    result = subprocess.run(argv, capture_output=True, check=False)
    if result.returncode:
        raise VideoAlignmentError(f"{argv[0]} failed while validating video media")
    return result.stdout


def video_sha256(path: Path) -> str:
    """Hash complete video bytes for input and evaluation lineage.

    Args:
        path: Existing local video.
    Returns:
        Lowercase SHA-256 digest.
    Raises:
        OSError: The video cannot be read.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timeline(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        stream, frames = payload["streams"][0], payload["frames"]
        stamps = [float(frame["best_effort_timestamp_time"]) for frame in frames]
        duration = float(stream.get("duration", payload.get("format", {}).get("duration")))
        dimensions = (int(stream["width"]), int(stream["height"]))
        valid = stamps and all(math.isfinite(value) for value in stamps)
        valid = valid and all(right > left for left, right in zip(stamps, stamps[1:]))
        valid = valid and math.isfinite(duration) and duration > 0
        valid = valid and all((frame["width"], frame["height"]) == dimensions for frame in frames)
        if not valid:
            raise ValueError("invalid presentation timeline")
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        raise VideoAlignmentError("Video has an invalid or incomplete presentation timeline") from exc
    return {"decoded_frames": len(stamps), "timestamps": stamps,
            "duration_seconds": duration, "width": dimensions[0], "height": dimensions[1]}


def probe_video(path: Path) -> dict[str, Any]:
    """Fully decode a video and measure its actual presentation timeline.

    Args:
        path: Nonempty regular local video.
    Returns:
        Decoded frame count, timestamps, dimensions, duration and SHA-256.
    Raises:
        VideoAlignmentError: Decoding, timestamps or dimensions are invalid.
        OSError: A required media executable or file is unavailable.
    """
    if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
        raise VideoAlignmentError("Video must be a nonempty regular file")
    raw = _command(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
                    "-show_entries", "stream=width,height,duration:format=duration:"
                    "frame=best_effort_timestamp_time,width,height", "-of", "json", str(path)])
    try:
        result = _timeline(json.loads(raw))
    except json.JSONDecodeError as exc:
        raise VideoAlignmentError("Video probe returned malformed evidence") from exc
    _command(["ffmpeg", "-v", "error", "-xerror", "-i", str(path),
              "-map", "0:v:0", "-f", "null", "-"])
    return {**result, "sha256": video_sha256(path), "full_decode_passed": True}


def validate_reference(timeline: dict[str, Any], fps: int = DEFAULT_FPS) -> None:
    """Require the prepared constant-rate model bucket without changing media.

    Args:
        timeline: Evidence from probe_video.
        fps: Requested prepared-source frame rate.
    Returns:
        None.
    Raises:
        VideoAlignmentError: Shape, duration or frame timestamps do not match.
    """
    if isinstance(fps, bool) or not isinstance(fps, int) or not 10 <= fps <= 30:
        raise VideoAlignmentError("conditioning_fps must be an integer from 10 through 30")
    expected_duration = timeline["decoded_frames"] / fps
    errors = [abs(stamp - index / fps) for index, stamp in enumerate(timeline["timestamps"])]
    if (timeline["decoded_frames"] < 6 or (timeline["width"], timeline["height"]) != (WIDTH, HEIGHT)
            or not errors or max(errors) > 0.00051
            or abs(timeline["duration_seconds"] - expected_duration) > 0.002):
        raise VideoAlignmentError("Prepared video does not satisfy its shape, frame-rate or timestamp contract")


def prepare_reference(source: Path, destination: Path, fps: int = DEFAULT_FPS) -> dict[str, Any]:
    """Normalize timing and letterbox the complete selected source for transfer.

    Args:
        source: Original selected video, retained by the caller.
        destination: New prepared video path.
        fps: Constant output frame rate between 10 and 30.
    Returns:
        Original/prepared measurements and the explicit normalization recipe.
    Raises:
        VideoAlignmentError: Source, conversion or resulting timeline is invalid.
        OSError: Required files or media executables are unavailable.
    """
    if isinstance(fps, bool) or not isinstance(fps, int) or not 10 <= fps <= 30:
        raise VideoAlignmentError("conditioning_fps must be an integer from 10 through 30")
    original = probe_video(source)
    filters = (f"setpts=PTS-STARTPTS,fps={fps}:start_time=0,"
               f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease:force_divisible_by=2,"
               f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2,setsar=1")
    _command(["ffmpeg", "-y", "-v", "error", "-xerror", "-i", str(source), "-an",
              "-vf", filters, "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(destination)])
    prepared = probe_video(destination)
    validate_reference(prepared, fps)
    if abs(prepared["duration_seconds"] - original["duration_seconds"]) > 1 / fps + 0.002:
        raise VideoAlignmentError("Normalization changed source duration by more than one output frame")
    return {"schema": "npa.paidf.cosmos3.timeline.v1", "status": "prepared",
            "fps": fps, "ffmpeg_filter": filters, "original": original, "prepared": prepared,
            "time_origin": "selected-source-start", "time_stretch": False}


def verify_pair(source: Path, augmented: Path, fps: int = DEFAULT_FPS) -> dict[str, Any]:
    """Verify that output frames correspond to the complete prepared timeline.

    Args:
        source: Prepared source used for structural controls and evaluation.
        augmented: Actual generated video after model guardrails.
        fps: Prepared-source frame rate.
    Returns:
        Hash-bound alignment evidence; visual quality remains separately evaluated.
    Raises:
        VideoAlignmentError: Either media or their frame correspondence is invalid.
        OSError: Required files or media executables are unavailable.
    """
    original, generated = probe_video(source), probe_video(augmented)
    validate_reference(original, fps)
    validate_reference(generated, fps)
    if original["decoded_frames"] != generated["decoded_frames"]:
        raise VideoAlignmentError("Generated video does not cover exactly the prepared source frames")
    error = max(abs(a - b) for a, b in zip(original["timestamps"], generated["timestamps"]))
    if error > 0.00051:
        raise VideoAlignmentError("Generated frames do not correspond to prepared source timestamps")
    return {"schema": "npa.paidf.cosmos3.alignment.v1", "status": "verified",
            "source_sha256": original["sha256"], "generated_sha256": generated["sha256"],
            "decoded_frames": original["decoded_frames"], "fps": fps,
            "duration_seconds": original["duration_seconds"], "max_timestamp_error_seconds": error,
            "full_decode_passed": True, "visual_quality_evaluated": False}
