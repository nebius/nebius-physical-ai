"""Validate scene bounds from preparation geometry or source-padding detection."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

from npa.workflows.paidf_cosmos3_media import _command, probe_video, video_sha256

SCHEMA = "npa.video.content-region.v1"
ORIGIN = "reference-aspect-ratio-padding"


def measure_scaled_size(source: Path, scale_filter: str) -> tuple[int, int]:
    """Measure the actual FFmpeg scaling result, including display rotation.

    Args:
        source: Original selected video.
        scale_filter: The same scaling filter used to prepare the reference.
    Returns:
        Scaled width and height in pixels.
    Raises:
        ValueError: FFmpeg did not return a PNG header.
        RuntimeError: FFmpeg could not decode the source.
    """
    header = _command(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(source),
            "-vf",
            scale_filter,
            "-frames:v",
            "1",
            "-c:v",
            "png",
            "-f",
            "image2pipe",
            "-",
        ]
    )
    if header[:8] != b"\x89PNG\r\n\x1a\n" or len(header) < 24:
        raise ValueError("Cannot measure the normalized scene rectangle")
    return struct.unpack(">II", header[16:24])


def content_region_record(
    prepared: dict[str, Any], bounds: list[int]
) -> dict[str, Any]:
    """Bind a preprocessing rectangle to the prepared reference bytes.

    Args:
        prepared: Fully decoded reference measurements.
        bounds: Left, top, exclusive right and exclusive bottom pixel coordinates.
    Returns:
        Serializable content-region provenance.
    Raises:
        KeyError: Required reference measurements are missing.
    """
    return {
        "schema": SCHEMA,
        "origin": ORIGIN,
        "source_sha256": prepared["sha256"],
        "canvas": [prepared["width"], prepared["height"]],
        "bounds": bounds,
    }


def validate_content_region(
    record: Any, source: Path, video: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify region provenance and require a matching complete video pair.

    Args:
        record: Content rectangle recorded during source preparation.
        source: Current full reference video.
        video: Current full generated video.
    Returns:
        Full reference and generated media measurements.
    Raises:
        ValueError: The record, reference hash or video correspondence is invalid.
        RuntimeError: A video cannot be decoded completely.
    """
    _validate_record(record)
    original, generated = probe_video(source), probe_video(video)
    if record["source_sha256"] != original["sha256"]:
        raise ValueError("Content region does not match the reference video hash")
    detection = record.get("padding_detection")
    if detection and detection["decoded_frames"] != original["decoded_frames"]:
        raise ValueError("Padding detection does not cover the complete reference")
    for media in (original, generated):
        if record["canvas"] != [media["width"], media["height"]]:
            raise ValueError("Content region canvas differs from video dimensions")
    if (
        original["decoded_frames"] != generated["decoded_frames"]
        or any(
            abs(left - right) > 0.00051
            for left, right in zip(original["timestamps"], generated["timestamps"])
        )
        or abs(original["duration_seconds"] - generated["duration_seconds"]) > 0.002
    ):
        raise ValueError("Content-region evaluation requires aligned complete videos")
    return original, generated


def _validate_record(record: Any) -> None:
    if not isinstance(record, dict) or record.get("schema") != SCHEMA:
        raise ValueError("Invalid content-region schema")
    if record.get("origin") == "source-padding-detection":
        _validate_detection_record(record)
        return
    if record.get("origin") != ORIGIN:
        raise ValueError("Content region must originate in reference preparation")
    digest = record.get("source_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("Content region requires the prepared source SHA-256")
    canvas, bounds = record.get("canvas"), record.get("bounds")
    if not isinstance(canvas, list) or len(canvas) != 2:
        raise ValueError("Content region requires a two-dimensional canvas")
    if not isinstance(bounds, list) or len(bounds) != 4:
        raise ValueError("Content region requires four pixel coordinates")
    if any(type(value) is not int for value in canvas + bounds):
        raise ValueError("Content-region coordinates must be integers")
    width, height = canvas
    left, top, right, bottom = bounds
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError("Content region lies outside its canvas")
    # Reference preparation only adds centered, even-sized YUV420 padding.
    if (
        any(value % 2 for value in canvas + bounds)
        or (abs(left - (width - right)) > 2 or abs(top - (height - bottom)) > 2)
        or (right - left != width and bottom - top != height)
    ):
        raise ValueError("Content region is not an aspect-ratio padding rectangle")


def _validate_detection_record(record):
    known = {**record, "origin": ORIGIN, "bounds": record.get("normalization_bounds")}
    _validate_record(known)
    evidence = record.get("padding_detection")
    bounds = record.get("bounds")
    if (
        not isinstance(evidence, dict)
        or not isinstance(bounds, list)
        or len(bounds) != 4
    ):
        raise ValueError("Detected padding requires measured rectangular evidence")
    if any(type(value) is not int or value % 2 for value in bounds):
        raise ValueError("Detected padding bounds must be even pixel coordinates")
    left, top, right, bottom = bounds
    x0, y0, x1, y1 = known["bounds"]
    if not (x0 <= left < right <= x1 and y0 <= top < bottom <= y1):
        raise ValueError("Detected padding falls outside the prepared scene")
    if (
        evidence.get("algorithm") != "persistent-paired-black-bands-v1"
        or evidence.get("source_sha256") != record["source_sha256"]
        or evidence.get("bounds") != bounds
        or type(evidence.get("decoded_frames")) is not int
        or evidence["decoded_frames"] < 1
        or evidence.get("black_level") != 12
        or evidence.get("boundary_level") != 24
        or evidence.get("boundary_transition_pixels") != 4
        or evidence.get("status")
        not in {"detected", "no-additional-padding", "ambiguous", "insufficient-frames"}
    ):
        raise ValueError("Detected padding evidence is invalid")
    if evidence["status"] == "detected":
        _validate_detected_bands(bounds, known["bounds"], evidence["decoded_frames"])
    if evidence["status"] != "detected" and bounds != known["bounds"]:
        raise ValueError("Uncertain padding cannot reduce the scene rectangle")


def _validate_detected_bands(bounds, known, frames):
    if frames < 6 or bounds == known:
        raise ValueError("Detected padding requires persistent additional bands")
    for axis in (0, 1):
        before = bounds[axis] - known[axis]
        after = known[axis + 2] - bounds[axis + 2]
        retained = bounds[axis + 2] - bounds[axis]
        extent = known[axis + 2] - known[axis]
        if before == after == 0:
            continue
        if min(before, after) < 2 or abs(before - after) > 2 or retained < extent / 4:
            raise ValueError("Detected padding requires conservative paired bands")


def crop_content(source: Path, destination: Path, bounds: list[int]) -> str:
    """Write a lossless RGB scene crop without resizing or dropping frames.

    Args:
        source: Full video with verified content bounds.
        destination: New temporary Matroska video.
        bounds: Validated left, top, right and bottom coordinates.
    Returns:
        SHA-256 of the derived evaluation video.
    Raises:
        RuntimeError: FFmpeg could not decode or encode the video.
        OSError: An executable or file is unavailable.
    """
    left, top, right, bottom = bounds
    # Reconstruct RGB before cropping to preserve actual boundary pixels losslessly.
    filters = f"format=rgb24,crop={right - left}:{bottom - top}:{left}:{top}:exact=1"
    command = ["ffmpeg", "-nostdin", "-y", "-v", "error", "-xerror", "-i", str(source)]
    _command(
        command
        + [
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            filters,
            "-fps_mode",
            "passthrough",
            "-c:v",
            "ffv1",
            "-fflags",
            "+bitexact",
            "-flags:v",
            "+bitexact",
            "-pix_fmt",
            "bgr0",
            str(destination),
        ]
    )
    return video_sha256(destination)
