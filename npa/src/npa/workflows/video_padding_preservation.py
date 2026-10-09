"""Restore verified source borders while retaining every generated scene pixel."""

from __future__ import annotations

import hashlib
import subprocess
import tempfile
from contextlib import closing
from fractions import Fraction
from itertools import zip_longest
from pathlib import Path
from typing import Any

import numpy as np

from npa.workflows.video_content_region import validate_content_region
from npa.workflows.video_padding_detection import detect_source_padding


def resolve_source_content_region(source: Path, record: Any = None) -> dict[str, Any]:
    """Validate supplied preparation evidence and detect embedded source bars.

    Args:
        source: Materialized prepared reference video.
        record: Existing content-region evidence, when present.
    Returns:
        Fresh hash-bound detection evidence for the complete reference.
    Raises:
        ValueError: Supplied evidence is invalid or contradicts measured padding.
        RuntimeError: Media decoding fails.
    """
    if record is None:
        return detect_source_padding(source)
    original, _ = validate_content_region(record, source, source)
    known = record.get("normalization_bounds", record["bounds"])
    resolved = detect_source_padding(source, original, known)
    if "padding_detection" in record and resolved["bounds"] != record["bounds"]:
        raise ValueError("Recorded padding differs from the complete source analysis")
    return resolved


def preserve_source_padding(
    source: Path, generated: Path, destination: Path, record: dict[str, Any]
) -> dict[str, Any] | None:
    """Restore only verified borders and prove scene pixels survived encoding.

    Args:
        source: Complete prepared reference video.
        generated: Retained raw model output.
        destination: New MP4 for publication; must differ from both input paths.
        record: Verified preparation and padding-detection evidence.
    Returns:
        Hash-bound restoration receipt, or None when no padding exists.
    Raises:
        ValueError: Correspondence, timing, paths or preserved pixels are invalid.
        RuntimeError: Decoding or lossless encoding fails.
    """
    if (
        destination.resolve() in {source.resolve(), generated.resolve()}
        or destination.exists()
    ):
        raise ValueError("Padding preservation requires a new output path")
    original, raw = validate_content_region(record, source, generated)
    width, height = record["canvas"]
    if record["bounds"] == [0, 0, width, height]:
        return None
    frame_rate = _frame_rate(original)
    expected = _encode_preserved(source, generated, destination, record, frame_rate)
    _, published = validate_content_region(record, source, destination)
    actual = _pixel_hashes(destination, record)
    if actual != expected:
        raise ValueError(
            "Lossless padding restoration changed the scene or reference border"
        )
    return _receipt(record, raw, published, expected)


def _frame_rate(media):
    rate = Fraction(
        media["decoded_frames"] / media["duration_seconds"]
    ).limit_denominator(1000000)
    if any(
        abs(stamp - index / float(rate)) > 0.00051
        for index, stamp in enumerate(media["timestamps"])
    ):
        raise ValueError(
            "Padding preservation requires the prepared constant-rate timeline"
        )
    return str(rate)


def _encoder_command(destination, record, frame_rate):
    width, height = record["canvas"]
    return [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-xerror",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        frame_rate,
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264rgb",
        "-preset",
        "veryfast",
        "-crf",
        "0",
        "-pix_fmt",
        "rgb24",
        "-fflags",
        "+bitexact",
        "-flags:v",
        "+bitexact",
        "-movflags",
        "+faststart",
        str(destination),
    ]


def _encode_preserved(source, generated, destination, record, frame_rate):
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            _encoder_command(destination, record, frame_rate),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=errors,
        )
        try:
            expected = _write_preserved_frames(source, generated, record, process.stdin)
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError("Lossless padding preservation encoding failed")
            return expected
        finally:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            if process.poll() is None:
                process.terminate()
                process.wait()


def _write_preserved_frames(source, generated, record, output):
    from npa.workbench.cosmos_evaluator.appearance_fidelity import _iter_rgb_frames

    width, height = record["canvas"]
    left, top, right, bottom = record["bounds"]
    mask = _border_mask(record)
    scene_hash, border_hash = hashlib.sha256(), hashlib.sha256()
    with (
        closing(_iter_rgb_frames(source, height, width)) as originals,
        closing(_iter_rgb_frames(generated, height, width)) as variants,
    ):
        for reference, variant in zip_longest(originals, variants):
            if reference is None or variant is None:
                raise ValueError("Padding preservation found mismatched frame counts")
            scene = variant[top:bottom, left:right]
            scene_hash.update(scene.tobytes())
            border_hash.update(reference[mask].tobytes())
            frame = reference.copy()
            frame[top:bottom, left:right] = scene
            output.write(frame.tobytes())
    return {
        "scene_rgb_sha256": scene_hash.hexdigest(),
        "padding_rgb_sha256": border_hash.hexdigest(),
    }


def _border_mask(record):
    width, height = record["canvas"]
    left, top, right, bottom = record["bounds"]
    mask = np.ones((height, width), dtype=bool)
    mask[top:bottom, left:right] = False
    return mask


def _pixel_hashes(video, record):
    from npa.workbench.cosmos_evaluator.appearance_fidelity import _iter_rgb_frames

    width, height = record["canvas"]
    left, top, right, bottom = record["bounds"]
    mask = _border_mask(record)
    scene_hash, border_hash = hashlib.sha256(), hashlib.sha256()
    for frame in _iter_rgb_frames(video, height, width):
        scene_hash.update(frame[top:bottom, left:right].tobytes())
        border_hash.update(frame[mask].tobytes())
    return {
        "scene_rgb_sha256": scene_hash.hexdigest(),
        "padding_rgb_sha256": border_hash.hexdigest(),
    }


def _receipt(record, raw, published, expected):
    return {
        "schema": "npa.video.padding-preservation.v1",
        "status": "verified",
        "policy": "restore-source-border-only",
        "source_sha256": record["source_sha256"],
        "raw_model_sha256": raw["sha256"],
        "published_sha256": published["sha256"],
        "bounds": record["bounds"],
        "decoded_frames": published["decoded_frames"],
        "scene_pixels_unchanged": True,
        "padding_matches_source": True,
        "encoding": "lossless-h264-rgb",
        **expected,
    }
