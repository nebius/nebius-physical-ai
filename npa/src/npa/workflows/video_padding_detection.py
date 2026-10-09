"""Detect persistent rectangular black padding without masking scene interiors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from npa.workflows.paidf_cosmos3_media import probe_video
from npa.workflows.video_content_region import content_region_record

DETECTION_ORIGIN = "source-padding-detection"
ALGORITHM = "persistent-paired-black-bands-v1"
BLACK_LEVEL = 12
BOUNDARY_LEVEL = 24
BOUNDARY_TRANSITION = 4


def detect_source_padding(
    source: Path,
    media: dict[str, Any] | None = None,
    normalization_bounds: list[int] | None = None,
) -> dict[str, Any]:
    """Measure stable paired black edge bands throughout the source video.

    Args:
        source: Fully prepared reference, including any embedded source borders.
        media: Optional complete reference measurements from preparation.
        normalization_bounds: Known scene bounds before inspecting embedded bars.
    Returns:
        Hash-bound content bounds and the measured detection decision.
    Raises:
        ValueError: The complete reference cannot be decoded consistently.
        RuntimeError: Decoding fails.
    """
    media = media or probe_video(source)
    width, height = media["width"], media["height"]
    known = normalization_bounds or [0, 0, width, height]
    measured, frames = _persistent_bounds(source, media, known)
    bounds, decision = _accept_bounds(measured, known, frames)
    record = content_region_record(media, bounds)
    record.update(origin=DETECTION_ORIGIN, normalization_bounds=known)
    record["padding_detection"] = {
        "algorithm": ALGORITHM,
        "status": decision,
        "decoded_frames": frames,
        "black_level": BLACK_LEVEL,
        "boundary_level": BOUNDARY_LEVEL,
        "boundary_transition_pixels": BOUNDARY_TRANSITION,
        "bounds": bounds,
        "source_sha256": media["sha256"],
    }
    return record


def _persistent_bounds(source, media, known):
    from npa.workbench.cosmos_evaluator.appearance_fidelity import _iter_rgb_frames

    left, top, right, bottom = known
    size = (bottom - top, right - left)
    row_black, column_black = np.ones(size[0], bool), np.ones(size[1], bool)
    row_bright, column_bright = np.zeros(size[0]), np.zeros(size[1])
    frames = 0
    for frame in _iter_rgb_frames(source, media["height"], media["width"]):
        scene = frame[top:bottom, left:right]
        # Every pixel/channel must stay near black in every decoded frame.
        # Interior black objects do not form paired full-width/full-height bands.
        row_black &= scene.max(axis=(1, 2)) <= BLACK_LEVEL
        column_black &= scene.max(axis=(0, 2)) <= BLACK_LEVEL
        brightness = scene.max(axis=2)
        row_bright += (brightness > BOUNDARY_LEVEL).mean(axis=1) >= 0.25
        column_bright += (brightness > BOUNDARY_LEVEL).mean(axis=0) >= 0.25
        frames += 1
    if frames != media["decoded_frames"]:
        raise ValueError("Padding detection did not decode the complete reference")
    return (row_black, column_black, row_bright, column_bright), frames


def _axis_bands(black, bright, frames):
    nonblack = np.flatnonzero(~black)
    if not len(nonblack):
        return 0, len(black), "ambiguous"
    first, last = int(nonblack[0]), int(nonblack[-1]) + 1
    before, after = first, len(black) - last
    if before == after == 0:
        return first, last, "none"
    if min(before, after) < 2 or abs(before - after) > 2:
        return 0, len(black), "ambiguous"
    if last - first < len(black) / 4:
        return 0, len(black), "ambiguous"
    # Resampling can soften the first few scene pixels. Keep the transition
    # inside the scene, but require a nearby persistent brightness step.
    start = bright[first : min(last, first + BOUNDARY_TRANSITION)].max()
    end = bright[max(first, last - BOUNDARY_TRANSITION) : last].max()
    if min(start, end) < 0.9 * frames:
        return 0, len(black), "ambiguous"
    # Keep uncertain boundary pixels with the scene, including odd-size bars.
    return first - first % 2, last + last % 2, "detected"


def _accept_bounds(measured, known, frames):
    if frames < 6:
        return list(known), "insufficient-frames"
    rows, columns, row_bright, column_bright = measured
    top, bottom, vertical = _axis_bands(rows, row_bright, frames)
    left, right, horizontal = _axis_bands(columns, column_bright, frames)
    statuses = {vertical, horizontal}
    if "ambiguous" in statuses:
        return list(known), "ambiguous"
    if "detected" not in statuses:
        return list(known), "no-additional-padding"
    x, y, _, _ = known
    return [left + x, top + y, right + x, bottom + y], "detected"
