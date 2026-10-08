"""Separate provenance-backed scene evidence from artificial frame padding."""

from __future__ import annotations

import json
from contextlib import closing
from itertools import zip_longest
from pathlib import Path
from typing import Any, Callable

import numpy as np

from npa.workbench.cosmos_evaluator.appearance_fidelity import _iter_rgb_frames
from npa.workflows.video_content_region import crop_content, validate_content_region


def prepare_spatial_evidence(
    record: Any, source: Path | None, video: Path | None, workdir: Path
) -> tuple[Path | None, Path | None, dict[str, Any]]:
    """Use recorded preparation bounds for all scene-quality evidence.

    Args:
        record: Optional content-region provenance from reference preparation.
        source: Original full reference video.
        video: Original full augmented video.
        workdir: Existing private directory for temporary evaluation crops.
    Returns:
        Evaluation paths and an auditable spatial-evidence report.
    Raises:
        ValueError: Provenance or complete source correspondence is invalid.
        RuntimeError: Media cannot be decoded or cropped.
    """
    if record is None:
        return source, video, {"mode": "full-frame", "reason": "no-content-provenance"}
    if source is None or video is None:
        raise ValueError("Content-region evaluation requires both complete videos")
    original, generated = validate_content_region(record, source, video)
    report = _evidence_record(record, original, generated)
    width, height = record["canvas"]
    if record["bounds"] == [0, 0, width, height]:
        return source, video, {**report, "mode": "full-frame", "padding": None}
    report["crop_encoding"] = "lossless-rgb-ffv1"
    report["padding"] = _measure_padding(source, video, record)
    scene_source, scene_video = (
        workdir / "source-scene.mkv",
        workdir / "variant-scene.mkv",
    )
    report["source_scene_sha256"] = crop_content(source, scene_source, record["bounds"])
    report["generated_scene_sha256"] = crop_content(
        video, scene_video, record["bounds"]
    )
    return scene_source, scene_video, report


def _evidence_record(record, original, generated):
    width, height = record["canvas"]
    left, top, right, bottom = record["bounds"]
    return {
        "mode": "prepared-scene",
        "content_region": record,
        "source_sha256": original["sha256"],
        "generated_sha256": generated["sha256"],
        "decoded_frames": original["decoded_frames"],
        "excluded_pixel_fraction": 1
        - (right - left) * (bottom - top) / (width * height),
    }


def _measure_padding(source, video, record):
    width, height = record["canvas"]
    left, top, right, bottom = record["bounds"]
    mask = np.ones((height, width), dtype=bool)
    mask[top:bottom, left:right] = False
    total = changed = error = frames = 0
    maximum = 0
    with (
        closing(_iter_rgb_frames(source, height, width)) as originals,
        closing(_iter_rgb_frames(video, height, width)) as generated,
    ):
        for original, augmented in zip_longest(originals, generated):
            if original is None or augmented is None:
                raise ValueError("Padding diagnostic found mismatched frame counts")
            difference = np.abs(original[mask].astype(np.int16) - augmented[mask])
            error += int(difference.sum())
            changed += int(np.count_nonzero(difference.max(axis=1) > 8))
            maximum = max(maximum, int(difference.max()))
            total += difference.shape[0]
            frames += 1
    if not total:
        raise ValueError("Padding diagnostic decoded no pixels")
    return {
        "status": "measured",
        "enforced": False,
        "included_in_scene_score": False,
        "decoded_frames": frames,
        "mean_absolute_rgb_error": error / (3 * total),
        "max_absolute_rgb_error": maximum,
        "changed_pixel_fraction": changed / total,
        "change_threshold_rgb_levels": 8,
    }


def remap_regions(regions_json: str, evidence: dict[str, Any], parser: Callable) -> str:
    """Intersect explicit full-canvas regions with the prepared scene rectangle.

    Args:
        regions_json: Existing normalized full-canvas metric regions, or empty.
        evidence: Verified spatial-evidence report.
        parser: The metric's existing region parser.
    Returns:
        Normalized crop-relative regions; defaults still tile the whole scene.
    Raises:
        ValueError: An explicit region contains only padding.
        RuntimeError: The metric parser rejects a malformed region.
    """
    if not regions_json or evidence["mode"] != "prepared-scene":
        return regions_json
    record = evidence["content_region"]
    width, height = record["canvas"]
    left, top, right, bottom = record["bounds"]
    remapped = []
    for region_id, (x0, y0, x1, y1) in parser(regions_json):
        x0, y0 = max(left, x0 * width), max(top, y0 * height)
        x1, y1 = min(right, x1 * width), min(bottom, y1 * height)
        if x0 >= x1 or y0 >= y1:
            raise ValueError("An explicit evaluation region contains only padding")
        remapped.append(
            {
                "id": region_id,
                "bounds": [
                    (x0 - left) / (right - left),
                    (y0 - top) / (bottom - top),
                    (x1 - left) / (right - left),
                    (y1 - top) / (bottom - top),
                ],
            }
        )
    return json.dumps(remapped)
