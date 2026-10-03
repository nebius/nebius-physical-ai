"""Embed only scored focal-episode renderer frames from sealed navigation artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image

from npa.workflows.navigation.artifacts import file_sha256
from npa.workflows.preview_html import image_preview


def _chronology(index):
    if index.get("robot_index") != 0 or index.get("renderer") != "isaac-replicator-rgb":
        raise ValueError("preview requires the actual focal Isaac renderer")
    rows = index["frames"]
    if any(type(row["step"]) is not int for row in rows):
        raise ValueError("native frame steps must be integers")
    if [row["step"] for row in rows] != list(range(len(rows))):
        raise ValueError("native frame chronology is incomplete")
    times = [row["simulation_seconds"] for row in rows]
    if any(
        not isinstance(time, (int, float)) or not math.isfinite(time) for time in times
    ):
        raise ValueError("native frame times must be finite")
    if (
        not times
        or times[0] != 0
        or any(left >= right for left, right in zip(times, times[1:]))
    ):
        raise ValueError("native frame simulation time is not strictly increasing")
    return rows


def scored_preview_frames(
    report: dict, index: dict, count: int = 32
) -> tuple[list, dict]:
    """Select display samples exclusively within the measured focal episode.

    Args:
        report: Native evaluation measurements from the sealed artifact bundle.
        index: Renderer frames.json from the same bundle.
        count: Maximum display samples; this does not change the evaluation.
    Returns:
        Selected frame rows and focal episode measurements.
    Raises:
        ValueError: Renderer identity, chronology or scored endpoint differs.
    """
    if type(count) is not int or count < 2:
        raise ValueError("preview count must allow both scored endpoints")
    rows = _chronology(index)
    focal = report["episodes"][0]
    terminal = focal["steps"]
    if type(terminal) is not int or terminal < 1 or terminal >= len(rows):
        raise ValueError("scored focal endpoint is absent from native frames")
    distance = focal["goal_distance_m"]
    if not math.isfinite(distance) or not math.isclose(
        distance, rows[terminal]["goal_distance_m"], rel_tol=1e-5, abs_tol=1e-5
    ):
        raise ValueError("scored focal endpoint differs from the renderer metadata")
    length = min(count, terminal + 1)
    indices = [round(index * terminal / (length - 1)) for index in range(length)]
    return [rows[index] for index in indices], focal


def _verified_member(root, relative, checksums):
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError("preview artifact escapes the sealed bundle")
    if relative not in checksums or file_sha256(path) != checksums[relative]:
        raise ValueError("preview artifact checksum differs from its seal")
    return path


def _embedded_frames(root, selected, checksums, title):
    frames = []
    for row in selected:
        name = f"rendered-rollout/{row['step']:06d}.png"
        path = _verified_member(root, name, checksums)
        with Image.open(path) as pixels:
            encoded = image_preview(pixels)
        frames.append(
            {
                "label": f"Step {row['step']} · t={row['simulation_seconds']:.2f}s",
                "images": [{"label": title, "data": encoded}],
            }
        )
    return frames


def scored_rollout_group(evaluation: Path, result: dict, *, title: str) -> dict:
    """Build an offline timeline while excluding post-termination observer motion.

    Args:
        evaluation: Materialized sealed native evaluation directory.
        result: Its decoded evaluation.json, already bound to the experiment.
        title: Display label for the evaluation arm and cohort.
    Returns:
        A self-contained group accepted by preview_html.write_preview.
    Raises:
        ValueError: Report identity, media hashes or scored-frame evidence differs.
        OSError: A required native artifact cannot be read.
    """
    checksums = json.loads((evaluation / "checksums.json").read_text())
    record = _verified_member(evaluation, "evaluation.json", checksums)
    if json.loads(record.read_text()) != result:
        raise ValueError("preview evaluation differs from the sealed report")
    index = _verified_member(evaluation, "rendered-rollout/frames.json", checksums)
    selected, focal = scored_preview_frames(result, json.loads(index.read_text()))
    return {
        "title": title,
        "frames": _embedded_frames(evaluation, selected, checksums, title),
        "note": (
            f"Actual Isaac frames of focal robot 0, scored through step {focal['steps']}; "
            f"success={focal['success']}, final goal distance={focal['goal_distance_m']:.2f} m. "
            "Sampled-frame playback is not realtime. Later observer frames are excluded. "
            "Cohort success includes all robots, not only the pictured robot."
        ),
    }
