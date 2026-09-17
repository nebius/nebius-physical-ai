"""Validate measured warehouse batches and hash their review artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

PHASES = (
    "ACCUMULATION_STOP",
    "APPROACH",
    "LOWER",
    "vacuum_attached",
    "LIFT",
    "TRANSFER",
    "LOWER_TO_PALLET",
    "vacuum_released",
    "carton_placed",
)
VIEWS = ("aisle", "overview", "packing")


def _finite(value: object) -> bool:
    return (
        isinstance(value, (float, int))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _below(value: object, threshold: float) -> bool:
    return _finite(value) and 0 <= value < threshold


def _ordered_events(events: list[dict], carton: int) -> bool:
    selected = [row for row in events if row.get("carton") == carton]
    cursor = 0
    previous = -1.0
    for row in selected:
        timestamp = row.get("sim_s")
        if not _finite(timestamp) or timestamp < previous:
            return False
        previous = timestamp
        if cursor < len(PHASES) and row.get("event") == PHASES[cursor]:
            cursor += 1
    return cursor == len(PHASES)


def _physics_checks(measured: dict, events: list[dict]) -> dict[str, bool]:
    placements = measured.get("settled", [])
    return {
        "physx_120hz": measured.get("physics_backend") == "physx"
        and _below(abs(measured.get("physics_dt", 0) - 1 / 120), 1e-9),
        "warehouse_geometry": measured.get("stage_prims", 0) > 4000,
        "eight_rigid_bodies": measured.get("rigid_bodies") == 8,
        "six_distinct_cartons": len(placements) == 6
        and {row.get("carton") for row in placements} == set(range(6)),
        "complete_batch": measured.get("completed_batches") == 1
        and measured.get("total_placements") == 6,
        "placement_error_under_2cm": bool(placements)
        and all(_below(row.get("error_m"), 0.02) for row in placements),
        "settled_speed_under_4cm_s": bool(placements)
        and all(_below(row.get("speed_m_s"), 0.04) for row in placements),
        "stopped_before_pickup": _below(measured.get("accumulation_speed_m_s"), 0.04),
        "attachment_error_under_4cm": _below(
            measured.get("max_attachment_error_m"), 0.04
        ),
        "contact_transport_over_3_5m": _finite(
            measured.get("max_conveyor_displacement_m")
        )
        and measured["max_conveyor_displacement_m"] > 3.5,
        "ordered_workflow_each_carton": all(
            _ordered_events(events, i) for i in range(6)
        ),
    }


def _frame_checks(directory: Path) -> tuple[dict, dict]:
    checks, frames = {}, {}
    for view in VIEWS:
        with Image.open(directory / f"{view}.png") as image:
            rgb = np.asarray(image.convert("RGB"))
        frames[view] = {
            "width": rgb.shape[1],
            "height": rgb.shape[0],
            "mean_rgb": float(rgb.mean()),
            "std_rgb": float(rgb.std()),
            "clipped_fraction": float(np.mean(np.min(rgb, axis=2) > 250)),
        }
        stats = frames[view]
        checks[f"{view}_dimensions"] = (stats["width"], stats["height"]) == (1280, 720)
        checks[f"{view}_exposure"] = (
            35 < stats["mean_rgb"] < 190 and stats["clipped_fraction"] < 0.01
        )
        checks[f"{view}_nonflat"] = stats["std_rgb"] > 10
    return checks, frames


def verify_evidence(directory: Path) -> dict:
    """Check all six placements, event order, telemetry, and decoded review images.

    Args:
        directory: Directory containing the batch's JSON and PNG artifacts.
    Returns:
        A JSON-compatible report with checks, pixel statistics, and content hashes.
    Raises:
        ValueError: JSON or measured telemetry is malformed.
        OSError: Required evidence is missing or an image cannot be decoded.
    """
    measured = json.loads((directory / "validation.json").read_text())
    events = json.loads((directory / "events.json").read_text())
    trajectory = json.loads((directory / "trajectory.json").read_text())
    checks = _physics_checks(measured, events)
    checks["measured_trajectory"] = _valid_trajectory(trajectory)
    frame_checks, frames = _frame_checks(directory)
    checks.update(frame_checks)
    names = [
        "validation.json",
        "events.json",
        "trajectory.json",
        *(f"{v}.png" for v in VIEWS),
    ]
    return {
        "schema": "npa.antioch-warehouse.verification.v1",
        "all_passed": all(checks.values()),
        "checks": checks,
        "frames": frames,
        "artifacts_sha256": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in names
        },
    }


def _valid_trajectory(samples: list[dict]) -> bool:
    if len(samples) < 2:
        return False
    previous = -1.0
    for row in samples:
        timestamp = row.get("sim_s")
        positions = row.get("cartons", [])
        if not _finite(timestamp) or timestamp <= previous or len(positions) != 6:
            return False
        if not all(
            len(position) == 3 and all(_finite(v) for v in position)
            for position in positions
        ):
            return False
        previous = timestamp
    return samples[-1].get("placed") == 6
