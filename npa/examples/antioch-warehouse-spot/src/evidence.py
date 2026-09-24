"""Measure simulated locomotion and publish footage with its physical-state evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _summarize(samples: list[dict], frame_count: int, expected_frames: int) -> dict:
    positions = np.array([sample["position"] for sample in samples])
    joints = np.array([sample["joints"] for sample in samples])
    orientations = np.array([sample["orientation"] for sample in samples])
    quaternion_norms = np.linalg.norm(orientations, axis=1)
    up_z = 1.0 - 2.0 * (orientations[:, 1] ** 2 + orientations[:, 2] ** 2)
    distance = float(np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1).sum())
    return {
        "frames": frame_count,
        "expected_frames": expected_frames,
        "distance_m": distance,
        "displacement_m": float(np.linalg.norm(positions[-1, :2] - positions[0, :2])),
        "minimum_base_height_m": float(positions[:, 2].min()),
        "maximum_base_height_m": float(positions[:, 2].max()),
        "maximum_joint_range_rad": float(np.ptp(joints, axis=0).max()),
        "minimum_body_up_z": float(up_z.min()),
        "finite_state": bool(
            np.isfinite(positions).all()
            and np.isfinite(joints).all()
            and np.isfinite(orientations).all()
            and np.all((quaternion_norms > 0.99) & (quaternion_norms < 1.01))
        ),
    }


def _checks(run, metrics, moving: bool) -> None:
    for name, value in metrics.items():
        run.add_result(name, value)
    run.check("finite physical state", metrics["finite_state"])
    run.check(
        "robot remains upright",
        metrics["minimum_base_height_m"] > 0.35 and metrics["minimum_body_up_z"] > 0.7,
    )
    run.check("camera frames advance", metrics["frames"] == metrics["expected_frames"])
    run.check("camera contains visible scene", metrics["minimum_frame_std"] > 10.0)
    run.check("policy advances", metrics["policy_steps"] > 0)
    if moving:
        run.check("robot travels", metrics["displacement_m"] > 1.0)
        run.check("joints articulate", metrics["maximum_joint_range_rad"] > 0.2)


def _publish(
    run,
    folder: Path,
    samples: list[dict],
    recording,
    expected: int,
    stepper,
    moving: bool,
) -> None:
    from isaacsim.core.simulation_manager import SimulationManager

    metrics = _summarize(samples, recording.frames, expected)
    metrics["policy_steps"] = stepper.steps
    metrics["physics_dt_s"] = SimulationManager.get_physics_dt()
    metrics["policy_physics_dt_s"] = stepper.controller._dt
    metrics["policy_decimation"] = stepper.controller._decimation
    metrics["minimum_frame_std"] = min(recording.pixel_variances, default=0.0)
    payload = {
        "metrics": metrics,
        "states": samples,
        "camera_timestamps": recording.timestamps,
    }
    (folder / "measurements.json").write_text(json.dumps(payload, indent=2))
    _checks(run, metrics, moving)
    run.add_artifact(
        folder / "measurements.json",
        name="measurements",
        content_type="application/json",
    )
    if recording.frames:
        run.add_artifact(
            folder / "warehouse-spot.mp4",
            name="warehouse_spot",
            content_type="video/mp4",
        )
