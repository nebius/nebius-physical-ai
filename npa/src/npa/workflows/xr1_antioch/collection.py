"""Record successful and failed demonstrations with synchronized actions and physical outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cameras import _Cameras
from .dataset import CAMERAS, validate_episode
from .scene import _Cell


def _append(columns: dict, values: dict) -> None:
    for name, value in values.items():
        columns.setdefault(name, []).append(value)


def _outcome(measurements: list[dict]) -> dict:
    positions = np.array([row["positions"] for row in measurements])
    final = measurements[-1]
    distances = np.linalg.norm(
        positions[-1, :, :2] - np.array(final["targets"])[:, :2], axis=1
    )
    # Contact solvers can report gravity integration velocity on a resting body.
    # Qualification uses observed motion over the final second of simulation.
    speeds = np.linalg.norm(positions[-1] - positions[-21], axis=1)
    drift = np.linalg.norm(positions[-20:] - positions[-1], axis=2).max(axis=0)
    heights = positions[:, :, 2].max(axis=0)
    settled = np.abs(positions[-1, :, 2] - 0.02) < 0.005
    released = np.array(final["gripper_apertures"]) > 0.075
    successes = (
        (heights > 0.12)
        & (distances < 0.035)
        & (speeds < 0.03)
        & (drift < 0.003)
        & settled
        & released
    )
    return {
        "success": bool(successes.all()),
        "arm_success": successes.tolist(),
        "maximum_height_m": heights.tolist(),
        "placement_error_m": distances.tolist(),
        "final_speed_m_s": speeds.tolist(),
        "settling_drift_m": drift.tolist(),
        "solver_velocity_m_s": final["velocities"],
        "released": released.tolist(),
    }


def _record_expert(cell: _Cell, cameras: _Cameras, episode: dict) -> None:
    started = cell.world.current_time
    completed_at = None
    for tick in range(1500):
        if tick % 3 == 0:
            cameras.follow_wrists(cell.robots)
            cell.world.render()
            cameras.capture()
            _append(episode["proprios"], cell.state())
            episode["timestamps"].append(cell.world.current_time - started)
            episode["measurements"].append(cell.measures())
        cell.expert_step()
        if tick % 3 == 0:
            _append(episode["actions"], cell.action_targets())
        if completed_at is None and all(expert.is_done() for expert in cell.experts):
            completed_at = tick
        if completed_at is not None and tick - completed_at >= 63 and tick % 3 == 2:
            break
    episode["num_frames"] = len(episode["timestamps"])
    episode.update(_outcome(episode["measurements"]))


def _episode_metadata(seed: int, split: str) -> dict:
    return {
        "schema": "npa.xr1-antioch.episode.v1",
        "episode_id": f"{split}-{seed}",
        "seed": seed,
        "split": split,
        "control_hz": 20,
        "physics_hz": 60,
        "grasp_mechanism": "finger_contact",
        "embodiment": "dual_franka",
        "ee_frame": "lula/right_gripper",
        "gripper_units": "full_aperture_metres",
        "action_semantics": "issued_cartesian_servo_targets_at_observation_time",
        "controller": "scripted_pick_place_expert_with_RMPflow_joint_actuation",
        "videos": {camera: f"{camera}.mp4" for camera in CAMERAS},
        "timestamps": [],
        "proprios": {},
        "actions": {},
        "measurements": [],
    }


def collect_episode(output: Path, seed: int, split: str) -> dict:
    """Execute one native episode and retain all outcomes before closing Isaac Sim.

    Args:
        output: New episode directory.
        seed: Deterministic scene randomization seed.
        split: Episode-level train, validation, or test assignment.
    Returns:
        Physical outcome and recording metadata.
    Raises:
        FileExistsError: Destination already exists.
        RuntimeError: Simulation or camera capture fails.
        ValueError: Exported state/action synchronization is invalid.
    """
    from isaacsim import SimulationApp

    output.mkdir(parents=True, exist_ok=False)
    app = SimulationApp({"headless": True, "width": 960, "height": 720})
    cameras = None
    try:
        cell = _Cell(seed)
        cameras = _Cameras(cell.random, output)
        for _ in range(30):
            cameras.follow_wrists(cell.robots)
            cell.world.step(render=True)
        episode = _episode_metadata(seed, split)
        _record_expert(cell, cameras, episode)
        validate_episode(episode)
        (output / "episode.json").write_text(json.dumps(episode, allow_nan=False))
        print(
            json.dumps(
                {
                    "episode_id": episode["episode_id"],
                    **_outcome(episode["measurements"]),
                }
            ),
            flush=True,
        )
        return episode
    finally:
        if cameras is not None:
            cameras.close()
        app.close()


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), required=True
    )
    arguments = parser.parse_args()
    collect_episode(arguments.output_path, arguments.seed, arguments.split)


if __name__ == "__main__":
    _main()
