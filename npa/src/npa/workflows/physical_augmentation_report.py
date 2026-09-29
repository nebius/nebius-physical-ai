"""Verify recorded transitions and export only successful physical demonstrations."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from npa.workflows.lerobot_transfer_data import file_sha256, write_json
from npa.workflows.physical_augmentation_contract import (
    accepted_steps,
    longest_hold,
    read_recipe,
)

_CHANNELS = (
    "rgb",
    "state",
    "actions",
    "next_state",
    "object",
    "tcp",
    "next_object",
    "next_tcp",
    "next_velocity",
    "timestamp",
)


def verify_episode(root: Path, recipe: dict, dt: float) -> dict:
    """Reject broken alignment, stale states, invalid images, and false success labels.

    Args:
        root: One recorded simulator attempt.
        recipe: Predeclared acceptance contract.
        dt: Measured simulation control interval.
    Returns:
        Recomputed attempt outcome with content hashes.
    Raises:
        ValueError: Recording bytes or claimed outcomes violate the contract.
        OSError: Required evidence is missing.
    """
    result = json.loads((root / "result.json").read_text())
    arrays = {
        key: np.load(root / f"{key}.npy", allow_pickle=False, mmap_mode="r")
        for key in _CHANNELS
    }
    count = result["length"]
    _validate_arrays(arrays, count, dt)
    if not np.array_equal(arrays["object"][0], result["initial_object_m"]):
        raise ValueError("Initial object pose differs from the recorded state")
    hold = longest_hold(
        accepted_steps(
            arrays, np.asarray(result["initial_object_m"]), recipe["success"]
        )
    )
    success = not result["terminated"] and hold >= recipe["success"]["hold_steps"]
    if result["success"] is not success or result["longest_hold_steps"] != hold:
        raise ValueError("Claimed outcome differs from measured physical acceptance")
    return {
        **result,
        "files": {f"{key}.npy": file_sha256(root / f"{key}.npy") for key in _CHANNELS},
    }


def _validate_arrays(arrays: dict, count: int, dt: float) -> None:
    if count < 2 or not np.isfinite(dt) or dt <= 0:
        raise ValueError(
            "An attempt needs at least two transitions and positive control_dt"
        )
    for key, value in arrays.items():
        if len(value) != count or not np.isfinite(value).all():
            raise ValueError(f"Nonfinite or misaligned {key}")
    shapes = {
        "state": (count, 9),
        "next_state": (count, 9),
        "object": (count, 3),
        "tcp": (count, 3),
        "actions": (count, 8),
        "next_object": (count, 3),
        "next_tcp": (count, 3),
        "next_velocity": (count, 3),
        "timestamp": (count,),
    }
    if any(arrays[key].shape != shape for key, shape in shapes.items()):
        raise ValueError(
            "Recorded channels differ from the public Franka transition schema"
        )
    if not np.allclose(arrays["timestamp"], np.arange(count) * dt, atol=1e-8, rtol=0):
        raise ValueError("Timestamps are not consecutive simulation control steps")
    if not np.array_equal(arrays["state"][1:], arrays["next_state"][:-1]):
        raise ValueError("State transitions are discontinuous or contain a reset")
    for field in ("object", "tcp"):
        if not np.array_equal(arrays[field][1:], arrays[f"next_{field}"][:-1]):
            raise ValueError(f"Measured {field} transitions are discontinuous")
    actions = arrays["actions"]
    if (
        not np.allclose(np.linalg.norm(actions[:, 3:7], axis=1), 1.0, atol=1e-4)
        or not np.isin(actions[:, 7], [-1.0, 1.0]).all()
    ):
        raise ValueError("Recorded IK quaternion or binary gripper action is invalid")
    _validate_images(arrays["rgb"], count)


def _validate_images(rgb: np.ndarray, count: int) -> None:
    if rgb.shape != (count, 480, 640, 3) or rgb.dtype != np.uint8:
        raise ValueError("Recording lacks genuine configured RGB frames")
    if any(np.ptp(frame) < 8 for frame in rgb) or all(
        np.array_equal(rgb[0], frame) for frame in rgb[1:]
    ):
        raise ValueError("RTX recording is blank or temporally frozen")


def _verified_attempts(source: Path, recipe: dict) -> tuple[list[dict], dict]:
    attempts, identity = [], None
    for condition in recipe["conditions"]:
        root = source / condition
        capture = json.loads((root / "capture.json").read_text())
        if (
            capture["condition"] != condition
            or capture["simulation_validity_checks"] <= 0
            or capture.get("tcp_contract") != recipe["tcp_contract"]
            or capture.get("tool_frame_checked") is not True
        ):
            raise ValueError("Capture lacks native physics validity evidence")
        current = {
            key: capture[key]
            for key in ("state_names", "control_dt", "runtime_version")
        }
        if identity is not None and current != identity:
            raise ValueError("Cases have inconsistent robot, timestep, or runtime")
        identity = current
        for index in range(recipe["episodes_per_condition"]):
            episode = root / f"episode_{index:06d}"
            result = verify_episode(episode, recipe, capture["control_dt"])
            if (result["condition"], result["attempt"], result["seed"]) != (
                condition,
                index,
                recipe["seed"] + index,
            ):
                raise ValueError("Attempt identity differs from its sealed case")
            attempts.append(
                {
                    **result,
                    "source": episode.relative_to(source).as_posix(),
                    "physics": capture["physics"],
                }
            )
    return attempts, identity


def _export_metadata(recipe: dict, accepted: list[dict], identity: dict) -> dict:
    return {
        "format": "npa_isaac_lab_rollout_v2",
        "robot_type": "franka",
        "run_id": recipe["run_id"],
        "task": recipe["task"],
        "state_names": identity["state_names"],
        "action_names": recipe["action_names"],
        "action_semantics": recipe["action_semantics"],
        "alignment": recipe["alignment"],
        "fps": 1 / identity["control_dt"],
        "num_episodes": len(accepted),
        "episode_results": accepted,
        "episode_lengths": [row["length"] for row in accepted],
        "genuine_simulator_pixels": True,
        "physical_robot_tested": False,
    }


def _export(
    source: Path, output: Path, recipe: dict, accepted: list[dict], identity: dict
) -> dict:
    from npa.adapter.isaac_lab_lerobot import LeRobotFeatureSpec, convert
    from npa.workflows.franka_rl_recording import write_recording

    metadata = _export_metadata(recipe, accepted, identity)
    with tempfile.TemporaryDirectory(prefix="physical-accepted-") as temporary:
        staging = Path(temporary)
        for index, row in enumerate(accepted):
            shutil.copytree(source / row["source"], staging / f"episode_{index:06d}")
        write_json(staging / "meta.json", metadata)
        spec = LeRobotFeatureSpec(
            identity["state_names"], recipe["action_names"], "franka"
        )
        convert(
            staging,
            output / "lerobot",
            fps=round(metadata["fps"]),
            robot_type="franka",
            task="Lift the cube and hold it steady",
            spec=spec,
        )
    counts = write_recording(
        output / "lerobot", output / "demonstrations.rrd", metadata
    )
    write_json(output / "accepted-provenance.json", metadata)
    return counts


def _summary(recipe: dict, attempts: list[dict], accepted: list[dict]) -> dict:
    return {
        "schema": "npa.physical-augmentation.report.v1",
        "run_id": recipe["run_id"],
        "attempted": len(attempts),
        "accepted": len(accepted),
        "attempts": attempts,
        "conditions": {
            name: {
                "attempts": sum(row["condition"] == name for row in attempts),
                "accepted": sum(row["condition"] == name for row in accepted),
            }
            for name in recipe["conditions"]
        },
        "controller": recipe["controller"],
        "physical_robot_tested": False,
        "policy_improvement_measured": False,
        "generated_video_actions_verified": False,
    }


def report_results(source: Path, output: Path) -> dict:
    """Publish measured coverage, successful LeRobot data, and factual Rerun views.

    Args:
        source: Verified collection stage, including every attempted case.
        output: New destination directory for validated exports.
    Returns:
        Summary containing successes and failures for all physical conditions.
    Raises:
        ValueError: Evidence is invalid or no attempt produced an accepted demonstration.
        OSError: Input or output files cannot be accessed.
    """
    recipe = read_recipe(source / "recipe.json")
    attempts, identity = _verified_attempts(source, recipe)
    accepted = [row for row in attempts if row["success"]]
    summary = _summary(recipe, attempts, accepted)
    write_json(
        output / "attempts.json",
        {**summary, "schema": "npa.physical-augmentation.attempts.v1"},
    )
    if not accepted:
        raise ValueError(
            "No physically accepted demonstrations; retained attempts are diagnostic data"
        )
    counts = _export(source, output, recipe, accepted, identity)
    write_json(output / "recording-validation.json", {"entity_counts": counts})
    write_json(output / "report.json", summary)
    return summary
