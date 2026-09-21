"""Replay durable cuRobo trajectories through independent FK and Pinocchio paths."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
from importlib.metadata import version
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np

from .benchmark_inventory import DATASET_FILES
from .schemas import DATASET_REVISION, SOURCE_REVISION


class ReplayError(RuntimeError):
    """Durable planner evidence cannot be independently replayed."""


QUATERNION_REPLAY_ATOL_RAD = 1e-5


def _runtime_source() -> Path:
    source = Path(os.environ.get("NPA_CUROBO_SOURCE", "/opt/curobo"))
    if (source / "NPA_SOURCE_REVISION").read_text().strip() != SOURCE_REVISION:
        raise ReplayError("replay source revision differs")
    if version("nvidia-curobo") != "0.8.0":
        raise ReplayError("replay cuRobo package version differs")
    return source


def _benchmark_module(source: Path):
    dataset = Path(os.environ.get("NPA_CUROBO_DATASET_SOURCE", "/opt/robometrics"))
    if (dataset / "NPA_SOURCE_REVISION").read_text().strip() != DATASET_REVISION:
        raise ReplayError("replay benchmark revision differs")
    for filename, expected in DATASET_FILES.values():
        path = dataset / "robometrics/content/dataset" / filename
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ReplayError("replay benchmark bytes differ")
    package = (dataset / "robometrics").resolve(strict=True)
    for name, module in tuple(sys.modules.items()):
        if name == "robometrics" or name.startswith("robometrics."):
            origin = getattr(module, "__file__", None)
            if not origin or not Path(origin).resolve().is_relative_to(package):
                raise ReplayError("replay benchmark import escaped verified source")
    sys.path.insert(0, str(dataset.resolve(strict=True)))
    importlib.invalidate_caches()
    spec = importlib.util.spec_from_file_location(
        "npa_curobo_independent_replay",
        source / "benchmark/motion_plan_benchmark.py",
    )
    if spec is None or spec.loader is None:
        raise ReplayError("pinned replay benchmark module is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tensor_array(tensor) -> np.ndarray:
    return tensor.detach().cpu().reshape(-1, tensor.shape[-1]).numpy()


def _quaternion_comparison(
    first: np.ndarray, second: np.ndarray
) -> tuple[float, float]:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if (
        first.shape != (4,)
        or second.shape != (4,)
        or not np.isfinite(first).all()
        or not np.isfinite(second).all()
        or not math.isfinite(first_norm)
        or not math.isfinite(second_norm)
        or first_norm <= 1e-12
        or second_norm <= 1e-12
    ):
        raise ReplayError("quaternion replay contains invalid values")
    component_delta = min(
        float(np.max(np.abs(first - second))),
        float(np.max(np.abs(first + second))),
    )
    first = first / first_norm
    second = second / second_norm
    difference = min(
        float(np.linalg.norm(first - second)),
        float(np.linalg.norm(first + second)),
    )
    summation = max(
        float(np.linalg.norm(first - second)),
        float(np.linalg.norm(first + second)),
    )
    angle = float(4.0 * np.arctan2(difference, summation))
    if not math.isfinite(angle) or not math.isfinite(component_delta):
        raise ReplayError("quaternion replay contains invalid values")
    return angle, component_delta


def _quaternion_distance(first: np.ndarray, second: np.ndarray) -> float:
    return _quaternion_comparison(first, second)[0]


def _require_quaternion_replay(
    actual: np.ndarray, expected: np.ndarray
) -> tuple[float, float]:
    if (
        actual.shape != expected.shape
        or actual.ndim != 2
        or actual.shape[1:] != (4,)
        or not len(actual)
    ):
        raise ReplayError("FK quaternion replay shape differs")
    comparisons = [
        _quaternion_comparison(first, second) for first, second in zip(actual, expected)
    ]
    max_angle = max(item[0] for item in comparisons)
    max_component_delta = max(item[1] for item in comparisons)
    if max_angle > QUATERNION_REPLAY_ATOL_RAD:
        raise ReplayError(
            "FK quaternion replay differs: "
            f"max angular error {max_angle:.17g} rad; "
            "max sign-invariant raw component delta "
            f"{max_component_delta:.17g}"
        )
    return max_angle, max_component_delta


def _require_close(
    actual: np.ndarray, expected: np.ndarray, *, name: str, atol: float
) -> float:
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ReplayError(f"{name} replay shape or values differ")
    error = float(np.max(np.abs(actual - expected)))
    if error > atol:
        raise ReplayError(f"{name} replay differs by {error}")
    return error


def replay_rows(rows: list[dict[str, Any]], report: dict[str, Any]) -> dict[str, Any]:
    """Recompute FK poses and inverse dynamics without producer validation helpers."""
    import torch
    from curobo._src.geom.types import SceneCfg
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import JointState

    if not torch.cuda.is_available():
        raise ReplayError("independent FK replay requires the claimed CUDA GPU")
    source = _runtime_source()
    benchmark = report.get("kind") == "benchmark"
    upstream = _benchmark_module(source) if benchmark else None
    planner = MotionPlanner(
        MotionPlannerCfg.create(
            robot="franka.yml", scene_model=SceneCfg.create({"cuboid": {}})
        )
    )
    dynamics_models = (
        {
            "kinematic": upstream.load_robot_model_for_dynamics(
                robot_name="franka", attached_object_mass=0.0
            ),
            "dynamics": upstream.load_robot_model_for_dynamics(
                robot_name="franka", attached_object_mass=3.0
            ),
        }
        if upstream
        else {}
    )
    fk_rows = 0
    dynamics_rows = 0
    max_position_replay_error = 0.0
    max_quaternion_replay_error = 0.0
    max_quaternion_component_replay_error = 0.0
    max_torque_replay_error = 0.0
    terminal_position_errors = []
    terminal_orientation_errors = []
    try:
        for row in rows:
            if row["status"] != "success":
                continue
            trajectory = row["trajectory"]
            names = trajectory["joint_names"]
            try:
                indices = [names.index(name) for name in planner.joint_names]
            except ValueError as exc:
                raise ReplayError("retained trajectory omits an active joint") from exc
            positions = np.asarray(trajectory["position"], dtype=float)[:, indices]
            state = JointState.from_position(
                planner.device_cfg.to_device(positions.tolist()),
                joint_names=list(planner.joint_names),
            )
            fk = planner.kinematics.compute_kinematics(state)
            pose = fk.tool_poses.get_link_pose(planner.tool_frames[0])
            replay_position = _tensor_array(pose.position)
            replay_quaternion = _tensor_array(pose.quaternion)
            max_position_replay_error = max(
                max_position_replay_error,
                _require_close(
                    replay_position,
                    np.asarray(trajectory["tool_position"], dtype=float),
                    name="FK position",
                    atol=1e-5,
                ),
            )
            retained_quaternion = np.asarray(trajectory["tool_quaternion"], dtype=float)
            row_max_quaternion_error, row_max_component_error = (
                _require_quaternion_replay(
                    replay_quaternion,
                    retained_quaternion,
                )
            )
            max_quaternion_replay_error = max(
                max_quaternion_replay_error, row_max_quaternion_error
            )
            max_quaternion_component_replay_error = max(
                max_quaternion_component_replay_error,
                row_max_component_error,
            )
            goal = row["query"]["goal_pose"]
            terminal_position_errors.append(
                float(
                    np.linalg.norm(
                        replay_position[-1]
                        - np.asarray(goal["position_xyz"], dtype=float)
                    )
                )
            )
            terminal_orientation_errors.append(
                _quaternion_distance(
                    replay_quaternion[-1],
                    np.asarray(goal["quaternion_wxyz"], dtype=float),
                )
            )
            fk_rows += 1
            if not benchmark:
                continue
            evidence = row["dynamics_evidence"]
            series = evidence["trajectory"]
            replay_input = SimpleNamespace(
                position=torch.as_tensor(series["position"], dtype=torch.float64),
                velocity=torch.as_tensor(series["velocity"], dtype=torch.float64),
                acceleration=torch.as_tensor(
                    series["acceleration"], dtype=torch.float64
                ),
                dt=torch.as_tensor(series["dt"], dtype=torch.float64),
            )
            replayed = upstream.compute_trajectory_energy(
                replay_input, dynamics_models[row["mode"]]
            )
            torque_error = _require_close(
                np.asarray(replayed["torques"], dtype=float),
                np.asarray(evidence["torques_nm"], dtype=float),
                name="inverse-dynamics torque",
                atol=1e-8,
            )
            max_torque_replay_error = max(max_torque_replay_error, torque_error)
            if (
                not math.isclose(
                    replayed["energy"],
                    row["metrics"]["energy_proxy_j"],
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
                or not math.isclose(
                    replayed["max_torque"],
                    row["metrics"]["max_torque_nm"],
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
                or int(replayed["torque_violation"])
                != row["metrics"]["torque_violation"]
            ):
                raise ReplayError(
                    "inverse-dynamics metrics differ on independent replay"
                )
            dynamics_rows += 1
    finally:
        planner.destroy()
    if not fk_rows:
        raise ReplayError("no successful trajectory exists for independent replay")
    return {
        "schema_version": "npa.curobo.replay.v1",
        "source_revision": SOURCE_REVISION,
        "dataset_revision": DATASET_REVISION if benchmark else None,
        "fk_replay_count": fk_rows,
        "dynamics_replay_count": dynamics_rows,
        "max_fk_position_replay_error_m": max_position_replay_error,
        "max_fk_quaternion_replay_error_rad": max_quaternion_replay_error,
        "max_fk_quaternion_component_replay_error": (
            max_quaternion_component_replay_error
        ),
        "max_torque_replay_error_nm": max_torque_replay_error,
        "terminal_goal_distance_m": {
            "max": max(terminal_position_errors),
            "mean": float(np.mean(terminal_position_errors)),
        },
        "terminal_goal_orientation_rad": {
            "max": max(terminal_orientation_errors),
            "mean": float(np.mean(terminal_orientation_errors)),
        },
        "valid": True,
        "limitations": [
            "Replays kinematics and inverse dynamics; does not independently certify collision freedom.",
            "Goal-pose agreement is not permission to execute on hardware.",
        ],
    }
