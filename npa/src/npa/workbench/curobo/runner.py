"""GPU subprocess: invoke pinned NVIDIA cuRobo V2 APIs, never surrogate plans."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
from importlib.metadata import version
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from .artifacts import (
    CuroboError,
    canonical,
    summarize,
    validate_joint_series,
    validate_report,
    validate_trajectory,
)
from .benchmark_inventory import DATASET_FILES
from .schemas import DATASET_REVISION, SOURCE_REVISION, BenchmarkManifest, PlanManifest


_DYNAMICS_DOF = 7
_TORQUE_LIMITS_NM = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)


def _runtime_source():
    source = Path(os.environ.get("NPA_CUROBO_SOURCE", "/opt/curobo"))
    if (source / "NPA_SOURCE_REVISION").read_text().strip() != SOURCE_REVISION:
        raise CuroboError(
            "cuRobo source revision does not match the reviewed V2 contract"
        )
    if version("nvidia-curobo") != "0.8.0":
        raise CuroboError(
            "installed cuRobo version does not match the reviewed V2 contract"
        )
    return source


def _benchmark_module():
    source = _runtime_source()
    dataset = Path(os.environ.get("NPA_CUROBO_DATASET_SOURCE", "/opt/robometrics"))
    if (dataset / "NPA_SOURCE_REVISION").read_text().strip() != DATASET_REVISION:
        raise CuroboError(
            "benchmark dataset revision does not match the reviewed contract"
        )
    for filename, expected in DATASET_FILES.values():
        path = dataset / "robometrics/content/dataset" / filename
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise CuroboError("benchmark YAML bytes do not match the pinned inventory")
    _activate_dataset_imports(dataset)
    spec = importlib.util.spec_from_file_location(
        "npa_curobo_upstream_benchmark", source / "benchmark/motion_plan_benchmark.py"
    )
    if spec is None or spec.loader is None:
        raise CuroboError("pinned upstream benchmark is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _activate_dataset_imports(dataset: Path):
    """Bind raw loaders to the verified tree even when SkyPilot clears PYTHONPATH."""
    package = (dataset / "robometrics").resolve(strict=True)
    # A cached module wins over sys.path; reject another installation instead of
    # validating one dataset tree and executing loaders from a different tree.
    for name, module in tuple(sys.modules.items()):
        if name == "robometrics" or name.startswith("robometrics."):
            origin = getattr(module, "__file__", None)
            if not origin or not Path(origin).resolve().is_relative_to(package):
                raise CuroboError(
                    "benchmark loader was imported outside the verified dataset tree"
                )
    sys.path.insert(0, str(dataset.resolve(strict=True)))
    importlib.invalidate_caches()
    module = importlib.import_module("robometrics.datasets")
    if Path(module.__file__).resolve() != package / "datasets.py":
        raise CuroboError("benchmark loader does not match the verified dataset tree")


def _array(tensor):
    return tensor.detach().cpu().reshape(-1, tensor.shape[-1]).numpy()


def _joint_series(state):
    return {
        "joint_names": list(state.joint_names),
        "dt": float(state.dt.item()),
        **{
            key: _array(getattr(state, key)).tolist()
            for key in ("position", "velocity", "acceleration", "jerk")
        },
    }


def _validated_names(value, *, label):
    if isinstance(value, (str, bytes)):
        raise CuroboError(f"{label} joint names are invalid")
    try:
        names = list(value)
    except TypeError as exc:
        raise CuroboError(f"{label} joint names are invalid") from exc
    if (
        not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(names) != len(set(names))
    ):
        raise CuroboError(f"{label} joint names are invalid")
    return names


def _validated_joint_series(state, *, label):
    try:
        series = _joint_series(state)
        validate_joint_series(series)
    except (
        AttributeError,
        CuroboError,
        IndexError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise CuroboError(f"{label} trajectory fields are invalid") from exc
    return series


def _active_dynamics_input(planner, state):
    active_names = _validated_names(
        getattr(planner, "joint_names", None), label="planner active"
    )
    kinematic_names = _validated_names(
        getattr(planner.kinematics, "joint_names", None), label="kinematics active"
    )
    available_names = _validated_names(
        getattr(planner.kinematics, "all_articulated_joint_names", None),
        label="kinematics articulated",
    )
    raw_names = _validated_names(
        getattr(state, "joint_names", None), label="raw dynamics"
    )
    if (
        len(active_names) != _DYNAMICS_DOF
        or active_names != kinematic_names
        or not set(active_names).issubset(raw_names)
        or not set(raw_names).issubset(available_names)
    ):
        raise CuroboError("inverse-dynamics joint identity is invalid")
    _validated_joint_series(state, label="raw dynamics")
    try:
        aligned = state.reorder(active_names)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise CuroboError("inverse-dynamics joint alignment failed") from exc
    trajectory = _validated_joint_series(aligned, label="active dynamics")
    if trajectory["joint_names"] != active_names:
        raise CuroboError("inverse-dynamics active joint order is invalid")
    return aligned, trajectory, active_names


def _model_dimension(value, *, label):
    if isinstance(value, bool):
        raise CuroboError(f"{label} is invalid")
    try:
        dimension = int(value)
    except (TypeError, ValueError) as exc:
        raise CuroboError(f"{label} is invalid") from exc
    if dimension < 0 or dimension != value:
        raise CuroboError(f"{label} is invalid")
    return dimension


def _dynamics_model_contract(dynamics_model, active_names):
    import numpy as np

    if not isinstance(dynamics_model, tuple) or len(dynamics_model) != 3:
        raise CuroboError("inverse-dynamics model identity is invalid")
    model, _data, raw_limits = dynamics_model
    model_names = _validated_names(
        getattr(model, "names", None), label="Pinocchio model"
    )
    model_nq = _model_dimension(
        getattr(model, "nq", None), label="Pinocchio model nq"
    )
    model_nv = _model_dimension(
        getattr(model, "nv", None), label="Pinocchio model nv"
    )
    try:
        limits = np.asarray(raw_limits, dtype=float)
    except (TypeError, ValueError) as exc:
        raise CuroboError("inverse-dynamics torque limits are invalid") from exc
    if (
        model_nq != _DYNAMICS_DOF
        or model_nv != _DYNAMICS_DOF
        or model_names[1:] != active_names
        or len(model_names) != len(active_names) + 1
        or limits.shape != (_DYNAMICS_DOF,)
        or not np.isfinite(limits).all()
        or not np.array_equal(limits, np.asarray(_TORQUE_LIMITS_NM))
    ):
        raise CuroboError(
            "inverse-dynamics model, joint order, or torque limits are invalid"
        )
    return limits


def _dynamics_scalars(dynamic):
    import numpy as np

    if not isinstance(dynamic, dict):
        raise CuroboError("inverse-dynamics result is invalid")
    try:
        energy = dynamic["energy"]
        max_torque = dynamic["max_torque"]
        torque_violation = dynamic["torque_violation"]
    except KeyError as exc:
        raise CuroboError("inverse-dynamics result is invalid") from exc
    numeric_types = (int, float, np.integer, np.floating)
    if (
        isinstance(energy, (bool, np.bool_))
        or not isinstance(energy, numeric_types)
        or not math.isfinite(float(energy))
        or isinstance(max_torque, (bool, np.bool_))
        or not isinstance(max_torque, numeric_types)
        or not math.isfinite(float(max_torque))
        or not isinstance(torque_violation, (bool, np.bool_))
    ):
        raise CuroboError("inverse-dynamics evidence shape or values are invalid")
    return float(energy), float(max_torque), bool(torque_violation)


def _dynamics_result(dynamic, trajectory, limits):
    import numpy as np

    energy, max_torque, torque_violation = _dynamics_scalars(dynamic)
    try:
        torques = np.asarray(dynamic["torques"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise CuroboError("inverse-dynamics result is invalid") from exc
    velocity = np.asarray(trajectory["velocity"], dtype=float)
    if (
        torques.shape != velocity.shape
        or torques.shape[1] != len(limits)
        or not np.isfinite(torques).all()
    ):
        raise CuroboError("inverse-dynamics evidence shape or values are invalid")
    expected_energy = float(np.abs(torques * velocity).sum() * trajectory["dt"])
    expected_max = float(np.abs(torques).max())
    expected_violation = bool(np.any(np.abs(torques).max(axis=0) > limits))
    if (
        not math.isclose(energy, expected_energy, rel_tol=1e-9, abs_tol=1e-9)
        or not math.isclose(
            max_torque, expected_max, rel_tol=1e-9, abs_tol=1e-9
        )
        or torque_violation != expected_violation
    ):
        raise CuroboError("inverse-dynamics metrics do not match retained evidence")
    return torques, {
        "energy_proxy_j": energy,
        "max_torque_nm": max_torque,
        "torque_violation": int(torque_violation),
    }


def _compute_dynamics_evidence(planner, result, benchmark_module, dynamics_model):
    aligned, trajectory, active_names = _active_dynamics_input(
        planner, result.js_solution
    )
    limits = _dynamics_model_contract(dynamics_model, active_names)
    dynamic = benchmark_module.compute_trajectory_energy(aligned, dynamics_model)
    torques, metrics = _dynamics_result(dynamic, trajectory, limits)
    return {
        "trajectory": trajectory,
        "torques_nm": torques.tolist(),
        "torque_limits_nm": limits.tolist(),
    }, metrics


def _durable_trajectory_metrics(trajectory):
    """Compute metrics from the values that will be written to the journal.

    Args:
        trajectory: Validated JSON-compatible trajectory payload.

    Returns:
        Metrics independently recomputable from the durable payload.

    Raises:
        KeyError: The validated trajectory is missing a required field.
        TypeError: A trajectory field is not a numeric sequence.
        ValueError: A trajectory value cannot be represented as a numeric array.
    """

    import numpy as np

    positions = np.asarray(trajectory["position"], dtype=float)
    tool_positions = np.asarray(trajectory["tool_position"], dtype=float)
    jerk = np.asarray(trajectory["jerk"], dtype=float)
    return {
        "joint_path_length_rad": float(
            np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()
        ),
        "tool_path_length_m": float(
            np.linalg.norm(np.diff(tool_positions, axis=0), axis=1).sum()
        ),
        "trajectory_duration_seconds": (len(positions) - 1) * trajectory["dt"],
        "max_abs_jerk_rad_s3": float(np.abs(jerk).max()),
    }


def _solve(
    planner,
    problem,
    *,
    benchmark_module=None,
    dynamics_model=None,
    attached_mass_kg=None,
):
    import torch
    from curobo.types import GoalToolPose, JointState

    q_start = JointState.from_position(
        planner.device_cfg.to_device([problem["start"]]),
        joint_names=planner.joint_names,
    )
    pose = problem["goal_pose"]
    goal = GoalToolPose(
        tool_frames=planner.tool_frames,
        position=planner.device_cfg.to_device(pose["position_xyz"]).reshape(
            1, 1, 1, 1, 3
        ),
        quaternion=planner.device_cfg.to_device(pose["quaternion_wxyz"]).reshape(
            1, 1, 1, 1, 4
        ),
    )
    planner.reset_seed()
    torch.cuda.synchronize()
    start = time.perf_counter()
    # This is the upstream benchmark's solver retry policy, not a workload cap.
    kwargs = (
        {"max_attempts": 100, "enable_graph_attempt": 1} if benchmark_module else {}
    )
    result = planner.plan_pose(goal, q_start, **kwargs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    record = {
        "status": "failed",
        "query": {
            "start": [float(value) for value in problem["start"]],
            "goal_pose": {
                "position_xyz": [
                    float(value) for value in problem["goal_pose"]["position_xyz"]
                ],
                "quaternion_wxyz": [
                    float(value) for value in problem["goal_pose"]["quaternion_wxyz"]
                ],
            },
        },
        "metrics": {"wall_plan_seconds": elapsed},
    }
    if result is None or not bool(result.success.item()):
        return record
    interpolated = result.get_interpolated_plan()
    # Interpolation includes locked/mimic joints (Franka's fingers), while the
    # kinematics model takes only active joints in its configured order.
    kinematic_state = interpolated.reorder(planner.joint_names)
    fk = planner.kinematics.compute_kinematics(
        JointState.from_position(
            kinematic_state.position.reshape(-1, len(planner.joint_names)),
            joint_names=kinematic_state.joint_names,
        )
    )
    tool_pose = fk.tool_poses.get_link_pose(planner.tool_frames[0])
    tool_positions = _array(tool_pose.position)
    tool_quaternions = _array(tool_pose.quaternion)
    trajectory = {
        **_joint_series(interpolated),
        "tool_position": tool_positions.tolist(),
        "tool_quaternion": tool_quaternions.tolist(),
    }
    validate_trajectory(trajectory)
    record.update(status="success", trajectory=trajectory)
    record["metrics"].update(
        {
            "planner_total_seconds": float(result.total_time),
            "solver_seconds": float(result.solve_time),
            "position_error_m": float(result.position_error.item()),
            "rotation_error_rad": float(result.rotation_error.item()),
            **_durable_trajectory_metrics(trajectory),
        }
    )
    if benchmark_module is not None:
        if dynamics_model is None or attached_mass_kg not in (0.0, 3.0):
            raise CuroboError("benchmark dynamics identity is unavailable")
        # Upstream inverse dynamics errors are fatal, never reported as zero energy.
        dynamics_evidence, dynamics_metrics = _compute_dynamics_evidence(
            planner, result, benchmark_module, dynamics_model
        )
        record["dynamics_evidence"] = {
            "attached_mass_kg": attached_mass_kg,
            **dynamics_evidence,
        }
        record["metrics"].update(dynamics_metrics)
    return record


def execute(kind: str, manifest: dict, output: Path, *, run_id: str):
    _runtime_source()
    import torch
    from curobo._src.geom.types import SceneCfg
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

    torch.manual_seed(2)
    if not torch.cuda.is_available():
        raise CuroboError("cuRobo requires a real CUDA GPU")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    warmups = []
    started = time.perf_counter()
    with (output / "problems.jsonl").open("x") as journal:

        def append(row):
            journal.write(canonical(row).decode() + "\n")
            journal.flush()
            os.fsync(journal.fileno())
            rows.append(row)

        if kind == "plan":
            inputs = PlanManifest.model_validate(manifest)
            for problem in inputs.problems:
                scene = {
                    "cuboid": {
                        name: box.model_dump(mode="json")
                        for name, box in problem.cuboids.items()
                    }
                }
                planner = MotionPlanner(
                    MotionPlannerCfg.create(
                        robot=inputs.robot, scene_model=SceneCfg.create(scene)
                    )
                )
                try:
                    start = time.perf_counter()
                    planner.warmup(enable_graph=True)
                    torch.cuda.synchronize()
                    warmups.append(time.perf_counter() - start)
                    row = _solve(planner, problem.model_dump(mode="json"))
                    append(
                        {
                            "mode": "kinematic",
                            "dataset": "operator",
                            "problem_id": problem.id,
                            **row,
                        }
                    )
                finally:
                    planner.destroy()
        else:
            inputs = BenchmarkManifest.model_validate(manifest)
            upstream = _benchmark_module()
            from robometrics.datasets import motion_benchmaker_raw, mpinets_raw

            for mode in inputs.modes:
                args = SimpleNamespace(
                    use_dynamics=mode == "dynamics",
                    mass=3.0 if mode == "dynamics" else 0.0,
                    mesh=False,
                    disable_cuda_graph=False,
                )
                for dataset, load in (
                    ("motion_benchmaker", motion_benchmaker_raw),
                    ("mpinets", mpinets_raw),
                ):
                    for group, problems in load().items():
                        planner, _robot_cfg = upstream.load_curobo(
                            upstream.check_problems(problems),
                            32,
                            4,
                            dataset == "mpinets",
                            collision_buffer=0.0,
                            args=args,
                        )
                        try:
                            start = time.perf_counter()
                            planner.warmup(enable_graph=True)
                            torch.cuda.synchronize()
                            warmups.append(time.perf_counter() - start)
                            dynamics_model = upstream.load_robot_model_for_dynamics(
                                robot_name="franka", attached_object_mass=args.mass
                            )
                            for index, problem in enumerate(problems):
                                identity = {
                                    "mode": mode,
                                    "dataset": dataset,
                                    "problem_id": f"{group}/{index}",
                                }
                                if problem["collision_buffer_ik"] < 0:
                                    append(
                                        {
                                            **identity,
                                            "status": "invalid",
                                            "reason": "upstream collision_buffer_ik is negative",
                                        }
                                    )
                                    continue
                                world = SceneCfg.create(
                                    problem["obstacles"]
                                ).get_obb_world()
                                planner.scene_collision_checker.clear_cache()
                                planner.update_world(world)
                                append(
                                    {
                                        **identity,
                                        **_solve(
                                            planner,
                                            problem,
                                            benchmark_module=upstream,
                                            dynamics_model=dynamics_model,
                                            attached_mass_kg=args.mass,
                                        ),
                                    }
                                )
                        finally:
                            planner.destroy()
    report = {
        "requested_modes": list(inputs.modes) if kind == "benchmark" else ["kinematic"],
        "schema_version": "npa.curobo.result.v1",
        "run_id": run_id,
        "kind": kind,
        "engine": "nvidia-curobo-v2",
        "seed": 2,
        "source_revision": SOURCE_REVISION,
        "dataset_revision": DATASET_REVISION if kind == "benchmark" else None,
        "gpu": {
            "name": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "warmup_seconds": warmups,
        "summary": summarize(rows),
        "limitations": [
            "Planner success is upstream feasibility, not independent collision certification.",
            "Benchmark follows upstream relaxed joint limits (+/-0.2 rad), OBB scene conversion, and optimizer settings.",
            "Dynamics benchmark carries 3 kg; energy is a Pinocchio inverse-dynamics proxy on the optimized trajectory.",
        ]
        if kind == "benchmark"
        else [
            "Franka pose planning only; no hardware execution or independent collision certification."
        ],
    }
    validate_report(report, rows, run_id=run_id)
    (output / "result.json").write_bytes(canonical(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("plan", "benchmark"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    execute(
        args.kind, json.loads(args.input.read_text()), args.output, run_id=args.run_id
    )


if __name__ == "__main__":
    main()
