"""GPU subprocess: invoke pinned NVIDIA cuRobo V2 APIs, never surrogate plans."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
from importlib.metadata import version
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from .artifacts import (
    CuroboError,
    canonical,
    summarize,
    validate_report,
    validate_trajectory,
)
from .benchmark_inventory import DATASET_FILES
from .schemas import DATASET_REVISION, SOURCE_REVISION, BenchmarkManifest, PlanManifest


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
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise CuroboError(
                    "benchmark YAML bytes do not match the pinned inventory"
                )
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
                raise CuroboError("benchmark loader was imported outside the verified dataset tree")
    sys.path.insert(0, str(dataset.resolve(strict=True)))
    importlib.invalidate_caches()
    module = importlib.import_module("robometrics.datasets")
    if Path(module.__file__).resolve() != package / "datasets.py":
        raise CuroboError("benchmark loader does not match the verified dataset tree")


def _array(tensor):
    return tensor.detach().cpu().reshape(-1, tensor.shape[-1]).numpy()


def _solve(planner, problem, *, benchmark_module=None, dynamics_model=None):
    import numpy as np
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
    record = {"status": "failed", "metrics": {"wall_plan_seconds": elapsed}}
    if result is None or not bool(result.success.item()):
        return record
    interpolated = result.get_interpolated_plan()
    positions = _array(interpolated.position)
    fk = planner.kinematics.compute_kinematics(
        JointState.from_position(
            interpolated.position.reshape(-1, positions.shape[1]),
            joint_names=planner.joint_names,
        )
    )
    tool_positions = _array(
        fk.tool_poses.get_link_pose(planner.tool_frames[0]).position
    )
    trajectory = {
        "joint_names": list(planner.joint_names),
        "dt": float(interpolated.dt.item()),
        **{
            key: _array(getattr(interpolated, key)).tolist()
            for key in ("position", "velocity", "acceleration", "jerk")
        },
        "tool_position": tool_positions.tolist(),
    }
    validate_trajectory(trajectory)
    record.update(status="success", trajectory=trajectory)
    record["metrics"].update(
        {
            "planner_total_seconds": float(result.total_time),
            "solver_seconds": float(result.solve_time),
            "position_error_m": float(result.position_error.item()),
            "rotation_error_rad": float(result.rotation_error.item()),
            "joint_path_length_rad": float(
                np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()
            ),
            "tool_path_length_m": float(
                np.linalg.norm(np.diff(tool_positions, axis=0), axis=1).sum()
            ),
            "trajectory_duration_seconds": (len(positions) - 1) * trajectory["dt"],
            "max_abs_jerk_rad_s3": float(np.abs(np.asarray(trajectory["jerk"])).max()),
        }
    )
    if benchmark_module is not None:
        # Upstream inverse dynamics errors are fatal, never reported as zero energy.
        dynamic = benchmark_module.compute_trajectory_energy(
            result.js_solution, dynamics_model
        )
        record["metrics"].update(
            energy_proxy_j=float(dynamic["energy"]),
            max_torque_nm=float(dynamic["max_torque"]),
            torque_violation=int(dynamic["torque_violation"]),
        )
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
