"""Execute prepared Workbench MuJoCo scenes and export real camera/action datasets."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import time

from matrix import file_digest, load_matrix


@contextmanager
def _controller(simulation, name):
    original = simulation._phases
    if name == "open_gripper":

        def open_gripper(position, goal):
            return [
                (phase, target, 1.0, steps)
                for phase, target, _, steps in original(position, goal)
            ]

        simulation._phases = open_gripper
    try:
        yield
    finally:
        simulation._phases = original


def _source_identity(simulation, artifacts):
    return {
        "robot_sim.py": file_digest(Path(simulation.__file__)),
        "robot_artifacts.py": file_digest(Path(artifacts.__file__)),
    }


def _episode(simulation, case, index, output):
    relative = f"episodes/episode_{index:04d}"
    with _controller(simulation, case["controller"]):
        result = simulation.simulate_robot_episode(
            case["scene"], seed=case["seed"], output=output / relative
        )
    return {
        "id": case["id"],
        "scene": case["scene"],
        "simulation_seed": case["seed"],
        "controller": case["controller"],
        "episode_path": relative,
        "status": "accepted" if result["accepted"] else "rejected",
        "simulation": result,
    }


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _finish(output, records, matrix_hash, source, versions, artifacts, started):
    count = artifacts.export_robot_dataset(output, records)
    (output / "provenance.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records)
    )
    report = {
        "schema": "npa.robot-workflow.result.v1",
        "status": "completed",
        "execution": "local_workbench_cpu",
        "cloud_submission_performed": False,
        "model_inference_performed": False,
        "matrix_sha256": matrix_hash,
        "source_sha256": source,
        "simulator_versions": versions,
        "case_count": len(records),
        "accepted_count": count,
        "rejected_count": sum(row["status"] == "rejected" for row in records),
        "total_frames": sum(
            row["simulation"]["frames"]
            for row in records
            if row["status"] == "accepted"
        ),
        "fps": 25,
        "seconds": time.monotonic() - started,
        "artifacts": artifacts.robot_artifact_hashes(output),
    }
    _write_json(output / "report.json", report)
    return report


def run_workflow(matrix_path: Path, output: Path) -> dict:
    """Run every prepared case once through Workbench's native simulator and exporter.

    Args: matrix_path: Validated scene JSON. output: New local artifact directory.
    Returns: Source-bound manifest with actual simulation and dataset counts.
    Raises: ValueError, RuntimeError, OSError: Validation, simulation or export fails.
    """
    from npa.workbench.token_factory import robot_artifacts, robot_sim

    matrix, digest = load_matrix(matrix_path)
    versions = robot_sim.runtime_versions()
    source = _source_identity(robot_sim, robot_artifacts)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "matrix.json", matrix)
    started, records = time.monotonic(), []
    try:
        for index, case in enumerate(matrix["cases"]):
            records.append(_episode(robot_sim, case, index, output))
            _write_json(
                output / "progress.json",
                {"cases_finished": len(records), "source_sha256": source},
            )
        if source != _source_identity(robot_sim, robot_artifacts):
            raise RuntimeError("source changed during workflow execution")
        return _finish(
            output, records, digest, source, versions, robot_artifacts, started
        )
    except (Exception, KeyboardInterrupt) as error:
        _write_json(
            output / "failure.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "source_sha256": source,
                "cases_finished": len(records),
            },
        )
        raise


def main() -> None:
    """Execute the local prepared-scenes workflow or validate its inputs only.

    Args: None; command-line --matrix, --output and optional --validate-only.
    Returns: None; prints one result JSON document.
    Raises: ValueError, RuntimeError, OSError: Inputs or actual execution fail.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        matrix, digest = load_matrix(args.matrix)
        result = {
            "status": "valid",
            "case_count": len(matrix["cases"]),
            "matrix_sha256": digest,
            "execution_performed": False,
        }
    else:
        if args.output is None:
            parser.error("--output is required for execution")
        result = run_workflow(args.matrix, args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
