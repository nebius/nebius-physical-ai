"""Independent byte-level audit of durable cuRobo result and journal artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .benchmark_inventory import benchmark_identities
from .schemas import DATASET_REVISION, SOURCE_REVISION


class AuditError(RuntimeError):
    """Durable planner artifacts cannot support the claimed result."""


_CELL_GATES = {
    "kinematic/motion_benchmaker": {
        "input_count": 800,
        "invalid": 1,
        "maximum_unusable_eligible": 7,
    },
    "kinematic/mpinets": {
        "input_count": 1800,
        "invalid": 9,
        "maximum_unusable_eligible": 17,
    },
    "dynamics/motion_benchmaker": {
        "input_count": 800,
        "invalid": 1,
        "maximum_unusable_eligible": 39,
    },
    "dynamics/mpinets": {
        "input_count": 1800,
        "invalid": 9,
        "maximum_unusable_eligible": 89,
    },
}


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    identities = [(row["mode"], row["dataset"], row["problem_id"]) for row in rows]
    if not rows or len(identities) != len(set(identities)):
        raise AuditError("planner journal is empty or contains duplicate identities")
    result: dict[str, Any] = {}
    for mode in sorted({row["mode"] for row in rows}):
        subset = [row for row in rows if row["mode"] == mode]
        counts = {
            status: sum(row["status"] == status for row in subset)
            for status in ("success", "failed", "invalid")
        }
        if sum(counts.values()) != len(subset):
            raise AuditError("planner journal contains an unknown status")
        eligible = counts["success"] + counts["failed"]
        metrics: dict[str, list[float]] = {}
        for row in subset:
            for name, value in row.get("metrics", {}).items():
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise AuditError("planner metric is not a finite number")
                metrics.setdefault(name, []).append(value)
        result[mode] = {
            "input_count": len(subset),
            "eligible_count": eligible,
            **counts,
            "success_fraction_all": counts["success"] / len(subset),
            "success_fraction_eligible": counts["success"] / eligible
            if eligible
            else None,
            "metrics": {
                name: {
                    "count": len(values),
                    "mean": float(np.mean(values)),
                    "p50": float(np.percentile(values, 50)),
                    "p95": float(np.percentile(values, 95)),
                    "p98": float(np.percentile(values, 98)),
                }
                for name, values in metrics.items()
            },
        }
    return result


def _array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or len(array) < 2 or not np.isfinite(array).all():
        raise AuditError(f"{name} is not a finite two-dimensional trajectory")
    return array


def _joint_names(value: Any, width: int, *, name: str) -> list[str]:
    if (
        not isinstance(value, list)
        or len(value) != width
        or any(not isinstance(joint, str) or not joint for joint in value)
        or len(value) != len(set(value))
    ):
        raise AuditError(f"{name} joint names are invalid")
    return value


def _close(observed: Any, expected: float, *, name: str) -> None:
    if (
        isinstance(observed, bool)
        or not isinstance(observed, (int, float))
        or not math.isclose(observed, expected, rel_tol=1e-9, abs_tol=1e-9)
    ):
        raise AuditError(f"{name} does not independently recompute")


def _query(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query = row.get("query")
    if not isinstance(query, dict):
        raise AuditError("executed planner query is absent")
    try:
        start = np.asarray(query["start"], dtype=float)
        goal = np.asarray(query["goal_pose"]["position_xyz"], dtype=float)
        quaternion = np.asarray(query["goal_pose"]["quaternion_wxyz"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise AuditError("executed planner query is malformed") from exc
    if (
        start.ndim != 1
        or not len(start)
        or goal.shape != (3,)
        or quaternion.shape != (4,)
        or not np.isfinite(start).all()
        or not np.isfinite(goal).all()
        or not np.isfinite(quaternion).all()
        or not math.isclose(
            float(np.linalg.norm(quaternion)), 1.0, rel_tol=1e-6, abs_tol=1e-6
        )
    ):
        raise AuditError("executed planner query has invalid values")
    return start, goal, quaternion


def _quaternion_distance(first: np.ndarray, second: np.ndarray) -> float:
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
        raise AuditError("quaternion evidence contains invalid values")
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
    if not math.isfinite(angle):
        raise AuditError("quaternion evidence contains invalid values")
    return angle


def _audit_success(row: dict[str, Any], *, benchmark: bool) -> dict[str, float]:
    _, goal, goal_quaternion = _query(row)
    trajectory = row.get("trajectory")
    if not isinstance(trajectory, dict):
        raise AuditError("successful row has no trajectory")
    position = _array(trajectory["position"], name="joint position")
    velocity = _array(trajectory["velocity"], name="joint velocity")
    acceleration = _array(trajectory["acceleration"], name="joint acceleration")
    jerk = _array(trajectory["jerk"], name="joint jerk")
    tool = _array(trajectory["tool_position"], name="FK tool position")
    tool_quaternion = _array(trajectory["tool_quaternion"], name="FK tool quaternion")
    _joint_names(
        trajectory.get("joint_names"), position.shape[1], name="successful trajectory"
    )
    dt = trajectory.get("dt")
    if (
        position.shape != velocity.shape
        or acceleration.shape != position.shape
        or jerk.shape != position.shape
        or tool.shape != (len(position), 3)
        or tool_quaternion.shape != (len(position), 4)
        or not np.allclose(np.linalg.norm(tool_quaternion, axis=1), 1.0, atol=1e-5)
        or isinstance(dt, bool)
        or not isinstance(dt, (int, float))
        or not math.isfinite(dt)
        or dt <= 0
    ):
        raise AuditError("successful trajectory shapes or timeline are invalid")
    metrics = row["metrics"]
    _close(
        metrics["joint_path_length_rad"],
        float(np.linalg.norm(np.diff(position, axis=0), axis=1).sum()),
        name="joint path length",
    )
    _close(
        metrics["tool_path_length_m"],
        float(np.linalg.norm(np.diff(tool, axis=0), axis=1).sum()),
        name="FK tool path length",
    )
    _close(
        metrics["trajectory_duration_seconds"],
        (len(position) - 1) * dt,
        name="trajectory duration",
    )
    _close(
        metrics["max_abs_jerk_rad_s3"],
        float(np.abs(jerk).max()),
        name="maximum jerk",
    )
    if benchmark:
        _audit_dynamics(row)
    elif "dynamics_evidence" in row:
        raise AuditError("operator plan unexpectedly carries benchmark dynamics")
    return {
        "terminal_goal_distance_m": float(np.linalg.norm(tool[-1] - goal)),
        "terminal_goal_orientation_rad": _quaternion_distance(
            tool_quaternion[-1], goal_quaternion
        ),
    }


def _dynamics_evidence_arrays(row: dict[str, Any]):
    evidence = row.get("dynamics_evidence")
    if not isinstance(evidence, dict):
        raise AuditError("benchmark success lacks dynamics evidence")
    trajectory = evidence.get("trajectory")
    if not isinstance(trajectory, dict):
        raise AuditError("dynamics trajectory is absent")
    position = _array(trajectory["position"], name="dynamics position")
    arrays = tuple(
        _array(trajectory[field], name=f"dynamics {field}")
        for field in ("velocity", "acceleration", "jerk")
    )
    torques = _array(evidence.get("torques_nm"), name="inverse-dynamics torque")
    limits = np.asarray(evidence.get("torque_limits_nm"), dtype=float)
    names = _joint_names(
        trajectory.get("joint_names"), position.shape[1], name="dynamics trajectory"
    )
    return evidence, trajectory, position, arrays, torques, limits, names


def _audit_dynamics(row: dict[str, Any]) -> None:
    evidence, trajectory, position, arrays, torques, limits, names = (
        _dynamics_evidence_arrays(row)
    )
    velocity, acceleration, jerk = arrays
    dt = trajectory.get("dt")
    retained_names = row["trajectory"]["joint_names"]
    expected_limits = np.asarray([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
    expected_mass = 3.0 if row["mode"] == "dynamics" else 0.0
    if (
        position.shape != velocity.shape
        or acceleration.shape != position.shape
        or jerk.shape != position.shape
        or torques.shape != position.shape
        or position.shape[1] != 7
        or not set(names).issubset(retained_names)
        or limits.shape != (7,)
        or not np.array_equal(limits, expected_limits)
        or evidence.get("attached_mass_kg") != expected_mass
        or isinstance(dt, bool)
        or not isinstance(dt, (int, float))
        or not math.isfinite(dt)
        or dt <= 0
    ):
        raise AuditError(
            "dynamics evidence shape, limits, mass, or timeline is invalid"
        )
    energy = float(np.abs(torques * velocity).sum() * dt)
    max_torque = float(np.abs(torques).max())
    violation = int(np.any(np.abs(torques).max(axis=0) > limits))
    metrics = row["metrics"]
    _close(metrics["energy_proxy_j"], energy, name="inverse-dynamics energy")
    _close(metrics["max_torque_nm"], max_torque, name="maximum torque")
    if metrics["torque_violation"] != violation:
        raise AuditError("torque-violation indicator does not independently recompute")


def benchmark_acceptance(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if set(cells) != set(_CELL_GATES):
        raise AuditError("benchmark acceptance cells differ")
    verdicts = {}
    for name, gate in _CELL_GATES.items():
        cell = cells[name]
        if (
            cell["input_count"] != gate["input_count"]
            or cell["invalid"] != gate["invalid"]
            or cell["success"] + cell["failed"] + cell["invalid"] != cell["input_count"]
            or not 0 <= cell["torque_violation_successes"] <= cell["success"]
        ):
            raise AuditError(f"{name} population differs")
        unusable = (
            cell["failed"]
            if name.startswith("kinematic/")
            else cell["failed"] + cell["torque_violation_successes"]
        )
        if unusable > gate["maximum_unusable_eligible"]:
            raise AuditError(f"{name} exceeds the frozen unusable-row gate")
        verdicts[name] = {
            "unusable_eligible": unusable,
            "maximum_unusable_eligible": gate["maximum_unusable_eligible"],
            "passed": True,
        }
    return {
        "schema_version": "npa.curobo.benchmark-acceptance.v1",
        "cells": verdicts,
        "passed": True,
    }


def audit_bytes(
    result_bytes: bytes, journal_bytes: bytes, *, run_id: str
) -> dict[str, Any]:
    try:
        report = json.loads(result_bytes)
        rows = [json.loads(line) for line in journal_bytes.splitlines() if line.strip()]
    except (UnicodeDecodeError, ValueError, TypeError) as exc:
        raise AuditError("result or journal is not valid JSON") from exc
    summary = _summary(rows)
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != "npa.curobo.result.v1"
        or report.get("engine") != "nvidia-curobo-v2"
        or report.get("source_revision") != SOURCE_REVISION
        or report.get("run_id") != run_id
        or report.get("summary") != summary
        or report.get("journal_sha256") != hashlib.sha256(journal_bytes).hexdigest()
    ):
        raise AuditError("result identity, summary, or journal binding differs")
    benchmark = report.get("kind") == "benchmark"
    if benchmark:
        modes = report.get("requested_modes")
        expected = benchmark_identities(modes) if isinstance(modes, list) else {}
        observed = {(row["mode"], row["dataset"], row["problem_id"]) for row in rows}
        if (
            not expected
            or observed != set(expected)
            or report.get("dataset_revision") != DATASET_REVISION
        ):
            raise AuditError("benchmark population or dataset revision differs")
        for row in rows:
            identity = (row["mode"], row["dataset"], row["problem_id"])
            if (row["status"] == "invalid") != expected[identity]:
                raise AuditError("benchmark invalid identities differ")
    elif (
        report.get("kind") != "plan"
        or report.get("requested_modes") != ["kinematic"]
        or report.get("dataset_revision") is not None
        or any(
            row["mode"] != "kinematic"
            or row["dataset"] != "operator"
            or row["status"] == "invalid"
            for row in rows
        )
    ):
        raise AuditError("operator-plan scope differs")
    terminal_distances = []
    terminal_orientations = []
    for row in rows:
        if row["status"] != "invalid":
            _query(row)
        if row["status"] == "success":
            replayed = _audit_success(row, benchmark=benchmark)
            terminal_distances.append(replayed["terminal_goal_distance_m"])
            terminal_orientations.append(replayed["terminal_goal_orientation_rad"])
        elif "trajectory" in row or "dynamics_evidence" in row:
            raise AuditError("unsolved row carries solution evidence")
    cells = {}
    for mode in sorted({row["mode"] for row in rows}):
        for dataset in sorted({row["dataset"] for row in rows if row["mode"] == mode}):
            subset = [
                row for row in rows if row["mode"] == mode and row["dataset"] == dataset
            ]
            counts = {
                status: sum(row["status"] == status for row in subset)
                for status in ("success", "failed", "invalid")
            }
            eligible = counts["success"] + counts["failed"]
            torque_violations = sum(
                row["status"] == "success"
                and row.get("metrics", {}).get("torque_violation") == 1
                for row in subset
            )
            cells[f"{mode}/{dataset}"] = {
                "input_count": len(subset),
                **counts,
                "torque_violation_successes": torque_violations,
                "usable_successes": counts["success"]
                - (torque_violations if mode == "dynamics" else 0),
                "eligible_success_fraction": counts["success"] / eligible
                if eligible
                else None,
            }
    result = {
        "schema_version": "npa.curobo.validation.v1",
        "run_id": run_id,
        "source_revision": SOURCE_REVISION,
        "dataset_revision": DATASET_REVISION if benchmark else None,
        "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "journal_sha256": hashlib.sha256(journal_bytes).hexdigest(),
        "problem_count": len(rows),
        "successful_trajectory_count": len(terminal_distances),
        "dynamics_recomputation_count": sum(
            row["status"] == "success" and benchmark for row in rows
        ),
        "summary": summary,
        "dataset_mode_cells": cells,
        "terminal_goal_distance_m": {
            "max": max(terminal_distances) if terminal_distances else None,
            "mean": float(np.mean(terminal_distances)) if terminal_distances else None,
        },
        "terminal_goal_orientation_rad": {
            "max": max(terminal_orientations) if terminal_orientations else None,
            "mean": float(np.mean(terminal_orientations))
            if terminal_orientations
            else None,
        },
        "valid": True,
        "limitations": [
            "Recomputes retained trajectory and inverse-dynamics facts; does not independently certify collision freedom.",
            "Terminal FK distance is a consumer check, not permission to execute on hardware.",
        ],
    }
    if benchmark:
        result["acceptance"] = benchmark_acceptance(cells)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audited = audit_bytes(
        args.result.read_bytes(), args.journal.read_bytes(), run_id=args.run_id
    )
    args.output.write_bytes(canonical(audited))


if __name__ == "__main__":
    main()
