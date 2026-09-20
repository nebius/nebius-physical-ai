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


def _close(observed: Any, expected: float, *, name: str) -> None:
    if (
        isinstance(observed, bool)
        or not isinstance(observed, (int, float))
        or not math.isclose(observed, expected, rel_tol=1e-9, abs_tol=1e-9)
    ):
        raise AuditError(f"{name} does not independently recompute")


def _query(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
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
    return start, goal


def _audit_success(row: dict[str, Any], *, benchmark: bool) -> dict[str, float]:
    _, goal = _query(row)
    trajectory = row.get("trajectory")
    if not isinstance(trajectory, dict):
        raise AuditError("successful row has no trajectory")
    position = _array(trajectory["position"], name="joint position")
    velocity = _array(trajectory["velocity"], name="joint velocity")
    acceleration = _array(trajectory["acceleration"], name="joint acceleration")
    jerk = _array(trajectory["jerk"], name="joint jerk")
    tool = _array(trajectory["tool_position"], name="FK tool position")
    names = trajectory.get("joint_names")
    dt = trajectory.get("dt")
    if (
        not isinstance(names, list)
        or len(names) != position.shape[1]
        or len(names) != len(set(names))
        or position.shape != velocity.shape
        or acceleration.shape != position.shape
        or jerk.shape != position.shape
        or tool.shape != (len(position), 3)
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
    return {"terminal_goal_distance_m": float(np.linalg.norm(tool[-1] - goal))}


def _audit_dynamics(row: dict[str, Any]) -> None:
    evidence = row.get("dynamics_evidence")
    if not isinstance(evidence, dict):
        raise AuditError("benchmark success lacks dynamics evidence")
    trajectory = evidence.get("trajectory")
    if not isinstance(trajectory, dict):
        raise AuditError("dynamics trajectory is absent")
    position = _array(trajectory["position"], name="dynamics position")
    velocity = _array(trajectory["velocity"], name="dynamics velocity")
    acceleration = _array(trajectory["acceleration"], name="dynamics acceleration")
    jerk = _array(trajectory["jerk"], name="dynamics jerk")
    torques = _array(evidence.get("torques_nm"), name="inverse-dynamics torque")
    limits = np.asarray(evidence.get("torque_limits_nm"), dtype=float)
    dt = trajectory.get("dt")
    names = trajectory.get("joint_names")
    expected_limits = np.asarray([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
    expected_mass = 3.0 if row["mode"] == "dynamics" else 0.0
    if (
        position.shape != velocity.shape
        or acceleration.shape != position.shape
        or jerk.shape != position.shape
        or torques.shape != position.shape
        or position.shape[1] != 7
        or not isinstance(names, list)
        or len(names) != 7
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
    for row in rows:
        if row["status"] != "invalid":
            _query(row)
        if row["status"] == "success":
            terminal_distances.append(
                _audit_success(row, benchmark=benchmark)["terminal_goal_distance_m"]
            )
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
            cells[f"{mode}/{dataset}"] = {
                "input_count": len(subset),
                **counts,
                "eligible_success_fraction": counts["success"] / eligible
                if eligible
                else None,
            }
    return {
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
        "valid": True,
        "limitations": [
            "Recomputes retained trajectory and inverse-dynamics facts; does not independently certify collision freedom.",
            "Terminal FK distance is a consumer check, not permission to execute on hardware.",
        ],
    }


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
