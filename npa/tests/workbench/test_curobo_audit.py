"""Independent cuRobo audit recomputes durable facts without producer helpers."""

from __future__ import annotations

import hashlib
import inspect
import json

import numpy as np
import pytest

from npa.workbench.curobo import audit
from npa.workbench.curobo.artifacts import canonical, summarize
from npa.workbench.curobo.schemas import SOURCE_REVISION


def _x_rotation(angle: float) -> np.ndarray:
    return np.asarray([np.cos(angle / 2.0), np.sin(angle / 2.0), 0.0, 0.0])


def plan_row():
    return {
        "mode": "kinematic",
        "dataset": "operator",
        "problem_id": "pose",
        "status": "success",
        "query": {
            "start": [0.0],
            "goal_pose": {
                "position_xyz": [0.1, 0.0, 0.0],
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
        },
        "metrics": {
            "wall_plan_seconds": 0.01,
            "planner_total_seconds": 0.008,
            "solver_seconds": 0.006,
            "position_error_m": 0.001,
            "rotation_error_rad": 0.002,
            "joint_path_length_rad": 0.2,
            "tool_path_length_m": 0.1,
            "trajectory_duration_seconds": 0.1,
            "max_abs_jerk_rad_s3": 0.0,
        },
        "trajectory": {
            "joint_names": ["joint"],
            "dt": 0.1,
            "position": [[0.0], [0.2]],
            "velocity": [[0.0], [0.1]],
            "acceleration": [[0.0], [0.0]],
            "jerk": [[0.0], [0.0]],
            "tool_position": [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
            "tool_quaternion": [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
        },
    }


def plan_bytes():
    row = plan_row()
    journal = canonical(row) + b"\n"
    report = {
        "schema_version": "npa.curobo.result.v1",
        "engine": "nvidia-curobo-v2",
        "source_revision": SOURCE_REVISION,
        "dataset_revision": None,
        "run_id": "audit-run",
        "kind": "plan",
        "requested_modes": ["kinematic"],
        "journal_sha256": hashlib.sha256(journal).hexdigest(),
        "summary": summarize([row]),
    }
    return canonical(report), journal


def test_audit_recomputes_plan_metrics_and_terminal_consumer():
    result_bytes, journal = plan_bytes()
    result = audit.audit_bytes(result_bytes, journal, run_id="audit-run")
    assert result["valid"] is True
    assert result["problem_count"] == 1
    assert result["successful_trajectory_count"] == 1
    assert result["dynamics_recomputation_count"] == 0
    assert result["terminal_goal_distance_m"] == {"max": 0.0, "mean": 0.0}
    assert result["terminal_goal_orientation_rad"] == {"max": 0.0, "mean": 0.0}
    assert result["result_sha256"] == hashlib.sha256(result_bytes).hexdigest()
    assert result["journal_sha256"] == hashlib.sha256(journal).hexdigest()


def test_audit_quaternion_distance_is_stable_for_retained_float32_boundary():
    # Exact durable B200 row waypoint 48 from retained receipt e18cc395.
    durable = np.asarray(
        [
            0.9929810762405396,
            -0.016400041058659554,
            -0.11199788004159927,
            -0.03429362550377846,
        ]
    )
    replayed = durable.astype(np.float32)
    assert audit._quaternion_distance(replayed, durable) == 0.0
    assert audit._quaternion_distance(replayed, -durable) == 0.0


@pytest.mark.parametrize("angle", [7.5e-6, 1.25e-5, 1e-3])
def test_audit_quaternion_distance_preserves_small_rotations(angle):
    assert audit._quaternion_distance(
        np.asarray([1.0, 0.0, 0.0, 0.0]), _x_rotation(angle)
    ) == pytest.approx(angle, abs=1e-15)


@pytest.mark.parametrize(
    "invalid",
    [
        np.asarray([np.nan, 0.0, 0.0, 0.0]),
        np.asarray([0.0, 0.0, 0.0, 0.0]),
        np.asarray([1.0, 0.0, 0.0]),
    ],
)
def test_audit_quaternion_distance_rejects_invalid_values(invalid):
    with pytest.raises(audit.AuditError, match="quaternion"):
        audit._quaternion_distance(invalid, np.asarray([1.0, 0.0, 0.0, 0.0]))


@pytest.mark.parametrize("mutation", ["path", "goal", "run_id", "journal_hash"])
def test_audit_fails_closed_on_inconsistent_durable_facts(mutation):
    result_bytes, journal = plan_bytes()
    report = json.loads(result_bytes)
    row = json.loads(journal)
    if mutation == "path":
        row["metrics"]["joint_path_length_rad"] = 0.1
        journal = canonical(row) + b"\n"
        report["journal_sha256"] = hashlib.sha256(journal).hexdigest()
        report["summary"] = summarize([row])
    elif mutation == "goal":
        row["query"]["goal_pose"]["quaternion_wxyz"] = [2.0, 0.0, 0.0, 0.0]
        journal = canonical(row) + b"\n"
        report["journal_sha256"] = hashlib.sha256(journal).hexdigest()
        report["summary"] = summarize([row])
    elif mutation == "run_id":
        report["run_id"] = "other"
    else:
        report["journal_sha256"] = "0" * 64
    with pytest.raises(audit.AuditError):
        audit.audit_bytes(canonical(report), journal, run_id="audit-run")


@pytest.mark.parametrize(
    "metric,changed",
    [
        ("joint_path_length_rad", 0.2001),
        ("tool_path_length_m", 0.1001),
        ("trajectory_duration_seconds", 0.1001),
        ("max_abs_jerk_rad_s3", 0.0001),
    ],
)
def test_audit_rejects_material_mutations_to_each_trajectory_metric(metric, changed):
    result_bytes, journal = plan_bytes()
    report = json.loads(result_bytes)
    row = json.loads(journal)
    row["metrics"][metric] = changed
    journal = canonical(row) + b"\n"
    report["journal_sha256"] = hashlib.sha256(journal).hexdigest()
    report["summary"] = summarize([row])

    with pytest.raises(audit.AuditError):
        audit.audit_bytes(canonical(report), journal, run_id="audit-run")


@pytest.mark.parametrize(
    "mutation",
    [
        "none",
        "torque",
        "limit",
        "mass",
        "energy",
        "max_torque",
        "violation",
        "duplicate_name",
        "unexpected_name",
    ],
)
def test_dynamics_audit_recomputes_torque_and_energy(mutation):
    joint_names = [f"joint{i}" for i in range(7)]
    row = {
        "mode": "kinematic",
        "trajectory": {"joint_names": joint_names},
        "metrics": {
            "energy_proxy_j": 0.07,
            "max_torque_nm": 1.0,
            "torque_violation": 0,
        },
        "dynamics_evidence": {
            "attached_mass_kg": 0.0,
            "trajectory": {
                "joint_names": joint_names.copy(),
                "dt": 0.1,
                "position": [[0.0] * 7, [0.1] * 7],
                "velocity": [[0.0] * 7, [0.1] * 7],
                "acceleration": [[0.0] * 7, [0.0] * 7],
                "jerk": [[0.0] * 7, [0.0] * 7],
            },
            "torques_nm": [[1.0] * 7, [1.0] * 7],
            "torque_limits_nm": [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0],
        },
    }
    if mutation == "torque":
        row["dynamics_evidence"]["torques_nm"][1][0] = 2.0
    elif mutation == "limit":
        row["dynamics_evidence"]["torque_limits_nm"][0] = 88.0
    elif mutation == "mass":
        row["dynamics_evidence"]["attached_mass_kg"] = 3.0
    elif mutation == "energy":
        row["metrics"]["energy_proxy_j"] = 1.0
    elif mutation == "max_torque":
        row["metrics"]["max_torque_nm"] = 1.0001
    elif mutation == "violation":
        row["metrics"]["torque_violation"] = 1
    elif mutation == "duplicate_name":
        row["dynamics_evidence"]["trajectory"]["joint_names"][0] = "joint1"
    elif mutation == "unexpected_name":
        row["dynamics_evidence"]["trajectory"]["joint_names"][0] = "other"
    if mutation == "none":
        audit._audit_dynamics(row)
    else:
        with pytest.raises(audit.AuditError):
            audit._audit_dynamics(row)


def test_independent_audit_does_not_import_producer_validation_helpers():
    source = inspect.getsource(audit)
    assert ".artifacts import" not in source
    assert ".runner import" not in source


def acceptance_cells():
    return {
        "kinematic/motion_benchmaker": {
            "input_count": 800,
            "invalid": 1,
            "success": 792,
            "failed": 7,
            "torque_violation_successes": 0,
        },
        "kinematic/mpinets": {
            "input_count": 1800,
            "invalid": 9,
            "success": 1774,
            "failed": 17,
            "torque_violation_successes": 0,
        },
        "dynamics/motion_benchmaker": {
            "input_count": 800,
            "invalid": 1,
            "success": 779,
            "failed": 20,
            "torque_violation_successes": 19,
        },
        "dynamics/mpinets": {
            "input_count": 1800,
            "invalid": 9,
            "success": 1752,
            "failed": 39,
            "torque_violation_successes": 50,
        },
    }


def test_benchmark_acceptance_uses_combined_failure_and_torque_gate():
    result = audit.benchmark_acceptance(acceptance_cells())
    assert result["passed"] is True
    assert result["cells"]["dynamics/motion_benchmaker"]["unusable_eligible"] == 39
    assert result["cells"]["dynamics/mpinets"]["unusable_eligible"] == 89


@pytest.mark.parametrize(
    "cell,field",
    [
        ("kinematic/motion_benchmaker", "failed"),
        ("kinematic/mpinets", "failed"),
        ("dynamics/motion_benchmaker", "failed"),
        ("dynamics/motion_benchmaker", "torque_violation_successes"),
        ("dynamics/mpinets", "failed"),
        ("dynamics/mpinets", "torque_violation_successes"),
    ],
)
def test_benchmark_acceptance_rejects_one_row_beyond_each_gate(cell, field):
    cells = acceptance_cells()
    cells[cell][field] += 1
    if field == "failed":
        cells[cell]["success"] -= 1
    with pytest.raises(audit.AuditError, match="unusable-row gate"):
        audit.benchmark_acceptance(cells)
