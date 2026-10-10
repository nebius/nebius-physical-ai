"""Reject cross-row or changed interior dynamics without conflating sample grids."""

from copy import deepcopy

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from npa.workbench.curobo import artifacts, audit


def _series(spline, steps, dt):
    grid = np.arange(20 * steps + 1) / steps
    return {
        "joint_names": [f"joint{i}" for i in range(7)],
        "dt": dt,
        **{
            field: (spline(grid, order) / (steps * dt) ** order).tolist()
            for order, field in enumerate(
                ("position", "velocity", "acceleration", "jerk")
            )
        },
    }


def _row(seed=1, raw_dt=0.02):
    # Independent SciPy cubic construction, not the implementation's Taylor
    # evaluator. Both paths have identical endpoints; interior knots differ.
    knots = np.random.default_rng(seed).uniform(-0.5, 0.5, size=(21, 7))
    knots[[0, -1]] = 0
    spline = CubicSpline(np.arange(21), knots)
    raw_dt = float(np.float32(raw_dt))
    dense_dt = float(np.float32(0.025))
    steps = int((np.float32(raw_dt) * 4 + np.float32(dense_dt)) / np.float32(dense_dt))
    optimized = _series(spline, 4, raw_dt)
    return {
        "mode": "kinematic",
        "trajectory": _series(spline, steps, dense_dt),
        "metrics": {
            "energy_proxy_j": float(np.abs(optimized["velocity"]).sum() * raw_dt),
            "max_torque_nm": 1.0,
            "torque_violation": 0,
        },
        "dynamics_evidence": {
            "attached_mass_kg": 0.0,
            "trajectory": optimized,
            "torques_nm": np.ones((81, 7)).tolist(),
            "torque_limits_nm": [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0],
        },
    }


@pytest.fixture(params=[artifacts._validate_dynamics_evidence, audit._audit_dynamics])
def validator(request):
    return request.param


@pytest.mark.parametrize("raw_dt", [0.003, 0.02, 0.049])
def test_binds_all_samples_across_actual_cubic_retiming_grids(validator, raw_dt):
    row = _row(raw_dt=raw_dt)
    assert row["trajectory"]["dt"] != row["dynamics_evidence"]["trajectory"]["dt"]
    validator(row)


def test_swapped_dynamics_rejected_even_with_identical_endpoints_and_metrics(validator):
    row, other = _row(1), _row(2)
    for index in (0, -1):
        assert row["trajectory"]["position"][index] == pytest.approx(
            other["trajectory"]["position"][index], abs=1e-14
        )
    row["dynamics_evidence"] = other["dynamics_evidence"]
    row["metrics"] = other["metrics"]
    with pytest.raises((artifacts.CuroboError, audit.AuditError), match="bound"):
        validator(row)


@pytest.mark.parametrize("side", ["optimized", "retained"])
@pytest.mark.parametrize("field", ["position", "velocity", "acceleration", "jerk"])
def test_changed_interior_sample_cannot_hide_behind_unchanged_endpoints(
    validator, side, field
):
    row = _row()
    series = (
        row["trajectory"]
        if side == "retained"
        else row["dynamics_evidence"]["trajectory"]
    )
    series[field][17][2] += 1
    if side == "optimized":
        row["metrics"]["energy_proxy_j"] = float(
            np.abs(series["velocity"]).sum() * series["dt"]
        )
    with pytest.raises((artifacts.CuroboError, audit.AuditError), match="bound"):
        validator(row)


@pytest.mark.parametrize("mutation", ["dt", "samples", "scaled_derivatives"])
def test_retained_timing_is_part_of_path_binding(validator, mutation):
    row = _row()
    series = row["trajectory"]
    if mutation == "dt":
        series["dt"] *= 2
    elif mutation == "samples":
        for field in ("position", "velocity", "acceleration", "jerk"):
            series[field].pop(17)
    else:
        series["velocity"] = deepcopy(
            row["dynamics_evidence"]["trajectory"]["velocity"]
        )
    with pytest.raises((artifacts.CuroboError, audit.AuditError), match="bound"):
        validator(row)


def test_shared_samples_bind_all_values_and_follow_joint_names(validator):
    row = _row()
    raw = row["dynamics_evidence"]["trajectory"]
    row["trajectory"] = deepcopy(raw)
    row["trajectory"]["joint_names"].reverse()
    for field in ("position", "velocity", "acceleration", "jerk"):
        row["trajectory"][field] = [list(reversed(sample)) for sample in raw[field]]
    validator(row)
    row["trajectory"]["position"][17][2] += 0.1
    with pytest.raises((artifacts.CuroboError, audit.AuditError), match="bound"):
        validator(row)


@pytest.mark.parametrize("side", ["optimized", "retained"])
@pytest.mark.parametrize("dt", [1e300, 1e-300])
def test_nonrepresentable_kernel_timing_is_a_typed_refusal(validator, side, dt):
    row = _row()
    series = (
        row["trajectory"]
        if side == "retained"
        else row["dynamics_evidence"]["trajectory"]
    )
    series["dt"] = dt
    if side == "optimized":
        row["metrics"]["energy_proxy_j"] = float(np.abs(series["velocity"]).sum() * dt)
    with pytest.raises((artifacts.CuroboError, audit.AuditError), match="bound"):
        validator(row)
