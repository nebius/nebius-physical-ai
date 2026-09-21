"""Prove cuRobo inverse dynamics consumes exactly named active Franka joints."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from npa.workbench.curobo import replay, runner
from npa.workbench.curobo.artifacts import CuroboError


ACTIVE_NAMES = [f"joint{index}" for index in range(7)]
EXTRA_NAMES = ["locked_finger", "mimic_finger"]
AVAILABLE_NAMES = [*ACTIVE_NAMES, *EXTRA_NAMES]
TORQUE_LIMITS = np.asarray([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])


class _Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    @property
    def shape(self):
        return self.values.shape

    def detach(self):
        return self

    def cpu(self):
        return self

    def reshape(self, *shape):
        return _Tensor(self.values.reshape(*shape))

    def numpy(self):
        return self.values

    def item(self):
        return self.values.item()


class _State:
    def __init__(self, names, arrays=None, *, dt=0.1):
        self.joint_names = list(names)
        arrays = arrays or _arrays_for(names)
        for field, values in arrays.items():
            setattr(self, field, _Tensor(values))
        self.dt = _Tensor(dt)

    def reorder(self, names):
        indices = [self.joint_names.index(name) for name in names]
        arrays = {
            field: getattr(self, field).values[:, indices]
            for field in ("position", "velocity", "acceleration", "jerk")
        }
        return _State(names, arrays, dt=self.dt.item())


def _arrays_for(names):
    codes = {name: AVAILABLE_NAMES.index(name) + 1.0 for name in names}
    position = np.asarray(
        [[codes[name] for name in names], [codes[name] + 0.25 for name in names]]
    )
    width = len(names)
    return {
        "position": position,
        "velocity": np.asarray([[0.0] * width, [0.1] * width]),
        "acceleration": np.zeros((2, width)),
        "jerk": np.zeros((2, width)),
    }


def _planner():
    return SimpleNamespace(
        joint_names=ACTIVE_NAMES.copy(),
        kinematics=SimpleNamespace(
            joint_names=ACTIVE_NAMES.copy(),
            all_articulated_joint_names=AVAILABLE_NAMES.copy(),
        ),
    )


def _model(*, names=None, nq=7, nv=7):
    ordered_names = list(names or ACTIVE_NAMES)
    return SimpleNamespace(
        names=["universe", *ordered_names],
        nq=nq,
        nv=nv,
    )


def _model_data(*, names=None, nq=7, nv=7, limits=None):
    if limits is None:
        limits = TORQUE_LIMITS.copy()
    return (_model(names=names, nq=nq, nv=nv), object(), limits)


@pytest.mark.parametrize(
    "raw_names",
    [
        [
            "locked_finger",
            "joint6",
            "joint2",
            "joint0",
            "mimic_finger",
            "joint5",
            "joint1",
            "joint4",
            "joint3",
        ],
        [
            "joint3",
            "mimic_finger",
            "joint1",
            "joint5",
            "joint0",
            "joint4",
            "locked_finger",
            "joint2",
            "joint6",
        ],
    ],
)
def test_dynamics_input_uses_names_to_remove_extra_joints(raw_names):
    _aligned, trajectory, names = runner._active_dynamics_input(
        _planner(), _State(raw_names)
    )

    assert names == ACTIVE_NAMES
    assert trajectory["joint_names"] == ACTIVE_NAMES
    assert np.asarray(trajectory["position"])[0].tolist() == [
        AVAILABLE_NAMES.index(name) + 1.0 for name in ACTIVE_NAMES
    ]
    for field in ("position", "velocity", "acceleration", "jerk"):
        assert np.asarray(trajectory[field]).shape == (2, 7)


@pytest.mark.parametrize(
    "raw_names",
    [
        [*ACTIVE_NAMES[:-1], *EXTRA_NAMES],
        [*ACTIVE_NAMES, ACTIVE_NAMES[0], *EXTRA_NAMES],
        [*ACTIVE_NAMES, *EXTRA_NAMES, "unexpected_joint"],
    ],
)
def test_dynamics_input_rejects_missing_duplicate_or_unexpected_names(raw_names):
    arrays = {
        field: np.zeros((2, len(raw_names)))
        for field in ("position", "velocity", "acceleration", "jerk")
    }
    with pytest.raises(CuroboError, match="joint"):
        runner._active_dynamics_input(_planner(), _State(raw_names, arrays))


@pytest.mark.parametrize(
    "mutation",
    ["position_width", "velocity_length", "acceleration_nan", "jerk_inf", "dt_nan"],
)
def test_dynamics_input_rejects_malformed_or_nonfinite_fields(mutation):
    state = _State(ACTIVE_NAMES)
    if mutation == "position_width":
        state.position = _Tensor(np.zeros((2, 6)))
    elif mutation == "velocity_length":
        state.velocity = _Tensor(np.zeros((1, 7)))
    elif mutation == "acceleration_nan":
        state.acceleration.values[0, 0] = np.nan
    elif mutation == "jerk_inf":
        state.jerk.values[0, 0] = np.inf
    else:
        state.dt = _Tensor(np.nan)

    with pytest.raises(CuroboError, match="trajectory fields"):
        runner._active_dynamics_input(_planner(), state)


@pytest.mark.parametrize(
    "model_data",
    [
        _model_data(nq=6),
        _model_data(nv=6),
        _model_data(names=[ACTIVE_NAMES[1], ACTIVE_NAMES[0], *ACTIVE_NAMES[2:]]),
        _model_data(limits=TORQUE_LIMITS[:-1]),
        _model_data(limits=np.asarray([*TORQUE_LIMITS[:-1], np.nan])),
    ],
)
def test_dynamics_model_rejects_dimension_order_or_limit_mismatch(model_data):
    with pytest.raises(CuroboError, match="(model|order|limits|dimensions)"):
        runner._dynamics_model_contract(model_data, ACTIVE_NAMES)


@pytest.mark.parametrize(
    "mutation",
    [
        "torque_width",
        "torque_nan",
        "energy_nan",
        "max_torque_nan",
        "nonboolean_violation",
        "wrong_energy",
        "missing_torques",
    ],
)
def test_dynamics_result_rejects_malformed_nonfinite_or_invented_values(mutation):
    trajectory = runner._validated_joint_series(
        _State(ACTIVE_NAMES), label="test dynamics"
    )
    dynamic = {
        "torques": np.ones((2, 7)),
        "energy": 0.07,
        "max_torque": 1.0,
        "torque_violation": False,
    }
    if mutation == "torque_width":
        dynamic["torques"] = np.ones((2, 6))
    elif mutation == "torque_nan":
        dynamic["torques"][0, 0] = np.nan
    elif mutation == "energy_nan":
        dynamic["energy"] = np.nan
    elif mutation == "max_torque_nan":
        dynamic["max_torque"] = np.nan
    elif mutation == "nonboolean_violation":
        dynamic["torque_violation"] = 0
    elif mutation == "wrong_energy":
        dynamic["energy"] = 0.08
    else:
        dynamic.pop("torques")

    with pytest.raises(CuroboError, match="inverse-dynamics"):
        runner._dynamics_result(dynamic, trajectory, TORQUE_LIMITS)


def test_durable_dynamics_evidence_matches_actual_active_order_input():
    raw_names = [EXTRA_NAMES[0], *reversed(ACTIVE_NAMES), EXTRA_NAMES[1]]
    state = _State(raw_names)
    observed = {}

    def compute_energy(aligned, _model_data):
        observed["names"] = list(aligned.joint_names)
        observed["position"] = aligned.position.numpy().copy()
        torques = np.ones((2, 7))
        velocity = aligned.velocity.numpy()
        return {
            "torques": torques,
            "energy": float(np.abs(torques * velocity).sum() * aligned.dt.item()),
            "max_torque": 1.0,
            "torque_violation": False,
        }

    evidence, metrics = runner._compute_dynamics_evidence(
        _planner(),
        SimpleNamespace(js_solution=state),
        SimpleNamespace(compute_trajectory_energy=compute_energy),
        _model_data(),
    )

    assert observed["names"] == ACTIVE_NAMES
    assert evidence["trajectory"]["joint_names"] == observed["names"]
    np.testing.assert_array_equal(
        np.asarray(evidence["trajectory"]["position"]), observed["position"]
    )
    assert np.asarray(evidence["torques_nm"]).shape == (2, 7)
    assert evidence["torque_limits_nm"] == TORQUE_LIMITS.tolist()
    assert metrics["energy_proxy_j"] == pytest.approx(0.07)
    assert metrics["max_torque_nm"] == 1.0
    assert metrics["torque_violation"] == 0


def test_replay_accepts_exact_model_and_active_order_evidence():
    np.testing.assert_array_equal(
        replay._dynamics_model_contract(_model_data(), ACTIVE_NAMES), TORQUE_LIMITS
    )
    series = runner._validated_joint_series(_State(ACTIVE_NAMES), label="test")
    replay._dynamics_arrays(series, ACTIVE_NAMES)


@pytest.mark.parametrize(
    "model_data",
    [
        _model_data(nq=6),
        _model_data(nv=6),
        _model_data(names=[ACTIVE_NAMES[1], ACTIVE_NAMES[0], *ACTIVE_NAMES[2:]]),
        _model_data(limits=TORQUE_LIMITS[:-1]),
        _model_data(limits=np.asarray([*TORQUE_LIMITS[:-1], np.nan])),
    ],
)
def test_replay_independently_rejects_model_contract_tampering(model_data):
    with pytest.raises(replay.ReplayError, match="(model|order|limits|dimensions)"):
        replay._dynamics_model_contract(model_data, ACTIVE_NAMES)


@pytest.mark.parametrize(
    "mutation", ["reordered", "duplicate", "unexpected", "nonfinite"]
)
def test_replay_independently_rejects_trajectory_tampering(mutation):
    series = runner._validated_joint_series(_State(ACTIVE_NAMES), label="test")
    if mutation == "reordered":
        series["joint_names"] = [*reversed(ACTIVE_NAMES)]
    elif mutation == "duplicate":
        series["joint_names"][0] = ACTIVE_NAMES[1]
    elif mutation == "unexpected":
        series["joint_names"][0] = "unexpected_joint"
    else:
        series["velocity"][0][0] = float("nan")

    with pytest.raises(replay.ReplayError, match="joint names|trajectory fields"):
        replay._dynamics_arrays(series, ACTIVE_NAMES)
