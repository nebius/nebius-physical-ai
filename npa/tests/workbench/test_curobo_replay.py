"""Independent cuRobo replay helpers stay separate and fail closed."""

from __future__ import annotations

import inspect
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from npa.workbench.curobo import replay


def _x_rotation(angle: float) -> np.ndarray:
    return np.asarray([np.cos(angle / 2.0), np.sin(angle / 2.0), 0.0, 0.0])


def test_quaternion_replay_distance_is_sign_invariant():
    identity = np.asarray([1.0, 0.0, 0.0, 0.0])
    assert replay._quaternion_distance(identity, identity) == 0.0
    assert replay._quaternion_distance(identity, -identity) == 0.0
    assert replay._quaternion_distance(
        identity, np.asarray([0.0, 1.0, 0.0, 0.0])
    ) == pytest.approx(np.pi)


def test_quaternion_replay_distance_is_stable_for_retained_float32_boundary():
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
    mixed_precision_acos = float(
        2.0
        * np.arccos(
            np.clip(
                abs(
                    np.dot(
                        replayed / np.linalg.norm(replayed),
                        durable / np.linalg.norm(durable),
                    )
                ),
                0.0,
                1.0,
            )
        )
    )
    assert mixed_precision_acos > replay.QUATERNION_REPLAY_ATOL_RAD
    assert replay._quaternion_comparison(replayed, durable) == (0.0, 0.0)


@pytest.mark.parametrize(
    ("angle", "expected_pass"),
    [(7.5e-6, True), (1.25e-5, False), (1e-3, False)],
)
def test_quaternion_replay_distance_preserves_angular_gate(angle, expected_pass):
    identity = np.asarray([1.0, 0.0, 0.0, 0.0])
    rotated = _x_rotation(angle)
    measured = replay._quaternion_distance(identity, rotated)
    assert replay.QUATERNION_REPLAY_ATOL_RAD == 1e-5
    assert measured == pytest.approx(angle, abs=1e-15)
    if expected_pass:
        replayed_angle, component_delta = replay._require_quaternion_replay(
            rotated[None, :], identity[None, :]
        )
        assert replayed_angle == pytest.approx(angle, abs=1e-15)
        assert component_delta > 0.0
    else:
        with pytest.raises(
            replay.ReplayError,
            match=(
                r"FK quaternion replay differs: max angular error .* rad; "
                r"max sign-invariant raw component delta .*"
            ),
        ):
            replay._require_quaternion_replay(rotated[None, :], identity[None, :])


@pytest.mark.parametrize(
    "invalid",
    [
        np.asarray([np.nan, 0.0, 0.0, 0.0]),
        np.asarray([0.0, 0.0, 0.0, 0.0]),
        np.asarray([1.0, 0.0, 0.0]),
    ],
)
def test_quaternion_replay_rejects_nonfinite_zero_or_wrong_shape(invalid):
    with pytest.raises(replay.ReplayError, match="quaternion"):
        replay._quaternion_distance(invalid, np.asarray([1.0, 0.0, 0.0, 0.0]))


def test_replay_comparison_rejects_shape_and_value_changes():
    expected = np.asarray([[0.0, 0.1], [0.2, 0.3]])
    assert (
        replay._require_close(expected.copy(), expected, name="fixture", atol=1e-9)
        == 0.0
    )
    with pytest.raises(replay.ReplayError, match="shape"):
        replay._require_close(expected[:, :1], expected, name="fixture", atol=1e-9)
    changed = expected.copy()
    changed[1, 1] += 1e-4
    with pytest.raises(replay.ReplayError, match="differs"):
        replay._require_close(changed, expected, name="fixture", atol=1e-9)


def test_replay_does_not_import_producer_or_validator_helpers():
    source = inspect.getsource(replay)
    assert ".runner import" not in source
    assert ".artifacts import" not in source
    assert ".audit import" not in source


def test_replay_rows_executes_independent_fk_path(monkeypatch):
    active_names = [f"j{index}" for index in range(7)]

    class Tensor:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=float)

        @property
        def shape(self):
            return self.values.shape

        def detach(self):
            return self

        def cpu(self):
            return self

        def reshape(self, *shape):
            return Tensor(self.values.reshape(*shape))

        def numpy(self):
            return self.values

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", torch)

    scene = types.ModuleType("curobo._src.geom.types")
    scene.SceneCfg = types.SimpleNamespace(create=lambda value: value)
    monkeypatch.setitem(sys.modules, "curobo", types.ModuleType("curobo"))
    monkeypatch.setitem(sys.modules, "curobo._src", types.ModuleType("curobo._src"))
    monkeypatch.setitem(
        sys.modules, "curobo._src.geom", types.ModuleType("curobo._src.geom")
    )
    monkeypatch.setitem(sys.modules, "curobo._src.geom.types", scene)

    destroyed = []

    class Planner:
        joint_names = active_names
        tool_frames = ["tool"]
        device_cfg = types.SimpleNamespace(to_device=lambda value: value)

        def __init__(self, _cfg):
            self.kinematics = types.SimpleNamespace(
                joint_names=active_names,
                all_articulated_joint_names=active_names,
                compute_kinematics=lambda _state: types.SimpleNamespace(
                    tool_poses=types.SimpleNamespace(
                        get_link_pose=lambda _frame: types.SimpleNamespace(
                            position=Tensor([[0.4, 0.0, 0.2], [0.5, 0.0, 0.3]]),
                            quaternion=Tensor(
                                [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
                            ),
                        )
                    )
                )
            )

        def destroy(self):
            destroyed.append(True)

    planner = types.ModuleType("curobo.motion_planner")
    planner.MotionPlanner = Planner
    planner.MotionPlannerCfg = types.SimpleNamespace(create=lambda **kwargs: kwargs)
    monkeypatch.setitem(sys.modules, "curobo.motion_planner", planner)
    curobo_types = types.ModuleType("curobo.types")
    curobo_types.JointState = types.SimpleNamespace(
        from_position=lambda values, **_kwargs: values
    )
    monkeypatch.setitem(sys.modules, "curobo.types", curobo_types)
    monkeypatch.setattr(replay, "_runtime_source", lambda: Path("/verified/curobo"))

    row = {
        "status": "success",
        "trajectory": {
            "joint_names": active_names,
            "dt": 0.1,
            "position": [[0.0] * 7, [0.1] * 7],
            "velocity": [[0.0] * 7, [0.0] * 7],
            "acceleration": [[0.0] * 7, [0.0] * 7],
            "jerk": [[0.0] * 7, [0.0] * 7],
            "tool_position": [[0.4, 0.0, 0.2], [0.5, 0.0, 0.3]],
            "tool_quaternion": [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
        },
        "query": {
            "goal_pose": {
                "position_xyz": [0.5, 0.0, 0.3],
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            }
        },
    }
    result = replay.replay_rows([row], {"kind": "plan"})
    assert result["fk_replay_count"] == 1
    assert result["max_fk_quaternion_component_replay_error"] == 0.0
    assert result["terminal_goal_distance_m"]["max"] == 0.0
    assert result["terminal_goal_orientation_rad"]["max"] == 0.0
    assert destroyed == [True]
