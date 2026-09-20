"""Independent cuRobo replay helpers stay separate and fail closed."""

from __future__ import annotations

import inspect
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from npa.workbench.curobo import replay


def test_quaternion_replay_distance_is_sign_invariant():
    identity = np.asarray([1.0, 0.0, 0.0, 0.0])
    assert replay._quaternion_distance(identity, identity) == 0.0
    assert replay._quaternion_distance(identity, -identity) == 0.0
    assert replay._quaternion_distance(
        identity, np.asarray([0.0, 1.0, 0.0, 0.0])
    ) == pytest.approx(np.pi)


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
        joint_names = ["j0"]
        tool_frames = ["tool"]
        device_cfg = types.SimpleNamespace(to_device=lambda value: value)

        def __init__(self, _cfg):
            self.kinematics = types.SimpleNamespace(
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
            "joint_names": ["j0"],
            "dt": 0.1,
            "position": [[0.0], [0.1]],
            "velocity": [[0.0], [0.0]],
            "acceleration": [[0.0], [0.0]],
            "jerk": [[0.0], [0.0]],
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
    assert result["terminal_goal_distance_m"]["max"] == 0.0
    assert result["terminal_goal_orientation_rad"]["max"] == 0.0
    assert destroyed == [True]
