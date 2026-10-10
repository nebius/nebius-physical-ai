"""Exercise the finger target ramp independently of native actuator tracking."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def servo(monkeypatch):
    class BinaryAction:
        def __init__(self, cfg, env):
            self._env = env
            self._asset = env.robot
            self._joint_ids = [0, 1]
            self._processed_actions = torch.zeros((2, 2))

        @property
        def processed_actions(self):
            return self._processed_actions

        def process_actions(self, actions):
            self.raw_actions = actions.clone()
            self._processed_actions = (
                torch.where(actions > 0, 0.04, 0.0).expand(2, 2).clone()
            )

        def reset(self, env_ids=None):
            pass

    native = ModuleType("isaaclab.envs.mdp.actions.binary_joint_actions")
    native.BinaryJointPositionAction = BinaryAction
    monkeypatch.setitem(sys.modules, native.__name__, native)
    path = (
        Path(__file__).resolve().parents[2]
        / "src/npa/workflows/physical_augmentation_servo.py"
    )
    spec = importlib.util.spec_from_file_location("tested_physical_gripper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = NS(
        joint_pos_limits=NS(torch=torch.tensor([[[0.0, 0.04]] * 2] * 2)),
        joint_vel_limits=NS(torch=torch.full((2, 2), 0.2)),
        joint_effort_limits=NS(torch=torch.full((2, 2), 20.0)),
        joint_pos=NS(torch=torch.full((2, 2), 0.04)),
    )
    env = NS(
        robot=NS(data=data),
        step_dt=0.02,
        cfg=NS(
            npa_gripper_servo={"native_velocity_fraction": 0.25, "effort_limit_n": 20.0}
        ),
    )
    return module.RampedGripperAction(None, env), data


def test_close_and_reverse_keep_binary_intent_and_bounded_targets(servo):
    action, data = servo
    close = -torch.ones((2, 1))
    for index in range(20):
        previous = action.target.clone()
        action.process_actions(close)
        assert torch.equal(action.raw_actions, close)
        assert torch.allclose(
            action.processed_actions, torch.full((2, 2), 0.04 - (index + 1) * 0.001)
        )
        assert torch.max(torch.abs(action.target - previous)) <= 0.001001
    # Tracking lag does not reset the commanded trajectory to measured position.
    assert torch.all(data.joint_pos.torch == 0.04)
    action.process_actions(-close)
    assert torch.allclose(action.target, torch.full((2, 2), 0.021))
    for _ in range(60):
        action.process_actions(close)
    assert torch.equal(action.target, torch.zeros((2, 2)))


def test_reset_uses_measured_fingers_and_preserves_other_environments(servo):
    action, data = servo
    action.process_actions(-torch.ones((2, 1)))
    untouched = action.target[1].clone()
    data.joint_pos.torch[0] = torch.tensor([0.023, 0.025])
    action.reset([0])
    assert torch.equal(action.target[0], data.joint_pos.torch[0])
    assert torch.equal(action.processed_actions[0], data.joint_pos.torch[0])
    assert torch.equal(action.target[1], untouched)


def test_nonfinite_gripper_intent_is_rejected(servo):
    action, _ = servo
    with pytest.raises(ValueError, match="Nonfinite"):
        action.process_actions(torch.full((2, 1), float("nan")))


def test_native_actuator_must_apply_the_sealed_force_limit(servo):
    action, data = servo
    data.joint_effort_limits.torch.fill_(200.0)
    with pytest.raises(ValueError, match="gripper limits"):
        type(action)(None, action._env)
