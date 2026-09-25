"""Refresh native derived reset observations without advancing simulation time."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from npa.workflows.navigation import reference


def test_reset_event_invalidates_kinematics_after_writes(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setitem(
        sys.modules,
        "isaaclab.envs.mdp.commands.pose_2d_command",
        SimpleNamespace(UniformPose2dCommand=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "isaaclab.utils.math",
        SimpleNamespace(
            quat_from_euler_xyz=lambda roll, pitch, yaw: torch.stack(
                (roll, pitch, torch.sin(yaw / 2), torch.cos(yaw / 2)), dim=-1
            )
        ),
    )
    path = Path(reference.__file__).with_name("reference_mdp.py")
    spec = importlib.util.spec_from_file_location("isolated_reference_reset", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names = (
        "_heading_w",
        "_projected_gravity_b",
        "_root_com_lin_vel_b",
        "_root_com_ang_vel_b",
        "_root_link_lin_vel_b",
        "_root_link_ang_vel_b",
        "_root_link_vel_w",
        "_root_com_pose_w",
    )
    data = SimpleNamespace(
        **{name: SimpleNamespace(timestamp=2.0) for name in names},
        _sim_timestamp=2.0,
        default_joint_pos=SimpleNamespace(torch=torch.zeros((1, 12))),
        default_joint_vel=SimpleNamespace(torch=torch.zeros((1, 12))),
    )
    writes = []

    def write(kind, *args, **kwargs):
        assert all(getattr(data, name).timestamp == 2.0 for name in names)
        writes.append(kind)

    robot = SimpleNamespace(
        data=data,
        write_root_pose_to_sim=lambda *a, **kw: write("pose", *a, **kw),
        write_root_velocity_to_sim=lambda *a, **kw: write("velocity", *a, **kw),
        write_joint_state_to_sim=lambda *a, **kw: write("joints", *a, **kw),
    )
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        scene={"robot": robot},
        npa_override_cases=[
            {"position_m": [0.0, 0.0, 0.6], "heading_rad": 1.2, "goal_m": [2.0, 0.0]}
        ],
    )
    module.reset_cases(env, torch.tensor([0]))
    assert writes == ["pose", "velocity", "joints"]
    assert all(getattr(data, name).timestamp < data._sim_timestamp for name in names)
    assert data._sim_timestamp == 2.0
    assert env.npa_goals.tolist() == [[2.0, 0.0]]


@pytest.mark.parametrize("selected", [[1], [0, 1]])
def test_reset_restores_targets_before_native_actuator_write(monkeypatch, selected):
    torch = pytest.importorskip("torch")
    monkeypatch.setitem(
        sys.modules, "isaaclab.envs", SimpleNamespace(ManagerBasedRLEnv=object)
    )
    path = Path(reference.__file__).with_name("reference_environment.py")
    spec = importlib.util.spec_from_file_location(
        "isolated_reference_environment", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    defaults = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    targets = torch.zeros_like(defaults)
    low = SimpleNamespace(
        _joint_ids=[0, 2], processed_actions=torch.zeros((2, 2)), reset=lambda ids: None
    )
    action = SimpleNamespace(
        low_level_actions=torch.ones((2, 2)),
        _raw_actions=torch.ones((2, 3)),
        _low_level_obs_manager=SimpleNamespace(reset=lambda ids: None),
        _low_level_action_term=low,
        _counter=7,
    )

    def write(target, *, joint_ids, env_ids):
        targets[env_ids[:, None], torch.tensor(joint_ids)] = target

    robot = SimpleNamespace(
        data=SimpleNamespace(default_joint_pos=SimpleNamespace(torch=defaults)),
        set_joint_position_target_index=write,
    )
    env = SimpleNamespace(
        num_envs=2,
        scene={"robot": robot},
        action_manager=SimpleNamespace(get_term=lambda name: action),
    )
    ids = torch.tensor(selected)
    network = torch.nn.LSTM(2, 3, batch_first=True).eval()
    hidden = []
    for stale in (4.0, -7.0):
        targets.fill_(stale)
        low.processed_actions.fill_(stale)
        module.ReferenceEnvironment._reset_controller(env, ids)
        assert torch.equal(low.processed_actions[ids], defaults[ids][:, [0, 2]])
        error = targets[ids][:, [0, 2]] - defaults[ids][:, [0, 2]]
        with torch.no_grad():
            _, (memory, _) = network(
                torch.stack((error, torch.zeros_like(error)), dim=-1).reshape(-1, 1, 2)
            )
        hidden.append(memory)
        if selected == [1]:
            assert torch.all(targets[0] == stale)
            assert torch.all(low.processed_actions[0] == stale)
    assert torch.equal(hidden[0], hidden[1])
    assert action._counter == (0 if len(selected) == 2 else 7)
