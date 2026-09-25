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
