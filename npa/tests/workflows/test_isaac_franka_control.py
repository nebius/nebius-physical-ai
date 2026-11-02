"""Verify native stock-Franka control fidelity, isolation and audit rejection."""

import copy
from types import SimpleNamespace
import sys

import pytest

from npa.workflows.sim2real.isaac_franka_control import (
    configure_stock_franka_control,
    franka_control_contract,
    stock_franka_control_audit,
)


def _stock_configuration():
    actuators = {
        name: SimpleNamespace(stiffness=80.0, damping=4.0, effort_limit_sim=12.0)
        for name in ("panda_shoulder", "panda_forearm")
    }
    actuators["panda_hand"] = SimpleNamespace(stiffness=2000.0, damping=100.0)
    robot = SimpleNamespace(
        spawn=SimpleNamespace(rigid_props=SimpleNamespace(disable_gravity=False)),
        actuators=actuators,
    )
    return SimpleNamespace(
        scene=SimpleNamespace(robot=robot, object=object()),
        sim=SimpleNamespace(dt=0.01, render_interval=2),
        decimation=2,
        episode_length_s=5.0,
    )


def _install_native_profile(monkeypatch):
    native = copy.deepcopy(_stock_configuration().scene.robot)
    native.spawn.rigid_props.disable_gravity = True
    for name in ("panda_shoulder", "panda_forearm"):
        native.actuators[name].stiffness = 400.0
        native.actuators[name].damping = 80.0
    monkeypatch.setitem(
        sys.modules,
        "isaaclab_assets.robots.franka",
        SimpleNamespace(FRANKA_PANDA_HIGH_PD_CFG=native),
    )
    return native


def test_finer_physics_preserves_policy_clock_episode_and_shared_robot(monkeypatch):
    native = _install_native_profile(monkeypatch)
    cfg = _stock_configuration()
    shared_robot = cfg.scene.robot
    object_configuration = cfg.scene.object
    configure_stock_franka_control(cfg)
    assert cfg.sim.dt == 0.005
    assert cfg.sim.dt * cfg.decimation == 0.02
    assert cfg.sim.render_interval == cfg.decimation
    assert cfg.episode_length_s / (cfg.sim.dt * cfg.decimation) == 250
    assert cfg.scene.object is object_configuration
    assert cfg.scene.robot is not shared_robot
    assert shared_robot.actuators["panda_shoulder"].stiffness == 80.0
    assert shared_robot.spawn.rigid_props.disable_gravity is False
    assert cfg.scene.robot.actuators["panda_hand"].stiffness == 2000.0
    assert cfg.scene.robot.actuators["panda_forearm"].effort_limit_sim == 12.0
    assert native.actuators["panda_forearm"].stiffness == 400.0
    assert stock_franka_control_audit(cfg) == franka_control_contract()


def test_control_refuses_clock_drift_before_changing_robot(monkeypatch):
    _install_native_profile(monkeypatch)
    cfg = _stock_configuration()
    cfg.sim.dt = 0.02
    shared_robot = cfg.scene.robot
    with pytest.raises(ValueError, match="50 Hz"):
        configure_stock_franka_control(cfg)
    assert cfg.scene.robot is shared_robot
    assert shared_robot.spawn.rigid_props.disable_gravity is False


@pytest.mark.parametrize("changed", ["clock", "gravity", "stiffness", "damping"])
def test_control_audit_reads_actual_native_settings(monkeypatch, changed):
    _install_native_profile(monkeypatch)
    cfg = _stock_configuration()
    configure_stock_franka_control(cfg)
    if changed == "clock":
        cfg.decimation = 2
    elif changed == "gravity":
        cfg.scene.robot.spawn.rigid_props.disable_gravity = False
    else:
        setattr(cfg.scene.robot.actuators["panda_forearm"], changed, 1.0)
    with pytest.raises(ValueError):
        stock_franka_control_audit(cfg)
