"""Configure and attest native stock-Franka control for stable manipulation."""

from __future__ import annotations

import copy
import math
from typing import Any


def franka_control_contract() -> dict[str, Any]:
    """Describe the canonical native manipulation control and physics clock.

    Args:
        None.
    Returns:
        Expected control parameters shared by training and held-out evaluation.
    Raises:
        None.
    """
    return {
        "profile": "stock-franka-high-pd-200hz-v1",
        "robot_gravity_compensated": True,
        "arm_stiffness": 400.0,
        "arm_damping": 80.0,
        "physics_dt_s": 0.005,
        "policy_dt_s": 0.02,
        "decimation": 4,
    }


def configure_stock_franka_control(env_cfg: Any) -> None:
    """Apply upstream task-space drives with finer physics at the native policy rate.

    Args:
        env_cfg: Stock-Franka scenario configuration before native environment creation.
    Returns:
        None; the configuration receives a private copy of the robot settings.
    Raises:
        ValueError: The upstream policy clock differs from the canonical contract.
        ImportError: The pinned Isaac Lab robot configurations are unavailable.
    """
    from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

    contract = franka_control_contract()
    policy_dt = float(env_cfg.sim.dt) * int(env_cfg.decimation)
    if not math.isclose(policy_dt, contract["policy_dt_s"], abs_tol=1e-9):
        raise ValueError("stock-Franka control requires the native 50 Hz policy clock")
    robot = copy.deepcopy(env_cfg.scene.robot)
    robot.spawn.rigid_props.disable_gravity = (
        FRANKA_PANDA_HIGH_PD_CFG.spawn.rigid_props.disable_gravity
    )
    for name in ("panda_shoulder", "panda_forearm"):
        native = FRANKA_PANDA_HIGH_PD_CFG.actuators[name]
        robot.actuators[name].stiffness = native.stiffness
        robot.actuators[name].damping = native.damping
    env_cfg.scene.robot = robot
    env_cfg.sim.dt = contract["physics_dt_s"]
    env_cfg.decimation = contract["decimation"]
    env_cfg.sim.render_interval = env_cfg.decimation
    stock_franka_control_audit(env_cfg)


def stock_franka_control_audit(env_cfg: Any) -> dict[str, Any]:
    """Verify the actual native configuration before attesting its control profile.

    Args:
        env_cfg: Instantiated or prepared stock-Franka scenario configuration.
    Returns:
        The verified control profile, physics interval and policy interval.
    Raises:
        ValueError: The actual native drives, gravity compensation or clock differ.
    """
    contract = franka_control_contract()
    robot = env_cfg.scene.robot
    actual = {
        "physics_dt_s": float(env_cfg.sim.dt),
        "policy_dt_s": float(env_cfg.sim.dt) * int(env_cfg.decimation),
        "decimation": int(env_cfg.decimation),
    }
    if any(actual[name] != contract[name] for name in actual):
        raise ValueError("native stock-Franka physics or policy clock changed")
    if robot.spawn.rigid_props.disable_gravity is not True:
        raise ValueError("native stock-Franka gravity compensation is missing")
    for name in ("panda_shoulder", "panda_forearm"):
        arm = robot.actuators[name]
        if (arm.stiffness, arm.damping) != (400.0, 80.0):
            raise ValueError(f"native stock-Franka task-space drive changed: {name}")
    return contract
