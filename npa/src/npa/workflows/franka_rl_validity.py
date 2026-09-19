"""Reject measured articulation contract violations before manipulation rewards are computed."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json

_CONTRACT = {
    "schema": "npa.manipulation-validity.v1",
    "sampling": "post-physics control-step, before reward and automatic reset",
    "position_allowance": "native joint velocity limit times physics timestep",
    "velocity_allowance": "floating-point rounding only",
    "rounding_epsilon_multiplier": 4,
    "quaternion_norm_tolerance": 0.0001,
    "failure_action": "abort stage and retain first-fault evidence",
    "task_domain": "absolute object XYZ relative to environment origin <= half scene spacing",
    "task_domain_exit": "ordinary episode failure, zero terminal reward, revoke earlier success",
    "checks": [
        "finite measured state",
        "native hard joint position limits",
        "native simulator joint velocity limits",
        "unit pose quaternions",
    ],
}


def validity_contract() -> dict:
    """Return the sealed measured-state contract shared by all embodiments.

    Args:
        None.
    Returns:
        Independent JSON-compatible contract.
    Raises:
        None.
    """
    return deepcopy(_CONTRACT)


class SimulationValidityError(RuntimeError):
    """Carry the first measured violation without interpreting it as task success.

    Args:
        evidence: JSON-compatible measured-state failure record.
    Returns:
        Exception with a retained diagnostic record.
    Raises:
        None.
    """

    def __init__(self, evidence: dict):
        self.evidence = evidence
        super().__init__(
            "Measured simulation state violates its sealed contract: "
            + evidence["constraint"]
        )


def configure_validity(config, recipe: dict) -> None:
    """Install a lazy pre-reward validity term for a newly sealed experiment.

    Args:
        config: Native Isaac environment configuration.
        recipe: Experiment recipe; historical recipes omit this contract.
    Returns:
        None.
    Raises:
        ValueError: The declared contract was changed.
    """
    contract = recipe.get("simulation_validity")
    if contract is None:
        return
    if contract != _CONTRACT:
        raise ValueError("Unsupported measured simulation validity contract")
    from isaaclab.managers import TerminationTermCfg

    term = TerminationTermCfg(
        func="npa.workflows.franka_rl_validity:check_simulation_state"
    )
    domain = TerminationTermCfg(
        func="npa.workflows.franka_rl_validity:outside_task_domain"
    )
    if isinstance(config.terminations, dict):
        config.terminations["npa_validity"] = term
        config.terminations["npa_workspace_exit"] = domain
    else:
        config.terminations.npa_validity = term
        config.terminations.npa_workspace_exit = domain
    config.npa_simulation_validity = deepcopy(contract)


def joint_contract_violations(
    positions, velocities, limits, maximum_velocity, physics_dt: float
):
    """Compare measured joints with their native bounds, including continuous joints.

    Args:
        positions: Measured joint positions, batch by joint.
        velocities: Measured joint velocities with the same layout.
        limits: Native hard lower/upper bounds, batch by joint by two.
        maximum_velocity: Simulator-configured positive joint velocity bounds.
        physics_dt: Native integration timestep, before control decimation.
    Returns:
        Named boolean violation arrays and their explicit comparison allowances.
    Raises:
        ValueError: Native limit tensors or integration timestep are invalid.
    """
    import math
    import torch

    if (
        limits.shape != (*positions.shape, 2)
        or velocities.shape != positions.shape
        or maximum_velocity.shape != positions.shape
        or not math.isfinite(physics_dt)
        or physics_dt <= 0
        or not torch.isfinite(limits).all()
        or (limits[..., 0] > limits[..., 1]).any()
        or not torch.isfinite(maximum_velocity).all()
        or (maximum_velocity <= 0).any()
    ):
        raise ValueError(
            "Native joint bounds do not define a valid measured-state contract"
        )
    rounding = (
        torch.finfo(positions.dtype).eps * _CONTRACT["rounding_epsilon_multiplier"]
    )
    position_allowance = (
        maximum_velocity * physics_dt
        + limits.abs().amax(dim=-1).clamp(min=1) * rounding
    )
    velocity_allowance = maximum_velocity.abs().clamp(min=1) * rounding
    outside = (positions < limits[..., 0] - position_allowance) | (
        positions > limits[..., 1] + position_allowance
    )
    return {
        "joint_position": outside,
        "joint_velocity": velocities.abs() > maximum_velocity + velocity_allowance,
        "position_allowance": position_allowance,
        "velocity_allowance": velocity_allowance,
    }


def _state_channels(env) -> dict:
    robot, obj = env.scene["robot"].data, env.scene["object"].data
    return {
        "joint_position": robot.joint_pos.torch,
        "joint_velocity": robot.joint_vel.torch,
        "robot_body_position": robot.body_link_pos_w.torch,
        "robot_body_quaternion": robot.body_link_quat_w.torch,
        "robot_body_velocity": robot.body_link_lin_vel_w.torch,
        "robot_body_angular_velocity": robot.body_link_ang_vel_w.torch,
        "robot_root_position": robot.root_pos_w.torch,
        "robot_root_quaternion": robot.root_quat_w.torch,
        "object_position": obj.root_pos_w.torch,
        "object_quaternion": obj.root_quat_w.torch,
        "object_velocity": obj.root_lin_vel_w.torch,
        "object_angular_velocity": obj.root_ang_vel_w.torch,
    }


def _violations(env, channels: dict) -> tuple[list, dict]:
    import torch

    data = env.scene["robot"].data
    joints = joint_contract_violations(
        channels["joint_position"],
        channels["joint_velocity"],
        data.joint_pos_limits.torch,
        data.joint_vel_limits.torch,
        float(env.cfg.sim.dt),
    )
    checks = [
        (name, "nonfinite", ~torch.isfinite(value)) for name, value in channels.items()
    ]
    checks.extend(
        (name, "native_limit_exceeded", joints[name])
        for name in ("joint_position", "joint_velocity")
    )
    for name in ("robot_body_quaternion", "robot_root_quaternion", "object_quaternion"):
        error = (torch.linalg.vector_norm(channels[name], dim=-1) - 1).abs()
        checks.append(
            (name, "nonunit_quaternion", error > _CONTRACT["quaternion_norm_tolerance"])
        )
    return checks, joints


def _scalar(value):
    import math

    number = float(value)
    return number if math.isfinite(number) else None


def _fault(env, channels, allowances, channel, constraint, indices, phase) -> dict:
    row = tuple(int(value) for value in indices.tolist())
    contract = env.cfg.npa_simulation_validity
    evidence = {
        "schema": "npa.manipulation-validity-failure.v1",
        "contract": contract,
        "contract_sha256": hashlib.sha256(
            json.dumps(contract, sort_keys=True).encode()
        ).hexdigest(),
        "constraint": constraint,
        "channel": channel,
        "indices": list(row),
        "phase": phase,
        "environment_index": row[0],
        "episode_step": int(env.episode_length_buf[row[0]]),
        "control_step": int(env.common_step_counter),
        "physics_dt": float(env.cfg.sim.dt),
        "control_dt": float(env.step_dt),
        "substeps_monitored": False,
    }
    value = channels[channel][row]
    evidence["measured"] = (
        [_scalar(x) for x in value.reshape(-1)] if value.ndim else _scalar(value)
    )
    if channel in ("joint_position", "joint_velocity"):
        data = env.scene["robot"].data
        evidence["joint_name"] = env.scene["robot"].joint_names[row[1]]
        evidence["native_position_bounds"] = [
            _scalar(x) for x in data.joint_pos_limits.torch[row]
        ]
        evidence["native_velocity_limit"] = _scalar(data.joint_vel_limits.torch[row])
        evidence["position_allowance"] = _scalar(allowances["position_allowance"][row])
        evidence["velocity_allowance"] = _scalar(allowances["velocity_allowance"][row])
    arm = env.action_manager.get_term("arm_action")
    target = getattr(arm, "latest_command_target", None)
    if target is not None:
        evidence["commanded_arm_target"] = [_scalar(x) for x in target[row[0]]]
    return evidence


def validate_simulation_state(env, *, phase: str = "before_policy") -> None:
    """Reject the first invalid measured state before policy or reward consumption.

    Args:
        env: Native initialized environment with current measured tensors.
        phase: Diagnostic sampling phase; the native termination uses pre-reward.
    Returns:
        None; increments the successfully checked state count.
    Raises:
        SimulationValidityError: A measured state violates the sealed contract.
        ValueError: The initialized native bounds are invalid.
    """
    if not hasattr(env.cfg, "npa_simulation_validity"):
        return
    import torch

    channels = _state_channels(env)
    checks, allowances = _violations(env, channels)
    failed = torch.stack([mask.any() for _, _, mask in checks])
    if failed.any():
        index = int(failed.nonzero()[0, 0])
        channel, constraint, mask = checks[index]
        evidence = _fault(
            env, channels, allowances, channel, constraint, mask.nonzero()[0], phase
        )
        env.npa_validity_failure = evidence
        raise SimulationValidityError(evidence)
    env.npa_validity_checks = getattr(env, "npa_validity_checks", 0) + 1


def check_simulation_state(env):
    """Run the pre-reward native termination hook without changing valid episodes.

    Args:
        env: Native initialized environment after the physics control step.
    Returns:
        False termination flags; a contract violation aborts the entire stage.
    Raises:
        SimulationValidityError: Invalid state must not enter learning or success metrics.
    """
    import torch

    validate_simulation_state(env, phase="post_physics_pre_reward")
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def build_validated_wrapper(env, clip_actions):
    """Check initial and automatically reset states before the learner receives them.

    Args:
        env: Native Gymnasium environment.
        clip_actions: Native RSL-RL action clipping setting.
    Returns:
        Native vector wrapper with measured-state checks around its public handoff.
    Raises:
        SimulationValidityError: A reset or initial state violates the contract.
    """
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    class CheckedWrapper(RslRlVecEnvWrapper):
        def reset(self):
            result = super().reset()
            validate_simulation_state(self.unwrapped, phase="reset_before_policy")
            return result

        def step(self, actions):
            result = super().step(actions)
            validate_simulation_state(self.unwrapped, phase="post_step_before_policy")
            if hasattr(self.unwrapped.cfg, "npa_simulation_validity"):
                obs, rewards, dones, extras = result
                rewards = rewards.masked_fill(task_domain_exits(self.unwrapped), 0)
                return obs, rewards, dones, extras
            return result

    wrapped = CheckedWrapper(env, clip_actions=clip_actions)
    validate_simulation_state(env.unwrapped, phase="initial_before_learner")
    return wrapped


def outside_task_domain(env):
    """Terminate object departures from the task envelope as ordinary failures.

    Args:
        env: Native environment after a physics control step, before reset.
    Returns:
        Per-environment departure flags; the vertical extent is a task envelope.
    Raises:
        ValueError: Scene spacing cannot define a finite task envelope.
    """
    import math

    spacing = float(env.cfg.scene.env_spacing)
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("Scene spacing must define a finite positive task envelope")
    local = env.scene["object"].data.root_pos_w.torch - env.scene.env_origins
    return (local.abs() > spacing / 2).any(dim=-1)


def task_domain_exits(env):
    """Read the pre-reset task-domain termination flags retained by Isaac.

    Args:
        env: Native environment immediately after stepping or during reward calculation.
    Returns:
        Per-environment departure flags, or false flags for a historical recipe.
    Raises:
        RuntimeError: A guarded environment lacks its required termination term.
    """
    import torch

    if not hasattr(env.cfg, "npa_simulation_validity"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return env.termination_manager.get_term("npa_workspace_exit")


def validity_evidence(env) -> dict:
    """Report measured-state coverage without certifying unsampled physics substeps.

    Args:
        env: Native environment whose first state was checked before policy use.
    Returns:
        Coverage and native constraint provenance, or an unverified legacy record.
    Raises:
        ValueError: A guarded environment completed without a measured check.
    """
    if not hasattr(env.cfg, "npa_simulation_validity"):
        return {
            "verified": False,
            "reason": "legacy run has no measured-state validity contract",
        }
    count = getattr(env, "npa_validity_checks", 0)
    if not count or hasattr(env, "npa_validity_failure"):
        raise ValueError("Simulation validity checks did not complete successfully")
    return {
        "verified": True,
        "contract": deepcopy(env.cfg.npa_simulation_validity),
        "checked_batches": count,
        "environments_per_batch": env.num_envs,
        "substeps_monitored": False,
        "joint_limit_source": "initialized native articulation joint_pos_limits and joint_vel_limits",
    }
