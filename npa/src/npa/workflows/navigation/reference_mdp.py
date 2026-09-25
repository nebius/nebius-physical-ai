"""Define native goal, obstacle observation, reset and reward terms for the reference."""

import torch
from isaaclab.envs.mdp.commands.pose_2d_command import UniformPose2dCommand


class FixedGoalCommand(UniformPose2dCommand):
    """Keep each explicit world-frame goal unchanged throughout its episode.

    Args:
        cfg: Native pose command configuration.
        env: Native reference environment.
    Returns:
        Native command term.
    Raises:
        None.
    """

    def _resample_command(self, env_ids):
        self.pos_command_w[env_ids, :2] = self._env.npa_goals[env_ids]
        self.pos_command_w[env_ids, 2] = self.robot.data.root_pos_w.torch[env_ids, 2]
        delta = (
            self.pos_command_w[env_ids, :2]
            - self.robot.data.root_pos_w.torch[env_ids, :2]
        )
        self.heading_command_w[env_ids] = torch.atan2(delta[:, 1], delta[:, 0])


def reset_cases(env, env_ids):
    """Write selected sealed case poses and default joints into real articulations.

    Args:
        env: Native reference environment.
        env_ids: Native indices requiring an episode reset.
    Returns:
        None.
    Raises:
        ValueError: Configured cases are absent.
    """
    from isaaclab.utils.math import quat_from_euler_xyz

    _case_buffers(env)
    selected = _selected_cases(env, env_ids)
    robot = env.scene["robot"]
    positions = torch.tensor(
        [case["position_m"] for case in selected], device=env.device
    )
    yaw = torch.tensor([case["heading_rad"] for case in selected], device=env.device)
    quaternion = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    robot.write_root_pose_to_sim(
        torch.cat((positions, quaternion), dim=1), env_ids=env_ids
    )
    robot.write_root_velocity_to_sim(
        torch.zeros((len(env_ids), 6), device=env.device), env_ids=env_ids
    )
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos.torch[env_ids],
        robot.data.default_joint_vel.torch[env_ids],
        env_ids=env_ids,
    )
    _invalidate_reset_kinematics(robot.data)
    env.npa_goals[env_ids] = torch.tensor(
        [case["goal_m"] for case in selected], device=env.device
    )
    env.npa_previous_distance[env_ids] = torch.linalg.vector_norm(
        env.npa_goals[env_ids] - positions[:, :2], dim=1
    )


def _invalidate_reset_kinematics(data):
    # Lab 3.0.0b2.post1 pose/velocity writers leave these same-timestamp caches
    # stale. Invalidate before command and observation resets without advancing
    # physical time. There is no public cache-invalidation API in this pin.
    for name in (
        "_heading_w",
        "_projected_gravity_b",
        "_root_com_lin_vel_b",
        "_root_com_ang_vel_b",
        "_root_link_lin_vel_b",
        "_root_link_ang_vel_b",
        "_root_link_vel_w",
        "_root_com_pose_w",
    ):
        getattr(data, name).timestamp = -1.0


def _case_buffers(env):
    if hasattr(env, "npa_goals"):
        return
    env.npa_goals = torch.zeros((env.num_envs, 2), device=env.device)
    env.npa_previous_distance = torch.zeros(env.num_envs, device=env.device)
    env.npa_episode_index = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.device
    )


def _selected_cases(env, env_ids):
    indices = env_ids.tolist()
    if env.npa_override_cases is not None:
        return [env.npa_override_cases[index] for index in indices]
    cases = env.cfg.npa_cases
    episodes = env.npa_episode_index[env_ids].tolist()
    env.npa_episode_index[env_ids] += 1
    return [
        cases[(index + episode * 104729) % len(cases)]
        for index, episode in zip(indices, episodes, strict=True)
    ]


def static_ranges(env):
    """Observe only rays intersecting the shared static warehouse geometry.

    Args:
        env: Native reference environment.
    Returns:
        Finite normalized ranges for the trainable navigation policy.
    Raises:
        RuntimeError: Native ray observations cannot be read.
    """
    sensor = env.scene["navigation_ranges"]
    distances = torch.linalg.vector_norm(
        sensor.data.ray_hits_w.torch - sensor.data.pos_w.torch[:, None, :], dim=-1
    )
    if torch.isnan(distances).any():
        raise RuntimeError("native static ray sensor produced NaN distances")
    return torch.nan_to_num(distances, posinf=12.0).clamp(0.0, 12.0) / 12.0


def progress(env):
    """Reward measured distance reduction, retaining penalties for obstacle detours.

    Args:
        env: Native reference environment.
    Returns:
        Per-robot progress rates from consecutive simulated root poses.
    Raises:
        None.
    """
    distance = _distance(env)
    reward = (env.npa_previous_distance - distance) / env.step_dt
    env.npa_previous_distance[:] = distance
    return reward


def collision(env):
    """Penalize actual obstacle or unexpected peer contact above sensor noise.

    Args:
        env: Native reference environment.
    Returns:
        Per-robot physical collision indicators.
    Raises:
        None.
    """
    return ((env.npa_contacts.obstacle > 0) | (env.npa_contacts.peer > 0)).float()


def terminate(env):
    """End training episodes on physical collision or successful goal arrival.

    Args:
        env: Native reference environment.
    Returns:
        Per-robot termination flags; evaluation and probes never auto-reset.
    Raises:
        None.
    """
    if env.npa_probe or not env.cfg.npa_training:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    return (failure(env) > 0) | (_distance(env) < 0.5)


def physical_failure(env):
    """Penalize measured falls, missing floor and unsupported reference poses.

    Args:
        env: Native reference environment.
    Returns:
        Per-robot physical failure indicators.
    Raises:
        RuntimeError: Physical measurements are invalid.
    """
    from npa.workflows.navigation.reference_validity import physical_state

    return physical_state(env)["physical_failure"].float()


def failure(env):
    """Combine physical and contact failures for native training termination.

    Args:
        env: Native reference environment.
    Returns:
        Per-robot failure indicators without conflating reported failure kinds.
    Raises:
        RuntimeError: Physical measurements are invalid.
    """
    return torch.maximum(collision(env), physical_failure(env))


def timeout(env):
    """Apply the native training episode horizon without resetting held-out probes.

    Args:
        env: Native reference environment.
    Returns:
        Per-robot training timeout flags.
    Raises:
        None.
    """
    enabled = env.cfg.npa_training and not env.npa_probe
    return (env.episode_length_buf >= env.max_episode_length) & enabled


def _distance(env):
    return torch.linalg.vector_norm(
        env.npa_goals - env.scene["robot"].data.root_pos_w.torch[:, :2], dim=1
    )
