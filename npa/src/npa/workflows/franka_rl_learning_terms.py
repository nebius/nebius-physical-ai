"""Compute observable manipulation rewards and advance training-only goal distributions."""

from __future__ import annotations

from copy import deepcopy

import torch
from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_apply_inverse,
    quat_inv,
    quat_mul,
)

from npa.workflows.franka_rl_curriculum import TrainingCurriculum, training_ranges
from npa.workflows.franka_rl_validity import task_domain_exits


def _geometry(env):
    scene = env.scene
    robot, obj = scene["robot"].data, scene["object"].data
    command = env.command_manager.get_command("object_pose")
    goal, _ = combine_frame_transforms(
        robot.root_pos_w.torch, robot.root_quat_w.torch, command[:, :3]
    )
    position = obj.root_pos_w.torch
    distance = torch.linalg.vector_norm(position - goal, dim=-1)
    speed = torch.linalg.vector_norm(obj.root_lin_vel_w.torch, dim=-1)
    height = position[:, 2] - scene.env_origins[:, 2]
    reach = torch.linalg.vector_norm(
        position - scene["ee_frame"].data.target_pos_w.torch[:, 0], dim=-1
    )
    return distance, speed, height, reach


def manipulation_state(env) -> torch.Tensor:
    """Expose relative geometry, object motion, and servo memory to the learned policy.

    Args:
        env: Live native environment.
    Returns:
        Observable task state, including the generic controller's target error.
    Raises:
        RuntimeError: Required native articulation or frame data is unavailable.
    """
    scene = env.scene
    robot, obj = scene["robot"].data, scene["object"].data
    frame = scene["ee_frame"].data
    inverse = quat_inv(robot.root_quat_w.torch)
    position = quat_apply_inverse(
        robot.root_quat_w.torch, obj.root_pos_w.torch - robot.root_pos_w.torch
    )
    tool_to_object = quat_apply_inverse(
        robot.root_quat_w.torch, obj.root_pos_w.torch - frame.target_pos_w.torch[:, 0]
    )
    goal_to_object = env.command_manager.get_command("object_pose")[:, :3] - position
    rotation = quat_mul(inverse, obj.root_quat_w.torch)
    tool_rotation = quat_mul(inverse, frame.target_quat_w.torch[:, 0])
    velocity = quat_apply_inverse(robot.root_quat_w.torch, obj.root_lin_vel_w.torch)
    angular = quat_apply_inverse(robot.root_quat_w.torch, obj.root_ang_vel_w.torch)
    action = env.action_manager.get_term("arm_action")
    tracking = action.target - robot.joint_pos.torch[:, action._joint_ids]
    reward = getattr(env, "npa_learning_reward", None)
    hold = (
        torch.zeros((env.num_envs, 1), device=env.device)
        if reward is None
        else reward.hold_fraction[:, None]
    )
    return torch.cat(
        (
            tool_to_object,
            goal_to_object,
            rotation,
            tool_rotation,
            velocity,
            angular,
            tracking,
            hold,
        ),
        dim=-1,
    )


class StableManipulationReward(ManagerTermBase):
    """Reward reaching, lifting, and continuously holding the unchanged geometric goal.

    Args:
        cfg: Native reward configuration containing the sealed recipe.
        env: Live environment.
    Returns:
        Stateful reward and training-success bookkeeping.
    Raises:
        ValueError: The training curriculum settings are invalid.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.recipe = cfg.params["recipe"]
        self.settings = self.recipe["learning"]
        self.training = cfg.params["training"]
        self.streak = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.succeeded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.hold_fraction = torch.zeros(env.num_envs, device=env.device)
        self.episode_fraction = torch.full((env.num_envs,), -1.0, device=env.device)
        self.previous_action = torch.zeros_like(env.action_manager.action)
        curriculum = self.settings["curriculum"]
        self.curriculum = TrainingCurriculum(
            **{
                key: curriculum[key]
                for key in (
                    "minimum_episodes",
                    "success_threshold",
                    "initial_fraction",
                    "increment",
                )
            }
        )
        self.reset_ranges = deepcopy(
            env.cfg.events.reset_object_position.params["pose_range"]
        )
        self.goal_ranges = {
            name: tuple(getattr(env.cfg.commands.object_pose.ranges, name))
            for name in ("pos_x", "pos_y", "pos_z")
        }
        self.completed_episodes = 0
        self.completed_successes = 0
        env.npa_learning_reward = self

    def __call__(self, env, recipe: dict, training: bool) -> torch.Tensor:
        """Compute a dense learning signal without relaxing the stable-success predicate.

        Args:
            env: Live native environment.
            recipe: Sealed experiment recipe.
            training: Whether this environment contributes curriculum evidence.
        Returns:
            Per-environment reward before native time-step scaling.
        Raises:
            ValueError: Simulator geometry is nonfinite.
        """
        distance, speed, height, reach = _geometry(env)
        if not all(
            torch.isfinite(value).all() for value in (distance, speed, height, reach)
        ):
            raise ValueError("Nonfinite manipulation geometry")
        valid = (
            (distance < recipe["success_distance_m"])
            & (speed < recipe["maximum_object_speed_m_s"])
            & (height > recipe["minimum_object_height_m"])
            & ~env.reset_buf
        )
        self.streak = torch.where(valid, self.streak + 1, 0)
        self.succeeded |= self.streak >= recipe["stable_steps"]
        if hasattr(env.cfg, "npa_simulation_validity"):
            self.succeeded &= ~task_domain_exits(env)
        self.hold_fraction = (self.streak / recipe["stable_steps"]).clamp(max=1)
        settings = self.settings["reward"]
        reaching = 1 - torch.tanh(reach / settings["reach_width_m"])
        lifted = (height > recipe["minimum_object_height_m"]).float()
        goal = 1 - torch.tanh(distance / settings["goal_width_m"])
        if settings.get("goal_requires_lift", True):
            goal *= lifted
        holding = (
            lifted
            * goal
            * torch.exp(-(speed / recipe["maximum_object_speed_m_s"]).square())
        )
        holding *= 1 + self.hold_fraction
        rate = (
            torch.tanh(env.action_manager.action - self.previous_action)
            .square()
            .mean(dim=-1)
        )
        self.previous_action.copy_(env.action_manager.action)
        return (
            settings["reach_weight"] * reaching
            + settings["lift_weight"] * lifted
            + settings["goal_weight"] * goal
            + settings["hold_weight"] * holding
            + settings["action_rate_weight"] * rate
        )

    def reset(self, env_ids) -> None:
        """Clear per-episode state after the curriculum has consumed completed outcomes.

        Args:
            env_ids: Environments reset by the native runtime.
        Returns:
            None; aggregate training metrics are added to the native logger.
        Raises:
            None.
        """
        self.streak[env_ids] = 0
        self.succeeded[env_ids] = False
        self.hold_fraction[env_ids] = 0
        self.previous_action[env_ids] = 0
        self.episode_fraction[env_ids] = self.curriculum.fraction
        self._env.extras.setdefault("log", {}).update(
            {
                "Learning/difficulty": self.curriculum.fraction,
                "Learning/completed_episodes": self.completed_episodes,
                "Learning/strict_successes": self.completed_successes,
            }
        )


def adapt_training(env, env_ids) -> float:
    """Advance training reset/goal ranges from completed episodes at the current difficulty.

    Args:
        env: Training environment with a stable-manipulation reward.
        env_ids: Environments about to reset.
    Returns:
        Current difficulty, also recorded by the native curriculum manager.
    Raises:
        ValueError: Invoked outside a training environment.
    """
    reward = env.npa_learning_reward
    if not reward.training:
        raise ValueError("Evaluation must not adapt the training distribution")
    current = reward.curriculum.fraction
    completed = reward.episode_fraction[env_ids] >= 0
    eligible = completed & torch.isclose(
        reward.episode_fraction[env_ids], torch.tensor(current, device=env.device)
    )
    success = reward.succeeded[env_ids]
    reward.completed_episodes += int(completed.sum())
    reward.completed_successes += int((completed & success).sum())
    reward.curriculum.observe(int((eligible & success).sum()), int(eligible.sum()))
    fraction = reward.curriculum.fraction
    ranges, goals = training_ranges(
        fraction,
        reward.reset_ranges,
        reward.goal_ranges,
        tuple(reward.settings["curriculum"]["easy_goal_height_m"]),
    )
    reset = env.event_manager.get_term_cfg("reset_object_position")
    reset.params["pose_range"] = ranges
    env.event_manager.set_term_cfg("reset_object_position", reset)
    command = env.command_manager.get_term("object_pose")
    for name, value in goals.items():
        setattr(command.cfg.ranges, name, value)
    return fraction
