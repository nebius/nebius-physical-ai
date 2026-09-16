"""Bound learned joint targets by physical limits and target slew."""

from __future__ import annotations

import torch
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction


class BoundedJointPositionAction(JointPositionAction):
    """Apply generic servo constraints without choosing a manipulation trajectory.

    Args:
        cfg: Native joint action configuration with the original action mapping.
        env: Native environment containing the sealed learning settings.
    Returns:
        A joint position action with persistent, observed servo targets.
    Raises:
        ValueError: Physical joint limits or actions are invalid.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        settings = env.cfg.npa_learning
        data = self._asset.data
        self._limits = data.soft_joint_pos_limits.torch[:, self._joint_ids].clone()
        velocities = data.joint_vel_limits.torch[:, self._joint_ids]
        if (not torch.isfinite(self._limits).all() or not torch.isfinite(velocities).all()
                or (velocities <= 0).any() or (self._limits[..., 0] > self._limits[..., 1]).any()):
            raise ValueError("The physical joint limits do not define a finite servo domain")
        self._step_limit = velocities.clamp(max=settings["target_velocity_rad_s"]) * env.step_dt
        self.target = data.joint_pos.torch[:, self._joint_ids].clone()
        self.latest_command_target = self.target.clone()

    def process_actions(self, actions: torch.Tensor) -> None:
        """Constrain physical targets while retaining the baseline's reachable action range.

        Args:
            actions: Learned normalized joint targets; never scripted task commands.
        Returns:
            None.
        Raises:
            ValueError: A policy command is nonfinite.
        """
        if not torch.isfinite(actions).all():
            raise ValueError("Nonfinite learned joint action")
        super().process_actions(actions)
        desired = self.processed_actions.clamp(self._limits[..., 0], self._limits[..., 1])
        self.target += (desired - self.target).clamp(-self._step_limit, self._step_limit)
        self.target.clamp_(self._limits[..., 0], self._limits[..., 1])
        self._processed_actions.copy_(self.target)
        self.latest_command_target.copy_(self.target)

    def reset(self, env_ids=None) -> None:
        """Reset servo memory from the actual joint state after a native scene reset.

        Args:
            env_ids: Environments reset by the native runtime, or all environments.
        Returns:
            None.
        Raises:
            None.
        """
        super().reset(env_ids)
        indices = slice(None) if env_ids is None else env_ids
        positions = self._asset.data.joint_pos.torch[:, self._joint_ids]
        self.target[indices] = positions[indices]
        self._processed_actions[indices] = self.target[indices]
