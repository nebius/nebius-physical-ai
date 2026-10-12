"""Ramp binary gripper commands within the native finger joint limits."""

from __future__ import annotations

import torch
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction


class RampedGripperAction(BinaryJointPositionAction):
    """Retain binary intent while avoiding an instantaneous finger target jump.

    Args:
        cfg: Native binary gripper action configuration.
        env: Native environment with the sealed gripper servo settings.
    Returns:
        Action term with persistent position targets initialized from measured joints.
    Raises:
        ValueError: Native joint limits do not define a valid servo domain.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        data = self._asset.data
        self._limits = data.joint_pos_limits.torch[:, self._joint_ids].clone()
        velocity = data.joint_vel_limits.torch[:, self._joint_ids]
        effort = data.joint_effort_limits.torch[:, self._joint_ids]
        expected_effort = env.cfg.npa_gripper_servo["effort_limit_n"]
        fraction = env.cfg.npa_gripper_servo["native_velocity_fraction"]
        if (
            not torch.isfinite(self._limits).all()
            or not torch.isfinite(velocity).all()
            or (velocity <= 0).any()
            or (self._limits[..., 0] > self._limits[..., 1]).any()
            or not 0 < fraction < 1
            or not torch.allclose(effort, torch.full_like(effort, expected_effort))
        ):
            raise ValueError(
                "Native gripper limits differ from the sealed servo settings"
            )
        self._step_limit = velocity * fraction * env.step_dt
        self.target = data.joint_pos.torch[:, self._joint_ids].clone()

    def process_actions(self, actions: torch.Tensor) -> None:
        """Advance the servo toward the requested open or close target.

        Args:
            actions: Recorded binary open/close intentions.
        Returns:
            None; the native action term applies the ramped joint targets.
        Raises:
            ValueError: A command is nonfinite.
        """
        if not torch.isfinite(actions).all():
            raise ValueError("Nonfinite binary gripper command")
        super().process_actions(actions)
        desired = self.processed_actions.clamp(
            self._limits[..., 0], self._limits[..., 1]
        )
        self.target += (desired - self.target).clamp(
            -self._step_limit, self._step_limit
        )
        self._processed_actions.copy_(self.target)

    def reset(self, env_ids=None) -> None:
        """Initialize servo memory from measured joints after a scene reset.

        Args:
            env_ids: Reset environments, or all environments when omitted.
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
