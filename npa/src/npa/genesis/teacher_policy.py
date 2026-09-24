"""Rebuild deterministic Genesis teacher inference from restricted checkpoint data."""

from dataclasses import dataclass
from typing import Any


@dataclass
class TeacherPolicy:
    """Wrap a deterministic actor with explicit observation and action dimensions.

    Args:
        model: Reconstructed inference module.
        num_obs: Required flat observation width.
        num_actions: Required action width.
    Returns:
        None.
    Raises:
        None.
    """

    model: Any
    num_obs: int
    num_actions: int

    def act_inference(self, observations):
        """Evaluate the actor mean using the checkpoint's normalization.

        Args:
            observations: Batched raw privileged observations.
        Returns:
            Deterministic actions with the declared action width.
        Raises:
            ValueError: Input or output dimensions differ from the checkpoint.
        """
        if observations.ndim != 2 or observations.shape[-1] != self.num_obs:
            raise ValueError("teacher observation shape differs from checkpoint")
        actions = self.model(observations)
        if actions.shape != (observations.shape[0], self.num_actions):
            raise ValueError("teacher action shape differs from checkpoint")
        return actions


def build_teacher_policy(checkpoint, architecture, env):
    """Reconstruct legacy or RSL-RL 5 actors without importing checkpoint classes.

    Args:
        checkpoint: Tensor mapping loaded with weights_only=True.
        architecture: Saved architecture metadata, or an empty mapping.
        env: Environment supplying fallback dimensions and the target device.
    Returns:
        An evaluation-only teacher preserving both checkpoint formats.
    Raises:
        ValueError: Architecture, normalization, or tensor shapes are incompatible.
    """
    import torch
    from npa.workflows.sim2real import policy_export as export

    state = export.load_state_dict_from_checkpoint(checkpoint)
    num_obs, num_actions, _ = export.infer_mlp_dims(export.actor_weight_shapes(state))
    if num_obs != architecture.get(
        "num_obs", env.obs_dim
    ) or num_actions != architecture.get("num_actions", env.act_dim):
        raise ValueError("teacher checkpoint dimensions disagree with architecture")
    config = architecture.get("actor", architecture.get("policy", {}))
    actor = export._build_actor_module(torch, state, config.get("activation", "elu"))
    normalization, _ = export._detect_normalization(torch, checkpoint, num_obs)
    model = export._make_forward_module(torch, actor, normalization).to(env.device)
    return TeacherPolicy(model, num_obs, num_actions)
