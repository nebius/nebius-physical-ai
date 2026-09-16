"""Seal an optional learned manipulation recipe and configure its native training terms."""

from __future__ import annotations

from copy import deepcopy

LEARNING_RECIPES = ("joint-baseline", "adaptive")
_ADAPTIVE = {
    "schema": "npa.manipulation-learning.v1", "name": "adaptive",
    "control": "bounded_joint_target", "target_velocity_rad_s": 2.0,
    "observations": "object-tool-goal-pose-velocity-and-servo-target",
    "reward": {"reach_weight": 2.0, "reach_width_m": 0.15,
               "lift_weight": 5.0, "goal_weight": 16.0, "goal_width_m": 0.1,
               "hold_weight": 25.0, "action_rate_weight": -0.01},
    "curriculum": {"initial_fraction": 0.25, "increment": 0.15,
                   "minimum_episodes": 8192, "success_threshold": 0.7,
                   "easy_goal_height_m": [0.12, 0.18]},
    "ppo": {"gamma": 0.99, "observation_normalization": True, "initial_std": 0.5},
}


def learning_profile(name: str) -> dict | None:
    """Return the immutable learning definition or the historical baseline.

    Args:
        name: Supported learning recipe.
    Returns:
        Independent settings, or None for the historical baseline.
    Raises:
        ValueError: The recipe is unsupported.
    """
    if name not in LEARNING_RECIPES:
        raise ValueError(f"Unsupported learning recipe: {name}")
    return deepcopy(_ADAPTIVE) if name == "adaptive" else None


def recipe_learning(recipe: dict) -> dict | None:
    """Reject modified learning definitions while accepting historical recipes.

    Args:
        recipe: Sealed experiment recipe.
    Returns:
        Validated learning settings or None for the baseline.
    Raises:
        ValueError: A declared learning definition differs from its versioned profile.
    """
    declared = recipe.get("learning")
    if declared is None:
        return None
    expected = learning_profile(declared["name"])
    if expected is None or declared != expected:
        raise ValueError("Sealed learning recipe differs from its supported definition")
    return expected


def configure_learning(config, recipe: dict, *, training: bool) -> None:
    """Install learned rewards, state observations, and training-only difficulty adaptation.

    Args:
        config: Native environment configuration.
        recipe: Sealed experiment recipe.
        training: Enable curriculum only for policy training.
    Returns:
        None.
    Raises:
        ValueError: The learning recipe is invalid.
        ImportError: The native Isaac runtime is unavailable.
    """
    settings = recipe_learning(recipe)
    if settings is None:
        return
    from isaaclab.managers import CurriculumTermCfg, ObservationTermCfg, RewardTermCfg
    from npa.workflows.franka_rl_learning_terms import StableManipulationReward, adapt_training, manipulation_state
    from npa.workflows.franka_rl_servo import BoundedJointPositionAction

    config.actions.arm_action.class_type = BoundedJointPositionAction
    config.observations.policy.manipulation = ObservationTermCfg(func=manipulation_state)
    config.rewards = {"manipulation": RewardTermCfg(func=StableManipulationReward, weight=1.0,
                                                   params={"recipe": recipe, "training": training})}
    config.curriculum = {"training_difficulty": CurriculumTermCfg(func=adapt_training)} if training else {}
    config.npa_learning = settings


def configure_learner(config, recipe: dict) -> None:
    """Apply sealed PPO normalization and temporal credit settings.

    Args:
        config: Resolved native RSL-RL agent configuration.
        recipe: Sealed experiment recipe.
    Returns:
        None.
    Raises:
        ValueError: The learning definition is invalid.
    """
    settings = recipe_learning(recipe)
    if settings is None:
        return
    config.actor.obs_normalization = settings["ppo"]["observation_normalization"]
    config.critic.obs_normalization = settings["ppo"]["observation_normalization"]
    config.actor.distribution_cfg.init_std = settings["ppo"]["initial_std"]
    config.algorithm.gamma = settings["ppo"]["gamma"]


def normalization_evidence(runner, recipe: dict) -> dict:
    """Fingerprint checkpoint normalizers so evaluation cannot quietly adapt them.

    Args:
        runner: Native runner after checkpoint loading.
        recipe: Sealed learning recipe.
    Returns:
        Normalizer sample count and byte hashes, or an empty baseline record.
    Raises:
        ValueError: Adaptive normalization buffers are missing or invalid.
    """
    if recipe_learning(recipe) is None:
        return {}
    import hashlib
    import torch

    state = runner.alg.get_policy().obs_normalizer.state_dict()
    if set(state) != {"_mean", "_var", "_std", "count"}:
        raise ValueError("Unexpected native observation normalizer state")
    count = state["count"]
    if count.dtype != torch.long or count.ndim != 0 or count.item() < 0:
        raise ValueError("Invalid native normalization sample count")
    if not all(torch.isfinite(value).all() for value in state.values()) or (state["_std"] < 0).any():
        raise ValueError("Nonfinite or invalid native normalization buffers")
    return {"sample_count": int(count), "buffers": {
        name: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": hashlib.sha256(
            value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
        for name, value in state.items()}}
