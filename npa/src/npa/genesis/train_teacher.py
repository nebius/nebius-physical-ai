"""PPO teacher training using rsl-rl and Genesis simulation.

Trains a privileged-state teacher policy for expert distillation.
The teacher sees object poses, contact forces, and goal positions —
observations that are only available in simulation.

After training, the teacher generates camera-only demonstrations that
are used to train a vision-based student policy via imitation learning.

Requires: genesis-world, rsl-rl-lib==2.2.4, torch (CUDA)
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


class TrainingError(Exception):
    pass


@dataclass
class PPOConfig:
    """PPO hyperparameters matching rsl-rl defaults for manipulation."""

    # Policy (passed to ActorCritic via train_cfg["policy"])
    actor_hidden_dims: list[int] | None = None
    critic_hidden_dims: list[int] | None = None
    activation: str = "elu"
    init_noise_std: float = 1.0

    # Algorithm (passed to PPO via train_cfg["algorithm"])
    learning_rate: float = 1e-3
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    clip_param: float = 0.2
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    use_clipped_value_loss: bool = True
    schedule: str = "adaptive"
    desired_kl: float = 0.01

    # Runner (top-level keys in train_cfg)
    num_steps_per_env: int = 24
    save_interval: int = 50
    empirical_normalization: bool = False

    def __post_init__(self) -> None:
        if self.actor_hidden_dims is None:
            self.actor_hidden_dims = [256, 256, 128]
        if self.critic_hidden_dims is None:
            self.critic_hidden_dims = [256, 256, 128]

    def to_train_cfg(self) -> dict[str, Any]:
        """Build the train_cfg dict that OnPolicyRunner expects.

        IMPORTANT: OnPolicyRunner pops "class_name" from the policy and
        algorithm sub-dicts, so the caller must deepcopy this if reuse is
        needed.
        """
        return {
            "policy": {
                "class_name": "ActorCritic",
                "actor_hidden_dims": list(self.actor_hidden_dims),
                "critic_hidden_dims": list(self.critic_hidden_dims),
                "activation": self.activation,
                "init_noise_std": self.init_noise_std,
            },
            "algorithm": {
                "class_name": "PPO",
                "num_learning_epochs": self.num_learning_epochs,
                "num_mini_batches": self.num_mini_batches,
                "clip_param": self.clip_param,
                "gamma": self.gamma,
                "lam": self.lam,
                "value_loss_coef": self.value_loss_coef,
                "entropy_coef": self.entropy_coef,
                "learning_rate": self.learning_rate,
                "max_grad_norm": self.max_grad_norm,
                "use_clipped_value_loss": self.use_clipped_value_loss,
                "schedule": self.schedule,
                "desired_kl": self.desired_kl,
            },
            "num_steps_per_env": self.num_steps_per_env,
            "save_interval": self.save_interval,
            "empirical_normalization": self.empirical_normalization,
        }


class GenesisEnvWrapper:
    """Wraps FrankaPickPlaceEnv for the rsl-rl VecEnv contract.

    rsl-rl OnPolicyRunner (v2.2.4) expects:

    Attributes:
        num_envs, num_actions, max_episode_length, episode_length_buf, device

    Methods:
        get_observations() -> (obs, extras)
            extras["observations"] is a dict; "critic" key holds privileged obs.
        step(actions)      -> (obs, rewards, dones, infos)   **4 values**
            infos["observations"]["critic"] = privileged obs
            infos["time_outs"] = timeout mask for value bootstrapping
    """

    def __init__(self, env) -> None:
        self._env = env
        self.num_envs: int = env.n_envs
        self.num_actions: int = env.act_dim
        self.max_episode_length: int = env.cfg.max_episode_steps
        self.device = torch.device(env.device)
        self.episode_length_buf = torch.zeros(
            env.n_envs, device=self.device, dtype=torch.long,
        )
        self._obs: torch.Tensor | None = None

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        if self._obs is None:
            obs_dict = self._env.get_privileged_obs()
            self._obs = obs_dict["flat"]
        return self._obs, {"observations": {"critic": self._obs}}

    def reset(self) -> tuple[torch.Tensor, dict]:
        obs_dict = self._env.reset()
        self._obs = obs_dict["flat"]
        self.episode_length_buf.zero_()
        return self._obs, {"observations": {"critic": self._obs}}

    def step(
        self, actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        obs_dict, rewards, dones, info = self._env.step(actions)
        self._obs = obs_dict["flat"]
        self.episode_length_buf += 1

        # Distinguish timeouts from true terminations so PPO can bootstrap
        time_outs = (
            self.episode_length_buf >= self.max_episode_length
        ) & ~info.get("success", torch.zeros_like(dones, dtype=torch.bool))

        infos: dict[str, Any] = {
            "observations": {"critic": self._obs},
            "time_outs": time_outs.float(),
        }

        # Reset episode counter for done envs
        self.episode_length_buf[dones] = 0

        return self._obs, rewards, dones, infos


def train_teacher(
    n_envs: int = 4096,
    max_iterations: int = 500,
    output_dir: Path = Path("./checkpoints/teacher/"),
    device: str = "cuda",
    log_dir: Path = Path("./logs/teacher/"),
    seed: int = 42,
    ppo_cfg: PPOConfig | None = None,
    action_space: str = "cartesian",
    env_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train an RL teacher policy using PPO.

    Args:
        n_envs: Number of parallel Genesis environments.
        max_iterations: PPO training iterations.
        output_dir: Where to save checkpoints.
        device: Torch device.
        log_dir: Tensorboard log directory.
        seed: Random seed.
        ppo_cfg: PPO hyperparameters (uses defaults if None).
        action_space: "cartesian" (4D: delta xyz + gripper) or
            "joint" (8D: delta joint positions + gripper).
        env_overrides: Optional dict of EnvConfig field overrides
            (e.g. {"approach_scale": 0, "domain_randomize": True}).

    Returns:
        Result dict with checkpoint_path, final metrics.

    Raises:
        TrainingError: On training failure.
    """
    if ppo_cfg is None:
        ppo_cfg = PPOConfig()

    output_dir = output_dir.resolve()
    log_dir = log_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)

    try:
        from rsl_rl.runners import OnPolicyRunner
    except ImportError as exc:
        raise TrainingError(
            "rsl-rl not installed. Install with: pip install rsl-rl-lib==2.2.4"
        ) from exc

    # Create Genesis environment
    logger.info("Creating Genesis environment (n_envs=%d)...", n_envs)
    try:
        from npa.genesis.env_pick_place import EnvConfig, FrankaPickPlaceEnv

        cfg_kwargs: dict[str, Any] = {
            "n_envs": n_envs,
            "enable_cameras": False,  # No cameras for teacher training (faster)
            "domain_randomize": False,
            "action_space": action_space,
        }
        if env_overrides:
            cfg_kwargs.update(env_overrides)
        env_cfg = EnvConfig(**cfg_kwargs)
        env = FrankaPickPlaceEnv(env_cfg)
        wrapped_env = GenesisEnvWrapper(env)
    except Exception as exc:
        raise TrainingError(f"Failed to create Genesis environment: {exc}") from exc

    # Build train_cfg dict (OnPolicyRunner pops class_name keys, so deepcopy)
    train_cfg = ppo_cfg.to_train_cfg()
    # Save a clean copy of the config alongside the checkpoint for later loading
    config_for_save = copy.deepcopy(train_cfg)

    # Create runner — it instantiates ActorCritic and PPO from the config dicts
    logger.info("Creating OnPolicyRunner...")
    try:
        runner = OnPolicyRunner(
            env=wrapped_env,
            train_cfg=train_cfg,
            log_dir=str(log_dir),
            device=device,
        )
    except Exception as exc:
        raise TrainingError(f"Failed to create OnPolicyRunner: {exc}") from exc

    # Train
    logger.info("Starting PPO training (max_iterations=%d)...", max_iterations)
    try:
        runner.learn(num_learning_iterations=max_iterations)
    except Exception as exc:
        raise TrainingError(f"PPO training failed: {exc}") from exc

    # Save final checkpoint
    checkpoint_path = output_dir / "model.pt"
    runner.save(str(checkpoint_path))
    logger.info("Teacher checkpoint saved to %s", checkpoint_path)

    # Save the architecture config so generate_demos can reconstruct the network
    arch_config = {
        "policy": config_for_save["policy"],
        "num_obs": env.obs_dim,
        "num_actions": env.act_dim,
        "action_space": action_space,
    }
    arch_path = output_dir / "arch_config.json"
    with arch_path.open("w") as f:
        json.dump(arch_config, f, indent=2)
    logger.info("Architecture config saved to %s", arch_path)

    return {
        "status": "success",
        "checkpoint_path": str(checkpoint_path),
        "arch_config_path": str(arch_path),
        "n_envs": n_envs,
        "max_iterations": max_iterations,
        "log_dir": str(log_dir),
    }
