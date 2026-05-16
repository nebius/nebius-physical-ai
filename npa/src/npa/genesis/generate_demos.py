"""Generate camera-only demonstrations using a trained teacher policy.

Loads the trained teacher checkpoint, runs rollouts in Genesis with
cameras and domain randomization enabled, records per-timestep data
that a real robot would have access to (camera RGB + joint positions),
and filters for successful episodes only.

The teacher uses get_privileged_obs() internally for action selection,
but the recorded dataset contains only camera observations and joint state.
This is the key step in expert distillation: privileged knowledge goes in,
vision-only demonstrations come out.

Output format (consumed by npa.adapter.sim_to_lerobot):
    demos/
    ├── episode_0000/
    │   ├── obs_workspace.npy  (T, H, W, 3) uint8
    │   ├── obs_wrist.npy      (T, H, W, 3) uint8
    │   ├── state.npy          (T, 9+1) float32 — joint positions + gripper
    │   └── actions.npy        (T, 8) float32
    └── episode_0001/ ...
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class DemoGenerationError(Exception):
    pass


def _read_checkpoint_action_space(checkpoint_path: Path) -> str | None:
    """Read the action_space from a checkpoint's arch_config.json.

    Returns the saved action_space string, or None if the file doesn't
    exist or doesn't contain the field (older checkpoints).
    """
    arch_path = checkpoint_path.parent / "arch_config.json"
    if not arch_path.exists():
        return None
    try:
        with arch_path.open() as f:
            arch = json.load(f)
        return arch.get("action_space")
    except (json.JSONDecodeError, OSError):
        return None


def _load_teacher_policy(
    checkpoint_path: Path,
    env,
) -> Any:
    """Load teacher ActorCritic from checkpoint + arch_config.json.

    The arch_config.json is saved alongside the checkpoint by train_teacher
    and records the exact network hidden dims and action/obs dimensions
    used during training so we reconstruct an identically-shaped network
    before loading weights.

    When arch_config.json contains ``num_actions`` or ``num_obs``, those
    values take precedence over the caller-supplied ``env`` — this prevents
    action-dimension mismatches when an existing joint-space checkpoint
    (num_actions=8) is loaded into an env configured for cartesian
    (act_dim=4).  The env is only used as a fallback for older checkpoints
    that lack these fields.
    """
    from rsl_rl.modules import ActorCritic

    # Find arch config: same directory as the checkpoint
    arch_path = checkpoint_path.parent / "arch_config.json"
    if arch_path.exists():
        with arch_path.open() as f:
            arch = json.load(f)
        policy_cfg = arch.get("policy", {})
        actor_hidden = policy_cfg.get("actor_hidden_dims", [256, 256, 128])
        critic_hidden = policy_cfg.get("critic_hidden_dims", [256, 256, 128])
        activation = policy_cfg.get("activation", "elu")
        init_noise_std = policy_cfg.get("init_noise_std", 1.0)

        # Use saved dimensions when available — they are authoritative.
        num_obs = arch.get("num_obs", env.obs_dim)
        num_actions = arch.get("num_actions", env.act_dim)

        if num_actions != env.act_dim:
            saved_space = arch.get("action_space", "unknown")
            logger.warning(
                "Checkpoint was trained with action_space=%s (num_actions=%d) "
                "but the current env has act_dim=%d (action_space=%s). "
                "Using the checkpoint's num_actions=%d for policy reconstruction.",
                saved_space, num_actions, env.act_dim,
                env.cfg.action_space, num_actions,
            )
    else:
        logger.warning(
            "arch_config.json not found at %s — using default network shape. "
            "This may cause a size mismatch if training used non-default dims.",
            arch_path,
        )
        actor_hidden = [256, 256, 128]
        critic_hidden = [256, 256, 128]
        activation = "elu"
        init_noise_std = 1.0
        num_obs = env.obs_dim
        num_actions = env.act_dim

    actor_critic = ActorCritic(
        num_actor_obs=num_obs,
        num_critic_obs=num_obs,
        num_actions=num_actions,
        actor_hidden_dims=actor_hidden,
        critic_hidden_dims=critic_hidden,
        activation=activation,
        init_noise_std=init_noise_std,
    ).to("cuda")

    checkpoint = torch.load(str(checkpoint_path), map_location="cuda", weights_only=False)
    actor_critic.load_state_dict(checkpoint["model_state_dict"])
    actor_critic.eval()
    return actor_critic


def eval_teacher(
    checkpoint_path: Path,
    n_envs: int = 1024,
    seed: int = 7777,
    action_space: str = "cartesian",
) -> float:
    """Evaluate the teacher under held-out conditions (no cameras).

    Runs a single batch of teacher rollouts with privileged state and
    returns the task success rate.  Uses a different seed from demo
    generation to ensure held-out evaluation.

    If the checkpoint's arch_config.json records a different action_space
    than the one passed by the caller, the checkpoint's value wins — this
    prevents dimension mismatches when loading older joint-space checkpoints.

    Returns:
        Teacher success rate in [0.0, 1.0].
    """
    from npa.genesis.env_pick_place import EnvConfig, FrankaPickPlaceEnv

    # Auto-detect action space from checkpoint metadata when available.
    saved_space = _read_checkpoint_action_space(checkpoint_path)
    if saved_space is not None and saved_space != action_space:
        logger.info(
            "Checkpoint was trained with action_space=%s (caller passed %s). "
            "Using the checkpoint's value.",
            saved_space, action_space,
        )
        action_space = saved_space

    # Seed all RNGs before env creation so domain randomization
    # (which uses torch RNG) produces a deterministic, held-out
    # distribution that differs from demo generation.
    torch.manual_seed(seed)
    np.random.seed(seed)

    env_cfg = EnvConfig(
        n_envs=n_envs,
        enable_cameras=False,
        domain_randomize=True,
        action_space=action_space,
    )
    env = FrankaPickPlaceEnv(env_cfg)
    env.reset()

    actor_critic = _load_teacher_policy(checkpoint_path, env)

    env_done = [False] * n_envs
    env_success = [False] * n_envs

    for step in range(env_cfg.max_episode_steps):
        priv_obs = env.get_privileged_obs()
        with torch.no_grad():
            actions = actor_critic.act_inference(priv_obs["flat"])

        _, _, dones, info = env.step(actions)
        success_flags = info["success"].cpu().numpy()

        for i in range(n_envs):
            if not env_done[i] and dones[i]:
                env_done[i] = True
                env_success[i] = bool(success_flags[i])

        if all(env_done):
            break

    successes = sum(env_success)
    rate = successes / n_envs
    logger.info(
        "Teacher eval (seed=%d): %d/%d succeeded (%.1f%%)",
        seed, successes, n_envs, rate * 100,
    )
    return rate


def generate_demos(
    checkpoint_path: Path,
    n_envs: int = 4096,
    n_episodes: int = 0,
    output_dir: Path = Path("./data/demos/"),
    domain_randomize: bool = True,
    fps: int = 20,
    seed: int = 42,
    allow_failure_demos: bool = False,
    action_space: str = "cartesian",
) -> dict[str, Any]:
    """Generate camera-only demonstrations from a trained teacher.

    Args:
        checkpoint_path: Path to teacher checkpoint (model.pt).
        n_envs: Number of parallel Genesis environments.
        n_episodes: Total episodes to collect. 0 = run exactly one batch of
            rollouts and keep only the successes (no retries).
        allow_failure_demos: If True, save all episodes even when no envs
            achieved task success (for development/debugging).  If False
            (default), raise DemoGenerationError when 0 successes.
        output_dir: Where to save demo numpy arrays.
        domain_randomize: Enable domain randomization.
        fps: Camera recording frame rate (controls step-to-frame ratio).
        seed: Random seed.
        action_space: "cartesian" (4D: delta xyz + gripper) or
            "joint" (8D: delta joint positions + gripper).

    Returns:
        Result dict with episode counts, success rate, output path.

    Raises:
        DemoGenerationError: On failure.
    """
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)

    if not checkpoint_path.exists():
        raise DemoGenerationError(f"Checkpoint not found: {checkpoint_path}")

    # Auto-detect action space from checkpoint metadata when available.
    saved_space = _read_checkpoint_action_space(checkpoint_path)
    if saved_space is not None and saved_space != action_space:
        logger.info(
            "Checkpoint was trained with action_space=%s (caller passed %s). "
            "Using the checkpoint's value.",
            saved_space, action_space,
        )
        action_space = saved_space

    try:
        from rsl_rl.modules import ActorCritic  # noqa: F401 — validate import
    except ImportError as exc:
        raise DemoGenerationError(
            "rsl-rl not installed. Install with: pip install rsl-rl-lib==2.2.4"
        ) from exc

    # Create Genesis environment with cameras enabled
    logger.info("Creating Genesis environment (n_envs=%d, cameras=True)...", n_envs)
    try:
        from npa.genesis.env_pick_place import EnvConfig, FrankaPickPlaceEnv

        env_cfg = EnvConfig(
            n_envs=n_envs,
            enable_cameras=True,
            domain_randomize=domain_randomize,
            camera_fps=fps,
            action_space=action_space,
        )
        env = FrankaPickPlaceEnv(env_cfg)
    except Exception as exc:
        raise DemoGenerationError(
            f"Failed to create Genesis environment: {exc}"
        ) from exc

    # Load teacher with matching architecture
    logger.info("Loading teacher checkpoint from %s", checkpoint_path)
    try:
        actor_critic = _load_teacher_policy(checkpoint_path, env)
    except Exception as exc:
        raise DemoGenerationError(f"Failed to load teacher model: {exc}") from exc

    # n_episodes=0 means "one batch, keep whatever succeeds"
    single_batch = n_episodes <= 0
    target_episodes = n_envs if single_batch else n_episodes
    collected_episodes = 0
    total_successes = 0
    total_attempted = 0
    batch_num = 0

    # Rendering step ratio: physics dt vs camera fps
    steps_per_frame = max(1, int(1.0 / (env_cfg.dt * fps)))

    logger.info(
        "Generating demos: target=%s, n_envs=%d, domain_randomize=%s",
        "one batch" if single_batch else f"{target_episodes} episodes",
        n_envs,
        domain_randomize,
    )

    while collected_episodes < target_episodes:
        batch_num += 1
        logger.info("Batch %d: resetting environments...", batch_num)

        env.reset()

        # Per-env recording buffers
        workspace_frames: list[list[np.ndarray]] = [[] for _ in range(n_envs)]
        wrist_frames: list[list[np.ndarray]] = [[] for _ in range(n_envs)]
        state_frames: list[list[np.ndarray]] = [[] for _ in range(n_envs)]
        action_frames: list[list[np.ndarray]] = [[] for _ in range(n_envs)]
        env_done: list[bool] = [False] * n_envs
        env_success: list[bool] = [False] * n_envs

        for step in range(env_cfg.max_episode_steps):
            # Teacher selects actions using PRIVILEGED observations
            priv_obs = env.get_privileged_obs()
            with torch.no_grad():
                actions = actor_critic.act_inference(priv_obs["flat"])

            # Record camera-only observations (what a real robot would see)
            if step % steps_per_frame == 0:
                cam_obs = env.get_camera_obs()
                ws_np = cam_obs["workspace"].cpu().numpy() if hasattr(cam_obs["workspace"], "cpu") else np.asarray(cam_obs["workspace"])
                wr_np = cam_obs["wrist"].cpu().numpy() if hasattr(cam_obs["wrist"], "cpu") else np.asarray(cam_obs["wrist"])
                jp_np = cam_obs["joint_pos"].cpu().numpy() if hasattr(cam_obs["joint_pos"], "cpu") else np.asarray(cam_obs["joint_pos"])
                gs_np = cam_obs["gripper_state"].cpu().numpy() if hasattr(cam_obs["gripper_state"], "cpu") else np.asarray(cam_obs["gripper_state"])
                state_np = np.concatenate([jp_np, gs_np], axis=-1)
                act_np = actions.cpu().numpy()

                for i in range(n_envs):
                    if not env_done[i]:
                        workspace_frames[i].append(ws_np[i])
                        wrist_frames[i].append(wr_np[i])
                        state_frames[i].append(state_np[i])
                        action_frames[i].append(act_np[i])

            _, _, dones, info = env.step(actions)
            success_flags = info["success"].cpu().numpy()

            for i in range(n_envs):
                if not env_done[i] and dones[i]:
                    env_done[i] = True
                    env_success[i] = bool(success_flags[i])

            if all(env_done):
                break

        # Count actual teacher successes separately from saved episodes.
        batch_success = sum(env_success)
        total_successes += batch_success
        total_attempted += n_envs

        # Save successful episodes.  If none succeeded and
        # allow_failure_demos is set, fall back to saving all episodes
        # (for development).  Otherwise fail — training on non-expert
        # rollouts degrades distillation quality.
        has_successes = any(env_success)
        if not has_successes and not allow_failure_demos:
            raise DemoGenerationError(
                f"Teacher achieved 0 task successes in batch {batch_num} "
                f"({n_envs} envs).  The student would train on failure "
                f"rollouts, which degrades distillation quality.  Either "
                f"increase --max-iterations for teacher training, or pass "
                f"--allow-failure-demos to proceed anyway."
            )
        for i in range(n_envs):
            if collected_episodes >= target_episodes:
                break
            if has_successes and not env_success[i]:
                continue
            if len(workspace_frames[i]) == 0:
                continue

            ep_dir = output_dir / f"episode_{collected_episodes:04d}"
            ep_dir.mkdir(parents=True, exist_ok=True)

            np.save(ep_dir / "obs_workspace.npy", np.stack(workspace_frames[i]))
            np.save(ep_dir / "obs_wrist.npy", np.stack(wrist_frames[i]))
            np.save(ep_dir / "state.npy", np.stack(state_frames[i]).astype(np.float32))
            np.save(ep_dir / "actions.npy", np.stack(action_frames[i]).astype(np.float32))

            collected_episodes += 1

        logger.info(
            "Batch %d: %d/%d envs succeeded, total collected: %d/%d",
            batch_num, batch_success, n_envs, collected_episodes, target_episodes,
        )

        # In single-batch mode, stop after one batch regardless of success count
        if single_batch:
            break

        # Multi-batch: if success rate is 0, teacher is broken
        if batch_success == 0:
            raise DemoGenerationError(
                f"Teacher achieved 0 successes in batch {batch_num} ({n_envs} envs). "
                f"Check that the teacher checkpoint is valid."
            )

    # success_rate reflects actual teacher task completion, not how many
    # episodes were saved (which may include non-successes as fallback).
    teacher_success_rate = total_successes / total_attempted if total_attempted > 0 else 0.0
    includes_failures = collected_episodes > total_successes

    if includes_failures:
        logger.warning(
            "Demo dataset includes non-successful episodes (teacher_success_rate=%.2f%%, "
            "%d/%d succeeded).  The student will be trained on failure rollouts — "
            "this degrades distillation quality.  Increase teacher training iterations "
            "or tune the reward to improve task completion.",
            teacher_success_rate * 100, total_successes, total_attempted,
        )

    result = {
        "status": "success",
        "output_dir": str(output_dir),
        "total_episodes": collected_episodes,
        "total_successes": total_successes,
        "total_attempted": total_attempted,
        "teacher_success_rate": round(teacher_success_rate, 4),
        "includes_failures": includes_failures,
        "domain_randomize": domain_randomize,
        "fps": fps,
    }
    logger.info("Demo generation complete: %s", result)
    return result
