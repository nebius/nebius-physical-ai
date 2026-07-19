"""Evaluate a student vision policy in Genesis simulation.

Loads a trained student policy (ACT, diffusion, etc.) from a LeRobot
checkpoint and evaluates it in the Genesis pick-and-place environment
using ONLY camera observations and joint state — never privileged info.

Uses held-out domain randomization seeds (different from demo generation)
to test generalization.

Outputs per-episode metrics and aggregate statistics including success
rate, steps to completion, failure mode breakdown, and the distillation
gap (teacher vs student success rate).

LeRobot policy loading (v0.5.x):
    - Use ACTPolicy.from_pretrained(path_to_pretrained_model_dir)
    - Checkpoint dir must contain config.json + model.safetensors
    - Call policy.reset() at the start of each episode
    - policy.select_action(obs_dict) returns (action_dim,) or (1, action_dim)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class EvalError(Exception):
    pass


def _resolve_pretrained_dir(checkpoint_path: Path) -> Path:
    """Resolve a checkpoint path to the pretrained_model/ directory.

    Users may pass:
        - .../student/                                 (parent output dir)
        - .../student/checkpoints/last/pretrained_model  (exact dir)
        - .../student/checkpoints/last/                  (step dir)
        - .../student/checkpoints/005000/pretrained_model
    """
    # If it already has config.json, it's the pretrained_model dir
    if (checkpoint_path / "config.json").exists():
        return checkpoint_path

    # Try standard LeRobot checkpoint layout
    candidates = [
        checkpoint_path / "checkpoints" / "last" / "pretrained_model",
        checkpoint_path / "pretrained_model",
    ]
    for c in candidates:
        if (c / "config.json").exists():
            return c

    # Try resolving symlink
    last = checkpoint_path / "checkpoints" / "last"
    if last.is_symlink() or last.is_dir():
        resolved = last / "pretrained_model"
        if (resolved / "config.json").exists():
            return resolved

    # Fallback: find the latest numbered checkpoint directory.
    # LeRobot 0.5.x saves to checkpoints/010000/ without a "last" symlink.
    ckpt_dir = checkpoint_path / "checkpoints"
    if ckpt_dir.is_dir():
        numbered = sorted(
            [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.isdigit()],
            key=lambda d: int(d.name),
            reverse=True,
        )
        for d in numbered:
            resolved = d / "pretrained_model"
            if (resolved / "config.json").exists():
                return resolved

    raise EvalError(
        f"Cannot find pretrained_model directory under {checkpoint_path}. "
        f"Expected config.json + model.safetensors in a pretrained_model/ dir."
    )


_POLICY_CLASS_MAP = {
    "act": "lerobot.policies.act.modeling_act.ACTPolicy",
    "diffusion": "lerobot.policies.diffusion.modeling_diffusion.DiffusionPolicy",
    "vla": "lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy",
    "smolvla": "lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy",
}


def _load_student_policy(checkpoint_path: Path) -> tuple[Any, Any, Any]:
    """Load a LeRobot policy plus its preprocessor and postprocessor.

    Returns (policy, preprocessor, postprocessor) so the eval loop can
    run observations through the same normalization/device pipeline that
    training used.
    """
    pretrained_dir = _resolve_pretrained_dir(checkpoint_path)
    logger.info("Resolved pretrained model dir: %s", pretrained_dir)

    try:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
    except ImportError as exc:
        raise EvalError(
            "LeRobot is required for student evaluation. "
            "Install with: pip install lerobot"
        ) from exc

    # Read config.json to determine policy type
    config_path = pretrained_dir / "config.json"
    with config_path.open() as f:
        config = json.load(f)

    policy_type = config.get("type", config.get("_target_", "act")).rsplit(".", 1)[-1].lower()

    # Resolve concrete policy class — do not fall back to the abstract base
    class_path = None
    for key, path in _POLICY_CLASS_MAP.items():
        if key in policy_type:
            class_path = path
            break

    if class_path is None:
        raise EvalError(
            f"Unsupported policy type '{policy_type}' in {config_path}. "
            f"Supported: {', '.join(_POLICY_CLASS_MAP.keys())}"
        )

    module_path, class_name = class_path.rsplit(".", 1)
    try:
        import importlib
        mod = importlib.import_module(module_path)
        policy_cls = getattr(mod, class_name)
    except (ImportError, AttributeError) as exc:
        raise EvalError(
            f"Cannot import policy class {class_path}: {exc}"
        ) from exc

    policy = policy_cls.from_pretrained(str(pretrained_dir))
    policy.eval()

    # Load the saved preprocessor/postprocessor pipeline so observations
    # get the same normalization and actions get unnormalized.
    try:
        pretrained_cfg = PreTrainedConfig.from_pretrained(str(pretrained_dir))
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=pretrained_cfg,
            pretrained_path=str(pretrained_dir),
        )
    except Exception as exc:
        logger.warning(
            "Could not load pre/post processors from %s: %s. "
            "Running without normalization — results may be degraded.",
            pretrained_dir, exc,
        )
        preprocessor = None
        postprocessor = None

    return policy, preprocessor, postprocessor


def eval_student(
    checkpoint_path: Path,
    n_envs: int = 1024,
    n_episodes: int = 1024,
    output_dir: Path = Path("./eval/"),
    domain_randomize: bool = True,
    seed: int = 42,
    teacher_success_rate: float | None = None,
    action_space: str = "cartesian",
) -> dict[str, Any]:
    """Evaluate a student vision policy in Genesis.

    Args:
        checkpoint_path: Path to the trained student policy checkpoint.
            Can be the output_dir from training, or the exact
            pretrained_model/ directory.
        n_envs: Number of parallel Genesis environments.
        n_episodes: Total evaluation episodes to run.
        output_dir: Where to save evaluation metrics.
        domain_randomize: Enable domain randomization with held-out seeds.
        seed: Random seed (should differ from training seeds).
        teacher_success_rate: If known, included in metrics for gap calculation.
        action_space: "cartesian" (4D: delta xyz + gripper) or
            "joint" (8D: delta joint positions + gripper). Must match
            the action space used for demo generation / student training.

    Returns:
        Evaluation metrics dict.

    Raises:
        EvalError: On failure.
    """
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)

    if not checkpoint_path.exists():
        raise EvalError(f"Checkpoint not found: {checkpoint_path}")

    # Load student policy with its saved pre/post processors
    logger.info("Loading student policy from %s", checkpoint_path)
    try:
        student_policy, preprocessor, postprocessor = _load_student_policy(checkpoint_path)
    except Exception as exc:
        raise EvalError(f"Failed to load student policy: {exc}") from exc

    # Create Genesis environment with cameras enabled
    logger.info("Creating Genesis environment (n_envs=%d, cameras=True)", n_envs)
    try:
        from npa.genesis.env_pick_place import EnvConfig, FrankaPickPlaceEnv

        env_cfg = EnvConfig(
            n_envs=n_envs,
            enable_cameras=True,
            domain_randomize=domain_randomize,
            action_space=action_space,
        )
        env = FrankaPickPlaceEnv(env_cfg)
    except Exception as exc:
        raise EvalError(f"Failed to create Genesis environment: {exc}") from exc

    # Run evaluation
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    logger.info("Starting evaluation: run_id=%s, n_episodes=%d", run_id, n_episodes)

    episode_results: list[dict[str, Any]] = []
    total_collected = 0
    batch_num = 0

    while total_collected < n_episodes:
        batch_num += 1
        remaining = n_episodes - total_collected
        batch_envs = min(n_envs, remaining)

        env.reset()
        # Reset policy action queues at the start of each batch of episodes
        student_policy.reset()

        env_done = [False] * batch_envs
        env_success = [False] * batch_envs
        env_steps = [0] * batch_envs
        env_failure_mode = ["timeout"] * batch_envs

        for step in range(env_cfg.max_episode_steps):
            # Student only gets camera observations — never privileged state
            cam_obs = env.get_camera_obs()

            # Prepare observation dict matching LeRobot feature keys.
            # Genesis cameras return (B, H, W, 3) uint8.  LeRobot's
            # inference path expects images as (B, C, H, W) float32 in
            # [0, 1], so we must convert before the preprocessor runs
            # (the preprocessor only applies learned normalization stats,
            # not the raw format conversion).
            # joint_pos/gripper_state are CUDA tensors; images may be
            # numpy arrays (CPU).  Move everything to the same device.
            device = cam_obs["joint_pos"].device
            observation = {
                "observation.images.workspace": _prepare_image(cam_obs["workspace"]).to(device),
                "observation.images.wrist": _prepare_image(cam_obs["wrist"]).to(device),
                "observation.state": torch.cat(
                    [cam_obs["joint_pos"], cam_obs["gripper_state"]], dim=-1
                ),
            }

            # Run through the saved preprocessor (normalization, device
            # placement) so the policy sees the same distribution it was
            # trained on.
            if preprocessor is not None:
                observation = preprocessor(observation)

            # Student inference
            with torch.no_grad():
                actions = student_policy.select_action(observation)

            # Run through the saved postprocessor (action unnormalization)
            # so the raw action space matches what Genesis expects.
            if postprocessor is not None:
                actions = postprocessor(actions)

            # Ensure actions have the right shape for the env
            if actions.dim() == 1:
                actions = actions.unsqueeze(0).expand(batch_envs, -1)
            elif actions.shape[0] == 1 and batch_envs > 1:
                actions = actions.expand(batch_envs, -1)

            # Ensure actions are on the same device as the simulation
            actions = actions.to(device)

            # Step simulation
            _, _, dones, info = env.step(actions)
            success_flags = info["success"].cpu().numpy()
            dones_np = dones.cpu().numpy()

            for i in range(batch_envs):
                if env_done[i]:
                    continue
                env_steps[i] = step + 1
                if dones_np[i]:
                    env_done[i] = True
                    if success_flags[i]:
                        env_success[i] = True
                        env_failure_mode[i] = "none"
                    else:
                        env_failure_mode[i] = _classify_failure(env, i, step)

            if all(env_done[:batch_envs]):
                break

        # Record results
        for i in range(batch_envs):
            if total_collected >= n_episodes:
                break
            episode_results.append({
                "episode_index": total_collected,
                "success": env_success[i],
                "steps": env_steps[i],
                "failure_mode": env_failure_mode[i],
            })
            total_collected += 1

        logger.info(
            "Batch %d: %d/%d succeeded, total: %d/%d",
            batch_num,
            sum(env_success[:batch_envs]),
            batch_envs,
            total_collected,
            n_episodes,
        )

    # Compute aggregate metrics
    successes = sum(1 for r in episode_results if r["success"])
    success_rate = successes / len(episode_results) if episode_results else 0.0

    success_steps = [r["steps"] for r in episode_results if r["success"]]
    mean_steps = float(np.mean(success_steps)) if success_steps else 0.0

    failure_modes: dict[str, int] = {}
    for r in episode_results:
        if not r["success"]:
            mode = r["failure_mode"]
            failure_modes[mode] = failure_modes.get(mode, 0) + 1

    metrics: dict[str, Any] = {
        "run_id": run_id,
        "n_episodes": len(episode_results),
        "success_rate": round(success_rate, 4),
        "mean_steps_to_success": round(mean_steps, 1),
        "failure_modes": failure_modes,
    }

    if teacher_success_rate is not None:
        metrics["teacher_success_rate"] = teacher_success_rate
        metrics["distillation_gap"] = round(teacher_success_rate - success_rate, 4)

    # Write metrics
    metrics_path = output_dir / f"eval_{run_id}.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    metrics["metrics_path"] = str(metrics_path)

    # Write per-episode results
    episodes_path = output_dir / f"episodes_{run_id}.json"
    with episodes_path.open("w") as f:
        json.dump(episode_results, f, indent=2)

    logger.info(
        "Evaluation complete: success_rate=%.2f%% (%d/%d)",
        success_rate * 100, successes, len(episode_results),
    )

    return metrics


def _prepare_image(img: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Convert a Genesis camera image to LeRobot's expected format.

    Genesis may return numpy arrays or torch tensors depending on the
    rendering backend.  Handles both.
    LeRobot expects (B, C, H, W) float32 in [0, 1].
    """
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    if img.dtype == torch.uint8:
        img = img.float() / 255.0
    if img.ndim == 4 and img.shape[-1] in (1, 3):
        # (B, H, W, C) → (B, C, H, W)
        img = img.permute(0, 3, 1, 2)
    return img


def _classify_failure(env, env_idx: int, step: int) -> str:
    """Classify why an episode failed.

    Categories:
        - timeout: reached max steps without success
        - drop: cube was grasped then dropped
        - collision: robot collided with table or exceeded joint limits
    """
    try:
        cube_pos = env._cube.get_pos()[env_idx].cpu().numpy()
        cube_init_z = env.cfg.cube_init_pos[2]

        # If cube is below table, it was dropped
        if cube_pos[2] < cube_init_z - 0.02:
            return "drop"

        # If cube moved far from initial position but not to target, likely drop
        cube_init = np.array(env.cfg.cube_init_pos)
        target = np.array(env.cfg.target_pos)
        dist_from_init = np.linalg.norm(cube_pos[:2] - cube_init[:2])
        dist_from_target = np.linalg.norm(cube_pos[:2] - target[:2])
        if dist_from_init > 0.2 and dist_from_target > env.cfg.target_threshold:
            return "drop"

        # Check joint limit violations
        joint_pos = env._robot.get_dofs_position()[env_idx].cpu().numpy()
        if np.any(np.abs(joint_pos[:7]) > 2.8):
            return "collision"
    except Exception:
        logging.getLogger(__name__).debug("suppressed exception", exc_info=True)

    return "timeout"
