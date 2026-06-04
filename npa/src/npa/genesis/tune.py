"""Auto-tune loop: diagnose → adjust → retrain → diagnose.

Takes a teacher checkpoint, diagnoses why it's failing, applies the
suggested config change, retrains with a short PPO run, and re-diagnoses.
Repeats until task success > 0% or max rounds exhausted.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TuneError(Exception):
    pass


def tune_teacher(
    checkpoint_path: Path,
    max_rounds: int = 5,
    retrain_iterations: int = 100,
    n_envs: int = 4096,
    diagnose_n_envs: int = 1024,
    seed: int = 42,
    output_dir: Path = Path("./checkpoints/tune/"),
    log_dir: Path = Path("./logs/tune/"),
    device: str = "cuda",
    action_space: str = "cartesian",
    env_overrides: dict[str, Any] | None = None,
    min_success_rate: float = 0.0,
) -> dict[str, Any]:
    """Run the diagnose → adjust → retrain loop.

    Args:
        checkpoint_path: Initial teacher checkpoint (model.pt).
        max_rounds: Maximum tune iterations before giving up.
        retrain_iterations: PPO iterations per retrain round (short).
        n_envs: Environments for retraining.
        diagnose_n_envs: Environments for diagnosis rollouts.
        seed: Base random seed (incremented per round).
        output_dir: Where to save per-round checkpoints.
        log_dir: Tensorboard log directory.
        device: Torch device.
        action_space: "cartesian" or "joint". Passed to diagnose
            (for tailored suggestions) and retrain (for env creation).
        env_overrides: Initial overrides from the caller (e.g. CLI
            --env-override flags). These are merged with per-round
            suggestions. May include diagnosis threshold keys
            (approach_threshold, etc.) which diagnose handles separately.
        min_success_rate: Stop when success rate exceeds this value.
            Default 0.0 means stop as soon as any episode succeeds.

    Returns:
        Result dict with per-round history and final outcome.

    Raises:
        TuneError: On unrecoverable failure.
    """
    from npa.genesis.diagnose import DiagnoseError, diagnose_teacher, save_diagnosis
    from npa.genesis.train_teacher import TrainingError

    checkpoint_path = Path(checkpoint_path).resolve()
    output_dir = Path(output_dir).resolve()
    log_dir = Path(log_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise TuneError(f"Checkpoint not found: {checkpoint_path}")

    current_checkpoint = checkpoint_path
    # Seed with caller-provided overrides (CLI --env-override flags).
    # Per-round suggestions merge on top of these.
    _initial_overrides: dict[str, Any] = dict(env_overrides) if env_overrides else {}
    env_overrides: dict[str, Any] = dict(_initial_overrides)
    rounds: list[dict[str, Any]] = []

    for round_num in range(1, max_rounds + 1):
        round_seed = seed + round_num
        logger.info("=== Tune round %d/%d ===", round_num, max_rounds)

        # ── Step 1: Diagnose ───────────────────────────────────────────
        logger.info("Diagnosing checkpoint: %s", current_checkpoint)
        try:
            diagnosis = diagnose_teacher(
                checkpoint_path=current_checkpoint,
                n_envs=diagnose_n_envs,
                seed=round_seed,
                env_overrides=env_overrides if env_overrides else None,
                action_space=action_space,
            )
        except DiagnoseError as exc:
            raise TuneError(
                f"Diagnosis failed in round {round_num}: {exc}"
            ) from exc

        # Save diagnosis artifact into a dedicated subdirectory so it
        # is unambiguously tied to the INPUT checkpoint, not the
        # retrained checkpoint that will be saved later.
        round_dir = output_dir / f"round_{round_num:02d}"
        diag_dir = round_dir / "diagnosis"
        diag_path = diag_dir / "diagnosis.json"
        save_diagnosis(diagnosis, diag_path)

        round_record: dict[str, Any] = {
            "round": round_num,
            "input_checkpoint": str(current_checkpoint),
            "success_rate": diagnosis["success_rate"],
            "bottleneck": diagnosis["bottleneck"],
            "phase_counts": diagnosis["phase_counts"],
        }

        # ── Step 2: Check exit condition ───────────────────────────────
        if diagnosis["success_rate"] > min_success_rate:
            logger.info(
                "Success rate %.1f%% > %.1f%% — stopping tune loop.",
                diagnosis["success_rate"] * 100, min_success_rate * 100,
            )
            round_record["action"] = "stop_success"
            # Write accumulated env overrides so the early-exit round
            # has a complete artifact set for the user to inspect.
            _save_env_overrides(diag_dir / "env_overrides.json", env_overrides)
            rounds.append(round_record)
            break

        if diagnosis["bottleneck"] == "none":
            logger.info("No failure bottleneck identified — stopping.")
            round_record["action"] = "stop_no_bottleneck"
            _save_env_overrides(diag_dir / "env_overrides.json", env_overrides)
            rounds.append(round_record)
            break

        # ── Step 3: Apply suggested config changes ─────────────────────
        suggestion = diagnosis.get("suggestion", {})
        config_changes = suggestion.get("config_changes", {})
        if not config_changes:
            logger.warning(
                "No config changes suggested for bottleneck '%s' — stopping.",
                diagnosis["bottleneck"],
            )
            round_record["action"] = "stop_no_suggestion"
            _save_env_overrides(diag_dir / "env_overrides.json", env_overrides)
            rounds.append(round_record)
            break

        # Merge changes into running overrides (accumulate across rounds)
        for key, value in config_changes.items():
            # Convert lists back to tuples for EnvConfig fields that expect tuples
            if isinstance(value, list):
                value = tuple(value)
            # If diagnose suggests switching action space, update the
            # action_space variable so subsequent rounds use the new space.
            if key == "action_space":
                action_space = value
                continue  # action_space is a ctor arg, not an env override
            env_overrides[key] = value

        logger.info(
            "Applying fix '%s': %s",
            suggestion.get("fix", "unknown"),
            config_changes,
        )
        round_record["fix_applied"] = suggestion.get("fix", "unknown")
        round_record["config_changes"] = config_changes
        round_record["action"] = "retrain"
        rounds.append(round_record)

        # ── Step 4: Retrain with adjusted config ───────────────────────
        retrain_dir = round_dir / "retrained"
        round_log = log_dir / f"round_{round_num:02d}"

        # Filter out diagnosis-threshold keys before passing to retrain —
        # they control episode classification, not the simulation.
        from npa.genesis.diagnose import _THRESHOLD_KEYS

        retrain_overrides = {
            k: v for k, v in env_overrides.items()
            if k not in _THRESHOLD_KEYS
        }
        logger.info(
            "Retraining: %d iterations, n_envs=%d, env_overrides=%s",
            retrain_iterations, n_envs, retrain_overrides,
        )
        try:
            train_result = _retrain_with_overrides(
                n_envs=n_envs,
                max_iterations=retrain_iterations,
                output_dir=retrain_dir,
                log_dir=round_log,
                device=device,
                seed=round_seed,
                env_overrides=retrain_overrides,
                action_space=action_space,
            )
        except TrainingError as exc:
            raise TuneError(
                f"Retraining failed in round {round_num}: {exc}"
            ) from exc

        current_checkpoint = Path(train_result["checkpoint_path"])
        logger.info("Round %d checkpoint: %s", round_num, current_checkpoint)

    else:
        # Loop exhausted without success
        logger.warning(
            "Max rounds (%d) reached without achieving success > 0%%.",
            max_rounds,
        )

    # ── Final diagnosis on the last checkpoint ─────────────────────────
    final_diagnosis = None
    if rounds and rounds[-1].get("action") != "stop_success":
        try:
            final_diagnosis = diagnose_teacher(
                checkpoint_path=current_checkpoint,
                n_envs=diagnose_n_envs,
                seed=seed + max_rounds + 1,
                env_overrides=env_overrides if env_overrides else None,
                action_space=action_space,
            )
            final_path = output_dir / "final_diagnosis.json"
            save_diagnosis(final_diagnosis, final_path)
        except DiagnoseError:
            logger.warning("Final diagnosis failed — skipping.")

    final_success_rate = (
        final_diagnosis["success_rate"]
        if final_diagnosis
        else rounds[-1]["success_rate"] if rounds else 0.0
    )

    result: dict[str, Any] = {
        "status": "success" if final_success_rate > min_success_rate else "no_improvement",
        "rounds_completed": len(rounds),
        "final_checkpoint": str(current_checkpoint),
        "final_success_rate": final_success_rate,
        "env_overrides_applied": _serialize_overrides(env_overrides),
        "rounds": rounds,
    }

    return result


def _save_env_overrides(path: Path, overrides: dict[str, Any]) -> None:
    """Write accumulated env overrides to a JSON file."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(_serialize_overrides(overrides), f, indent=2)


def _retrain_with_overrides(
    n_envs: int,
    max_iterations: int,
    output_dir: Path,
    log_dir: Path,
    device: str,
    seed: int,
    env_overrides: dict[str, Any],
    action_space: str = "cartesian",
) -> dict[str, Any]:
    """Train teacher with modified EnvConfig parameters.

    This is a thin wrapper around the Genesis env + rsl-rl training loop
    that applies env_overrides to EnvConfig before creating the environment.
    """
    import copy as _copy

    import torch

    from npa.genesis.train_teacher import (
        GenesisEnvWrapper,
        PPOConfig,
        TrainingError,
    )

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

    # Create environment WITH overrides
    try:
        from npa.genesis.env_pick_place import EnvConfig, FrankaPickPlaceEnv

        cfg_kwargs: dict[str, Any] = {
            "n_envs": n_envs,
            "enable_cameras": False,
            "domain_randomize": False,
            "action_space": action_space,
        }
        cfg_kwargs.update(env_overrides)
        env_cfg = EnvConfig(**cfg_kwargs)
        env = FrankaPickPlaceEnv(env_cfg)
        wrapped_env = GenesisEnvWrapper(env)
    except Exception as exc:
        raise TrainingError(f"Failed to create Genesis environment: {exc}") from exc

    ppo_cfg = PPOConfig()
    train_cfg = ppo_cfg.to_train_cfg()
    config_for_save = _copy.deepcopy(train_cfg)

    try:
        runner = OnPolicyRunner(
            env=wrapped_env,
            train_cfg=train_cfg,
            log_dir=str(log_dir),
            device=device,
        )
    except Exception as exc:
        raise TrainingError(f"Failed to create OnPolicyRunner: {exc}") from exc

    try:
        runner.learn(num_learning_iterations=max_iterations)
    except Exception as exc:
        raise TrainingError(f"PPO training failed: {exc}") from exc

    # Save checkpoint
    checkpoint_path = output_dir / "model.pt"
    runner.save(str(checkpoint_path))

    # Save arch config for downstream loading
    import json

    arch_config = {
        "policy": config_for_save["policy"],
        "num_obs": env.obs_dim,
        "num_actions": env.act_dim,
        "action_space": action_space,
    }
    arch_path = output_dir / "arch_config.json"
    with arch_path.open("w") as f:
        json.dump(arch_config, f, indent=2)

    # Save the env overrides so the next round knows what was changed
    overrides_path = output_dir / "env_overrides.json"
    with overrides_path.open("w") as f:
        json.dump(_serialize_overrides(env_overrides), f, indent=2)

    return {
        "status": "success",
        "checkpoint_path": str(checkpoint_path),
        "arch_config_path": str(arch_path),
        "n_envs": n_envs,
        "max_iterations": max_iterations,
        "log_dir": str(log_dir),
    }


def _serialize_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    """Make overrides JSON-serializable (tuples → lists)."""
    out: dict[str, Any] = {}
    for k, v in overrides.items():
        out[k] = list(v) if isinstance(v, tuple) else v
    return out
