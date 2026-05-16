"""Student policy training via LeRobot.

Thin wrapper that configures and launches LeRobot training for a vision-based
student policy (ACT, diffusion, etc.) on a dataset produced by the
SimToLeRobot adapter. The student only sees camera observations and joint
state — never privileged simulator state.

LeRobot API notes (v0.5.x):
    - Training is step-based (--steps), not epoch-based.
    - Local datasets use --dataset.repo_id=<name> --dataset.root=<path>.
      There is no local:// prefix.
    - lerobot-train rejects an existing output_dir unless --resume=true.
    - Flag names are flattened dataclass paths via draccus.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Approximate steps per epoch assuming ~200 transitions/episode and
# ~50 episodes in a typical distillation dataset.  The CLI accepts
# --epochs for ergonomics; we convert to steps internally.
_DEFAULT_TRANSITIONS_PER_EPOCH = 10_000


class StudentTrainingError(Exception):
    pass


def _estimate_steps(dataset_path: Path, num_epochs: int, batch_size: int) -> int:
    """Convert epoch count to optimizer steps based on dataset size.

    In LeRobot, ``--steps`` counts optimizer updates, not frames.
    One epoch = total_frames / batch_size updates.
    """
    info_path = dataset_path / "meta" / "info.json"
    if info_path.exists():
        with info_path.open() as f:
            info = json.load(f)
        total_frames = info.get("total_frames", _DEFAULT_TRANSITIONS_PER_EPOCH)
    else:
        total_frames = _DEFAULT_TRANSITIONS_PER_EPOCH
    steps_per_epoch = max(1, total_frames // batch_size)
    return steps_per_epoch * num_epochs


def build_train_command(
    dataset_path: str,
    output_dir: str,
    *,
    policy_type: str = "act",
    steps: int = 100_000,
    batch_size: int = 64,
    device: str = "cuda",
    num_workers: int = 4,
    save_freq: int | None = None,
    extra_args: dict[str, str] | None = None,
) -> list[str]:
    """Build the lerobot-train command line.

    Args:
        dataset_path: Resolved path to the local LeRobotDataset v3 directory.
        output_dir: Where to save training checkpoints.
        policy_type: Policy architecture (act, diffusion).
        steps: Total training steps.
        batch_size: Batch size for training.
        device: Torch device (cuda, cpu).
        num_workers: Dataloader workers.
        save_freq: Checkpoint save frequency in steps (None = at end).
        extra_args: Additional key=value args to pass to lerobot-train.

    Returns:
        Command as a list of strings suitable for subprocess.
    """
    # LeRobot local datasets: when --dataset.root is set, LeRobot uses
    # that path directly (repo_id is NOT appended).  So root must be
    # the full path to the dataset directory containing meta/info.json.
    ds_path = Path(dataset_path)
    repo_id = ds_path.name
    root = str(ds_path)

    if save_freq is None:
        save_freq = steps  # Save once at the end

    cmd = [
        "lerobot-train",
        f"--policy.type={policy_type}",
        "--policy.push_to_hub=false",
        f"--policy.device={device}",
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={root}",
        f"--output_dir={output_dir}",
        f"--steps={steps}",
        f"--batch_size={batch_size}",
        f"--num_workers={num_workers}",
        f"--save_freq={save_freq}",
        # Disable eval during distillation (eval is done in Genesis separately)
        "--eval_freq=1000000",
        "--wandb.enable=false",
    ]

    if extra_args:
        for key, value in extra_args.items():
            cmd.append(f"--{key}={value}")

    return cmd


def train_student(
    dataset_path: Path,
    output_dir: Path,
    *,
    policy_type: str = "act",
    num_epochs: int = 100,
    batch_size: int = 64,
    device: str = "cuda",
    num_workers: int = 4,
    save_freq: int | None = None,
    extra_args: dict[str, str] | None = None,
    stream: bool = True,
) -> dict[str, Any]:
    """Run student policy training via LeRobot.

    Args:
        dataset_path: Path to the local LeRobotDataset v3 directory.
        output_dir: Where to save training checkpoints. Must not already
            exist (LeRobot rejects existing dirs unless --resume=true).
        policy_type: Policy architecture (act, diffusion).
        num_epochs: Approximate epoch count — converted to steps based on
            dataset size (LeRobot is step-based, not epoch-based).
        batch_size: Batch size for training.
        device: Torch device.
        num_workers: Dataloader workers.
        save_freq: Checkpoint save frequency in steps.
        extra_args: Additional key=value args.
        stream: If True, forward stdout to the terminal.

    Returns:
        Result dict with status, checkpoint_path, etc.

    Raises:
        StudentTrainingError: If training fails.
    """
    dataset_path = dataset_path.resolve()
    output_dir = output_dir.resolve()

    if not dataset_path.exists():
        raise StudentTrainingError(f"Dataset not found: {dataset_path}")
    if not (dataset_path / "meta" / "info.json").exists():
        raise StudentTrainingError(
            f"Not a valid LeRobotDataset v3 directory: {dataset_path} "
            f"(missing meta/info.json)"
        )

    # Don't pre-create output_dir — LeRobot rejects existing directories
    if output_dir.exists():
        raise StudentTrainingError(
            f"Output directory already exists: {output_dir}. "
            f"LeRobot requires a fresh directory. Remove it or choose a different path."
        )

    steps = _estimate_steps(dataset_path, num_epochs, batch_size)
    logger.info("Converted %d epochs to %d optimizer steps (dataset: %s, batch_size: %d)", num_epochs, steps, dataset_path, batch_size)

    cmd = build_train_command(
        str(dataset_path),
        str(output_dir),
        policy_type=policy_type,
        steps=steps,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        save_freq=save_freq,
        extra_args=extra_args,
    )

    logger.info("Running: %s", " ".join(cmd))

    # Force offline mode so LeRobot loads the local dataset without
    # trying to resolve repo_id on HuggingFace Hub.
    import os
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}

    if stream:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
        proc.wait()
        exit_code = proc.returncode
    else:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=86400, env=env)
        exit_code = result.returncode

    checkpoint_path = output_dir / "checkpoints" / "last" / "pretrained_model"

    outcome: dict[str, Any] = {
        "status": "success" if exit_code == 0 else "failed",
        "exit_code": exit_code,
        "policy_type": policy_type,
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "checkpoint_path": str(checkpoint_path) if exit_code == 0 else None,
    }

    if exit_code != 0:
        raise StudentTrainingError(
            f"lerobot-train failed (exit {exit_code}). "
            f"Check logs in {output_dir}"
        )

    return outcome
