"""Train matched native LeRobot ACT policies from a sealed demonstration recipe."""

from __future__ import annotations

import importlib.metadata
import json
import shutil
import subprocess
import time
from pathlib import Path

from npa.workbench.lerobot.policy_container import (
    build_lerobot_train_command,
    validate_lerobot_checkpoint,
)
from npa.workflows.lerobot_transfer_data import file_sha256, tree_hashes, write_json


def training_command(prepared: Path, output: Path, arm: str) -> list[str]:
    """Build an ACT experiment with identical settings except photometric augmentation.

    Args:
        prepared: Verified dataset and recipe directory.
        output: Native LeRobot training output directory.
        arm: Baseline or robust candidate.
    Returns:
        Native lerobot-train argument vector.
    Raises:
        ValueError: The arm or recipe is invalid.
    """
    if arm not in {"baseline", "robust"}:
        raise ValueError("arm must be baseline or robust")
    recipe = json.loads((prepared / "recipe.json").read_text())
    overrides = [
        f"--dataset.revision={recipe['dataset_revision']}",
        f"--dataset.episodes={json.dumps(recipe['train_episodes'])}",
        "--dataset.video_backend=pyav", "--dataset.use_imagenet_stats=true",
        f"--seed={recipe['seed']}", "--cudnn_deterministic=true",
        "--policy.chunk_size=16", "--policy.n_action_steps=8",
        "--policy.optimizer_lr=0.0001", "--policy.optimizer_lr_backbone=0.00001",
        "--policy.use_amp=true",
        f"--dataset.image_transforms.enable={str(arm == 'robust').lower()}",
        "--dataset.image_transforms.tfs.affine.weight=0",
        "--dataset.image_transforms.tfs.brightness.kwargs.brightness=[0.4,1.4]",
        "--dataset.image_transforms.tfs.contrast.kwargs.contrast=[0.6,1.4]",
    ]
    return build_lerobot_train_command(
        dataset_path=prepared / "dataset", dataset_repo_id=recipe["dataset_repo"],
        output_dir=output, steps=recipe["train_steps"], batch_size=recipe["batch_size"],
        policy_type="act", device="cuda", num_workers=4, eval_freq=0,
        log_freq=100, extra_args=overrides,
    )


def train_policy(prepared: Path, output: Path, arm: str) -> None:
    """Execute real training and export the exact final checkpoint and provenance.

    Args:
        prepared: Verified preparation artifact.
        output: New stage directory.
        arm: Baseline or robust candidate.
    Returns:
        None.
    Raises:
        RuntimeError: LeRobot version or CUDA runtime is incompatible.
        subprocess.CalledProcessError: Native training failed.
        OSError: Checkpoint export failed.
    """
    runtime = runtime_versions()
    output.mkdir(parents=True)
    command = training_command(prepared, output / "training", arm)
    started = time.monotonic()
    _run_training(command, output / "train.log")
    checkpoint = output / "training/checkpoints/last/pretrained_model"
    validate_lerobot_checkpoint(checkpoint)
    shutil.copytree(checkpoint, output / "checkpoint")
    last = output / "training/checkpoints/last"
    if last.is_symlink():
        last.unlink()
    write_json(output / "training.json", {
        "schema": "npa.lerobot-transfer.training.v1", "arm": arm,
        "recipe_sha256": file_sha256(prepared / "recipe.json"),
        "checkpoint_hashes": tree_hashes(output / "checkpoint"),
        "duration_seconds": time.monotonic() - started,
        "runtime": runtime, "command": command,
    })


def _run_training(command: list[str], log_path: Path) -> None:
    with log_path.open("w") as log, subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    ) as process:
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, command)


def runtime_versions() -> dict:
    """Verify the supported CUDA workload and record software and accelerator versions.

    Args:
        None.
    Returns:
        Non-identifying runtime provenance.
    Raises:
        RuntimeError: LeRobot is not 0.5.1 or CUDA is unavailable.
    """
    import torch

    version = importlib.metadata.version("lerobot")
    if version != "0.5.1" or not torch.cuda.is_available():
        raise RuntimeError("The transfer benchmark requires LeRobot 0.5.1 and working CUDA")
    return {
        "lerobot": version, "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(), "compute_capability": list(torch.cuda.get_device_capability()),
        "lerobot_source": json.loads(importlib.metadata.distribution("lerobot").read_text("direct_url.json") or "{}"),
        "packages": dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())),
    }
