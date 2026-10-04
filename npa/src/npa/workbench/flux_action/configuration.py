"""Translate a robot recipe into the pinned upstream index and trainer contracts."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any

from .schemas import BASE_REPOSITORY, BASE_REVISION, Recipe


def validate_dataset(root: Path, recipe: Recipe) -> dict[str, Any]:
    """Check dataset metadata against the operator's embodiment before GPU work.

    Args: root is a LeRobot directory; recipe declares channel semantics.
    Returns: Parsed LeRobot metadata.
    Raises: ValueError for incompatible metadata or unsafe file templates.
    """
    info = json.loads((root / "meta/info.json").read_text())
    robot = recipe.robot
    if str(info.get("codebase_version", "")).split(".")[0] not in {"v2", "v3"}:
        raise ValueError("expected a LeRobot v2.x or v3.x dataset")
    if info.get("fps") != robot.fps:
        raise ValueError(
            "dataset fps differs from robot fps; metadata does not resample data"
        )
    features = info["features"]
    for key, channels in (
        (robot.state_key, robot.states),
        (robot.action_key, robot.actions),
    ):
        feature = features.get(key, {})
        if feature.get("shape") != [len(channels)]:
            raise ValueError(f"dataset {key} width differs from robot contract")
        names = feature.get("names")
        if isinstance(names, dict) and len(names) == 1:
            names = next(iter(names.values()))
        if names is not None and names != [channel.name for channel in channels]:
            raise ValueError(f"dataset {key} channel order differs from robot contract")
    for feature in robot.cameras.values():
        if features.get(feature, {}).get("dtype") != "video":
            raise ValueError(f"camera {feature} must be an existing video feature")
    for key in ("data_path", "video_path"):
        path = PurePosixPath(info[key])
        if path.is_absolute() or ".." in path.parts or "\\" in info[key]:
            raise ValueError(f"unsafe dataset {key}")
    return info


def index_arguments(recipe: Recipe, dataset: Path, index: Path) -> list[str]:
    """Build the native LeRobot index command with explicit robot semantics.

    Args: recipe, dataset root, and destination index directory.
    Returns: Arguments following python -m flux_action.cli.
    Raises: None.
    """
    robot = recipe.robot
    args = [
        "index-lerobot",
        "--source-root",
        str(dataset),
        "--output-dir",
        str(index),
        "--state-key",
        robot.state_key,
        "--action-key",
        robot.action_key,
        "--action-parameterization",
        robot.action_parameterization,
        "--absolute-action-dims",
        ",".join(map(str, robot.absolute_action_dims)),
        "--chunk-size",
        "32",
        "--val-episodes",
        str(recipe.val_episodes),
        "--hash-files",
    ]
    for stream, feature in robot.cameras.items():
        args.extend(["--camera", f"{stream}={feature}"])
    return args


def training_config(
    recipe: Recipe, dataset: Path, output: Path, processes: int
) -> dict:
    """Build a native trainer config with pinned encoders and a new robot head.

    Args: recipe, staged dataset, artifact directory, and local GPU count.
    Returns: JSON-serializable upstream TrainConfig fields.
    Raises: None.
    """
    settings = recipe.training.model_dump()
    policy = _policy_config(recipe, output)
    for name in ("optimizer_lr", "optimizer_lr_heads_multiplier", "caption_dropout"):
        policy[name] = settings.pop(name)
    return {
        **settings,
        "dataset": "index",
        "source_root": str(dataset),
        "index_dir": str(output / "index"),
        "output_dir": str(output / "train"),
        "policy": policy,
        "shard_size": processes,
        "resume": "none",
        "cooldown_start": None,
        "reseed_on_resume": False,
        "log_every": 1,
        "compute_dtype": "bfloat16",
        "reduce_dtype": "float32",
        "activation_checkpointing": True,
        "keep_checkpoints": None,
        "keep_every": None,
    }


def _policy_config(recipe: Recipe, output: Path) -> dict:
    robot = recipe.robot
    return {
        "trunk_weights": f"{BASE_REPOSITORY}:flux-3-action-base.safetensors@{BASE_REVISION}",
        "video_vae_id": f"{BASE_REPOSITORY}:video_vae.safetensors@{BASE_REVISION}",
        "text_encoder_id": f"{BASE_REPOSITORY}:text_encoder@{BASE_REVISION}",
        "action_modality": f"robot_{robot.embodiment}",
        "action_dim": len(robot.actions),
        "camera_layout": robot.camera_layout,
        "camera_keys": [f"images.{stream}" for stream in robot.cameras],
        "canvas_hw": list(robot.canvas_hw),
        "fps": float(robot.fps),
        "chunk_size": 32,
        "n_action_steps": robot.n_action_steps,
        "action_parameterization": robot.action_parameterization,
        "absolute_action_dims": robot.absolute_action_dims,
        "gripper_flip_dims": robot.gripper_flip_dims,
        "single_frame_encode": True,
        "augment": False,
        "attn_mode": "torch",
        "action_scale": 1.0,
    }
