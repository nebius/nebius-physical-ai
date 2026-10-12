"""Exercise a saved ACT checkpoint through the real NPA RoboCasa adapter."""

from __future__ import annotations

import json
import math
import tempfile
from importlib.metadata import version
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from npa.workbench.robocasa.capabilities import (
    RoboCasaError,
    _act_action_selector,
    _load_act_policy,
)
from verify_lerobot_act_derivative import _config


class _AdapterEnv:
    """Minimal real Gymnasium action-space boundary used by policy evaluation."""

    action_space = gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32)


def _write_dataset(root: Path) -> tuple[dict, int]:
    """Round-trip state and action features through the real dataset facade."""
    features = {
        "observation.state": {"dtype": "float32", "shape": (16,), "names": None},
        "action": {"dtype": "float32", "shape": (7,), "names": None},
    }
    dataset = LeRobotDataset.create(
        repo_id="npa/robocasa-act-adapter-verifier",
        fps=10,
        features=features,
        root=root,
    )
    for value in (-0.5, 0.5):
        dataset.add_frame(
            {
                "task": "RoboCasa ACT adapter verification",
                "observation.state": torch.full((16,), value),
                "action": torch.full((7,), value),
            }
        )
    dataset.save_episode()
    dataset.finalize()
    reopened = LeRobotDataset(repo_id=dataset.repo_id, root=root)
    assert len(reopened) == 2
    assert tuple(reopened[0]["observation.state"].shape) == (16,)
    assert tuple(reopened[0]["action"].shape) == (7,)
    return dict(reopened.meta.stats), len(reopened)


def _processor_stats(root: Path) -> tuple[dict, int]:
    """Combine real dataset statistics with deterministic image statistics."""
    stats, frames = _write_dataset(root)
    for key in ("observation.images.workspace", "observation.images.wrist"):
        stats[key] = {
            "mean": torch.zeros((3, 1, 1)),
            "std": torch.ones((3, 1, 1)),
        }
    return stats, frames


def _observation() -> dict[str, np.ndarray]:
    """Build one observation with the exact RoboCasa-to-ACT feature layout."""
    return {
        "video.robot0_agentview_left": np.full((64, 64, 3), 31, dtype=np.uint8),
        "video.robot0_eye_in_hand": np.full((64, 64, 3), 127, dtype=np.uint8),
        "state.base_position": np.zeros(3, dtype=np.float32),
        "state.base_rotation": np.array([1, 0, 0, 0], dtype=np.float32),
        "state.end_effector_position_relative": np.zeros(3, dtype=np.float32),
        "state.end_effector_rotation_relative": np.array(
            [1, 0, 0, 0], dtype=np.float32
        ),
        "state.gripper_qpos": np.zeros(2, dtype=np.float32),
    }


def _save_checkpoint(path: Path, stats: dict) -> list[str]:
    """Save policy and processor files in the layout loaded by NPA evaluation."""
    torch.manual_seed(595)
    policy = ACTPolicy(_config()).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config, dataset_stats=stats
    )
    policy.save_pretrained(path)
    preprocessor.save_pretrained(path)
    postprocessor.save_pretrained(path)
    names = sorted(item.name for item in path.iterdir())
    assert {"config.json", "model.safetensors"} <= set(names), names
    assert "policy_preprocessor.json" in names, names
    assert "policy_postprocessor.json" in names, names
    return names


def _run_adapter(checkpoint: Path) -> tuple[list[int], str]:
    """Load the checkpoint through NPA and exercise success and missing-input paths."""
    selector = _act_action_selector(_load_act_policy(checkpoint))
    action = np.asarray(selector(_AdapterEnv(), _observation()), dtype=np.float32)
    assert action.shape == (7,), action.shape
    assert all(math.isfinite(float(value)) for value in action)
    assert _AdapterEnv.action_space.contains(action), action
    missing = _observation()
    missing.pop("video.robot0_agentview_left")
    try:
        selector(_AdapterEnv(), missing)
    except RoboCasaError as exc:
        missing_failure = type(exc).__name__
    else:
        raise AssertionError("NPA ACT adapter accepted a missing workspace image")
    return list(action.shape), missing_failure


def main() -> None:
    """Validate the dataset and real NPA checkpoint/action entrypoints."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        stats, frames = _processor_stats(root / "dataset")
        checkpoint = root / "checkpoint"
        checkpoint.mkdir()
        files = _save_checkpoint(checkpoint, stats)
        action_shape, missing_failure = _run_adapter(checkpoint)
    print(
        json.dumps(
            {
                "action_shape": action_shape,
                "checkpoint_files": files,
                "dataset_frames": frames,
                "lerobot": version("lerobot"),
                "missing_image_rejected": missing_failure,
                "npa_adapter": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
