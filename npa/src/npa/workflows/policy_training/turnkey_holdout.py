"""Evaluate actual policy loss on reserved public episodes without refitting normalization."""

from __future__ import annotations

import math
from pathlib import Path

from .public_vla_data import PINS
from .turnkey_store import read


def holdout_loss(
    workspace: Path, prepared: Path, policy_path: Path, partition: str, task: str | None
) -> dict:
    """Compute native SmolVLA loss over every frame of the selected reserved episodes.

    Args:
        workspace: Worker containing the immutable public dataset.
        prepared: Curated corpus and immutable recipe.
        policy_path: Exact candidate with its training-fitted processors.
        partition: First or second reserved demonstration set.
        task: Optional specialist task restriction.
    Returns:
        Native loss, episode identities and complete observed frame count.
    Raises:
        ValueError: The partition is empty, incomplete or nonfinite.
    """
    import torch
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    recipe = read(prepared, "recipe.json")
    rows = _reserved_rows(prepared, partition, task)
    torch.manual_seed(recipe["seed"])
    policy = SmolVLAPolicy.from_pretrained(policy_path).to("cuda").eval()
    pre, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=policy_path,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    dataset = _reserved_dataset(workspace, policy, rows)
    result = _loss(policy, pre, dataset, recipe, rows, partition)
    del policy
    torch.cuda.empty_cache()
    return result


def _reserved_rows(prepared, partition, task):
    rows = [
        r
        for r in read(prepared, "corpus.json")["episodes"]
        if r["partition"] == partition and (task is None or task in r["tasks"])
    ]
    if not rows:
        raise ValueError("reserved demonstration partition is empty")
    return rows


def _loss(policy, pre, dataset, recipe, rows, partition):
    import torch

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=recipe["batch_size"],
        shuffle=False,
        num_workers=recipe["workers"],
    )
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            loss, _ = policy.forward(pre(batch))
            value = float(loss.item())
            size = len(batch["episode_index"])
            if not math.isfinite(value):
                raise ValueError("nonfinite native reserved-episode loss")
            total += value * size
            count += size
    if count != sum(r["frames"] for r in rows):
        raise ValueError("reserved episode evaluation did not cover every frame")
    return {
        "engine": "native-smolvla-forward",
        "partition": partition,
        "frames": count,
        "episodes": [r["episode_index"] for r in rows],
        "mean_loss": total / count,
        "normalization": "saved training processors; no holdout refit",
    }


def _reserved_dataset(workspace, policy, rows):
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    dataset_root = workspace / "inputs/dataset"
    metadata = LeRobotDatasetMetadata(PINS["dataset"]["repo"], root=dataset_root)
    return LeRobotDataset(
        PINS["dataset"]["repo"],
        root=dataset_root,
        episodes=[r["episode_index"] for r in rows],
        video_backend="torchcodec",
        delta_timestamps=resolve_delta_timestamps(policy.config, metadata),
        # SmolVLA's forward expects float images, as after native training's conversion.
        return_uint8=False,
    )
