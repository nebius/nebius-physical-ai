"""Generate original reaching trajectories, camera frames, and a LeRobot v3 dataset."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from npa.adapter.sim_to_lerobot import convert
from npa.workbench.dataset.storage import write_json_uri


def _frame(position, target, dark):
    image = Image.new("RGB", (128, 128), (2, 3, 4) if dark else (32, 44, 56))
    if dark:
        return image
    draw = ImageDraw.Draw(image)
    for offset in range(0, 128, 16):
        draw.line((offset, 0, offset, 128), fill=(49, 64, 78))
        draw.line((0, offset, 128, offset), fill=(49, 64, 78))
    end = (64 + position[0] * 58, 112 - position[1] * 58)
    goal = (64 + target[0] * 58, 112 - target[1] * 58)
    elbow = (32, 70)
    draw.line([(64, 112), elbow, end], fill=(165, 190, 210), width=7)
    draw.ellipse((end[0] - 4, end[1] - 4, end[0] + 4, end[1] + 4), fill=(54, 219, 188))
    draw.rectangle(
        (goal[0] - 4, goal[1] - 4, goal[0] + 4, goal[1] + 4), fill=(243, 188, 93)
    )
    return image


def _episode(index, raw, previews, generator):
    angle = generator.uniform(0.45, 2.65)
    radius = generator.uniform(0.6, 1.15)
    target = np.array([radius * math.cos(angle), radius * math.sin(angle)])
    position = np.array([generator.uniform(-0.5, 0.5), generator.uniform(0.3, 0.7)])
    states, actions, frames = [], [], []
    dark = index >= 120
    for _ in range(24):
        states.append(np.r_[position, target])
        action = 0.28 * (target - position)
        actions.append(action)
        frames.append(np.asarray(_frame(position, target, dark)))
        position = position + action
    directory = raw / f"episode_{index:04d}"
    directory.mkdir()
    for name, data in {
        "state": states,
        "actions": actions,
        "obs_workspace": frames,
        "obs_wrist": frames,
    }.items():
        np.save(directory / f"{name}.npy", np.asarray(data), allow_pickle=False)
    preview = previews / f"{index:04d}.png"
    Image.fromarray(frames[0]).save(preview)
    return preview


def prepare(directory: Path) -> None:
    """Create an entirely generated dataset and explicit demo gate settings.

    Args:
        directory: Empty private workspace used by the local reference run.
    Returns:
        None; writes raw episodes, LeRobot v3 data, previews, and pipeline inputs.
    Raises:
        OSError: Dataset files cannot be written.
        RuntimeError: Video encoding or dataset conversion fails.
    """
    raw, previews = directory / "raw", directory / "previews"
    raw.mkdir(parents=True)
    previews.mkdir()
    generator, episodes = np.random.default_rng(7), []
    dataset = directory / "dataset"
    for index in range(132):
        preview = _episode(index, raw, previews, generator)
        episodes.append(
            {
                "dataset_uri": str(dataset),
                "episode_index": index,
                "group_id": f"generated-{index:04d}",
                "source": "synthetic",
                "preview_uri": str(preview),
                "detections": ["target"],
                "inhouse_keep": True,
            }
        )
    convert(
        raw,
        dataset,
        fps=20,
        robot_type="npa_generated_planar_reacher",
        task="Reach a target in an analytical Cartesian reference environment",
    )
    _inputs(directory, episodes)


def _inputs(directory, episodes):
    payloads = {
        "episodes": {"schema": "npa.policy.episodes.v1", "episodes": episodes},
        "curation-policy": {
            "min_brightness": 0.08,
            "max_brightness": 0.9,
            "min_sharpness": 0.001,
            "min_detections": 1,
        },
        "gate-policy": {
            "pretrain": {"nominal": 0.9, "perturbed": 0.9},
            "finetune": {"slow-actuator": 0.9},
        },
        "batch-settings": {"transport": "reference-local"},
    }
    for name, payload in payloads.items():
        write_json_uri(str(directory / "input" / f"{name}.json"), payload)
