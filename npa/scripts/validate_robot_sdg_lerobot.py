"""Prove recorded robot SDG data loads through native LeRobot and preserves observations/actions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import av
import numpy as np


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _verify_hashes(root, report):
    for relative, expected in report["artifacts"].items():
        path = (root / relative).resolve()
        _check(path.is_relative_to(root.resolve()), "Artifact path escapes run")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        _check(digest.hexdigest() == expected["sha256"], f"Hash mismatch: {relative}")


def _decode_video(path, expected_frames):
    with av.open(str(path)) as container:
        frames = sum(1 for _ in container.decode(video=0))
    _check(frames == expected_frames, f"Wrong video frame count: {path.name}")
    return frames


def _check_episode(dataset, record, root, start):
    episode = root / record["episode_path"]
    states = np.load(episode / "state.npy", allow_pickle=False)
    actions = np.load(episode / "actions.npy", allow_pickle=False)
    count = len(actions)
    _check(np.ptp(states, axis=0).max() > 0.01, "Robot joints did not move")
    _check(np.isfinite(actions).all(), "Nonfinite robot actions")
    sampled = 0
    for offset in (0, count // 2, count - 1):
        row = dataset[start + offset]
        _check(
            row["task"] == record["simulation"]["task"], "LeRobot task label was lost"
        )
        np.testing.assert_allclose(row["action"].numpy(), actions[offset], atol=1e-6)
        np.testing.assert_allclose(
            row["observation.state"].numpy(), states[offset], atol=1e-6
        )
        for camera in ("workspace", "wrist"):
            original = np.load(
                episode / f"obs_{camera}.npy", mmap_mode="r", allow_pickle=False
            )[offset]
            decoded = (
                row[f"observation.images.{camera}"].numpy().transpose(1, 2, 0) * 255
            )
            _check(decoded.shape == original.shape, "Camera shape changed")
            _check(
                float(np.abs(decoded - original.astype(float)).mean()) < 10,
                "Camera frame is not aligned with the recorded timestep",
            )
            sampled += 1
    return count, sampled


def _native_validation(root, report):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from torch.utils.data import DataLoader

    records = [
        json.loads(line)
        for line in (root / "provenance.jsonl").read_text().splitlines()
    ]
    accepted = sorted(
        (record for record in records if record["status"] == "accepted"),
        key=lambda row: row["dataset_episode_index"],
    )
    dataset = LeRobotDataset(
        "npa/robot-sdg-validation", root=root / "dataset", video_backend="pyav"
    )
    _check(
        dataset.num_episodes == report["accepted_count"], "Wrong native episode count"
    )
    _check(dataset.num_frames == report["total_frames"], "Wrong native frame count")
    start = sampled = decoded = 0
    for record in accepted:
        count, images = _check_episode(dataset, record, root, start)
        index = record["dataset_episode_index"]
        decoded += _decode_episode_videos(root, index, count)
        start += count
        sampled += images
    batch = next(iter(DataLoader(dataset, batch_size=4, num_workers=0)))
    return {
        "native_episodes": dataset.num_episodes,
        "native_frames": dataset.num_frames,
        "decoded_camera_frames": decoded,
        "aligned_camera_samples": sampled,
        "training_batch_shapes": {
            key: list(value.shape)
            for key, value in batch.items()
            if hasattr(value, "shape")
        },
        "task_labels_preserved": True,
        "state_action_samples_match": True,
    }


def _decode_episode_videos(root, index, count):
    decoded = 0
    for camera in ("workspace", "wrist"):
        path = (
            root
            / f"dataset/videos/observation.images.{camera}/chunk-000/file-{index:03d}.mp4"
        )
        decoded += _decode_video(path, count)
    return decoded


def main():
    """Validate a robot run in an isolated native LeRobot environment.

    Args:
        None; command-line --input-path and --output-path select local artifacts.
    Returns:
        None; writes and prints the native-reader validation result.
    Raises:
        ValueError: Any hash, video, task, alignment, or native-reader check fails.
    """
    from importlib.metadata import version

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.input_path / "report.json").read_text())
    _check(report["schema"] == "npa.token_factory.robot_sdg.v1", "Not a robot SDG run")
    _verify_hashes(args.input_path, report)
    result = {
        "status": "passed",
        "lerobot_version": version("lerobot"),
        **_native_validation(args.input_path, report),
    }
    args.output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
