"""Verify the real NPA conversion against the image's actual LeRobot reader."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from npa.adapter.sim_to_lerobot import AdapterError, convert


def _write_input(root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Freeze two episodes whose pixels identify each state/action row."""
    root.mkdir()
    states = np.arange(6 * 16, dtype=np.float32).reshape(6, 16) / 100
    actions = -np.arange(6 * 7, dtype=np.float32).reshape(6, 7) / 100
    for episode in range(2):
        directory = root / f"episode_{episode:06d}"
        directory.mkdir()
        rows = slice(episode * 3, episode * 3 + 3)
        np.save(directory / "state.npy", states[rows])
        np.save(directory / "actions.npy", actions[rows])
        for camera, offset in (("obs_workspace", 0), ("obs_wrist", 7)):
            frames = np.stack(
                [
                    np.full((32, 32, 3), 24 + index * 30 + offset, dtype=np.uint8)
                    for index in range(episode * 3, episode * 3 + 3)
                ]
            )
            np.save(directory / f"{camera}.npy", frames)
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "episodes": [
                    {"episode_index": index, "task": f"conversion-control-{index}"}
                    for index in range(2)
                ]
            }
        ),
        encoding="utf-8",
    )
    return states, actions


def _read_and_compare(root: Path, states: np.ndarray, actions: np.ndarray) -> float:
    """Decode every camera/frame and compare exact scalar rows and pooled stats."""
    dataset = LeRobotDataset(
        repo_id="npa/robocasa-conversion-verifier",
        root=root,
        download_videos=False,
        video_backend="pyav",
    )
    assert len(dataset) == 6
    max_pixel_error = 0.0
    for index in range(6):
        sample = dataset[index]
        torch.testing.assert_close(
            sample["observation.state"], torch.from_numpy(states[index])
        )
        torch.testing.assert_close(sample["action"], torch.from_numpy(actions[index]))
        assert sample["task"] == f"conversion-control-{index // 3}"
        for camera, offset in (("workspace", 0), ("wrist", 7)):
            decoded = sample[f"observation.images.{camera}"]
            assert tuple(decoded.shape) == (3, 32, 32)
            error = float((decoded * 255 - (24 + index * 30 + offset)).abs().max())
            assert error <= 4.0, (index, camera, error)
            max_pixel_error = max(max_pixel_error, error)
    stats = dataset.meta.stats["observation.state"]
    np.testing.assert_allclose(stats["mean"], states.mean(axis=0), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(stats["std"], states.std(axis=0), rtol=1e-6, atol=1e-6)
    return max_pixel_error


def main() -> None:
    """Run conversion, all-frame readback and the invalid-frame-rate refusal."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        states, actions = _write_input(root / "source")
        convert(root / "source", root / "dataset", fps=10, task_from_metadata=True)
        pixel_error = _read_and_compare(root / "dataset", states, actions)
        try:
            convert(root / "source", root / "invalid", fps=float("nan"))
        except AdapterError:
            assert not (root / "invalid").exists()
        else:
            raise AssertionError("NPA conversion accepted a non-finite frame rate")
        videos = sorted((root / "dataset").rglob("*.mp4"))
        assert len(videos) == 4, videos
        video_hashes = [
            hashlib.sha256(path.read_bytes()).hexdigest() for path in videos
        ]
    print(
        json.dumps(
            {
                "episodes": 2,
                "state_action_rows": 6,
                "decoded_camera_frames": 12,
                "video_sha256": video_hashes,
                "maximum_pixel_error_255": pixel_error,
                "pixel_error_tolerance_255": 4.0,
                "invalid_fps_rejected_before_output": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
