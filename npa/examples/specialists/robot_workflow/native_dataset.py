"""Validate every exported timestep through native LeRobot without Workbench producer imports."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, closing
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

from media import reference_frames, verify_video

_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "upperarm_roll_joint",
    "elbow_flex_joint",
    "forearm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
    "r_gripper_finger_joint",
    "l_gripper_finger_joint",
]
_ACTIONS = ["cartesian_dx", "cartesian_dy", "cartesian_dz", "gripper"]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _feature(info, name, dtype, shape, names):
    feature = info["features"][name]
    _require(feature["dtype"] == dtype, f"dataset feature dtype differs: {name}")
    _require(feature["shape"] == shape, f"dataset feature shape differs: {name}")
    _require(feature["names"] == names, f"dataset feature names differ: {name}")


def _metadata(root, expected):
    info = json.loads((root / "dataset/meta/info.json").read_text())
    _require(info["codebase_version"] == "v3.0", "dataset format differs")
    _require(
        info["robot_type"] == "fetch" and info["fps"] == 25, "dataset robot/FPS differs"
    )
    _require(
        info["total_episodes"] == len(expected),
        "dataset declared episode count differs",
    )
    _require(
        info["total_frames"] == 185 * len(expected),
        "dataset declared frame count differs",
    )
    _feature(info, "observation.state", "float32", [9], _JOINTS)
    _feature(info, "action", "float32", [4], _ACTIONS)
    for name in ("episode_index", "frame_index", "index", "task_index"):
        _feature(info, name, "int64", [1], None)
    _feature(info, "timestamp", "float32", [1], None)
    for camera in ("workspace", "wrist"):
        name = "observation.images." + camera
        _feature(info, name, "video", [360, 480, 3], ["height", "width", "channel"])
        video = info["features"][name]["video_info"]
        _require(
            video["video.fps"] == 25 and not video["video.is_depth_map"],
            "dataset video metadata differs",
        )
    return info


def _episode(dataset, root, row, start):
    episode = root / row["episode_path"]
    actions = np.load(episode / "actions.npy", mmap_mode="r", allow_pickle=False)
    states = np.load(episode / "state.npy", mmap_mode="r", allow_pickle=False)
    images = {
        key: np.load(episode / f"obs_{key}.npy", mmap_mode="r", allow_pickle=False)
        for key in ("workspace", "wrist")
    }
    with ExitStack() as stack:
        references = {
            key: stack.enter_context(closing(reference_frames([frames])))
            for key, frames in images.items()
        }
        for index in range(len(actions)):
            sample = dataset[start + index]
            _sample_metadata(sample, row, index, start)
            np.testing.assert_allclose(
                sample["action"].numpy(), actions[index], rtol=0, atol=1e-6
            )
            np.testing.assert_allclose(
                sample["observation.state"].numpy(), states[index], rtol=0, atol=1e-6
            )
            _sample_images(sample, references)
    return len(actions)


def _sample_metadata(sample, row, index, start):
    _require(sample["task"] == row["simulation"]["task"], "dataset task label differs")
    _require(
        int(sample["episode_index"]) == row["dataset_episode_index"],
        "dataset episode index differs",
    )
    _require(int(sample["frame_index"]) == index, "dataset frame index differs")
    _require(int(sample["index"]) == start + index, "dataset global index differs")
    _require(
        abs(float(sample["timestamp"]) - index / 25) < 1e-5, "dataset timestamp differs"
    )


def _sample_images(sample, references):
    for key, frames in references.items():
        observed = sample[f"observation.images.{key}"].numpy().transpose(1, 2, 0)
        _require(
            np.isfinite(observed).all()
            and (observed >= 0).all()
            and (observed <= 1).all(),
            "dataset camera values differ",
        )
        pixels = np.rint(observed * 255).astype(np.uint8)
        np.testing.assert_array_equal(
            pixels, next(frames), err_msg="dataset camera timeline differs"
        )


def _summary(dataset, start, decoded):
    from torch.utils.data import DataLoader

    batch = next(iter(DataLoader(dataset, batch_size=4, num_workers=0)))
    return {
        "status": "passed",
        "lerobot_version": version("lerobot"),
        "native_episodes": dataset.num_episodes,
        "native_frames": dataset.num_frames,
        "decoded_dataset_camera_frames": decoded,
        "aligned_dataset_camera_samples": 2 * start,
        "training_batch_shapes": {
            name: list(value.shape)
            for name, value in batch.items()
            if hasattr(value, "shape")
        },
    }


def validate_dataset(root: Path, expected: list[dict]) -> dict:
    """Read all accepted native dataset samples and decode all retained episode previews.

    Args: root: Recorded workflow directory. expected: Independently accepted records.
    Returns: Native reader counts and data-loader batch evidence.
    Raises: ValueError, AssertionError, OSError: Dataset membership or contents differ.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    _metadata(root, expected)
    dataset = LeRobotDataset(
        "npa/prepared-fetch-validation", root=root / "dataset", video_backend="pyav"
    )
    _require(
        dataset.num_episodes == len(expected), "native dataset includes wrong episodes"
    )
    start, decoded = 0, 0
    for row in expected:
        count = _episode(dataset, root, row, start)
        for camera in ("workspace", "wrist"):
            video = (
                root
                / f"dataset/videos/observation.images.{camera}/chunk-000/file-{row['dataset_episode_index']:03d}.mp4"
            )
            frames = np.load(
                root / row["episode_path"] / f"obs_{camera}.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
            decoded += verify_video(video, [frames])
        start += count
    _require(dataset.num_frames == start, "native dataset frame count differs")
    return _summary(dataset, start, decoded)


def main() -> None:
    """Run the native reader in an operator-selected isolated interpreter.

    Args: None; --input, --expected and --output are local paths.
    Returns: None; writes verification JSON.
    Raises: ValueError, AssertionError, OSError: Native dataset verification fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate_dataset(args.input, json.loads(args.expected.read_text()))
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
