"""Validate synchronized robot trajectories and export XR1's native JSON contract."""

from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np

CAMERAS = ("ego", "wrist_left", "wrist_right")
STATE_WIDTHS = {
    "left_ee_pos": 3, "left_ee_rotm": 9, "left_arm_joint": 7,
    "left_gripper_pos": 1, "right_ee_pos": 3, "right_ee_rotm": 9,
    "right_arm_joint": 7, "right_gripper_pos": 1, "waist_pos": 1,
}
ACTION_WIDTHS = {
    name: width for name, width in STATE_WIDTHS.items() if "arm_joint" not in name
} | {"base_vel": 3}
INSTRUCTION = "Pick up both colored blocks and place each on its green target."


def _validate_arrays(arrays: dict, widths: dict, frames: int, label: str) -> None:
    for name, width in widths.items():
        values = np.asarray(arrays.get(name), dtype=np.float64)
        if values.shape != (frames, width) or not np.isfinite(values).all():
            raise ValueError(f"{label}.{name} must contain {frames} finite {width}-vectors")
        if name.endswith("rotm"):
            rotations = values.reshape(-1, 3, 3)
            identity = rotations.transpose(0, 2, 1) @ rotations
            if not np.allclose(identity, np.eye(3), atol=1e-4):
                raise ValueError(f"{label}.{name} contains non-orthonormal rotations")
            if not np.allclose(np.linalg.det(rotations), 1, atol=1e-4):
                raise ValueError(f"{label}.{name} contains reflected rotations")


def validate_episode(episode: dict) -> None:
    """Reject incomplete trajectories and inconsistent simulation timestamps.

    Args:
        episode: Recorded observations, issued action targets, and provenance.
    Returns:
        None.
    Raises:
        ValueError: When any synchronized field or physical provenance is invalid.
    """
    frames = episode.get("num_frames", 0)
    if not isinstance(frames, int) or frames < 30:
        raise ValueError("An XR1 episode needs at least one complete 30-frame action window")
    _validate_arrays(episode["proprios"], STATE_WIDTHS, frames, "proprios")
    _validate_arrays(episode["actions"], ACTION_WIDTHS, frames, "actions")
    timestamps = np.asarray(episode.get("timestamps"), dtype=np.float64)
    period = 1 / episode["control_hz"]
    if timestamps.shape != (frames,) or not np.isfinite(timestamps).all():
        raise ValueError("Each frame needs a finite simulation timestamp")
    if not np.allclose(np.diff(timestamps), period, atol=1e-6):
        raise ValueError("Camera/state/action samples must share a uniform simulation clock")
    if episode.get("grasp_mechanism") != "finger_contact":
        raise ValueError("Training requires physical finger contact, without object attachment")
    if set(episode["videos"]) != set(CAMERAS):
        raise ValueError("XR1 requires ego, left-wrist, and right-wrist cameras")


def _instruction() -> dict:
    views = "# Ego View\n<image>\n# Left-Wrist View\n<image>\n# Right-Wrist View\n<image>"
    prompt = f"The following observations are captured from multiple views.\n{views}"
    return {"general": [{
        "images": [f"observations.{camera}" for camera in CAMERAS],
        "conversations": [
            {"from": "human", "value": f"{prompt}\nGenerate robot actions for the task:\n{INSTRUCTION}"},
            {"from": "gpt", "value": ""},
        ],
    }]}


def native_annotation(episode: dict, video_root: Path) -> dict:
    """Create an upstream XR1 annotation from a validated successful demonstration.

    Args:
        episode: Synchronized, physically measured demonstration.
        video_root: Materialized directory containing its three videos.
    Returns:
        JSON-serializable annotation accepted by XR1's JsonDataset.
    Raises:
        ValueError: If the demonstration is incomplete, unsuccessful, or escapes its root.
    """
    validate_episode(episode)
    if episode.get("success") is not True:
        raise ValueError("Failed demonstrations remain evidence and cannot enter behavior cloning")
    observations = {}
    for camera, relative in episode["videos"].items():
        path = (video_root / relative).resolve()
        if not path.is_relative_to(video_root.resolve()) or not path.is_file():
            raise ValueError(f"Missing or out-of-root video for {camera}")
        observations[camera] = [{"path": str(path), "start": 0, "crop_bbox": None}]
    return {
        "trajectory_type": "success", "time": episode["episode_id"],
        "num_frames": episode["num_frames"], "instruction": _instruction(),
        "observations": observations, "proprios": episode["proprios"],
        "actions": episode["actions"],
    }


def verify_video(path: Path, frames: int, control_hz: int) -> dict:
    """Decode every frame to verify count, clock, and changing visual observations.

    Args:
        path: Recorded video to validate.
        frames: Expected synchronized frame count.
        control_hz: Expected recording rate in simulation seconds.
    Returns:
        Video metadata, decoded frame count, and SHA-256 digest.
    Raises:
        ValueError: If the video is static, corrupt, or inconsistent with the trajectory.
    """
    import av

    count, hashes, times = 0, set(), []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            count += 1
            hashes.add(hashlib.sha256(frame.to_ndarray(format="rgb24").tobytes()).digest())
            times.append(float(frame.time))
    if count != frames or len(hashes) < 2:
        raise ValueError(f"{path.name}: expected {frames} moving frames; got {count}/{len(hashes)}")
    if not np.allclose(np.diff(times), 1 / control_hz, atol=1e-5):
        raise ValueError(f"{path.name}: decoded video clock disagrees with robot control clock")
    return {"frames": count, "unique_frames": len(hashes), "sha256": _file_digest(path)}


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_splits(manifest: dict) -> None:
    """Ensure whole episodes and scenario seeds belong to exactly one split.

    Args:
        manifest: Train, validation, and test episode lists with seeds and identifiers.
    Returns:
        None.
    Raises:
        ValueError: If a split is empty or an episode/seed crosses split boundaries.
    """
    identifiers, seeds = set(), set()
    for split in ("train", "validation", "test"):
        episodes = manifest.get(split, [])
        if not episodes:
            raise ValueError(f"Missing {split} episodes")
        for episode in episodes:
            identifier, seed = episode["episode_id"], episode["seed"]
            if identifier in identifiers or seed in seeds:
                raise ValueError("Episode identifiers and seeds must be disjoint across all splits")
            identifiers.add(identifier)
            seeds.add(seed)
