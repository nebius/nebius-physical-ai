"""Restrict the RLC policy to the official 2026 RGB and local proprioception fields."""

import numpy as np

# The 2025 vector had 256 entries; using its offsets on the 2026 vector is invalid.
PROPRIOCEPTION_INDICES = {
    "R1Pro": {
        "base_qvel": slice(0, 3),
        "arm_left_qpos": slice(3, 10),
        "gripper_left_qpos": slice(24, 26),
        "arm_right_qpos": slice(28, 35),
        "gripper_right_qpos": slice(49, 51),
        "trunk_qpos": slice(53, 57),
    }
}
CAMERAS = ("zed_link", "left_realsense_link", "right_realsense_link")


def policy_observation(observation: dict) -> dict:
    """Select onboard inputs, excluding evaluator metadata and task-state fields.

    Args:
        observation: Flattened official RGBDFullResWrapper observation.
    Returns:
        Three RGB images and the 61-element permitted proprioception vector.
    Raises:
        KeyError: A required camera or proprioception field is absent.
        ValueError: An input has an incompatible shape or contains nonfinite state.
    """
    state = np.asarray(observation["robot_r1::proprio"])
    if state.shape != (61,) or not np.isfinite(state).all():
        raise ValueError("Expected finite 61-element BEHAVIOR 2026 proprioception")
    result = {"robot_r1::proprio": state}
    for camera in CAMERAS:
        key = f"robot_r1::robot_r1:{camera}:Camera:0::rgb"
        image = np.asarray(observation[key])
        size = 720 if camera == "zed_link" else 480
        if image.shape not in {(size, size, 3), (size, size, 4)}:
            raise ValueError("Expected official full-resolution RGB camera shape")
        if image.dtype != np.uint8:
            raise ValueError("Expected uint8 RGB camera pixels")
        result[key] = image[..., :3]
    return result
