"""Reject replay pose interpretations that contradict recorded action-target matrices."""

from __future__ import annotations

from typing import Any

from .errors import IsaacArenaError
from .replay_quaternions import dataset_format_version


def _rotation_matrices(quaternions: Any, legacy: bool, np: Any) -> Any:
    values = quaternions.astype(np.float64)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(norms).all() or not (norms > 0).all():
        raise IsaacArenaError("GR1 Pink replay poses require finite nonzero quaternions")
    values = values / norms
    if legacy:
        values = np.roll(values, -1, axis=-1)
    x, y, z, w = values.T
    return np.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(-1, 3, 3)


def _target_matrices(item: Any, steps: int, h5py: Any, np: Any) -> Any:
    error = "recorded GR1 Pink target poses must be finite per-action 4x4 matrices"
    if not isinstance(item, h5py.Dataset):
        raise IsaacArenaError(error)
    matrices = np.asarray(item)
    if (
        matrices.shape != (steps, 4, 4)
        or matrices.dtype.kind not in "biuf"
        or not np.isfinite(matrices).all()
        or not np.allclose(matrices[:, 3, :], [0, 0, 0, 1], atol=1e-5, rtol=0)
    ):
        raise IsaacArenaError(error)
    return matrices


def _validate_target(actions: Any, matrices: Any, start: int, legacy: bool, np: Any) -> None:
    rotations = _rotation_matrices(actions[:, start + 3:start + 7], legacy, np)
    # Native datagen targets describe the same action in the environment frame.
    # Check translation as well, so a frame or sample offset cannot select an
    # apparently matching orientation. Tolerance covers float32 serialization.
    if not (
        np.allclose(actions[:, start:start + 3], matrices[:, :3, 3], atol=1e-5, rtol=0)
        and np.allclose(rotations, matrices[:, :3, :3], atol=1e-5, rtol=0)
    ):
        raise IsaacArenaError(
            "recorded GR1 Pink target poses contradict the replay quaternion format, "
            "frame, or sample alignment; normalize this source explicitly before replay"
        )


def validate_recorded_target_poses(
    source: Any, episode_name: str, embodiment: str
) -> dict[str, Any] | None:
    """Check optional native datagen target poses before creating execution input.

    Args:
        source: Open source HDF5 recording, retained without modification.
        episode_name: Selected episode under the recording's data group.
        embodiment: Resolved upstream embodiment; only gr1_pink uses this contract.

    Returns:
        Source-consistency metadata, or None when target matrices are not provided.

    Raises:
        IsaacArenaError: Provided targets are malformed or contradict the input contract.
    """
    if embodiment != "gr1_pink":
        return None
    import h5py
    import numpy as np

    episode = source["data"][episode_name]
    targets = [(episode.get(f"obs/datagen_info/target_eef_pose/{hand}"), start)
               for hand, start in (("left", 0), ("right", 7))]
    targets = [(item, start) for item, start in targets if item is not None]
    if not targets:
        return None
    actions = np.asarray(episode["actions"])
    if actions.ndim != 2 or actions.shape[1] != 36:
        raise IsaacArenaError("gr1_pink replay requires exactly 36 action columns")
    legacy = dataset_format_version(source) == 0
    for item, start in targets:
        matrices = _target_matrices(item, len(actions), h5py, np)
        _validate_target(actions, matrices, start, legacy, np)
    return {"status": "consistent", "checked_hand_targets": len(targets) * len(actions),
            "runtime_outcome_claim": False}
