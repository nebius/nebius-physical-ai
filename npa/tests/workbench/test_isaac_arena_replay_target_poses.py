"""Check source-target consistency without using source observations as task outcomes."""

from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from npa.workbench.isaac_arena.errors import IsaacArenaError
from npa.workbench.isaac_arena.replay_input import (
    _prepare_replay_execution_input,
    _replay_input_evidence,
)


def _recording(path: Path, version: int | None, *, mixed: bool = False) -> None:
    rotations = Rotation.from_euler(
        "xyz", [[0, 10, 70], [10, 20, 80], [20, 30, 90]], degrees=True
    )
    quaternions = rotations.as_quat()
    if version != 1 and not mixed:
        quaternions = quaternions[:, [3, 0, 1, 2]]
    actions = np.arange(108, dtype=np.float32).reshape(3, 36) / 100
    with h5py.File(path, "w") as dataset:
        if version is not None:
            dataset.attrs["format_version"] = version
        episode = dataset.create_group("data/demo_0")
        for hand, start in (("left", 0), ("right", 7)):
            actions[:, start + 3 : start + 7] = quaternions
            matrices = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()
            matrices[:, :3, :3] = rotations.as_matrix()
            matrices[:, :3, 3] = actions[:, start : start + 3]
            episode.create_dataset(
                f"obs/datagen_info/target_eef_pose/{hand}", data=matrices
            )
        episode.create_dataset("actions", data=actions)
        root = [0, 0, 0, 0, 0, 0, 1] if version == 1 else [0, 0, 0, 1, 0, 0, 0]
        episode.create_dataset(
            "initial_state/articulation/robot/root_pose", data=[root]
        )


def _prepare(source: Path, embodiment: str = "gr1_pink") -> tuple[Path, dict]:
    return _prepare_replay_execution_input(
        source,
        source.parent / "private",
        embodiment=embodiment,
        source_evidence={"trajectory": _replay_input_evidence(source)},
    )


@pytest.mark.parametrize("version", [None, 0, 1])
@pytest.mark.parametrize("quaternion_scale", [1, -3])
def test_consistent_targets_accept_equivalent_quaternion_representations(
    tmp_path: Path, version: int | None, quaternion_scale: int
) -> None:
    source = tmp_path / "recording.hdf5"
    _recording(source, version)
    with h5py.File(source, "a") as dataset:
        for start in (3, 10):
            dataset["data/demo_0/actions"][:, start : start + 4] *= quaternion_scale
    before = source.read_bytes()
    prepared, metadata = _prepare(source)
    with h5py.File(prepared) as execution, h5py.File(source) as original:
        actions = execution["data/demo_0/actions"][:]
        for hand, start in (("left", 3), ("right", 10)):
            expected = original[f"data/demo_0/obs/datagen_info/target_eef_pose/{hand}"][
                :, :3, :3
            ]
            actual = Rotation.from_quat(actions[:, start : start + 4]).as_matrix()
            np.testing.assert_allclose(actual, expected, atol=1e-6)
    assert source.read_bytes() == before
    assert metadata["pose_representation"]["recorded_target_poses"] == {
        "status": "consistent",
        "checked_hand_targets": 6,
        "runtime_outcome_claim": False,
    }


def test_mixed_source_conventions_refuse_before_creating_execution_input(
    tmp_path: Path,
) -> None:
    source = tmp_path / "recording.hdf5"
    _recording(source, None, mixed=True)
    before = source.read_bytes()
    with pytest.raises(IsaacArenaError, match="normalize this source explicitly"):
        _prepare(source)
    assert source.read_bytes() == before
    assert not (tmp_path / "private/replay-execution.hdf5").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "position",
        "rotation",
        "sample_order",
        "bottom_row",
        "nonfinite",
        "shape",
        "group",
    ],
)
def test_malformed_or_misaligned_recorded_targets_refuse(
    tmp_path: Path, mutation: str
) -> None:
    source = tmp_path / "recording.hdf5"
    _recording(source, 1)
    with h5py.File(source, "a") as dataset:
        name = "data/demo_0/obs/datagen_info/target_eef_pose/right"
        matrices = dataset[name][:]
        del dataset[name]
        if mutation == "position":
            matrices[:, 0, 3] += 0.1
        elif mutation == "rotation":
            matrices[:, :3, :3] = np.eye(3)
        elif mutation == "sample_order":
            matrices = matrices[::-1]
        elif mutation == "bottom_row":
            matrices[:, 3, 0] = 1
        elif mutation == "nonfinite":
            matrices[:, 0, 0] = np.nan
        elif mutation == "shape":
            matrices = matrices[:-1]
        if mutation == "group":
            dataset.create_group(name)
        else:
            dataset.create_dataset(name, data=matrices)
    with pytest.raises(IsaacArenaError, match="recorded GR1 Pink target poses"):
        _prepare(source)
    assert not (tmp_path / "private/replay-execution.hdf5").exists()


def test_explicit_source_normalization_preserves_original_hand_commands(
    tmp_path: Path,
) -> None:
    source = tmp_path / "recording.hdf5"
    _recording(source, None, mixed=True)
    with h5py.File(source, "a") as dataset:
        original_actions = dataset["data/demo_0/actions"][:]
        root = dataset["data/demo_0/initial_state/articulation/robot/root_pose"]
        values = root[:]
        values[:, 3:7] = values[:, [4, 5, 6, 3]]
        root[...] = values
        dataset.attrs["format_version"] = 1
    prepared, metadata = _prepare(source)
    with h5py.File(prepared) as execution:
        np.testing.assert_array_equal(
            execution["data/demo_0/actions"], original_actions
        )
    assert metadata["pose_representation"]["conversion"] == "none"
    assert metadata["action_padding_steps"] == 0


def test_joint_controller_does_not_interpret_optional_matrices_as_pink_targets(
    tmp_path: Path,
) -> None:
    source = tmp_path / "recording.hdf5"
    _recording(source, None, mixed=True)
    _, metadata = _prepare(source, "gr1_joint")
    assert "recorded_target_poses" not in metadata["pose_representation"]


def test_one_recorded_hand_target_is_checked_independently(tmp_path: Path) -> None:
    source = tmp_path / "recording.hdf5"
    _recording(source, 1)
    with h5py.File(source, "a") as dataset:
        del dataset["data/demo_0/obs/datagen_info/target_eef_pose/left"]
    _, metadata = _prepare(source)
    assert (
        metadata["pose_representation"]["recorded_target_poses"]["checked_hand_targets"]
        == 3
    )
