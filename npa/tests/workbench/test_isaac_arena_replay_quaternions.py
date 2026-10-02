"""Check legacy Pink replay geometry against an independent rotation implementation."""

from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from npa.workbench.isaac_arena.errors import IsaacArenaError
from npa.workbench.isaac_arena.replay_input import (
    _input_evidence,
    _prepare_replay_execution_input,
)
from npa.workbench.isaac_arena.runtime import IsaacArenaRequest, _prepare_inputs


def _recording(path: Path, version: object = None) -> np.ndarray:
    angles = np.array([[0, 0, 90], [10, 20, 80], [20, 30, 70]])
    rotations = Rotation.from_euler("xyz", angles, degrees=True)
    xyzw = rotations.as_quat()
    quaternions = xyzw if version == 1 else xyzw[:, [3, 0, 1, 2]]
    actions = np.arange(108, dtype=np.float64).reshape(3, 36) / 100
    actions[:, 3:7] = quaternions
    actions[:, 10:14] = quaternions
    with h5py.File(path, "w") as dataset:
        if version is not None:
            dataset.attrs["format_version"] = version
        episode = dataset.create_group("data/demo_0")
        episode.create_dataset("actions", data=actions)
        root = [1, 2, 3, 0, 0, 0, 1] if version == 1 else [1, 2, 3, 1, 0, 0, 0]
        for asset in ("articulation/robot", "rigid_object/box"):
            pose = episode.create_dataset(
                f"initial_state/{asset}/root_pose", data=[root]
            )
            pose.attrs["retained_attribute"] = "example"
        episode.create_dataset(
            "initial_state/articulation/robot/root_velocity", data=[[1] * 6]
        )
        episode.create_dataset(
            "initial_state/articulation/robot/joint_position", data=[[2] * 54]
        )
    return rotations.as_matrix()


def _prepare(source: Path, embodiment: str = "gr1_pink") -> tuple[Path, dict]:
    request = IsaacArenaRequest(
        output_path=str(source.parent / "out"),
        policy_type="replay",
        input_path=str(source),
    )
    evidence = _input_evidence(request, source)
    assert evidence is not None
    return _prepare_replay_execution_input(
        source,
        source.parent / "private",
        source_evidence=evidence,
        embodiment=embodiment,
    )


@pytest.mark.parametrize("version", [None, 0])
def test_legacy_pink_targets_preserve_rotation_and_every_other_command(
    tmp_path: Path, version: int | None
) -> None:
    source = tmp_path / "source.hdf5"
    expected = _recording(source, version)
    source_bytes = source.read_bytes()
    execution, evidence = _prepare(source)
    with h5py.File(source) as original, h5py.File(execution) as prepared:
        before = original["data/demo_0/actions"][:]
        after = prepared["data/demo_0/actions"][:]
        for start in (3, 10):
            actual = Rotation.from_quat(after[:, start : start + 4]).as_matrix()
            np.testing.assert_allclose(actual, expected, atol=1e-14)
            wrong = Rotation.from_quat(before[:, start : start + 4]).as_matrix()
            assert np.linalg.norm(wrong - expected) > 1
        columns = [0, 1, 2, 7, 8, 9, *range(14, 36)]
        np.testing.assert_array_equal(after[:, columns], before[:, columns])
        assert after.shape == before.shape
        assert after.dtype == before.dtype
        assert prepared.attrs["format_version"] == 1
    assert source.read_bytes() == source_bytes
    assert evidence["source_steps"] == evidence["prepared_steps"] == 3
    assert evidence["action_padding_steps"] == 0
    assert evidence["pose_representation"]["conversion"] == "wxyz_to_xyzw_gr1_pink"
    assert evidence["pose_representation"]["action_quaternion_slices"] == [
        [3, 7],
        [10, 14],
    ]


def test_legacy_initial_poses_convert_before_disabling_native_conversion(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.hdf5"
    _recording(source)
    execution, evidence = _prepare(source)
    with h5py.File(source) as original, h5py.File(execution) as prepared:
        for asset in ("articulation/robot", "rigid_object/box"):
            key = f"data/demo_0/initial_state/{asset}/root_pose"
            np.testing.assert_array_equal(
                prepared[key][..., :3], original[key][..., :3]
            )
            actual = Rotation.from_quat(prepared[key][..., 3:7]).as_matrix()
            np.testing.assert_allclose(actual, [np.eye(3)], atol=1e-14)
            assert dict(prepared[key].attrs) == dict(original[key].attrs)
        for field in ("root_velocity", "joint_position"):
            key = f"data/demo_0/initial_state/articulation/robot/{field}"
            np.testing.assert_array_equal(prepared[key], original[key])
        assert prepared.attrs["format_version"] == 1
    assert evidence["pose_representation"]["initial_root_poses_converted"] == 2


def test_native_xyzw_pink_recording_is_not_converted_twice(tmp_path: Path) -> None:
    source = tmp_path / "source.hdf5"
    _recording(source, 1)
    execution, evidence = _prepare(source)
    with h5py.File(source) as original, h5py.File(execution) as prepared:
        for key in ("actions", "initial_state/articulation/robot/root_pose"):
            np.testing.assert_array_equal(
                prepared[f"data/demo_0/{key}"], original[f"data/demo_0/{key}"]
            )
        assert prepared.attrs["format_version"] == 1
    assert evidence["pose_representation"]["conversion"] == "none"


def test_same_width_joint_controller_does_not_receive_pink_conversion(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.hdf5"
    _recording(source)
    execution, evidence = _prepare(source, "gr1_joint")
    with h5py.File(source) as original, h5py.File(execution) as prepared:
        for key in ("actions", "initial_state/articulation/robot/root_pose"):
            np.testing.assert_array_equal(
                prepared[f"data/demo_0/{key}"], original[f"data/demo_0/{key}"]
            )
        assert "format_version" not in prepared.attrs
    assert evidence["pose_representation"]["conversion"] == "upstream_root_pose_only"


@pytest.mark.parametrize("version", [-1, 2, True, 1.0, "1"])
def test_unknown_or_malformed_version_is_rejected_without_guessing(
    tmp_path: Path, version: object
) -> None:
    source = tmp_path / "source.hdf5"
    _recording(source)
    with h5py.File(source, "a") as dataset:
        dataset.attrs["format_version"] = version
    with pytest.raises(IsaacArenaError, match="format_version"):
        _prepare(source)


@pytest.mark.parametrize("width", [14, 35, 37])
def test_pink_conversion_rejects_another_action_layout(
    tmp_path: Path, width: int
) -> None:
    source = tmp_path / "source.hdf5"
    _recording(source)
    with h5py.File(source, "a") as dataset:
        del dataset["data/demo_0/actions"]
        dataset.create_dataset("data/demo_0/actions", data=np.ones((3, width)))
    with pytest.raises(IsaacArenaError, match="36 action columns"):
        _prepare(source)


@pytest.mark.parametrize(
    "key", ["actions", "initial_state/articulation/robot/root_pose"]
)
def test_zero_norm_pose_is_rejected_before_native_rotation_math(
    tmp_path: Path, key: str
) -> None:
    source = tmp_path / "source.hdf5"
    _recording(source)
    with h5py.File(source, "a") as dataset:
        dataset[f"data/demo_0/{key}"][:, 3:7] = 0
    with pytest.raises(IsaacArenaError, match="nonzero quaternions"):
        _prepare(source)


@pytest.mark.parametrize("shape", [(7,), (2, 7), (1, 6)])
def test_pink_root_pose_requires_the_single_environment_layout(
    tmp_path: Path, shape: tuple[int, ...]
) -> None:
    source = tmp_path / "source.hdf5"
    _recording(source)
    with h5py.File(source, "a") as dataset:
        key = "data/demo_0/initial_state/articulation/robot/root_pose"
        del dataset[key]
        dataset.create_dataset(key, data=np.ones(shape))
    with pytest.raises(IsaacArenaError, match="single-environment shape"):
        _prepare(source)


def test_pink_recording_requires_the_initial_robot_pose(tmp_path: Path) -> None:
    source = tmp_path / "source.hdf5"
    _recording(source)
    with h5py.File(source, "a") as dataset:
        del dataset["data/demo_0/initial_state/articulation/robot/root_pose"]
    with pytest.raises(IsaacArenaError, match="initial robot root_pose"):
        _prepare(source)


@pytest.mark.parametrize(
    "embodiment,converted", [("", True), ("gr1_pink", True), ("gr1_joint", False)]
)
def test_runtime_resolves_environment_default_before_selecting_conversion(
    tmp_path: Path, embodiment: str, converted: bool
) -> None:
    source = tmp_path / "source.hdf5"
    _recording(source)
    request = IsaacArenaRequest(
        output_path=str(tmp_path / "out"),
        environment="gr1_open_microwave",
        policy_type="replay",
        input_path=str(source),
        embodiment=embodiment,
    )
    _, evidence = _prepare_inputs(request, tmp_path / "run-private")
    assert evidence is not None
    actual = evidence["execution"]["pose_representation"]["conversion"]
    assert (actual == "wxyz_to_xyzw_gr1_pink") is converted
