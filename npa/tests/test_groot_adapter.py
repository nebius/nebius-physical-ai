from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from npa.adapter.groot import GR00TAdapterError, groot_to_lerobot, lerobot_to_groot


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


G1_JOINT_NAMES_26 = [
    "kLeftShoulderPitch",
    "kLeftShoulderRoll",
    "kLeftShoulderYaw",
    "kLeftElbow",
    "kLeftWristRoll",
    "kLeftWristPitch",
    "kLeftWristYaw",
    "kRightShoulderPitch",
    "kRightShoulderRoll",
    "kRightShoulderYaw",
    "kRightElbow",
    "kRightWristRoll",
    "kRightWristPitch",
    "kRightWristYaw",
    "kLeftHandPinky",
    "kLeftHandRing",
    "kLeftHandMiddle",
    "kLeftHandIndex",
    "kLeftHandThumbBend",
    "kLeftHandThumbRotation",
    "kRightHandPinky",
    "kRightHandRing",
    "kRightHandMiddle",
    "kRightHandIndex",
    "kRightHandThumbBend",
    "kRightHandThumbRotation",
]


@pytest.fixture()
def standard_lerobot_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "lerobot"
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)

    data = pa.table(
        {
            "observation.state": pa.array(
                [
                    [1.0, 2.0, 0.0],
                    [1.1, 2.1, 1.0],
                    [3.0, 4.0, 0.0],
                    [3.1, 4.1, 1.0],
                ],
                type=pa.list_(pa.float32(), 3),
            ),
            "action": pa.array(
                [[0.1, 0.2, 0.0], [0.2, 0.3, 1.0], [0.3, 0.4, 0.0], [0.4, 0.5, 1.0]],
                type=pa.list_(pa.float32(), 3),
            ),
            "episode_index": pa.array([0, 0, 1, 1], type=pa.int64()),
            "frame_index": pa.array([0, 1, 0, 1], type=pa.int64()),
            "timestamp": pa.array([0.0, 0.05, 0.0, 0.05], type=pa.float32()),
            "index": pa.array([0, 1, 2, 3], type=pa.int64()),
            "task_index": pa.array([0, 0, 0, 0], type=pa.int64()),
        }
    )
    pq.write_table(data, root / "data" / "chunk-000" / "file-000.parquet")

    episodes = pa.table(
        {
            "episode_index": pa.array([0, 1], type=pa.int64()),
            "data/chunk_index": pa.array([0, 0], type=pa.int64()),
            "data/file_index": pa.array([0, 0], type=pa.int64()),
            "dataset_from_index": pa.array([0, 2], type=pa.int64()),
            "dataset_to_index": pa.array([2, 4], type=pa.int64()),
            "length": pa.array([2, 2], type=pa.int64()),
            "tasks": pa.array([["pick"], ["pick"]]),
            "meta/episodes/chunk_index": pa.array([0, 0], type=pa.int64()),
            "meta/episodes/file_index": pa.array([0, 0], type=pa.int64()),
        }
    )
    pq.write_table(episodes, root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    pq.write_table(
        pa.table(
            {
                "task_index": pa.array([0], type=pa.int64()),
                "task": pa.array(["pick"], type=pa.string()),
            }
        ),
        root / "meta" / "tasks.parquet",
    )
    _write_json(
        root / "meta" / "info.json",
        {
            "codebase_version": "v3.0",
            "robot_type": "testbot",
            "total_episodes": 2,
            "total_frames": 4,
            "total_tasks": 1,
            "chunks_size": 1000,
            "fps": 20,
            "splits": {"train": "0:2"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "features": {
                "observation.state": {
                    "dtype": "float32",
                    "shape": [3],
                    "names": None,
                },
                "action": {
                    "dtype": "float32",
                    "shape": [3],
                    "names": None,
                },
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None},
            },
        },
    )
    _write_json(
        root / "meta" / "stats.json",
        {
            "observation.state": {
                "min": [1.0, 2.0, 0.0],
                "max": [3.1, 4.1, 1.0],
                "mean": [2.05, 3.05, 0.5],
                "std": [1.0, 1.0, 0.5],
            },
            "action": {
                "min": [0.1, 0.2, 0.0],
                "max": [0.4, 0.5, 1.0],
                "mean": [0.25, 0.35, 0.5],
                "std": [0.1, 0.1, 0.5],
            },
        },
    )
    return root


@pytest.fixture()
def cartesian_lerobot_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "cartesian-lerobot"
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)

    data = pa.table(
        {
            "observation.state": pa.array(
                [
                    [0.0, -0.7, 0.0, -2.3, 0.0, 1.5, 0.7, 0.04, 0.04, 0.04],
                    [0.1, -0.6, 0.1, -2.2, 0.1, 1.4, 0.8, 0.04, 0.04, 0.04],
                ],
                type=pa.list_(pa.float32(), 10),
            ),
            "action": pa.array(
                [[0.01, -0.02, 0.03, 0.1], [0.02, -0.01, 0.04, 0.2]],
                type=pa.list_(pa.float32(), 4),
            ),
            "episode_index": pa.array([0, 0], type=pa.int64()),
            "frame_index": pa.array([0, 1], type=pa.int64()),
            "timestamp": pa.array([0.0, 0.05], type=pa.float32()),
            "index": pa.array([0, 1], type=pa.int64()),
            "task_index": pa.array([0, 0], type=pa.int64()),
        }
    )
    pq.write_table(data, root / "data" / "chunk-000" / "file-000.parquet")
    pq.write_table(
        pa.table(
            {
                "episode_index": pa.array([0], type=pa.int64()),
                "data/chunk_index": pa.array([0], type=pa.int64()),
                "data/file_index": pa.array([0], type=pa.int64()),
                "dataset_from_index": pa.array([0], type=pa.int64()),
                "dataset_to_index": pa.array([2], type=pa.int64()),
                "length": pa.array([2], type=pa.int64()),
                "tasks": pa.array([["pick"]]),
                "meta/episodes/chunk_index": pa.array([0], type=pa.int64()),
                "meta/episodes/file_index": pa.array([0], type=pa.int64()),
            }
        ),
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "task_index": pa.array([0], type=pa.int64()),
                "task": pa.array(["pick"], type=pa.string()),
            }
        ),
        root / "meta" / "tasks.parquet",
    )
    _write_json(
        root / "meta" / "info.json",
        {
            "codebase_version": "v3.0",
            "robot_type": "genesis-franka",
            "total_episodes": 1,
            "total_frames": 2,
            "total_tasks": 1,
            "chunks_size": 1000,
            "fps": 20,
            "splits": {"train": "0:1"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "features": {
                "observation.state": {"dtype": "float32", "shape": [10], "names": None},
                "action": {"dtype": "float32", "shape": [4], "names": None},
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None},
            },
        },
    )
    _write_json(root / "meta" / "stats.json", {})
    return root


@pytest.fixture()
def real_g1_lerobot_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "real-g1-lerobot"
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)

    state_rows = [
        [float(idx) for idx in range(26)],
        [float(idx) + 0.5 for idx in range(26)],
    ]
    action_rows = [
        [float(idx) * 0.1 for idx in range(26)],
        [float(idx) * 0.1 + 0.05 for idx in range(26)],
    ]
    data = pa.table(
        {
            "observation.state": pa.array(state_rows, type=pa.list_(pa.float32(), 26)),
            "action": pa.array(action_rows, type=pa.list_(pa.float32(), 26)),
            "episode_index": pa.array([0, 0], type=pa.int64()),
            "frame_index": pa.array([0, 1], type=pa.int64()),
            "timestamp": pa.array([0.0, 0.05], type=pa.float32()),
            "index": pa.array([0, 1], type=pa.int64()),
            "task_index": pa.array([0, 0], type=pa.int64()),
        }
    )
    pq.write_table(data, root / "data" / "chunk-000" / "file-000.parquet")
    pq.write_table(
        pa.table(
            {
                "episode_index": pa.array([0], type=pa.int64()),
                "data/chunk_index": pa.array([0], type=pa.int64()),
                "data/file_index": pa.array([0], type=pa.int64()),
                "dataset_from_index": pa.array([0], type=pa.int64()),
                "dataset_to_index": pa.array([2], type=pa.int64()),
                "length": pa.array([2], type=pa.int64()),
                "tasks": pa.array([["pick cube"]]),
                "meta/episodes/chunk_index": pa.array([0], type=pa.int64()),
                "meta/episodes/file_index": pa.array([0], type=pa.int64()),
                "videos/observation.images.color_0/chunk_index": pa.array([0], type=pa.int64()),
                "videos/observation.images.color_0/file_index": pa.array([0], type=pa.int64()),
                "videos/observation.images.color_0/from_timestamp": pa.array([0.0], type=pa.float64()),
                "videos/observation.images.color_0/to_timestamp": pa.array([0.1], type=pa.float64()),
            }
        ),
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "task_index": pa.array([0], type=pa.int64()),
                "task": pa.array(["pick cube"], type=pa.string()),
            }
        ),
        root / "meta" / "tasks.parquet",
    )
    _write_json(
        root / "meta" / "info.json",
        {
            "codebase_version": "v3.0",
            "robot_type": "Unitree_G1_Inspire_FTP",
            "total_episodes": 1,
            "total_frames": 2,
            "total_tasks": 1,
            "chunks_size": 1000,
            "fps": 20,
            "splits": {"train": "0:1"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": {
                "observation.state": {
                    "dtype": "float32",
                    "shape": [26],
                    "names": [G1_JOINT_NAMES_26],
                },
                "action": {
                    "dtype": "float32",
                    "shape": [26],
                    "names": [G1_JOINT_NAMES_26],
                },
                "observation.images.color_0": {
                    "dtype": "video",
                    "shape": [480, 640, 3],
                    "names": ["height", "width", "channel"],
                },
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None},
            },
        },
    )
    _write_json(
        root / "meta" / "stats.json",
        {
            "observation.state": {
                "min": [0.0] * 26,
                "max": [1.0] * 26,
                "mean": [0.5] * 26,
                "std": [1.0] * 26,
            },
            "action": {
                "min": [0.0] * 26,
                "max": [1.0] * 26,
                "mean": [0.5] * 26,
                "std": [1.0] * 26,
            },
        },
    )
    return root


def test_lerobot_to_groot_writes_modality_and_episode_parquets(
    standard_lerobot_dataset: Path,
    tmp_path: Path,
) -> None:
    out = lerobot_to_groot(
        standard_lerobot_dataset,
        tmp_path / "groot",
        robot_embodiment="NEW_EMBODIMENT",
    )

    modality = json.loads((out / "meta" / "modality.json").read_text())
    assert modality["state"]["single_arm"] == {"start": 0, "end": 2}
    assert modality["state"]["gripper"] == {"start": 2, "end": 3}
    assert modality["action"]["single_arm"] == {"start": 0, "end": 2}
    assert modality["action"]["gripper"] == {"start": 2, "end": 3}
    assert modality["annotation"]["human.task_description"]["original_key"] == "task_index"
    assert (out / "data" / "chunk-000" / "episode_000000.parquet").exists()
    assert (out / "data" / "chunk-000" / "episode_000001.parquet").exists()
    assert '"robot_embodiment": "NEW_EMBODIMENT"' in (
        out / "meta" / "npa_groot_adapter.json"
    ).read_text()


def test_lerobot_to_groot_detects_cartesian_actions_and_writes_config(
    cartesian_lerobot_dataset: Path,
    tmp_path: Path,
) -> None:
    out = lerobot_to_groot(
        cartesian_lerobot_dataset,
        tmp_path / "groot-cartesian",
        robot_embodiment="NEW_EMBODIMENT",
    )

    modality = json.loads((out / "meta" / "modality.json").read_text())
    assert modality["state"]["joint_position"] == {"start": 0, "end": 10}
    assert modality["action"] == {
        "x": {"start": 0, "end": 1},
        "y": {"start": 1, "end": 2},
        "z": {"start": 2, "end": 3},
        "gripper": {"start": 3, "end": 4},
    }
    generated_config = out / "meta" / "npa_groot_modality_config.py"
    assert generated_config.exists()
    config_text = generated_config.read_text()
    assert "ActionRepresentation.ABSOLUTE" in config_text
    assert 'embodiment_tag = EmbodimentTag.resolve("NEW_EMBODIMENT")' in config_text

    manifest = json.loads((out / "meta" / "npa_groot_adapter.json").read_text())
    assert manifest["action_space"] == "cartesian_xyz_gripper"
    assert manifest["state_dim"] == 10
    assert manifest["action_dim"] == 4


def test_lerobot_to_groot_rejects_cartesian_dims_for_builtin_joint_tag(
    cartesian_lerobot_dataset: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(GR00TAdapterError, match="Use --embodiment-tag NEW_EMBODIMENT"):
        lerobot_to_groot(
            cartesian_lerobot_dataset,
            tmp_path / "groot-cartesian",
            robot_embodiment="UNITREE_G1",
        )


def test_lerobot_to_groot_real_g1_writes_canonical_modality(
    real_g1_lerobot_dataset: Path,
    tmp_path: Path,
) -> None:
    out = lerobot_to_groot(
        real_g1_lerobot_dataset,
        tmp_path / "groot-real-g1",
        robot_embodiment="REAL_G1",
    )

    modality = json.loads((out / "meta" / "modality.json").read_text())
    assert list(modality["state"]) == [
        "left_wrist_eef_9d",
        "right_wrist_eef_9d",
        "left_hand",
        "right_hand",
        "left_arm",
        "right_arm",
        "waist",
    ]
    assert list(modality["action"]) == [
        "left_wrist_eef_9d",
        "right_wrist_eef_9d",
        "left_hand",
        "right_hand",
        "left_arm",
        "right_arm",
        "waist",
        "base_height_command",
        "navigate_command",
    ]
    assert modality["video"] == {
        "ego_view": {"original_key": "observation.images.color_0"}
    }
    assert modality["state"]["left_arm"] == {
        "start": 0,
        "end": 7,
        "original_key": "observation.state",
    }
    assert modality["state"]["left_hand"] == {
        "start": 0,
        "end": 7,
        "original_key": "observation.real_g1.left_hand",
    }
    assert modality["state"]["left_wrist_eef_9d"] == {
        "start": 0,
        "end": 9,
        "original_key": "observation.real_g1.left_wrist_eef_9d",
    }

    info = json.loads((out / "meta" / "info.json").read_text())
    assert info["features"]["observation.real_g1.left_wrist_eef_9d"]["shape"] == [9]
    assert info["features"]["observation.real_g1.left_hand"]["shape"] == [7]
    assert info["features"]["action.real_g1.navigate_command"]["shape"] == [3]

    episode = pq.read_table(out / "data" / "chunk-000" / "episode_000000.parquet")
    assert "observation.real_g1.left_wrist_eef_9d" in episode.column_names
    assert "observation.real_g1.left_hand" in episode.column_names
    assert "action.real_g1.navigate_command" in episode.column_names
    assert episode["observation.real_g1.left_wrist_eef_9d"][0].as_py() == [
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ]
    assert episode["observation.real_g1.left_hand"][0].as_py() == [
        14.0,
        15.0,
        16.0,
        17.0,
        18.0,
        19.0,
        0.0,
    ]
    assert episode["action.real_g1.navigate_command"][0].as_py() == [0.0] * 3

    stats = json.loads((out / "meta" / "stats.json").read_text())
    assert stats["observation.real_g1.left_wrist_eef_9d"]["mean"] == [
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ]
    assert stats["observation.real_g1.left_hand"]["min"] == [
        14.0,
        15.0,
        16.0,
        17.0,
        18.0,
        19.0,
        0.0,
    ]
    assert stats["action.real_g1.navigate_command"]["std"] == [1.0] * 3


def test_lerobot_to_groot_ignores_appledouble_sidecar_parquets(
    standard_lerobot_dataset: Path,
    tmp_path: Path,
) -> None:
    (standard_lerobot_dataset / "data" / "chunk-000" / "._file-000.parquet").write_text(
        "not a parquet file"
    )
    (
        standard_lerobot_dataset
        / "meta"
        / "episodes"
        / "chunk-000"
        / "._file-000.parquet"
    ).write_text("not a parquet file")

    out = lerobot_to_groot(
        standard_lerobot_dataset,
        tmp_path / "groot-sidecar",
        robot_embodiment="NEW_EMBODIMENT",
    )

    assert (out / "data" / "chunk-000" / "episode_000000.parquet").exists()
    assert (out / "data" / "chunk-000" / "episode_000001.parquet").exists()


def test_groot_to_lerobot_restores_standard_metadata(
    standard_lerobot_dataset: Path,
    tmp_path: Path,
) -> None:
    groot = lerobot_to_groot(standard_lerobot_dataset, tmp_path / "groot")

    restored = groot_to_lerobot(groot, tmp_path / "restored")

    info = json.loads((restored / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v3.0"
    assert info["data_path"] == "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    assert (restored / "meta" / "tasks.parquet").exists()
    assert (restored / "meta" / "episodes" / "chunk-000" / "file-000.parquet").exists()
    table = pq.read_table(restored / "data" / "chunk-000" / "file-000.parquet")
    assert table.num_rows == 4


def test_lerobot_groot_round_trip_preserves_rows(
    standard_lerobot_dataset: Path,
    tmp_path: Path,
) -> None:
    groot = lerobot_to_groot(standard_lerobot_dataset, tmp_path / "groot")
    restored = groot_to_lerobot(groot, tmp_path / "restored")
    roundtrip = lerobot_to_groot(restored, tmp_path / "roundtrip")

    original_rows = sum(
        pq.read_table(path).num_rows
        for path in sorted((groot / "data").rglob("episode_*.parquet"))
    )
    roundtrip_rows = sum(
        pq.read_table(path).num_rows
        for path in sorted((roundtrip / "data").rglob("episode_*.parquet"))
    )
    assert roundtrip_rows == original_rows == 4
    assert (roundtrip / "meta" / "tasks.jsonl").read_text() == (
        groot / "meta" / "tasks.jsonl"
    ).read_text()


def test_groot_output_loadable_by_gr00t_loader_when_installed(
    standard_lerobot_dataset: Path,
    tmp_path: Path,
) -> None:
    loader_mod = pytest.importorskip("gr00t.data.dataset.lerobot_episode_loader")
    types_mod = pytest.importorskip("gr00t.data.types")
    out = lerobot_to_groot(standard_lerobot_dataset, tmp_path / "groot")

    loader = loader_mod.LeRobotEpisodeLoader(
        dataset_path=out,
        modality_configs={
            "state": types_mod.ModalityConfig(
                delta_indices=[0],
                modality_keys=["single_arm", "gripper"],
            ),
            "action": types_mod.ModalityConfig(
                delta_indices=[0],
                modality_keys=["single_arm", "gripper"],
            ),
            "language": types_mod.ModalityConfig(
                delta_indices=[0],
                modality_keys=["annotation.human.task_description"],
            ),
        },
    )

    episode = loader[0]
    assert "state.single_arm" in episode.columns
    assert "action.gripper" in episode.columns
    assert "language.annotation.human.task_description" in episode.columns
