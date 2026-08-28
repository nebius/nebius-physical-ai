from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from npa.adapter.isaac_lab_lerobot import (
    G1_STATE_DIM,
    G1_STATE_NAMES_43,
    IsaacLabLeRobotError,
    LeRobotFeatureSpec,
    WORKSPACE_VIEW_KEY,
    convert,
    discover_episodes,
)


def _write_episode(root: Path, index: int, frames: int = 3) -> None:
    episode = root / f"episode_{index:06d}"
    episode.mkdir(parents=True)
    state = np.arange(frames * G1_STATE_DIM, dtype=np.float32).reshape(frames, G1_STATE_DIM)
    actions = state + 0.25
    np.save(episode / "state.npy", state)
    np.save(episode / "actions.npy", actions)


def test_recorded_isaac_lab_sample_converts_to_lerobot(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_episode(raw, 0, frames=2)
    _write_episode(raw, 1, frames=3)
    (raw / "meta.json").write_text(
        json.dumps(
            {
                "format": "npa_isaac_lab_g1_rollout_v1",
                "task": "Isaac-Velocity-Flat-G1-v0",
                "robot_type": "unitree_g1",
                "state_names": G1_STATE_NAMES_43,
                "action_names": G1_STATE_NAMES_43,
            }
        )
    )

    out = convert(raw, tmp_path / "lerobot", fps=50)

    info = json.loads((out / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v3.0"
    assert info["robot_type"] == "unitree_g1"
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 5
    assert info["features"]["observation.state"]["shape"] == [G1_STATE_DIM]
    assert info["features"]["observation.state"]["names"] == [G1_STATE_NAMES_43]
    assert "video_path" not in info

    data = pq.read_table(out / "data" / "chunk-000" / "file-000.parquet")
    assert data.num_rows == 5
    assert data.schema.field("observation.state").type.list_size == G1_STATE_DIM
    assert data["episode_index"].to_pylist() == [0, 0, 1, 1, 1]
    assert data["frame_index"].to_pylist() == [0, 1, 0, 1, 2]

    episodes = pq.read_table(out / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    assert episodes["length"].to_pylist() == [2, 3]
    tasks = pq.read_table(out / "meta" / "tasks.parquet")
    assert tasks["task"].to_pylist() == ["Isaac-Velocity-Flat-G1-v0"]


def test_convert_rejects_state_action_length_mismatch(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    episode = raw / "episode_000000"
    episode.mkdir(parents=True)
    np.save(episode / "state.npy", np.zeros((3, G1_STATE_DIM), dtype=np.float32))
    np.save(episode / "actions.npy", np.zeros((2, G1_STATE_DIM), dtype=np.float32))

    with pytest.raises(IsaacLabLeRobotError, match="length mismatch"):
        convert(raw, tmp_path / "out")


def test_discover_episodes_rejects_empty_input(tmp_path: Path) -> None:
    with pytest.raises(IsaacLabLeRobotError, match="No episode_"):
        discover_episodes(tmp_path)


def test_convert_with_custom_feature_spec(tmp_path):
    from npa.adapter.isaac_lab_lerobot import LeRobotFeatureSpec

    spec = LeRobotFeatureSpec(
        state_names=[f"joint_{i}" for i in range(9)],
        action_names=[f"act_{i}" for i in range(7)],
        robot_type="franka_panda",
    )
    assert spec.state_dim == 9
    assert spec.action_dim == 7

    input_dir = tmp_path / "raw"
    episode = input_dir / "episode_000000"
    episode.mkdir(parents=True)
    frames = 4
    np.save(episode / "state.npy", np.zeros((frames, 9), dtype=np.float32))
    np.save(episode / "actions.npy", np.ones((frames, 7), dtype=np.float32))

    output_dir = tmp_path / "lerobot"
    convert(input_dir, output_dir, spec=spec)

    info = json.loads((output_dir / "meta" / "info.json").read_text())
    assert info["robot_type"] == "franka_panda"
    assert info["features"]["observation.state"]["shape"] == [9]
    assert info["features"]["action"]["shape"] == [7]
    assert info["features"]["observation.state"]["names"] == [spec.state_names]

    import pyarrow.parquet as pq

    data = pq.read_table(output_dir / "data" / "chunk-000" / "file-000.parquet")
    assert data.schema.field("observation.state").type.list_size == 9
    assert data.schema.field("action").type.list_size == 7


def test_convert_with_spec_rejects_mismatched_dims(tmp_path):
    from npa.adapter.isaac_lab_lerobot import LeRobotFeatureSpec

    spec = LeRobotFeatureSpec(
        state_names=["a", "b", "c"],
        action_names=["a", "b", "c"],
        robot_type="tiny_bot",
    )
    input_dir = tmp_path / "raw"
    episode = input_dir / "episode_000000"
    episode.mkdir(parents=True)
    np.save(episode / "state.npy", np.zeros((2, 5), dtype=np.float32))
    np.save(episode / "actions.npy", np.zeros((2, 5), dtype=np.float32))

    with pytest.raises(IsaacLabLeRobotError, match="tiny_bot"):
        convert(input_dir, tmp_path / "lerobot", spec=spec)


def test_convert_carries_real_isaac_rgb_and_provenance(tmp_path, mocker) -> None:
    spec = LeRobotFeatureSpec(
        state_names=["cart", "pole"],
        action_names=["force"],
        robot_type="cartpole",
    )
    raw = tmp_path / "raw"
    for episode_index, frame_count in enumerate((3, 2)):
        episode = raw / f"episode_{episode_index:06d}"
        episode.mkdir(parents=True)
        np.save(episode / "state.npy", np.zeros((frame_count, 2), dtype=np.float32))
        np.save(episode / "actions.npy", np.zeros((frame_count, 1), dtype=np.float32))
        pixels = np.zeros((frame_count, 24, 32, 3), dtype=np.uint8)
        pixels[..., 0] = 20 + episode_index
        pixels[:, 5:10, 8:16, 1] = 220
        np.save(episode / "rgb.npy", pixels)
    (raw / "meta.json").write_text(
        json.dumps(
            {
                "task": "Isaac-Cartpole-v0",
                "runtime_version": "3.0.0b2.post1",
                "policy_loaded": True,
                "checkpoint_sha256": "a" * 64,
                "renderer": "isaac_sim_rgb_array",
            }
        )
    )

    def fake_encode(frames: np.ndarray, output_path: Path, *, fps: int) -> None:
        assert frames.dtype == np.uint8
        assert frames.shape[1:] == (24, 32, 3)
        assert fps == 50
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"encoded-isaac-rgb")

    mocker.patch("npa.adapter.isaac_lab_lerobot._encode_video", side_effect=fake_encode)
    out = convert(raw, tmp_path / "lerobot", spec=spec)

    info = json.loads((out / "meta" / "info.json").read_text())
    assert info["features"][WORKSPACE_VIEW_KEY]["dtype"] == "video"
    assert info["features"][WORKSPACE_VIEW_KEY]["shape"] == [24, 32, 3]
    assert info["visual_provenance"] == {
        "source": "isaac_sim_rgb_array",
        "genuine_simulator_pixels": True,
        "synchronized_timeline": "episode_index/frame_index/timestamp",
        "frame_count": 5,
        "dimensions": [24, 32, 3],
        "task": "Isaac-Cartpole-v0",
        "runtime_version": "3.0.0b2.post1",
        "policy_loaded": True,
        "checkpoint_sha256": "a" * 64,
        "renderer": "isaac_sim_rgb_array",
    }
    episodes = pq.read_table(
        out / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    )
    assert episodes[f"videos/{WORKSPACE_VIEW_KEY}/file_index"].to_pylist() == [0, 1]


def test_convert_rejects_partial_or_unsynchronized_rgb(tmp_path) -> None:
    spec = LeRobotFeatureSpec(
        state_names=["cart", "pole"],
        action_names=["force"],
        robot_type="cartpole",
    )
    raw = tmp_path / "raw"
    for episode_index in range(2):
        episode = raw / f"episode_{episode_index:06d}"
        episode.mkdir(parents=True)
        np.save(episode / "state.npy", np.zeros((3, 2), dtype=np.float32))
        np.save(episode / "actions.npy", np.zeros((3, 1), dtype=np.float32))
    np.save(raw / "episode_000000" / "rgb.npy", np.zeros((3, 8, 8, 3), dtype=np.uint8))

    with pytest.raises(IsaacLabLeRobotError, match="cover every episode"):
        convert(raw, tmp_path / "partial", spec=spec)

    np.save(raw / "episode_000001" / "rgb.npy", np.zeros((2, 8, 8, 3), dtype=np.uint8))
    with pytest.raises(IsaacLabLeRobotError, match="RGB/state length mismatch"):
        convert(raw, tmp_path / "unsynchronized", spec=spec)
