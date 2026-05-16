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
