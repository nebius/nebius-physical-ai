"""Test SimToLeRobot adapter with dummy numpy data."""

from __future__ import annotations

import json
import shutil
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from npa.adapter import sim_to_lerobot
from npa.adapter.sim_to_lerobot import (
    AdapterError,
    _build_data_schema,
    _compute_feature_stats,
    _write_episodes_parquet,
    _write_tasks_parquet,
    convert,
    discover_episodes,
    encode_video,
)


N_EPISODES = 5
N_TIMESTEPS = 30
IMG_H, IMG_W = 480, 640
N_STATE_DIM = 10  # 9 joint positions + 1 gripper state
N_ACTIONS = 8
FPS = 20


@pytest.fixture()
def demo_dir(tmp_path: Path) -> Path:
    """Create a dummy demo directory with random numpy arrays."""
    for ep_idx in range(N_EPISODES):
        ep_dir = tmp_path / f"episode_{ep_idx:04d}"
        ep_dir.mkdir()
        np.save(
            ep_dir / "obs_workspace.npy",
            np.random.randint(0, 256, (N_TIMESTEPS, IMG_H, IMG_W, 3), dtype=np.uint8),
        )
        np.save(
            ep_dir / "obs_wrist.npy",
            np.random.randint(0, 256, (N_TIMESTEPS, IMG_H, IMG_W, 3), dtype=np.uint8),
        )
        np.save(
            ep_dir / "state.npy",
            np.random.randn(N_TIMESTEPS, N_STATE_DIM).astype(np.float32),
        )
        np.save(
            ep_dir / "actions.npy",
            np.random.randn(N_TIMESTEPS, N_ACTIONS).astype(np.float32),
        )
    return tmp_path


@pytest.fixture()
def output_dir(tmp_path: Path) -> Path:
    out = tmp_path / "dataset_output"
    out.mkdir()
    return out


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


needs_ffmpeg = pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg not installed")


class TestDiscoverEpisodes:
    def test_finds_episodes(self, demo_dir: Path) -> None:
        episodes = discover_episodes(demo_dir)
        assert len(episodes) == N_EPISODES
        assert all(d.name.startswith("episode_") for d in episodes)

    def test_sorted_order(self, demo_dir: Path) -> None:
        episodes = discover_episodes(demo_dir)
        names = [d.name for d in episodes]
        assert names == sorted(names)

    def test_empty_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(AdapterError, match="No episode_"):
            discover_episodes(tmp_path)


class TestAdapterHelpers:
    def test_compute_feature_stats_for_vectors(self) -> None:
        stats = _compute_feature_stats(
            [
                np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                np.array([[5.0, 6.0]], dtype=np.float32),
            ]
        )

        assert stats["min"] == [1.0, 2.0]
        assert stats["max"] == [5.0, 6.0]
        assert stats["mean"] == [3.0, 4.0]
        assert stats["count"] == [3]

    def test_compute_feature_stats_for_1d_arrays(self) -> None:
        stats = _compute_feature_stats(
            [
                np.array([1, 2, 3], dtype=np.int64),
                np.array([4], dtype=np.int64),
            ]
        )

        assert stats["min"] == [1.0]
        assert stats["max"] == [4.0]
        assert stats["count"] == [4]

    def test_compute_feature_stats_for_video_normalizes_channels(self) -> None:
        frames = np.array(
            [
                [
                    [[0, 128, 255], [255, 128, 0]],
                    [[0, 0, 0], [255, 255, 255]],
                ]
            ],
            dtype=np.uint8,
        )

        stats = _compute_feature_stats([frames], is_video=True)

        assert stats["min"][0][0][0] == 0.0
        assert stats["max"][0][0][0] == 1.0
        assert stats["max"][2][0][0] == 1.0
        assert stats["count"] == [1]

    def test_build_data_schema_has_fixed_size_lists(self) -> None:
        schema = _build_data_schema(n_state=10, n_actions=8)

        assert schema.field("observation.state").type.list_size == 10
        assert schema.field("action").type.list_size == 8
        assert schema.field("timestamp").type == pa.float32()

    def test_write_tasks_parquet_creates_parent_dirs(self, tmp_path: Path) -> None:
        out = tmp_path / "nested" / "tasks.parquet"

        _write_tasks_parquet("Do the task", out)

        table = pq.read_table(out)
        assert table.column("task_index").to_pylist() == [0]
        assert table.column("task").to_pylist() == ["Do the task"]

    def test_write_episodes_parquet_empty_rows_noops(self, tmp_path: Path) -> None:
        out = tmp_path / "episodes.parquet"

        _write_episodes_parquet([], out)

        assert not out.exists()

    def test_write_episodes_parquet_preserves_dynamic_columns(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "meta" / "episodes.parquet"

        _write_episodes_parquet(
            [
                {
                    "episode_index": 0,
                    "length": 3,
                    "tasks": ["pick"],
                    "score": 1.5,
                    "label": "episode-zero",
                }
            ],
            out,
        )

        table = pq.read_table(out)
        assert table.column("episode_index").to_pylist() == [0]
        assert table.column("tasks").to_pylist() == [["pick"]]
        assert table.column("label").to_pylist() == ["episode-zero"]

    def test_encode_video_rejects_non_rgb_frames(self, tmp_path: Path) -> None:
        frames = np.zeros((2, 4, 4), dtype=np.uint8)

        with pytest.raises(AdapterError, match="Expected"):
            encode_video(frames, tmp_path / "out.mp4", fps=20)

    def test_encode_video_maps_ffmpeg_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        frames = np.zeros((2, 4, 4, 3), dtype=np.uint8)

        def fake_run(*_args, **_kwargs):
            return SimpleNamespace(returncode=1, stderr=b"bad codec")

        monkeypatch.setattr(sim_to_lerobot.subprocess, "run", fake_run)

        with pytest.raises(AdapterError, match="ffmpeg failed.*bad codec"):
            encode_video(frames, tmp_path / "out.mp4", fps=20)


@needs_ffmpeg
class TestConvert:
    def test_output_structure(self, demo_dir: Path, output_dir: Path) -> None:
        convert(demo_dir, output_dir, fps=FPS, robot_type="franka_panda")

        # meta/
        assert (output_dir / "meta" / "info.json").exists()
        assert (output_dir / "meta" / "stats.json").exists()
        assert (output_dir / "meta" / "tasks.parquet").exists()
        assert (output_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet").exists()

        # data/
        assert (output_dir / "data" / "chunk-000" / "file-000.parquet").exists()

        # videos/
        for cam in ["observation.images.workspace", "observation.images.wrist"]:
            for ep_idx in range(N_EPISODES):
                vid = output_dir / "videos" / cam / "chunk-000" / f"file-{ep_idx:03d}.mp4"
                assert vid.exists(), f"Missing video: {vid}"
                assert vid.stat().st_size > 0

    def test_info_json(self, demo_dir: Path, output_dir: Path) -> None:
        convert(demo_dir, output_dir, fps=FPS, robot_type="test_robot")
        info = json.loads((output_dir / "meta" / "info.json").read_text())

        assert info["codebase_version"] == "v3.0"
        assert info["robot_type"] == "test_robot"
        assert info["total_episodes"] == N_EPISODES
        assert info["total_frames"] == N_EPISODES * N_TIMESTEPS
        assert info["fps"] == FPS
        assert info["total_tasks"] == 1
        assert info["splits"] == {"train": f"0:{N_EPISODES}"}

        feats = info["features"]
        assert feats["observation.images.workspace"]["dtype"] == "video"
        assert feats["observation.images.workspace"]["shape"] == [IMG_H, IMG_W, 3]
        assert feats["observation.state"]["dtype"] == "float32"
        assert feats["observation.state"]["shape"] == [N_STATE_DIM]
        assert feats["action"]["shape"] == [N_ACTIONS]

    def test_data_parquet_schema(self, demo_dir: Path, output_dir: Path) -> None:
        convert(demo_dir, output_dir, fps=FPS)
        table = pq.read_table(output_dir / "data" / "chunk-000" / "file-000.parquet")

        assert table.num_rows == N_EPISODES * N_TIMESTEPS
        assert "observation.state" in table.column_names
        assert "action" in table.column_names
        assert "episode_index" in table.column_names
        assert "frame_index" in table.column_names
        assert "timestamp" in table.column_names
        assert "index" in table.column_names
        assert "task_index" in table.column_names

    def test_data_parquet_values(self, demo_dir: Path, output_dir: Path) -> None:
        convert(demo_dir, output_dir, fps=FPS)
        table = pq.read_table(output_dir / "data" / "chunk-000" / "file-000.parquet")

        ep_indices = table.column("episode_index").to_pylist()
        frame_indices = table.column("frame_index").to_pylist()
        global_indices = table.column("index").to_pylist()
        timestamps = table.column("timestamp").to_pylist()

        # Global index is monotonically increasing
        assert global_indices == list(range(N_EPISODES * N_TIMESTEPS))

        # Frame indices reset per episode
        for ep_idx in range(N_EPISODES):
            ep_frames = [
                frame_indices[i]
                for i in range(len(ep_indices))
                if ep_indices[i] == ep_idx
            ]
            assert ep_frames == list(range(N_TIMESTEPS))

        # Timestamps are frame_index / fps
        for i, ts in enumerate(timestamps):
            expected = frame_indices[i] / FPS
            assert abs(ts - expected) < 1e-5

    def test_episodes_parquet(self, demo_dir: Path, output_dir: Path) -> None:
        convert(demo_dir, output_dir, fps=FPS)
        table = pq.read_table(
            output_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        )

        assert table.num_rows == N_EPISODES
        assert "episode_index" in table.column_names
        assert "length" in table.column_names
        assert "dataset_from_index" in table.column_names
        assert "dataset_to_index" in table.column_names

        lengths = table.column("length").to_pylist()
        assert all(length == N_TIMESTEPS for length in lengths)

        from_indices = table.column("dataset_from_index").to_pylist()
        to_indices = table.column("dataset_to_index").to_pylist()
        for i in range(N_EPISODES):
            assert from_indices[i] == i * N_TIMESTEPS
            assert to_indices[i] == (i + 1) * N_TIMESTEPS

    def test_tasks_parquet(self, demo_dir: Path, output_dir: Path) -> None:
        task_str = "Test task description"
        convert(demo_dir, output_dir, fps=FPS, task=task_str)
        table = pq.read_table(output_dir / "meta" / "tasks.parquet")

        assert table.num_rows == 1
        assert table.column("task_index").to_pylist() == [0]
        assert table.column("task").to_pylist() == [task_str]

    def test_stats_json(self, demo_dir: Path, output_dir: Path) -> None:
        convert(demo_dir, output_dir, fps=FPS)
        stats = json.loads((output_dir / "meta" / "stats.json").read_text())

        # All expected feature keys present
        for key in [
            "observation.images.workspace",
            "observation.images.wrist",
            "observation.state",
            "action",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        ]:
            assert key in stats, f"Missing stats key: {key}"
            assert "min" in stats[key]
            assert "max" in stats[key]
            assert "mean" in stats[key]
            assert "std" in stats[key]
            assert "count" in stats[key]

        # Video stats are per-channel (C, 1, 1)
        ws_stats = stats["observation.images.workspace"]
        assert len(ws_stats["min"]) == 3  # RGB channels
        assert len(ws_stats["min"][0]) == 1
        assert len(ws_stats["min"][0][0]) == 1

        # State stats match n_joints
        assert len(stats["observation.state"]["min"]) == N_STATE_DIM
        assert len(stats["action"]["min"]) == N_ACTIONS

    def test_video_stats_range(self, demo_dir: Path, output_dir: Path) -> None:
        convert(demo_dir, output_dir, fps=FPS)
        stats = json.loads((output_dir / "meta" / "stats.json").read_text())

        # Video stats should be in [0, 1] range (normalized)
        for cam_key in ["observation.images.workspace", "observation.images.wrist"]:
            for ch in range(3):
                assert stats[cam_key]["min"][ch][0][0] >= 0.0
                assert stats[cam_key]["max"][ch][0][0] <= 1.0


class TestConvertErrors:
    def test_convert_missing_required_array_raises_file_not_found(
        self, tmp_path: Path
    ) -> None:
        ep_dir = tmp_path / "episode_0000"
        ep_dir.mkdir()
        np.save(ep_dir / "obs_workspace.npy", np.zeros((2, 4, 4, 3), dtype=np.uint8))
        np.save(ep_dir / "state.npy", np.zeros((2, 3), dtype=np.float32))
        np.save(ep_dir / "actions.npy", np.zeros((2, 2), dtype=np.float32))

        with pytest.raises(FileNotFoundError, match="obs_wrist.npy"):
            convert(tmp_path, tmp_path / "out")

    def test_convert_workspace_state_length_mismatch_raises_adapter_error(
        self, tmp_path: Path
    ) -> None:
        ep_dir = tmp_path / "episode_0000"
        ep_dir.mkdir()
        np.save(ep_dir / "obs_workspace.npy", np.zeros((3, 4, 4, 3), dtype=np.uint8))
        np.save(ep_dir / "obs_wrist.npy", np.zeros((2, 4, 4, 3), dtype=np.uint8))
        np.save(ep_dir / "state.npy", np.zeros((2, 3), dtype=np.float32))
        np.save(ep_dir / "actions.npy", np.zeros((2, 2), dtype=np.float32))

        with pytest.raises(AdapterError, match="obs_workspace has 3"):
            convert(tmp_path, tmp_path / "out")
