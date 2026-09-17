from __future__ import annotations

import json
import shutil
import subprocess
import tracemalloc
import weakref
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from npa.adapter import isaac_lab_lerobot as adapter
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


def test_rgb_stats_match_pixel_reference_for_unequal_episodes() -> None:
    generator = np.random.default_rng(281)
    episodes = [generator.integers(0, 256, (length, 6, 8, 3), dtype=np.uint8)
                for length in (1, 7, 2)]
    episodes[0][:] = 0
    episodes[1][:] = 255
    reference = np.concatenate(episodes).astype(np.float64) / 255.0
    accumulator = adapter._RgbStats()
    for frames in episodes:
        accumulator.update(frames)

    actual = accumulator.as_dict()

    assert actual["count"] == [10]
    for name in ("min", "max", "mean", "std"):
        expected = getattr(reference, name)(axis=(0, 1, 2)).reshape(3, 1, 1)
        np.testing.assert_allclose(actual[name], expected, rtol=1e-13, atol=1e-15)


def test_rgb_stats_constant_channels_have_exactly_zero_variance() -> None:
    accumulator = adapter._RgbStats()
    colors = np.array([0, 128, 255], dtype=np.uint8)
    for length in (3, 11):
        accumulator.update(np.broadcast_to(colors, (length, 4, 6, 3)))

    actual = accumulator.as_dict()

    assert actual["count"] == [14]
    assert actual["std"] == [[[0.0]], [[0.0]], [[0.0]]]
    for name in ("min", "max", "mean"):
        np.testing.assert_array_equal(actual[name], (colors / 255.0).reshape(3, 1, 1))


def test_rgb_stats_preserve_rare_intensity_variation() -> None:
    frames = np.full((3, 32, 48, 3), 255, dtype=np.uint8)
    frames[0, 0, 0] = [254, 253, 252]
    expected = (frames.astype(np.float64) / 255.0).std(axis=(0, 1, 2))

    actual = adapter._compute_feature_stats([frames], is_video=True)

    np.testing.assert_allclose(np.asarray(actual["std"]).reshape(3), expected, rtol=1e-12)


@pytest.mark.parametrize("dtype", [np.float32, np.int16])
def test_rgb_stats_reject_non_uint8_pixels(dtype) -> None:
    with pytest.raises(IsaacLabLeRobotError, match="uint8"):
        adapter._RgbStats().update(np.zeros((2, 4, 6, 3), dtype=dtype))


def test_rgb_stats_memory_is_bounded_by_frame_size() -> None:
    frame = np.arange(128 * 128 * 3, dtype=np.uint8).reshape(128, 128, 3)
    accumulator = adapter._RgbStats()
    # This view represents 12 MiB of pixels without allocating the repeated frames.
    frames = np.broadcast_to(frame, (256, *frame.shape))
    reference = weakref.ref(frames)
    tracemalloc.start()
    try:
        accumulator.update(frames)
        accumulator.as_dict()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    del frames

    assert reference() is None
    assert peak < 1024 * 1024
    assert accumulator.as_dict()["count"] == [256]


@pytest.mark.parametrize("channels", [3, 4])
def test_rgb_loading_is_readonly_mapped_and_preserves_rgb(tmp_path, channels) -> None:
    pixels = np.arange(3 * 6 * 8 * channels, dtype=np.uint8).reshape(3, 6, 8, channels)
    np.save(tmp_path / "rgb.npy", pixels)

    frames = adapter._load_rgb_frames(tmp_path, expected_frames=3)

    assert isinstance(frames, np.memmap)
    assert frames.mode == "r"
    assert not frames.flags.writeable
    np.testing.assert_array_equal(frames, pixels[..., :3])


def test_convert_releases_rgb_between_episodes_and_writes_exact_stats(tmp_path, monkeypatch) -> None:
    raw = tmp_path / "raw"
    references = []
    original_load = adapter._load_rgb_frames

    def load_episode(*args, **kwargs):
        assert all(reference() is None for reference in references)
        frames = original_load(*args, **kwargs)
        references.append(weakref.ref(frames))
        return frames

    def encode_episode(frames, path, *, fps):
        assert isinstance(frames, np.memmap)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"video-placeholder-for-stats-test")

    for index, length in enumerate((2, 5, 3)):
        _write_episode(raw, index, frames=length)
        np.save(raw / f"episode_{index:06d}" / "rgb.npy",
                np.full((length, 6, 8, 4), index * 100, dtype=np.uint8))
    monkeypatch.setattr(adapter, "_load_rgb_frames", load_episode)
    monkeypatch.setattr(adapter, "_encode_video", encode_episode)

    output = convert(raw, tmp_path / "lerobot")

    assert len(references) == 3 and all(reference() is None for reference in references)
    actual = json.loads((output / "meta" / "stats.json").read_text())[WORKSPACE_VIEW_KEY]
    reference = np.repeat([0, 100, 200], [2, 5, 3]).astype(np.float64) / 255.0
    assert actual["count"] == [10]
    for name in ("min", "max", "mean", "std"):
        np.testing.assert_allclose(actual[name], np.full((3, 1, 1), getattr(reference, name)()))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_streaming_encoder_preserves_noncontiguous_rgb_timeline(tmp_path) -> None:
    rgba = np.zeros((3, 12, 16, 4), dtype=np.uint8)
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    rgba[..., :3] = colors[:, None, None, :]
    frames = rgba[..., :3]
    assert not frames.flags.c_contiguous
    output = tmp_path / "output.mp4"

    adapter._encode_video(frames, output, fps=20)

    decoded = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(output), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:"],
        capture_output=True, check=True, timeout=10,
    )
    pixels = np.frombuffer(decoded.stdout, dtype=np.uint8).reshape(frames.shape)
    np.testing.assert_allclose(pixels.mean(axis=(1, 2)), colors, atol=8)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_streaming_encoder_reports_real_ffmpeg_failure_and_broken_pipe(tmp_path, monkeypatch) -> None:
    frames = np.broadcast_to(np.zeros((64, 64, 3), dtype=np.uint8), (1000, 64, 64, 3))
    write_errors = []
    original_write = adapter._write_video_frames

    def record_write_errors(stream, frames, errors):
        original_write(stream, frames, errors)
        write_errors.extend(errors)

    monkeypatch.setattr(adapter, "_write_video_frames", record_write_errors)
    command = ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", "64x64", "-i", "pipe:", "-c:v", "npa_missing_codec", str(tmp_path / "bad.mp4")]

    with pytest.raises(IsaacLabLeRobotError, match="ffmpeg failed.*npa_missing_codec"):
        adapter._run_video_encoder(command, frames, timeout=10)

    assert any(isinstance(error, BrokenPipeError) for error in write_errors)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_streaming_encoder_timeout_reaps_child_and_closes_blocked_writer(monkeypatch) -> None:
    frames = np.broadcast_to(np.zeros((64, 64, 3), dtype=np.uint8), (1000, 64, 64, 3))
    processes = []
    original_popen = subprocess.Popen

    def start_process(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(adapter.subprocess, "Popen", start_process)
    # An infinite generated stream leaves stdin unread, so the RGB writer also blocks.
    command = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=size=16x16", "-f", "null", "-"]

    with pytest.raises(subprocess.TimeoutExpired):
        adapter._run_video_encoder(command, frames, timeout=1)

    assert len(processes) == 1
    assert processes[0].poll() is not None
    assert processes[0].stdin.closed
