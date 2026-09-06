"""Simulation streams must align within each episode before video encoding."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import av
import numpy as np
import pyarrow.parquet as pq
import pytest

from npa.adapter import sim_to_lerobot


STREAMS = ("obs_workspace", "obs_wrist", "actions")
CAMERAS = ("workspace", "wrist")
FPS = 20
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _write_episode(root: Path, index: int, length: int) -> dict[str, np.ndarray]:
    episode = root / f"episode_{index:04d}"
    episode.mkdir(parents=True)
    frames = np.broadcast_to(
        np.arange(length, dtype=np.uint8)[:, None, None, None] * 24 + 32,
        (length, 16, 16, 3),
    ).copy()
    arrays = {
        "state": np.arange(length * 3, dtype=np.float32).reshape(length, 3) / 4 + index,
        "actions": np.arange(length * 2, dtype=np.float32).reshape(length, 2) / 8 - index,
        "obs_workspace": frames,
        "obs_wrist": frames + 16,
    }
    for name, values in arrays.items():
        np.save(episode / f"{name}.npy", values)
    return arrays


def _change_length(root: Path, index: int, stream: str, length: int) -> None:
    path = root / f"episode_{index:04d}" / f"{stream}.npy"
    values = np.load(path)
    np.save(path, np.resize(values, (length, *values.shape[1:])))


def _assert_no_dataset_metadata(output: Path) -> None:
    assert not list(output.rglob("*.parquet"))
    assert not (output / "meta" / "info.json").exists()
    assert not (output / "meta" / "stats.json").exists()


@pytest.mark.parametrize("stream", STREAMS)
@pytest.mark.parametrize("length", [0, 3, 5])
def test_mismatched_stream_rejected_before_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stream: str, length: int,
) -> None:
    raw, output = tmp_path / "raw", tmp_path / "dataset"
    _write_episode(raw, 0, 4)
    _change_length(raw, 0, stream, length)

    def unexpected_encode(*args, **kwargs):
        pytest.fail("Mismatched episode reached video encoding")

    monkeypatch.setattr(sim_to_lerobot, "encode_video", unexpected_encode)
    with pytest.raises(
        sim_to_lerobot.AdapterError,
        match=rf"Episode 0: {stream} has {length} frames but state has 4",
    ):
        sim_to_lerobot.convert(raw, output)

    assert not list(output.rglob("*.mp4"))
    _assert_no_dataset_metadata(output)


@needs_ffmpeg
@pytest.mark.parametrize("stream", STREAMS)
@pytest.mark.parametrize("length", [3, 5])
def test_later_mismatched_episode_is_not_encoded_or_published(
    tmp_path: Path, stream: str, length: int,
) -> None:
    raw, output = tmp_path / "raw", tmp_path / "dataset"
    _write_episode(raw, 0, 2)
    _write_episode(raw, 1, 4)
    _change_length(raw, 1, stream, length)

    with pytest.raises(
        sim_to_lerobot.AdapterError,
        match=rf"Episode 1: {stream} has {length} frames but state has 4",
    ):
        sim_to_lerobot.convert(raw, output)

    for camera in CAMERAS:
        videos = output / "videos" / f"observation.images.{camera}" / "chunk-000"
        with av.open(str(videos / "file-000.mp4")) as container:
            assert len(list(container.decode(video=0))) == 2
        assert not (videos / "file-001.mp4").exists()
    _assert_no_dataset_metadata(output)


@needs_ffmpeg
@pytest.mark.parametrize("lengths", [(4,), (3, 5)])
def test_aligned_streams_preserve_video_rows_and_statistics(
    tmp_path: Path, lengths: tuple[int, ...],
) -> None:
    raw, output = tmp_path / "raw", tmp_path / "dataset"
    episodes = [_write_episode(raw, index, length) for index, length in enumerate(lengths)]
    assert sim_to_lerobot.convert(raw, output, fps=FPS) == output

    info = json.loads((output / "meta" / "info.json").read_text())
    stats = json.loads((output / "meta" / "stats.json").read_text())
    rows = pq.read_table(output / "data/chunk-000/file-000.parquet").to_pydict()
    metadata = pq.read_table(output / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    assert info["total_episodes"] == len(lengths)
    assert info["total_frames"] == sum(lengths)
    assert rows["index"] == list(range(sum(lengths)))
    assert rows["task_index"] == [0] * sum(lengths)
    for column, source in [("observation.state", "state"), ("action", "actions")]:
        expected = np.concatenate([episode[source] for episode in episodes])
        np.testing.assert_array_equal(rows[column], expected)
        assert stats[column]["count"] == [sum(lengths)]
        for stat in ("min", "max", "mean", "std"):
            np.testing.assert_allclose(stats[column][stat], getattr(expected.astype(float), stat)(axis=0))

    offset = 0
    for index, (length, episode, meta) in enumerate(zip(lengths, episodes, metadata, strict=True)):
        stop = offset + length
        assert meta["length"] == length
        assert (meta["dataset_from_index"], meta["dataset_to_index"]) == (offset, stop)
        assert rows["episode_index"][offset:stop] == [index] * length
        assert rows["frame_index"][offset:stop] == list(range(length))
        np.testing.assert_array_equal(rows["timestamp"][offset:stop], np.arange(length, dtype=np.float32) / FPS)
        for feature in ("observation.state", "action"):
            assert meta[f"stats/{feature}/count"] == [length]
        for camera in CAMERAS:
            key = f"observation.images.{camera}"
            video = output / "videos" / key / "chunk-000" / f"file-{index:03d}.mp4"
            with av.open(str(video)) as container:
                assert container.streams.video[0].average_rate == FPS
                decoded = np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])
            assert decoded.shape == episode[f"obs_{camera}"].shape
            # H.264 is lossy: verify frame order/content within codec rounding.
            np.testing.assert_allclose(decoded.astype(float), episode[f"obs_{camera}"], atol=3)
            assert stats[key]["count"] == [sum(lengths)]
            assert meta[f"stats/{key}/count"] == [length]
            assert meta[f"videos/{key}/from_timestamp"] == 0.0
            assert meta[f"videos/{key}/to_timestamp"] == length / FPS
        offset = stop


def test_missing_actions_keeps_existing_required_file_contract(tmp_path: Path) -> None:
    raw, output = tmp_path / "raw", tmp_path / "dataset"
    _write_episode(raw, 0, 4)
    (raw / "episode_0000" / "actions.npy").unlink()
    with pytest.raises(FileNotFoundError, match="actions.npy"):
        sim_to_lerobot.convert(raw, output)
    assert not list(output.rglob("*.mp4"))
    _assert_no_dataset_metadata(output)
