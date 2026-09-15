"""Real media regressions for selected episodes from consolidated LeRobot videos."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import av
import imageio_ffmpeg
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from npa.adapter import groot

FPS = 20
CAMERA = "observation.images.front"
WRIST = "observation.images.wrist"


@pytest.fixture()
def ffmpeg(monkeypatch: pytest.MonkeyPatch) -> str:
    """Use the dev dependency's real encoder even without a host ffmpeg install."""
    executable = imageio_ffmpeg.get_ffmpeg_exe()
    original = shutil.which
    monkeypatch.setattr(
        groot.shutil, "which", lambda name: executable if name == "ffmpeg" else original(name)
    )
    return executable


def _encode(path: Path, ffmpeg: str, frame_count: int = 20) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.stack([np.full((32, 32, 3), 20 + 10 * i, dtype=np.uint8) for i in range(frame_count)])
    subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s", "32x32", "-r", str(FPS), "-i", "pipe:0",
            "-c:v", "libx264", "-g", "60", "-keyint_min", "60", "-sc_threshold", "0",
            "-crf", "18", "-pix_fmt", "yuv420p", "-y", str(path),
        ],
        input=frames.tobytes(), capture_output=True, check=True,
    )


def _decode(path: Path) -> tuple[np.ndarray, list[bool]]:
    with av.open(str(path)) as container:
        frames = list(container.decode(video=0))
    return np.stack([frame.to_ndarray(format="rgb24") for frame in frames]), [
        frame.key_frame for frame in frames
    ]


def _dataset(
    root: Path,
    ffmpeg: str,
    windows: dict[str, list[tuple[float | None, float | None]]],
    episodes: tuple[int, ...] = (17,),
    frame_count: int = 20,
) -> tuple[Path, pa.Table]:
    """Selected numeric episodes retain references into longer synthetic videos."""
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    rows = [
        {
            "observation.state": [float(episode * 100 + frame), float(frame), 0.5],
            "action": [float(-episode * 100 - frame), float(frame + 1), 1.0],
            "episode_index": episode, "frame_index": frame, "timestamp": frame / FPS,
            "index": episode * 5 + frame, "task_index": 3,
        }
        for episode in episodes for frame in range(5)
    ]
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, root / "data/chunk-000/file-000.parquet")
    metadata = []
    for position, episode in enumerate(episodes):
        row = {"episode_index": episode, "length": 5, "tasks": ["Synthetic camera alignment"]}
        for camera, camera_windows in windows.items():
            start, end = camera_windows[position]
            row.update({
                f"videos/{camera}/chunk_index": 0, f"videos/{camera}/file_index": 0,
                f"videos/{camera}/from_timestamp": start, f"videos/{camera}/to_timestamp": end,
            })
        metadata.append(row)
    pq.write_table(pa.Table.from_pylist(metadata), root / "meta/episodes/chunk-000/file-000.parquet")
    pq.write_table(pa.Table.from_pylist([
        {"task_index": 3, "task": "Synthetic camera alignment"},
    ]), root / "meta/tasks.parquet")
    info = {
        "codebase_version": "v3.0", "fps": FPS, "chunks_size": 1000,
        "total_episodes": len(episodes), "total_frames": table.num_rows,
        "data_path": groot.LEROBOT_DATA_PATH_TPL, "video_path": groot.LEROBOT_VIDEO_PATH_TPL,
        "features": {
            "observation.state": {"dtype": "float64", "shape": [3]},
            "action": {"dtype": "float64", "shape": [3]},
            **{camera: {"dtype": "video", "shape": [32, 32, 3]} for camera in windows},
        },
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    for camera in windows:
        _encode(_source(root, camera), ffmpeg, frame_count)
    return root, table


def _source(root: Path, camera: str = CAMERA) -> Path:
    return root / groot.LEROBOT_VIDEO_PATH_TPL.format(video_key=camera, chunk_index=0, file_index=0)


def _output_video(root: Path, camera: str = CAMERA, episode: int = 17) -> Path:
    return root / groot.GROOT_VIDEO_PATH_TPL.format(
        video_key=camera, episode_chunk=0, episode_index=episode,
    )


def _assert_identity(output: Path, table: pa.Table, episodes: tuple[int, ...] = (17,)) -> None:
    for episode in episodes:
        converted = pq.read_table(output / groot.GROOT_DATA_PATH_TPL.format(
            episode_chunk=0, episode_index=episode,
        ))
        expected = table.take([i for i, value in enumerate(table["episode_index"].to_pylist())
                               if value == episode])
        assert converted.select(table.column_names).equals(expected)
    metadata = [json.loads(line) for line in (output / "meta/episodes.jsonl").read_text().splitlines()]
    assert [(row["episode_index"], row["length"]) for row in metadata] == [(e, 5) for e in episodes]
    assert all(row["tasks"] == ["Synthetic camera alignment"] for row in metadata)
    assert json.loads((output / "meta/tasks.jsonl").read_text()) == {
        "task_index": 3, "task": "Synthetic camera alignment",
    }


def _assert_slice(source: Path, output: Path, start_frame: int) -> None:
    source_frames, keyframes = _decode(source)
    if start_frame:
        assert not keyframes[start_frame], "The fixture must exercise seeking between keyframes"
    frames, _ = _decode(output)
    # Check pixels before length so the baseline exposes wrong episode identity.
    np.testing.assert_allclose(
        frames[:5].astype(float), source_frames[start_frame:start_frame + 5], atol=3, rtol=0,
    )
    assert len(frames) == 5


def test_single_later_episode_seeks_between_keyframes(tmp_path: Path, ffmpeg: str) -> None:
    source, table = _dataset(tmp_path / "source", ffmpeg, {CAMERA: [(0.35, 0.6)]})
    output = groot.lerobot_to_groot(source, tmp_path / "converted")
    _assert_identity(output, table)
    _assert_slice(_source(source), _output_video(output), 7)


def test_each_camera_uses_its_own_window(tmp_path: Path, ffmpeg: str) -> None:
    source, table = _dataset(tmp_path / "source", ffmpeg, {
        CAMERA: [(0.35, 0.6)], WRIST: [(0.55, 0.8)],
    })
    output = groot.lerobot_to_groot(source, tmp_path / "converted")
    _assert_identity(output, table)
    for camera, start in [(CAMERA, 7), (WRIST, 11)]:
        _assert_slice(_source(source, camera), _output_video(output, camera), start)


def test_shared_source_still_splits_zero_and_positive_offsets(tmp_path: Path, ffmpeg: str) -> None:
    source, table = _dataset(
        tmp_path / "source", ffmpeg, {CAMERA: [(0.0, 0.25), (0.35, 0.6)]}, episodes=(0, 17),
    )
    output = groot.lerobot_to_groot(source, tmp_path / "converted")
    _assert_identity(output, table, (0, 17))
    for episode, start in [(0, 0), (17, 7)]:
        _assert_slice(_source(source), _output_video(output, episode=episode), start)


@pytest.mark.parametrize("metadata", ["zero-offset", "null-timestamps", "absent-timestamps"])
def test_single_episode_copy_compatibility(
    tmp_path: Path, ffmpeg: str, monkeypatch: pytest.MonkeyPatch, metadata: str,
) -> None:
    window = (0.0, 0.25) if metadata == "zero-offset" else (None, None)
    source, table = _dataset(tmp_path / "source", ffmpeg, {CAMERA: [window]}, frame_count=5)
    if metadata == "absent-timestamps":
        episodes = source / "meta/episodes/chunk-000/file-000.parquet"
        pq.write_table(pq.read_table(episodes).drop([
            f"videos/{CAMERA}/from_timestamp", f"videos/{CAMERA}/to_timestamp",
        ]), episodes)
    monkeypatch.setattr(groot.shutil, "which", lambda name: None)
    output = groot.lerobot_to_groot(source, tmp_path / "converted")
    _assert_identity(output, table)
    assert _output_video(output).read_bytes() == _source(source).read_bytes()
    _assert_slice(_source(source), _output_video(output), 0)


@pytest.mark.parametrize("window, message", [
    ((0.35, None), "Incomplete video timestamps"),
    ((None, 0.6), "Incomplete video timestamps"),
    ((0.6, 0.35), "Invalid video timestamps"),
    ((0.35, 0.35), "Invalid video timestamps"),
    ((-0.1, 0.6), "Invalid video timestamps"),
])
def test_invalid_window_fails_closed(tmp_path: Path, ffmpeg: str, window: tuple, message: str) -> None:
    source, _ = _dataset(tmp_path / "source", ffmpeg, {CAMERA: [window]})
    with pytest.raises(groot.GR00TAdapterError, match=message):
        groot.lerobot_to_groot(source, tmp_path / "converted")
    assert not _output_video(tmp_path / "converted").exists()


def test_positive_offset_requires_ffmpeg(tmp_path: Path, ffmpeg: str, monkeypatch: pytest.MonkeyPatch) -> None:
    source, _ = _dataset(tmp_path / "source", ffmpeg, {CAMERA: [(0.35, 0.6)]})
    monkeypatch.setattr(groot.shutil, "which", lambda name: None)
    with pytest.raises(groot.GR00TAdapterError, match="ffmpeg is required"):
        groot.lerobot_to_groot(source, tmp_path / "converted")
    assert not _output_video(tmp_path / "converted").exists()


def test_failed_decode_does_not_leave_episode_video(tmp_path: Path, ffmpeg: str) -> None:
    source, _ = _dataset(tmp_path / "source", ffmpeg, {CAMERA: [(0.35, 0.6)]})
    _source(source).write_bytes(b"not an encoded video")
    with pytest.raises(groot.GR00TAdapterError, match="Failed to split consolidated LeRobot video"):
        groot.lerobot_to_groot(source, tmp_path / "converted")
    assert not list((tmp_path / "converted/videos").rglob("*.mp4"))


def test_shared_source_requires_timestamps(tmp_path: Path, ffmpeg: str) -> None:
    source, _ = _dataset(
        tmp_path / "source", ffmpeg, {CAMERA: [(None, None), (0.35, 0.6)]}, episodes=(0, 17),
    )
    with pytest.raises(groot.GR00TAdapterError, match="Shared source video .* lacks episode timestamps"):
        groot.lerobot_to_groot(source, tmp_path / "converted")
