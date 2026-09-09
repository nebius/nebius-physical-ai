"""Regression: LeRobot v3 shared MP4 segments retain episode-relative timelines."""

from pathlib import Path
import importlib
import io
import json
import subprocess
import shutil

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rerun.recording import load_recording

adapter = importlib.import_module("npa.viz.adapters.lerobot_to_rerun")
CAMERA = "observation.images.workspace"


def build_dataset(root: Path, *, shared: bool, include_offset: bool) -> Path:
    dataset = root / "dataset"
    (dataset / "meta/episodes/chunk-000").mkdir(parents=True)
    (dataset / "data/chunk-000").mkdir(parents=True)
    videos = dataset / "videos" / CAMERA / "chunk-000"
    videos.mkdir(parents=True)
    metadata = {
        "fps": 4,
        "codebase_version": "v3.0",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {CAMERA: {"dtype": "video", "shape": [16, 16, 3]}},
    }
    (dataset / "meta/info.json").write_text(json.dumps(metadata))
    rows = []
    locations = []
    red = np.zeros((4, 16, 16, 3), dtype=np.uint8)
    red[..., 0] = 220
    green = np.zeros_like(red)
    green[..., 1] = 220
    streams = [np.concatenate([red, green])] if shared else [red, green]
    for file_index, frames in enumerate(streams):
        proc = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                "16x16",
                "-r",
                "4",
                "-i",
                "pipe:",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(videos / f"file-{file_index:03d}.mp4"),
            ],
            input=frames.tobytes(),
            capture_output=True,
        )
        assert proc.returncode == 0, proc.stderr.decode()
    for episode in range(2):
        for frame in range(4):
            rows.append(
                {
                    "episode_index": episode,
                    "frame_index": frame,
                    "timestamp": frame / 4,
                    "observation.state": [float(episode), float(frame)],
                    "action": [float(episode)],
                }
            )
        location = {
            "episode_index": episode,
            f"videos/{CAMERA}/chunk_index": 0,
            f"videos/{CAMERA}/file_index": 0 if shared else episode,
        }
        if include_offset:
            location[f"videos/{CAMERA}/from_timestamp"] = (
                float(episode) if shared else 0.0
            )
            location[f"videos/{CAMERA}/to_timestamp"] = (
                float(episode + 1) if shared else 1.0
            )
        locations.append(location)
    pq.write_table(
        pa.Table.from_pylist(rows), dataset / "data/chunk-000/file-000.parquet"
    )
    pq.write_table(
        pa.Table.from_pylist(locations),
        dataset / "meta/episodes/chunk-000/file-000.parquet",
    )
    return dataset


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
@pytest.mark.parametrize(
    ("shared", "include_offset"), [(True, True), (False, True), (False, False)]
)
def test_episode_camera_frame_matches_state_and_local_timeline(
    tmp_path, shared, include_offset
):
    from rerun.recording import load_recording

    dataset = build_dataset(tmp_path, shared=shared, include_offset=include_offset)
    output = tmp_path / "recording.rrd"
    adapter.lerobot_dataset_logical_to_rerun(
        dataset,
        output,
        input_episode_indices=[0],
        rollout_episode_indices=[1],
        feedback_by_episode={},
        max_frames_per_episode=4,
    )
    chunks = list(load_recording(output).chunks())
    camera = (
        "/policy_rollout/episodes/episode_000001/camera/observation_images_workspace"
    )
    batches = [
        c.to_record_batch()
        for c in chunks
        if str(c.entity_path) == camera and not c.is_static
    ]
    assert batches
    table = pa.Table.from_batches(batches)
    (tmp_path / "decoded-camera.json").write_text(
        json.dumps(
            {
                "schema": str(table.schema),
                "columns": {
                    name: table[name].to_pylist() for name in table.column_names
                },
            },
            default=str,
            indent=2,
        )
    )
    timestamp_column = next(
        n for n in table.column_names if "timestamp" in n.lower() and n != "frame_time"
    )
    reference_ns = [v[0] for v in table[timestamp_column].to_pylist()]
    expected_ns = [
        int(((1.0 if shared else 0.0) + frame / 4) * 1e9) for frame in range(4)
    ]
    assert reference_ns == expected_ns, (
        "Camera video offsets must identify the same episode as the state/action rows"
    )
    timeline = table["frame_time"].cast(pa.int64()).to_pylist()
    assert timeline == [0, 250000000, 500000000, 750000000], (
        "RRD episode timeline must stay episode-relative"
    )
    file_index = 0 if shared else 1
    video_path = (
        dataset / "videos" / CAMERA / "chunk-000" / f"file-{file_index:03d}.mp4"
    )
    for value in reference_ns:
        decoded = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-ss",
                str(value / 1e9),
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:",
            ],
            capture_output=True,
        )
        assert decoded.returncode == 0, decoded.stderr.decode()
        frame = np.frombuffer(decoded.stdout, dtype=np.uint8).reshape(16, 16, 3)
        assert frame[..., 1].mean() > 180 and frame[..., 0].mean() < 20, (
            "Episode 1 should show its green camera frames"
        )


def test_real_rrd_video_offsets_without_encoder(tmp_path, monkeypatch):
    from rerun.recording import load_recording

    dataset = tmp_path / "dataset"
    (dataset / "meta/episodes/chunk-000").mkdir(parents=True)
    (dataset / "data/chunk-000").mkdir(parents=True)
    cameras = ["observation.images.front", "observation.images.wrist"]
    (dataset / "meta/info.json").write_text(
        json.dumps(
            {
                "fps": 8,
                "codebase_version": "v3.0",
                "features": {camera: {"dtype": "video"} for camera in cameras},
            }
        )
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "episode_index": 3,
                    f"videos/{cameras[0]}/from_timestamp": 2.125,
                    f"videos/{cameras[1]}/from_timestamp": 0.375,
                }
            ]
        ),
        dataset / "meta/episodes/chunk-000/file-000.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "episode_index": 3,
                    "frame_index": i,
                    "timestamp": t,
                    "observation.state": [i, 1.0],
                    "action": [3.0 + i],
                }
                for i, t in enumerate([0.125, 0.375])
            ]
        ),
        dataset / "data/chunk-000/file-000.parquet",
    )
    # Only video-asset I/O is replaced. Metadata loading, episode dispatch, Rerun
    # stream creation, real VideoFrameReference serialization and decoding run.
    monkeypatch.setattr(
        adapter,
        "_log_dataset_videos",
        lambda *_args, **_kwargs: {
            3: {camera: f"videos/{camera}" for camera in cameras}
        },
    )
    output = tmp_path / "offsets.rrd"
    adapter.lerobot_dataset_logical_to_rerun(
        dataset,
        output,
        input_episode_indices=[],
        rollout_episode_indices=[3],
        feedback_by_episode={},
        max_frames_per_episode=2,
    )
    chunks = list(load_recording(output).chunks())
    actual = {}
    expected = {"front": [2250000000, 2500000000], "wrist": [500000000, 750000000]}
    for camera in ["front", "wrist"]:
        path = f"/policy_rollout/episodes/episode_000003/camera/observation_images_{camera}"
        batches = [
            chunk.to_record_batch()
            for chunk in chunks
            if str(chunk.entity_path) == path and not chunk.is_static
        ]
        assert sum(batch.num_rows for batch in batches) == 2
        timeline = [
            v
            for batch in batches
            for v in batch.column("frame_time").cast(pa.int64()).to_pylist()
        ]
        assert timeline == [125000000, 375000000]
        actual[camera] = [
            v[0]
            for batch in batches
            for v in batch.column("VideoFrameReference:timestamp").to_pylist()
        ]
    (tmp_path / "observed.json").write_text(
        json.dumps({"actual": actual, "expected": expected}, indent=2)
    )
    assert actual == expected


CAMERAS = ("observation.images.front", "observation.images.wrist")
FPS = 4
FRAME_COUNT = 4


def _color(episode: int, camera: int, frame: int) -> np.ndarray:
    return np.array([30 + episode * 70, 40 + camera * 130, 30 + frame * 50])


def _write_video(path: Path, colors: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width, stream.height = 32, 24
        stream.pix_fmt = "yuv444p"
        stream.options = {"crf": "0"}
        for color in colors:
            pixels = np.broadcast_to(color, (24, 32, 3)).astype(np.uint8).copy()
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _write_alignment_dataset(tmp_path: Path, *, shared: bool) -> tuple[Path, dict]:
    dataset = tmp_path / "synthetic-lerobot"
    (dataset / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (dataset / "data" / "chunk-000").mkdir(parents=True)
    (dataset / "meta" / "info.json").write_text(json.dumps({
        "fps": FPS,
        "features": {key: {"dtype": "video"} for key in CAMERAS},
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }))
    rows, episodes, expected = [], [], {}
    for episode in range(3):
        metadata = {"episode_index": episode}
        for camera, key in enumerate(CAMERAS):
            # The wrist stream has gaps between episodes: offsets are per camera.
            offset = episode * (1.0 + camera * 0.5) if shared else 0.0
            file_index = 0 if shared else episode
            path = dataset / "videos" / key / "chunk-000" / f"file-{file_index:03d}.mp4"
            expected[episode, key] = (path, offset)
            metadata[f"videos/{key}/chunk_index"] = 0
            metadata[f"videos/{key}/file_index"] = file_index
            metadata[f"videos/{key}/from_timestamp"] = offset if shared else None
        episodes.append(metadata)
        rows.extend({
            "episode_index": episode,
            "frame_index": frame,
            "timestamp": frame / FPS,
            "observation.state": [episode * 10 + frame, episode * 10 + frame + 0.5],
            "action": [-episode * 10 - frame, episode + frame * 0.25],
        } for frame in range(FRAME_COUNT))
    pq.write_table(pa.Table.from_pylist(rows), dataset / "data" / "chunk-000" / "file-000.parquet")
    pq.write_table(
        pa.Table.from_pylist(episodes),
        dataset / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    for camera, key in enumerate(CAMERAS):
        videos = {}
        for episode in range(3):
            path, offset = expected[episode, key]
            first = round(offset * FPS)
            colors = videos.setdefault(path, [])
            colors.extend([np.array([255, 0, 255])] * (first + FRAME_COUNT - len(colors)))
            for frame in range(FRAME_COUNT):
                colors[first + frame] = _color(episode, camera, frame)
        for path, colors in videos.items():
            _write_video(path, colors)
    return dataset, expected


def _dynamic_rows(chunks: list, entity: str, component: str) -> list[tuple[int, object]]:
    rows = []
    for chunk in chunks:
        if str(chunk.entity_path) != entity or chunk.is_static:
            continue
        batch = chunk.to_record_batch()
        times = batch.column("frame_time").cast(pa.int64()).to_pylist()
        values = batch.column(component).to_pylist()
        rows.extend(zip(times, (value[0] for value in values), strict=True))
    return sorted(rows)


def _video_bytes(chunks: list, entity: str) -> bytes:
    blobs = [
        chunk.to_record_batch().column("AssetVideo:blob").to_pylist()[0][0]
        for chunk in chunks
        if str(chunk.entity_path) == entity and chunk.is_static
    ]
    assert len(blobs) == 1
    return bytes(blobs[0])


def _assert_media_signal_alignment(tmp_path: Path, *, shared: bool, frames: list[int]) -> None:
    dataset, expected = _write_alignment_dataset(tmp_path, shared=shared)
    output = tmp_path / "synthetic-episodes.rrd"
    adapter.lerobot_dataset_logical_to_rerun(
        dataset,
        output,
        input_episode_indices=[0, 1],
        rollout_episode_indices=[1, 2],
        feedback_by_episode={},
        max_frames_per_episode=len(frames),
    )
    recording = load_recording(output)
    assert recording.application_id() == "npa_lerobot_to_rerun"
    chunks = list(recording.chunks())
    timeline = [round(frame / FPS * 1e9) for frame in frames]
    for role, episodes in (("input_dataset", [0, 1]), ("policy_rollout", [1, 2])):
        for episode in episodes:
            root = f"/{role}/episodes/episode_{episode:06d}"
            for camera, key in enumerate(CAMERAS):
                entity_key = key.replace(".", "_")
                entity = f"{root}/camera/{entity_key}"
                video_entity = f"videos/episode_{episode:06d}/{entity_key}"
                path, offset = expected[episode, key]
                references = _dynamic_rows(chunks, entity, "VideoFrameReference:timestamp")
                assert [time for time, _ in references] == timeline
                assert _dynamic_rows(chunks, entity, "VideoFrameReference:video_reference") == [
                    (time, video_entity) for time in timeline
                ]
                payload = _video_bytes(chunks, "/" + video_entity)
                assert payload == path.read_bytes()
                with av.open(io.BytesIO(payload)) as container:
                    decoded = {
                        round(float(frame.pts * frame.time_base) * 1e9):
                        frame.to_ndarray(format="rgb24").mean(axis=(0, 1))
                        for frame in container.decode(video=0)
                    }
                for (_, media_time), frame in zip(references, frames, strict=True):
                    # Decode the embedded asset at the actual reference, independently
                    # of the adapter's metadata arithmetic, to catch wrong episode pixels.
                    np.testing.assert_allclose(decoded[media_time], _color(episode, camera, frame), atol=2)
                assert [value for _, value in references] == [
                    round((offset + frame / FPS) * 1e9) for frame in frames
                ]
            expected_signals = {
                "state/dim_00": [episode * 10 + frame for frame in frames],
                "state/dim_01": [episode * 10 + frame + 0.5 for frame in frames],
                "actions/dim_00": [-episode * 10 - frame for frame in frames],
                "actions/dim_01": [episode + frame * 0.25 for frame in frames],
            }
            for signal, values in expected_signals.items():
                assert _dynamic_rows(chunks, f"{root}/{signal}", "Scalars:scalars") == list(
                    zip(timeline, values, strict=True)
                )
            assert _dynamic_rows(chunks, f"{root}/state/transform", "Transform3D:translation") == [
                (time, [episode * 10 + frame, episode * 10 + frame + 0.5, 0.0])
                for time, frame in zip(timeline, frames, strict=True)
            ]


def test_sampled_shared_videos_align_embedded_media_and_signals(tmp_path: Path) -> None:
    # Sampling must preserve the original frame timestamps and each camera's offset.
    _assert_media_signal_alignment(tmp_path, shared=True, frames=[0, 3])


def test_null_video_offsets_align_embedded_media_and_signals(tmp_path: Path) -> None:
    _assert_media_signal_alignment(tmp_path, shared=False, frames=list(range(FRAME_COUNT)))
