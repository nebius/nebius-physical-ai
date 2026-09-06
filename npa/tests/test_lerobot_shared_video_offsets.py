"""Decode synthetic MP4/RRD bytes to check episode media and signal alignment."""

from __future__ import annotations

import io
import json
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rerun.recording import load_recording

from npa.viz.adapters.lerobot_to_rerun import lerobot_dataset_logical_to_rerun


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


def _write_dataset(tmp_path: Path, *, shared: bool, offset_mode: str) -> tuple[Path, dict]:
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
            if offset_mode != "missing":
                metadata[f"videos/{key}/from_timestamp"] = None if offset_mode == "null" else offset
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


def _assert_recording(tmp_path: Path, *, shared: bool, offset_mode: str, max_frames: int) -> None:
    dataset, expected = _write_dataset(tmp_path, shared=shared, offset_mode=offset_mode)
    output = tmp_path / "synthetic-episodes.rrd"
    lerobot_dataset_logical_to_rerun(
        dataset,
        output,
        input_episode_indices=[0, 1],
        rollout_episode_indices=[1, 2],
        feedback_by_episode={},
        max_frames_per_episode=max_frames,
    )
    recording = load_recording(output)
    assert recording.application_id() == "npa_lerobot_to_rerun"
    chunks = list(recording.chunks())
    frames = list(range(FRAME_COUNT)) if max_frames == FRAME_COUNT else [0, 3]
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


@pytest.mark.parametrize("max_frames", [FRAME_COUNT, 2], ids=["all-frames", "sampled-frames"])
def test_shared_videos_use_each_camera_episode_offset(tmp_path: Path, max_frames: int) -> None:
    _assert_recording(tmp_path, shared=True, offset_mode="explicit", max_frames=max_frames)


@pytest.mark.parametrize("offset_mode", ["explicit", "missing", "null"], ids=["zero", "missing", "null"])
def test_episode_videos_default_to_zero_offset(tmp_path: Path, offset_mode: str) -> None:
    _assert_recording(tmp_path, shared=False, offset_mode=offset_mode, max_frames=FRAME_COUNT)
