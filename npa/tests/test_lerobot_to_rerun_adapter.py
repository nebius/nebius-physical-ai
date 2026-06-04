from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from npa.adapter.isaac_lab_lerobot import G1_BONE_PAIRS, G1_STATE_DIM, convert
from npa.viz.adapters.lerobot_to_rerun import REPRESENTATIVE_JOINTS, lerobot_to_rerun, verify_rerun_entities
from npa.viz.lerobot import VizDataError


def _write_g1_raw_dataset(root: Path, *, frames: int = 10) -> Path:
    raw = root / "raw"
    episode = raw / "episode_000000"
    episode.mkdir(parents=True)
    t = np.linspace(0.0, 1.0, frames, dtype=np.float32)
    state = np.zeros((frames, G1_STATE_DIM), dtype=np.float32)
    state[:, 0] = np.sin(t * np.pi) * 0.10
    state[:, 6] = np.sin(t * np.pi * 2.0) * 0.25
    state[:, 15] = np.cos(t * np.pi * 2.0) * 0.20
    state[:, 29] = np.cos(t * np.pi * 2.0) * -0.20
    np.save(episode / "state.npy", state)
    np.save(episode / "actions.npy", state + 0.05)
    return raw


def _write_lerobot_dataset(root: Path, *, frames: int = 10, fps: int = 10) -> Path:
    return convert(
        _write_g1_raw_dataset(root, frames=frames),
        root / "lerobot",
        fps=fps,
        task="Isaac-Velocity-Flat-G1-v0",
    )


def _write_empty_lerobot_dataset(root: Path, *, fps: int = 10) -> Path:
    dataset = root / "empty-lerobot"
    data_path = dataset / "data" / "chunk-000" / "file-000.parquet"
    data_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "observation.state": pa.array([], type=pa.list_(pa.float32(), G1_STATE_DIM)),
                "index": pa.array([], type=pa.int64()),
            }
        ),
        data_path,
    )
    meta_path = dataset / "meta" / "info.json"
    meta_path.parent.mkdir(parents=True)
    meta_path.write_text(f'{{"fps": {fps}, "robot_type": "unitree_g1"}}')
    return dataset


def _recording_chunks(path: Path):
    from rerun.recording import load_recording

    return list(load_recording(path).chunks())


def _entity_paths(chunks) -> set[str]:
    return {str(chunk.entity_path) for chunk in chunks}


def _dynamic_row_count(chunks, entity_path: str) -> int:
    return sum(
        int(chunk.num_rows)
        for chunk in chunks
        if str(chunk.entity_path) == entity_path and not chunk.is_static
    )


def test_lerobot_to_rerun_writes_expected_entities_and_frame_count(tmp_path: Path) -> None:
    dataset = _write_lerobot_dataset(tmp_path, frames=10, fps=10)
    output = tmp_path / "isaac-lab-trajectory.rrd"

    lerobot_to_rerun(dataset, output)

    assert output.exists()
    assert output.stat().st_size > 0
    chunks = _recording_chunks(output)
    entity_paths = _entity_paths(chunks)
    assert "/world/skeleton/joints" in entity_paths
    assert "/world/skeleton/bones" in entity_paths
    for joint_name in REPRESENTATIVE_JOINTS:
        assert f"/world/skeleton/angles/{joint_name}" in entity_paths

    assert _dynamic_row_count(chunks, "/world/skeleton/joints") == 10
    assert _dynamic_row_count(chunks, "/world/skeleton/bones") == 10
    for joint_name in REPRESENTATIVE_JOINTS:
        assert _dynamic_row_count(chunks, f"/world/skeleton/angles/{joint_name}") == 10


def test_lerobot_to_rerun_duration_cap_subsamples_to_five_seconds(tmp_path: Path) -> None:
    dataset = _write_lerobot_dataset(tmp_path, frames=100, fps=10)
    output = tmp_path / "capped.rrd"

    lerobot_to_rerun(dataset, output)

    chunks = _recording_chunks(output)
    assert _dynamic_row_count(chunks, "/world/skeleton/joints") == 50
    assert _dynamic_row_count(chunks, "/world/skeleton/bones") == 50


def test_lerobot_to_rerun_caps_trajectory_longer_than_ten_seconds(tmp_path: Path) -> None:
    dataset = _write_lerobot_dataset(tmp_path, frames=120, fps=10)
    output = tmp_path / "capped-long.rrd"

    lerobot_to_rerun(dataset, output)

    chunks = _recording_chunks(output)
    assert _dynamic_row_count(chunks, "/world/skeleton/joints") == 50


def test_lerobot_to_rerun_single_frame_dataset(tmp_path: Path) -> None:
    dataset = _write_lerobot_dataset(tmp_path, frames=1, fps=10)
    output = tmp_path / "single-frame.rrd"

    lerobot_to_rerun(dataset, output)

    chunks = _recording_chunks(output)
    assert _dynamic_row_count(chunks, "/world/skeleton/joints") == 1
    assert _dynamic_row_count(chunks, "/world/skeleton/bones") == 1


def test_lerobot_to_rerun_empty_dataset_is_clean_error(tmp_path: Path) -> None:
    dataset = _write_empty_lerobot_dataset(tmp_path)

    with pytest.raises(VizDataError, match="No observation.state rows found"):
        lerobot_to_rerun(dataset, tmp_path / "empty.rrd")


def test_lerobot_to_rerun_records_bone_segments(tmp_path: Path) -> None:
    dataset = _write_lerobot_dataset(tmp_path, frames=10, fps=10)
    output = tmp_path / "bones.rrd"

    lerobot_to_rerun(dataset, output)

    chunks = _recording_chunks(output)
    bone_chunk = next(
        chunk for chunk in chunks if str(chunk.entity_path) == "/world/skeleton/bones" and not chunk.is_static
    )
    batch = bone_chunk.to_record_batch()
    strips = batch.column("LineStrips3D:strips").to_pylist()[0]
    assert len(strips) == len(G1_BONE_PAIRS)
    assert len(strips[0]) == 2
    assert len(strips[0][0]) == 3


def test_lerobot_to_rerun_uploads_s3_output_after_local_save(tmp_path: Path, mocker) -> None:
    dataset = _write_lerobot_dataset(tmp_path, frames=10, fps=10)
    storage = mocker.Mock()

    def upload_file(local_file: str, destination: str) -> str:
        local_path = Path(local_file)
        assert local_path.exists()
        assert local_path.suffix == ".rrd"
        assert local_path.stat().st_size > 0
        assert destination == "s3://bucket/visuals/out.rrd"
        return destination

    storage.upload_file.side_effect = upload_file
    adapter_module = importlib.import_module("npa.viz.adapters.lerobot_to_rerun")
    mocker.patch.object(adapter_module, "_storage_client", return_value=storage)

    lerobot_to_rerun(dataset, "s3://bucket/visuals/out.rrd")

    storage.upload_file.assert_called_once()


def test_verify_rerun_entities_uses_fallback_counts_without_recording_loader(tmp_path: Path, mocker) -> None:
    output = tmp_path / "logical.rrd"
    output.write_bytes(b"rrd")
    counts = {"/input_dataset/episodes/episode_000000/state/dim_00": 3}
    real_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "rerun.recording":
            raise ImportError("recording loader unavailable")
        return real_import(name, globals, locals, fromlist, level)

    mocker.patch("builtins.__import__", side_effect=fake_import)

    assert verify_rerun_entities(
        output,
        ["input_dataset/episodes/episode_000000/state/dim_00"],
        fallback_counts=counts,
    ) == counts
