from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from npa.adapter.isaac_lab_lerobot import G1_STATE_DIM, convert
from npa.viz.adapters.groot_predictions_to_rerun import groot_predictions_to_rerun
from npa.viz.adapters.lerobot_to_rerun import REPRESENTATIVE_JOINTS, RerunAdapterError
from npa.viz.lerobot import REAL_G1_ACTION_DIM


CYAN_PACKED = 0x00D9FFFF
ORANGE_PACKED = 0xFF8800FF


def _write_g1_raw_dataset(root: Path, *, frames: int = 10) -> Path:
    raw = root / "raw"
    episode = raw / "episode_000000"
    episode.mkdir(parents=True)
    t = np.linspace(0.0, 1.0, frames, dtype=np.float32)
    state = np.zeros((frames, G1_STATE_DIM), dtype=np.float32)
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


def _write_predictions_json(root: Path, *, frames: int = 10) -> Path:
    path = root / "predictions.json"
    t = np.linspace(0.0, 1.0, frames, dtype=np.float32)
    predictions = np.zeros((frames, REAL_G1_ACTION_DIM), dtype=np.float32)
    predictions[:, 32] = np.sin(t * np.pi * 2.0) * 0.30
    predictions[:, 39] = np.cos(t * np.pi * 2.0) * -0.30
    predictions[:, 46] = np.sin(t * np.pi) * 0.10
    path.write_text(json.dumps({"predicted_actions": predictions.tolist()}))
    return path


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


def _first_packed_color(chunks, entity_path: str, column_name: str) -> int:
    chunk = next(chunk for chunk in chunks if str(chunk.entity_path) == entity_path and not chunk.is_static)
    return int(chunk.to_record_batch().column(column_name).to_pylist()[0][0])


def test_groot_predictions_to_rerun_writes_overlay_hierarchies(tmp_path: Path) -> None:
    dataset = _write_lerobot_dataset(tmp_path, frames=10, fps=10)
    predictions = _write_predictions_json(tmp_path, frames=10)
    output = tmp_path / "groot-predictions-overlay.rrd"

    groot_predictions_to_rerun(predictions, dataset, output)

    assert output.exists()
    assert output.stat().st_size > 0
    chunks = _recording_chunks(output)
    entity_paths = _entity_paths(chunks)
    for root in ("/world/skeleton", "/world/predictions"):
        assert f"{root}/joints" in entity_paths
        assert f"{root}/bones" in entity_paths
        for joint_name in REPRESENTATIVE_JOINTS:
            assert f"{root}/angles/{joint_name}" in entity_paths
        assert _dynamic_row_count(chunks, f"{root}/joints") == 10
        assert _dynamic_row_count(chunks, f"{root}/bones") == 10


def test_groot_predictions_to_rerun_missing_predictions_is_clean_error(tmp_path: Path) -> None:
    dataset = _write_lerobot_dataset(tmp_path, frames=10, fps=10)

    with pytest.raises(RerunAdapterError, match="Predictions path does not exist"):
        groot_predictions_to_rerun(tmp_path / "missing.json", dataset, tmp_path / "missing.rrd")


def test_groot_predictions_to_rerun_color_differentiates_input_and_predictions(tmp_path: Path) -> None:
    dataset = _write_lerobot_dataset(tmp_path, frames=10, fps=10)
    predictions = _write_predictions_json(tmp_path, frames=10)
    output = tmp_path / "colors.rrd"

    groot_predictions_to_rerun(predictions, dataset, output)

    chunks = _recording_chunks(output)
    assert _first_packed_color(chunks, "/world/skeleton/joints", "Points3D:colors") == CYAN_PACKED
    assert _first_packed_color(chunks, "/world/skeleton/bones", "LineStrips3D:colors") == CYAN_PACKED
    assert _first_packed_color(chunks, "/world/predictions/joints", "Points3D:colors") == ORANGE_PACKED
    assert _first_packed_color(chunks, "/world/predictions/bones", "LineStrips3D:colors") == ORANGE_PACKED


def test_groot_predictions_to_rerun_repeats_short_prediction_horizon(tmp_path: Path) -> None:
    dataset = _write_lerobot_dataset(tmp_path, frames=50, fps=10)
    predictions = _write_predictions_json(tmp_path, frames=4)
    output = tmp_path / "short-horizon.rrd"

    groot_predictions_to_rerun(predictions, dataset, output)

    chunks = _recording_chunks(output)
    assert _dynamic_row_count(chunks, "/world/skeleton/joints") == 50
    assert _dynamic_row_count(chunks, "/world/predictions/joints") == 50
