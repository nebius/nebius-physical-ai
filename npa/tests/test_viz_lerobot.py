from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from npa.adapter.isaac_lab_lerobot import G1_STATE_DIM, convert
from npa.viz.lerobot import (
    REAL_G1_ACTION_DIM,
    VizDataError,
    load_lerobot_state_vectors,
    load_predictions_skeleton,
    load_render_inputs,
    real_g1_action_vectors_to_g1_state_vectors,
    resolve_duration_s,
    select_frames,
)


def _write_g1_raw_dataset(root: Path, *, frames: int = 6) -> Path:
    raw = root / "raw"
    episode = raw / "episode_000000"
    episode.mkdir(parents=True)
    t = np.linspace(0.0, 1.0, frames, dtype=np.float32)
    state = np.zeros((frames, G1_STATE_DIM), dtype=np.float32)
    state[:, 0] = np.sin(t * np.pi)
    state[:, 6] = np.sin(t * np.pi * 2.0) * 0.25
    state[:, 18] = np.cos(t * np.pi * 2.0) * 0.20
    np.save(episode / "state.npy", state)
    np.save(episode / "actions.npy", state + 0.1)
    return raw


def _write_lerobot_dataset(root: Path, *, frames: int = 6, fps: int = 30) -> Path:
    return convert(
        _write_g1_raw_dataset(root, frames=frames),
        root / "lerobot",
        fps=fps,
        task="Isaac-Velocity-Flat-G1-v0",
    )


def test_load_render_inputs_reads_dataset_and_caps_default_duration(tmp_path: Path) -> None:
    dataset = _write_lerobot_dataset(tmp_path, frames=20, fps=1)

    loaded = load_render_inputs(dataset, output_fps=2)

    assert loaded.duration_s == 10.0
    assert loaded.skeleton_data.shape == (20, G1_STATE_DIM, 3)
    assert loaded.predictions_data is None
    assert loaded.source_fps == 1
    assert loaded.title == "Isaac-Velocity-Flat-G1-v0"


def test_resolve_duration_uses_requested_duration_without_cap() -> None:
    assert resolve_duration_s(frame_count=100, source_fps=10, requested_duration_s=5.0) == 5.0


def test_select_frames_subsamples_evenly_when_source_is_longer() -> None:
    data = np.arange(10, dtype=np.float32).reshape(10, 1)

    selected, indices = select_frames(data, source_fps=10, output_fps=2, duration_s=1.0)

    assert indices.tolist() == [0, 9]
    assert selected.ravel().tolist() == [0.0, 9.0]


def test_select_frames_holds_last_frame_when_source_is_shorter() -> None:
    data = np.arange(2, dtype=np.float32).reshape(2, 1)

    selected, indices = select_frames(data, source_fps=1, output_fps=2, duration_s=3.0)

    assert indices.tolist() == [0, 0, 1, 1, 1, 1]
    assert selected.ravel().tolist() == [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]


def test_overlay_layout_requires_predictions(tmp_path: Path) -> None:
    dataset = _write_lerobot_dataset(tmp_path, frames=4)

    with pytest.raises(VizDataError, match="predictions-path is required"):
        load_render_inputs(dataset, layout="overlay", output_fps=2)


def test_load_predictions_npz_to_skeleton(tmp_path: Path) -> None:
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    actions = np.zeros((2, 3, G1_STATE_DIM), dtype=np.float32)
    actions[:, :, 6] = 0.4
    np.savez_compressed(pred_dir / "predicted_actions.npz", trajectory_0=actions)

    predictions = load_predictions_skeleton(
        pred_dir,
        source_fps=6,
        output_fps=2,
        duration_s=1.0,
        target_joint_count=G1_STATE_DIM,
    )

    assert predictions.shape == (6, G1_STATE_DIM, 3)


def test_load_real_g1_action_predictions_to_skeleton(tmp_path: Path) -> None:
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    actions = np.zeros((2, REAL_G1_ACTION_DIM), dtype=np.float32)
    actions[:, 32] = 0.5
    actions[:, 46] = -0.25
    np.savez_compressed(pred_dir / "predicted_actions.npz", trajectory_0=actions)

    predictions = load_predictions_skeleton(
        pred_dir,
        source_fps=2,
        output_fps=2,
        duration_s=1.0,
        target_joint_count=G1_STATE_DIM,
    )

    assert predictions.shape == (2, G1_STATE_DIM, 3)


def test_load_render_inputs_preserves_short_prediction_horizon(tmp_path: Path) -> None:
    dataset = _write_lerobot_dataset(tmp_path, frames=8, fps=4)
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    actions = np.zeros((3, REAL_G1_ACTION_DIM), dtype=np.float32)
    actions[:, 32] = 0.5
    np.savez_compressed(pred_dir / "predicted_actions.npz", trajectory_0=actions)

    loaded = load_render_inputs(
        dataset,
        predictions_path=pred_dir,
        layout="overlay",
        duration_s=2.0,
        output_fps=4,
    )

    assert loaded.skeleton_data.shape == (8, G1_STATE_DIM, 3)
    assert loaded.predictions_data is not None
    assert loaded.predictions_data.shape == (3, G1_STATE_DIM, 3)


def test_load_render_inputs_rejects_predictions_longer_than_input(tmp_path: Path) -> None:
    dataset = _write_lerobot_dataset(tmp_path, frames=4, fps=4)
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    actions = np.zeros((5, REAL_G1_ACTION_DIM), dtype=np.float32)
    np.savez_compressed(pred_dir / "predicted_actions.npz", trajectory_0=actions)

    with pytest.raises(VizDataError, match="Prediction frame count cannot exceed input frame count"):
        load_render_inputs(
            dataset,
            predictions_path=pred_dir,
            layout="overlay",
            duration_s=1.0,
            output_fps=4,
        )


def test_real_g1_action_mapping_rejects_wrong_dim() -> None:
    with pytest.raises(VizDataError, match="Expected 53D"):
        real_g1_action_vectors_to_g1_state_vectors(np.zeros((2, 52), dtype=np.float32))


def test_load_lerobot_state_vectors_rejects_missing_data(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(VizDataError, match="data directory is missing"):
        load_lerobot_state_vectors(empty)
