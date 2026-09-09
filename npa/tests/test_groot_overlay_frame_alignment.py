"""Decode real RRDs made from synthetic trajectories to check overlay alignment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from npa.adapter.isaac_lab_lerobot import G1_BONE_PAIRS, G1_STATE_DIM, G1_STATE_NAMES_43, convert
from npa.viz.adapters.groot_predictions_to_rerun import groot_predictions_to_rerun
from npa.viz.adapters.lerobot_to_rerun import REPRESENTATIVE_JOINTS, RerunAdapterError
from npa.viz.lerobot import g1_state_vectors_to_skeleton


_DEFAULT_INDICES = list(range(50))
_TWO_SECOND_INDICES = list(range(20))


def _synthetic_dataset(root: Path, frames: int) -> tuple[Path, np.ndarray]:
    # Every joint encodes its source frame, so a shifted overlay cannot pass by
    # comparing two constant poses. These are synthetic states, not model output.
    states = (
        np.arange(frames, dtype=np.float32)[:, None] * 0.005
        + np.arange(G1_STATE_DIM, dtype=np.float32)[None, :] * 0.001
    )
    episode = root / "raw" / "episode_000000"
    episode.mkdir(parents=True)
    np.save(episode / "state.npy", states)
    np.save(episode / "actions.npy", states)
    dataset = convert(episode.parent, root / "lerobot", fps=10, task="Synthetic overlay alignment")
    return dataset, states


def _predictions(root: Path, values: np.ndarray) -> Path:
    path = root / "predictions.json"
    path.write_text(json.dumps({"predicted_actions": values.tolist()}))
    return path


def _decoded_rows(chunks, entity: str, component: str) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for chunk in chunks:
        if str(chunk.entity_path) != entity or chunk.is_static:
            continue
        batch = chunk.to_record_batch()
        times = batch.column("frame_time").cast(pa.int64()).to_pylist()
        rows.extend(zip(times, batch.column(component).to_pylist(), strict=True))
    rows.sort(key=lambda row: row[0])
    assert rows, f"No decoded rows for {entity}"
    times, values = zip(*rows, strict=True)
    return np.asarray(times), np.asarray(values)


@pytest.mark.parametrize(
    ("frames", "horizon", "duration", "source_indices"),
    [
        pytest.param(100, 100, None, _DEFAULT_INDICES, id="full-horizon-default-cap"),
        pytest.param(100, 4, None, _DEFAULT_INDICES, id="short-prefix"),
        pytest.param(100, 49, None, _DEFAULT_INDICES, id="prefix-below-display-count"),
        pytest.param(100, 51, None, _DEFAULT_INDICES, id="prefix-longer-than-display-count"),
        pytest.param(100, 100, 2.0, _TWO_SECOND_INDICES, id="custom-duration"),
        pytest.param(100, 4, 2.0, _TWO_SECOND_INDICES, id="custom-duration-short-prefix"),
        pytest.param(100, 100, 8.0, _DEFAULT_INDICES, id="duration-above-default-cap"),
        pytest.param(100, 100, 0.01, [0], id="sub-frame-duration"),
        pytest.param(10, 10, None, list(range(10)), id="short-episode-full-horizon"),
        pytest.param(10, 4, None, list(range(10)), id="short-episode-prefix"),
        pytest.param(1, 1, None, [0], id="single-frame"),
    ],
)
def test_overlay_uses_matching_source_frames_and_times(
    tmp_path: Path, frames: int, horizon: int, duration: float | None, source_indices: list[int]
) -> None:
    from rerun.recording import load_recording

    dataset, states = _synthetic_dataset(tmp_path, frames)
    predictions = _predictions(tmp_path, states[:horizon])
    output = tmp_path / "synthetic-overlay.rrd"

    groot_predictions_to_rerun(predictions, dataset, output, duration_s=duration)

    assert output.stat().st_size > 0
    chunks = list(load_recording(output).chunks())
    skeleton = g1_state_vectors_to_skeleton(states)
    for root, indices in (
        ("/world/skeleton", source_indices),
        ("/world/predictions", [index for index in source_indices if index < horizon]),
    ):
        # Check every timestamp and value across the native-time source prefix.
        # A duration cap must not resample later source states onto earlier times.
        expected_times = np.asarray(indices, dtype=np.int64) * 100_000_000
        times, positions = _decoded_rows(chunks, f"{root}/joints", "Points3D:positions")
        np.testing.assert_array_equal(times, expected_times)
        np.testing.assert_allclose(positions, skeleton[indices])
        times, bones = _decoded_rows(chunks, f"{root}/bones", "LineStrips3D:strips")
        np.testing.assert_array_equal(times, expected_times)
        np.testing.assert_allclose(bones, skeleton[indices][:, np.asarray(G1_BONE_PAIRS)])
        for joint in REPRESENTATIVE_JOINTS:
            times, angles = _decoded_rows(chunks, f"{root}/angles/{joint}", "Scalars:scalars")
            np.testing.assert_array_equal(times, expected_times)
            np.testing.assert_allclose(angles[:, 0], states[indices, G1_STATE_NAMES_43.index(joint)])


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ((101, G1_STATE_DIM), "Prediction frame count cannot exceed input frame count.*101 > 100"),
        ((100, G1_STATE_DIM - 1), "Predictions must be either G1 state vectors"),
        ((100, G1_STATE_DIM - 1, 3), "Prediction joint count must match input joint count"),
    ],
)
def test_overlay_rejects_invalid_raw_predictions(tmp_path: Path, shape: tuple[int, ...], message: str) -> None:
    dataset, _states = _synthetic_dataset(tmp_path, 100)
    predictions = _predictions(tmp_path, np.zeros(shape, dtype=np.float32))
    output = tmp_path / "invalid.rrd"

    with pytest.raises(RerunAdapterError, match=message):
        groot_predictions_to_rerun(predictions, dataset, output)

    assert not output.exists()


@pytest.mark.parametrize("duration", [0.0, -1.0])
def test_overlay_rejects_nonpositive_duration(tmp_path: Path, duration: float) -> None:
    dataset, states = _synthetic_dataset(tmp_path, 10)
    predictions = _predictions(tmp_path, states)

    with pytest.raises(RerunAdapterError, match="duration_s must be positive"):
        groot_predictions_to_rerun(predictions, dataset, tmp_path / "invalid.rrd", duration_s=duration)
