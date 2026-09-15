from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rerun.recording import load_recording

from npa.adapter.isaac_lab_lerobot import G1_STATE_DIM, G1_STATE_NAMES_43
from npa.viz.adapters.lerobot_to_rerun import lerobot_to_rerun
from npa.viz.lerobot import (
    VizDataError,
    g1_state_vectors_to_skeleton,
    load_lerobot_state_vectors,
    load_render_inputs,
)


def _dataset(root: Path, info: object, *, frames: int = 61) -> np.ndarray:
    """Write distinguishable G1 states out of order in real Parquet files."""
    states = np.arange(frames * G1_STATE_DIM, dtype=np.float32).reshape(frames, G1_STATE_DIM) / 10000
    data = root / "data" / "chunk-000"
    data.mkdir(parents=True)
    for part, indices in enumerate((np.arange(frames - 1, -1, -2), np.arange(frames - 2, -1, -2))):
        pq.write_table(
            pa.table({"index": indices, "observation.state": states[indices].tolist()}),
            data / f"file-{part:03d}.parquet",
        )
    meta = root / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps(info))
    return states


@pytest.mark.parametrize("fps", [29.97, 59.94, 0.5, 30])
def test_source_cadence_preserves_numeric_states_and_render_duration(tmp_path: Path, fps: float) -> None:
    states = _dataset(tmp_path, {"fps": fps, "task": "Synthetic G1 cadence"})

    loaded_states, source_fps, title = load_lerobot_state_vectors(tmp_path)

    np.testing.assert_array_equal(loaded_states, states)
    assert source_fps == fps
    assert title == "Synthetic G1 cadence"
    rendered = load_render_inputs(tmp_path, output_fps=30)
    duration = min(len(states) / fps, 10.0)
    assert rendered.source_fps == fps
    assert rendered.duration_s == pytest.approx(duration)
    # Preserve the established selection across the full source under a cap.
    indices = np.rint(np.linspace(0, len(states) - 1, round(duration * 30))).astype(np.int64)
    np.testing.assert_array_equal(rendered.frame_indices, indices)
    np.testing.assert_array_equal(rendered.skeleton_data, g1_state_vectors_to_skeleton(states[indices]))


def test_fractional_source_resampling_holds_last_frame_at_requested_output_rate(tmp_path: Path) -> None:
    states = _dataset(tmp_path, {"fps": 2.5}, frames=3)

    rendered = load_render_inputs(tmp_path, duration_s=2.0, output_fps=5)

    assert rendered.duration_s == 2.0
    assert rendered.source_fps == 2.5
    indices = np.array([0, 0, 1, 1, 2, 2, 2, 2, 2, 2])
    np.testing.assert_array_equal(rendered.frame_indices, indices)
    np.testing.assert_array_equal(rendered.skeleton_data, g1_state_vectors_to_skeleton(states[indices]))


@pytest.mark.parametrize("info", [{}, {"fps": None}, {"fps": 0}, {"fps": -1}, {"fps": -0.5},
                                  {"fps": ""}, {"fps": False}, {"fps": []}, [], None])
def test_missing_or_nonpositive_metadata_keeps_default_cadence(tmp_path: Path, info: object) -> None:
    _dataset(tmp_path, info, frames=3)
    assert load_lerobot_state_vectors(tmp_path)[1] == 30
    assert load_render_inputs(tmp_path).duration_s == pytest.approx(0.1)


def test_absent_metadata_keeps_default_cadence(tmp_path: Path) -> None:
    _dataset(tmp_path, {}, frames=3)
    (tmp_path / "meta" / "info.json").unlink()
    assert load_lerobot_state_vectors(tmp_path)[1] == 30


@pytest.mark.parametrize("fps, expected", [("30", 30), (True, 1), (30.0, 30)])
def test_existing_integer_metadata_coercion(tmp_path: Path, fps: object, expected: int) -> None:
    _dataset(tmp_path, {"fps": fps})
    assert load_lerobot_state_vectors(tmp_path)[1] == expected


@pytest.mark.parametrize("fps, error", [("invalid", ValueError), ("29.97", ValueError),
                                       (float("nan"), ValueError), (float("inf"), OverflowError),
                                       (-float("inf"), OverflowError), ([30], TypeError),
                                       ({"value": 30}, TypeError)])
def test_invalid_metadata_keeps_existing_rejection(tmp_path: Path, fps: object, error: type[Exception]) -> None:
    _dataset(tmp_path, {"fps": fps})
    with pytest.raises(error):
        load_lerobot_state_vectors(tmp_path)


def test_malformed_metadata_keeps_existing_error(tmp_path: Path) -> None:
    _dataset(tmp_path, {})
    (tmp_path / "meta" / "info.json").write_text("{")
    with pytest.raises(VizDataError, match="Invalid LeRobotDataset info.json"):
        load_lerobot_state_vectors(tmp_path)


@pytest.mark.parametrize("fps, indices", [(29.97, list(range(7))), (59.94, list(range(7))),
                                         (0.5, [0, 6]), (30, list(range(7)))])
def test_g1_rrd_decodes_recorded_cadence_and_selected_states(
    tmp_path: Path, fps: float, indices: list[int],
) -> None:
    dataset = tmp_path / "dataset"
    states = _dataset(dataset, {"fps": fps}, frames=7)
    output = tmp_path / "synthetic-g1.rrd"

    lerobot_to_rerun(dataset, output)

    chunks = list(load_recording(output).chunks())
    joint_name = "left_shoulder_pitch_joint"
    angle_rows = []
    skeleton_rows = []
    for chunk in chunks:
        if chunk.is_static:
            continue
        batch = chunk.to_record_batch()
        times = batch.column("frame_time").cast(pa.int64()).to_pylist()
        if str(chunk.entity_path) == f"/world/skeleton/angles/{joint_name}":
            angle_rows.extend(zip(times, batch.column("Scalars:scalars").to_pylist(), strict=True))
        if str(chunk.entity_path) == "/world/skeleton/joints":
            skeleton_rows.extend(zip(times, batch.column("Points3D:positions").to_pylist(), strict=True))
    angle_rows.sort(key=lambda row: row[0])
    skeleton_rows.sort(key=lambda row: row[0])
    expected_times = np.arange(len(indices)) / fps * 1_000_000_000
    np.testing.assert_allclose([row[0] for row in angle_rows], expected_times, rtol=0, atol=1)
    np.testing.assert_allclose([row[0] for row in skeleton_rows], expected_times, rtol=0, atol=1)
    np.testing.assert_array_equal(
        np.asarray([row[1] for row in angle_rows]).ravel(),
        states[indices, list(G1_STATE_NAMES_43).index(joint_name)],
    )
    np.testing.assert_array_equal(
        np.asarray([row[1] for row in skeleton_rows], dtype=np.float32),
        g1_state_vectors_to_skeleton(states[indices]),
    )
