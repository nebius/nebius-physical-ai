"""Decode real frame recordings to check layout and prediction horizons."""

from pathlib import Path

import numpy as np
import pytest
from rerun.recording import load_recording

from npa.viz.backends import rerun as backend


CONNECTIONS = [(0, 6), (6, 7), (7, 18), (18, 32), (32, 33)]
JOINT_INDICES = (6, 7, 18, 32)
FRAME_COUNT = 4
FPS = 4


def _trajectory(moving: bool) -> np.ndarray:
    data = np.zeros((FRAME_COUNT, 34, 3), dtype=np.float32)
    data[:, :, 0] = np.linspace(-0.2, 0.2, 34)
    data[:, :, 2] = np.linspace(0.0, 1.0, 34)
    if moving:
        for frame in range(FRAME_COUNT):
            data[frame] += np.array([frame * 0.4, frame * 0.1, frame * 0.125])
    return data


def _static_geometry(chunks, entity: str, column: str) -> np.ndarray:
    matches = [chunk for chunk in chunks if str(chunk.entity_path) == entity]
    assert len(matches) == 1
    assert matches[0].is_static
    batch = matches[0].to_record_batch()
    assert batch.num_rows == 1
    return np.asarray(batch.column(column).to_pylist()[0])


def _assert_geometry(chunks, root: str, expected: np.ndarray) -> None:
    np.testing.assert_allclose(
        _static_geometry(chunks, f"{root}/joints", "Points3D:positions"),
        expected,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        _static_geometry(chunks, f"{root}/bones", "LineStrips3D:strips"),
        expected[np.asarray(CONNECTIONS)],
        atol=1e-6,
    )


def _assert_motion_series(chunks, root: str, source: np.ndarray) -> None:
    for joint in JOINT_INDICES:
        rows = []
        for chunk in chunks:
            if str(chunk.entity_path) != f"{root}/angles/joint_{joint}" or chunk.is_static:
                continue
            batch = chunk.to_record_batch()
            timestamps = batch.column("frame_time").cast("int64").to_pylist()
            values = batch.column("Scalars:scalars").to_pylist()
            rows.extend(zip(timestamps, values, strict=True))
        rows.sort(key=lambda row: row[0])
        assert [row[0] for row in rows] == [frame * 250_000_000 for frame in range(len(source))]
        np.testing.assert_allclose([row[1][0] for row in rows], source[:, joint, 2])


@pytest.mark.parametrize("moving", [False, True], ids=["stationary", "moving"])
@pytest.mark.parametrize(
    ("layout", "horizon"),
    [(layout, horizon) for layout in ("side-by-side", "overlay", "single") for horizon in (1, 2, 4)]
    + [("single", None)],
)
def test_frame_recordings_preserve_layout_and_prediction_horizon(
    tmp_path: Path, moving: bool, layout: str, horizon: int | None
) -> None:
    skeleton = _trajectory(moving)
    predictions = (
        None
        if horizon is None
        else skeleton[:horizon] + np.array([0.1, 0.2, 0.3], dtype=np.float32)
    )
    original_skeleton = skeleton.copy()
    original_predictions = None if predictions is None else predictions.copy()
    # These fixtures exercise both the minimum separation and trajectory-wide span.
    separation = (3.06 if horizon == FRAME_COUNT else 2.88) if moving else 1.2
    input_shift = np.array([-separation / 2, 0, 0]) if layout == "side-by-side" else np.zeros(3)
    paths = backend._write_frame_recordings(
        skeleton, predictions, layout, tmp_path / "recordings", FPS, 1.0,
        "Synthetic prediction horizon", CONNECTIONS,
    )

    assert [path.name for path in paths] == [f"frame_{frame:06d}.rrd" for frame in range(FRAME_COUNT)]
    recording_ids = set()
    for frame, path in enumerate(paths):
        assert path.stat().st_size > 0
        recording = load_recording(path)
        assert recording.application_id() == backend.APPLICATION_ID
        assert recording.recording_id()
        recording_ids.add(recording.recording_id())
        chunks = list(recording.chunks())
        _assert_geometry(chunks, "/world/input", skeleton[frame] + input_shift)
        if predictions is not None and frame < len(predictions) and layout != "single":
            _assert_geometry(chunks, "/world/predictions", predictions[frame] - input_shift)
        else:
            entities = {str(chunk.entity_path) for chunk in chunks}
            assert "/world/predictions/joints" not in entities
            assert "/world/predictions/bones" not in entities
        _assert_motion_series(chunks, "/world/input", skeleton)
        if predictions is not None:
            _assert_motion_series(chunks, "/world/predictions", predictions)
        else:
            assert not any(str(chunk.entity_path).startswith("/world/predictions/") for chunk in chunks)

    assert len(recording_ids) == FRAME_COUNT
    np.testing.assert_array_equal(skeleton, original_skeleton)
    if predictions is not None:
        np.testing.assert_array_equal(predictions, original_predictions)
