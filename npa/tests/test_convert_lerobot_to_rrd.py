"""SDK destination routing with synthetic LeRobot data and real RRD bytes."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from npa import convert
from npa.adapter.isaac_lab_lerobot import G1_STATE_DIM


@pytest.fixture(params=[False, True], ids=["dataset", "predictions"])
def conversion_input(tmp_path: Path, request) -> dict:
    dataset = tmp_path / "dataset"
    data = dataset / "data" / "chunk-000" / "file-000.parquet"
    data.parent.mkdir(parents=True)
    states = np.zeros((4, G1_STATE_DIM), dtype=np.float32)
    states[:, 6] = [0.0, 0.1, 0.2, 0.3]
    pq.write_table(
        pa.table({"observation.state": states.tolist(), "index": [0, 1, 2, 3]}),
        data,
    )
    metadata = dataset / "meta" / "info.json"
    metadata.parent.mkdir()
    metadata.write_text(json.dumps({"fps": 4, "robot_type": "unitree_g1"}))
    predictions = tmp_path / "predictions.json"
    predictions.write_text(json.dumps({"predicted_actions": states.tolist()}))
    return {
        "input_path": dataset,
        "predictions_path": predictions if request.param else None,
    }


def _assert_recording(path: Path, *, predictions: bool) -> None:
    from rerun.recording import load_recording

    assert path.stat().st_size > 0
    chunks = list(load_recording(path).chunks())
    roots = ["/world/skeleton"] + (["/world/predictions"] if predictions else [])
    for root in roots:
        timestamps = []
        for chunk in chunks:
            if str(chunk.entity_path) == f"{root}/joints" and not chunk.is_static:
                batch = chunk.to_record_batch()
                timestamps.extend(batch.column("frame_time").cast(pa.int64()).to_pylist())
        assert sorted(timestamps) == [0, 250_000_000, 500_000_000, 750_000_000]


def _mock_storage(monkeypatch, conversion_input: dict) -> Mock:
    name = "groot_predictions_to_rerun" if conversion_input["predictions_path"] else "lerobot_to_rerun"
    adapter = importlib.import_module(f"npa.viz.adapters.{name}")
    storage = Mock()
    monkeypatch.setattr(adapter, "_storage_client", lambda *_args: storage)
    return storage


def test_s3_output_uploads_recording_to_exact_uri(tmp_path: Path, monkeypatch, conversion_input: dict) -> None:
    monkeypatch.chdir(tmp_path)
    destination = "s3://synthetic-bucket/reports/nested/trajectory.rrd"
    received = tmp_path / "received.rrd"
    storage = _mock_storage(monkeypatch, conversion_input)

    def upload_file(local_file: str, output_uri: str) -> str:
        assert output_uri == destination
        received.write_bytes(Path(local_file).read_bytes())
        return output_uri

    storage.upload_file.side_effect = upload_file

    result = convert.lerobot_to_rrd(**conversion_input, output_path=destination)

    storage.upload_file.assert_called_once()
    assert result == destination
    assert not (tmp_path / "s3:").exists()
    _assert_recording(received, predictions=bool(conversion_input["predictions_path"]))


@pytest.mark.parametrize("path_type", [str, Path], ids=["string", "path"])
def test_local_output_returns_path(tmp_path: Path, monkeypatch, conversion_input: dict, path_type) -> None:
    monkeypatch.chdir(tmp_path)
    destination = Path("reports") / "trajectory.rrd"

    result = convert.lerobot_to_rrd(**conversion_input, output_path=path_type(destination))

    assert isinstance(result, Path)
    assert result == destination
    _assert_recording(result, predictions=bool(conversion_input["predictions_path"]))


def test_s3_upload_failure_propagates(tmp_path: Path, monkeypatch, conversion_input: dict) -> None:
    monkeypatch.chdir(tmp_path)
    storage = _mock_storage(monkeypatch, conversion_input)
    failure = OSError("synthetic upload failure")

    def fail_upload(local_file: str, output_uri: str) -> None:
        _assert_recording(Path(local_file), predictions=bool(conversion_input["predictions_path"]))
        raise failure

    storage.upload_file.side_effect = fail_upload

    with pytest.raises(OSError, match="synthetic upload failure") as caught:
        convert.lerobot_to_rrd(
            **conversion_input,
            output_path="s3://synthetic-bucket/reports/trajectory.rrd",
        )

    assert caught.value is failure
    storage.upload_file.assert_called_once()
    assert not (tmp_path / "s3:").exists()
