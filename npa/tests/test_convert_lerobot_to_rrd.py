"""Exercise SDK output routing with a real local dataset and Rerun encoder."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from npa import convert
from npa.adapter.isaac_lab_lerobot import G1_STATE_DIM


def _dataset(root: Path) -> tuple[Path, Path]:
    dataset = root / 'dataset'
    data = dataset / 'data' / 'chunk-000' / 'file-000.parquet'
    data.parent.mkdir(parents=True)
    states = np.zeros((3, G1_STATE_DIM), dtype=np.float32)
    states[:, 6] = [0.0, 0.1, 0.2]
    pq.write_table(pa.table({'observation.state': states.tolist(), 'index': [0, 1, 2]}), data)
    meta = dataset / 'meta' / 'info.json'
    meta.parent.mkdir()
    meta.write_text(json.dumps({'fps': 10, 'robot_type': 'unitree_g1'}))
    predictions = root / 'predictions.json'
    predictions.write_text(json.dumps({'predicted_actions': states.tolist()}))
    return dataset, predictions


def _assert_recording(path: Path, *, overlay: bool) -> None:
    from rerun.recording import load_recording

    assert path.stat().st_size > 0
    chunks = list(load_recording(path).chunks())
    roots = ['/world/skeleton'] + (['/world/predictions'] if overlay else [])
    for root in roots:
        assert sum(int(c.num_rows) for c in chunks if str(c.entity_path) == f'{root}/joints' and not c.is_static) == 3


@pytest.mark.parametrize('overlay', [False, True], ids=['dataset', 'predictions'])
def test_sdk_rrd_s3_output_uploads_exact_uri(tmp_path: Path, monkeypatch, overlay: bool) -> None:
    dataset, predictions = _dataset(tmp_path)
    monkeypatch.chdir(tmp_path)
    destination = 's3://synthetic-bucket/reports/conversion.rrd'
    uploaded = []

    class Storage:
        def upload_file(self, local_file: str, output_uri: str) -> str:
            _assert_recording(Path(local_file), overlay=overlay)
            uploaded.append(output_uri)
            return output_uri

    for name in ('lerobot_to_rerun', 'groot_predictions_to_rerun'):
        adapter = importlib.import_module(f'npa.viz.adapters.{name}')
        monkeypatch.setattr(adapter, '_storage_client', lambda *_args: Storage())

    result = convert.lerobot_to_rrd(
        input_path=dataset,
        output_path=destination,
        predictions_path=predictions if overlay else None,
    )

    assert uploaded == [destination], f'SDK returned {result!r}; local s3: directory exists={(tmp_path / "s3:").exists()}'
    assert result == destination
    assert not (tmp_path / 's3:').exists()


@pytest.mark.parametrize('overlay', [False, True], ids=['dataset', 'predictions'])
def test_sdk_rrd_local_output_remains_path(tmp_path: Path, overlay: bool) -> None:
    dataset, predictions = _dataset(tmp_path)
    destination = tmp_path / 'reports' / 'conversion.rrd'
    result = convert.lerobot_to_rrd(
        input_path=dataset,
        output_path=str(destination),
        predictions_path=predictions if overlay else None,
    )
    assert isinstance(result, Path)
    assert result == destination
    _assert_recording(destination, overlay=overlay)
