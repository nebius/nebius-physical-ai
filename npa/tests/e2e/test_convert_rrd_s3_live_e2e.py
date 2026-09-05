"""Live SDK upload/readback proof using a synthetic four-frame LeRobot dataset.

Run with NPA_INTEGRATION_E2E=1 and NPA_E2E_CONVERT_RRD_S3_DESTINATIONS_FILE
pointing to private JSON with exact, unused ``dataset`` and ``predictions`` S3
object URIs. Existing NPA storage credentials must permit GET and PUT on those
two objects. No bucket or IAM resources are created. The caller owns cleanup of
the retained RRD objects; repeated runs require fresh explicitly assigned keys.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rerun.recording import load_recording

from npa import convert
from npa.adapter.isaac_lab_lerobot import G1_STATE_DIM
from npa.clients.storage import StorageClient
from npa.viz.adapters.lerobot_to_rerun import _storage_client

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def destinations() -> dict[str, str]:
    source = os.environ.get("NPA_E2E_CONVERT_RRD_S3_DESTINATIONS_FILE", "")
    if os.environ.get("NPA_INTEGRATION_E2E") != "1" or not source:
        pytest.skip("Requires live opt-in and explicit unused S3 destination keys")
    values = json.loads(Path(source).read_text())
    if not isinstance(values, dict) or set(values) != {"dataset", "predictions"}:
        pytest.fail("Destination file must map dataset and predictions to exact S3 object URIs")
    for value in values.values():
        if not isinstance(value, str):
            pytest.fail("Each destination must be an S3 URI string")
        parsed = urlparse(value)
        if (
            parsed.scheme != "s3"
            or not parsed.netloc
            or not parsed.path.endswith(".rrd")
            or parsed.query
            or parsed.fragment
        ):
            pytest.fail("Each destination must be an exact S3 RRD object URI")
    if len(set(values.values())) != 2:
        pytest.fail("Each conversion mode requires its own destination object")
    return values


@pytest.mark.parametrize("mode", ["dataset", "predictions"])
def test_sdk_rrd_s3_upload_and_readback(
    tmp_path: Path, monkeypatch, record_property, destinations: dict[str, str], mode: str
) -> None:
    destination = destinations[mode]
    storage = _storage_client(destination)
    if storage.read_bytes_with_etag(destination) is not None:
        pytest.fail("Refusing to overwrite an existing destination object")

    dataset = tmp_path / "synthetic-dataset"
    data = dataset / "data" / "chunk-000" / "file-000.parquet"
    data.parent.mkdir(parents=True)
    states = np.zeros((4, G1_STATE_DIM), dtype=np.float32)
    states[:, 6] = [0.0, 0.1, 0.2, 0.3]
    pq.write_table(pa.table({"observation.state": states.tolist(), "index": [0, 1, 2, 3]}), data)
    metadata = dataset / "meta" / "info.json"
    metadata.parent.mkdir()
    metadata.write_text(json.dumps({"fps": 4, "robot_type": "unitree_g1"}))
    predictions = tmp_path / "synthetic-predictions.json"
    predictions.write_text(json.dumps({"predicted_actions": states.tolist()}))
    uploaded = []
    original_upload = StorageClient.upload_file

    def capture_real_upload(self, local_file: str, output_uri: str) -> str:
        if output_uri != destination:
            pytest.fail("SDK attempted an upload outside the assigned destination")
        uploaded.append(Path(local_file).read_bytes())
        return original_upload(self, local_file, output_uri)

    monkeypatch.setattr(StorageClient, "upload_file", capture_real_upload)
    monkeypatch.chdir(tmp_path)
    returned = convert.lerobot_to_rrd(
        input_path=dataset,
        output_path=destination,
        predictions_path=predictions if mode == "predictions" else None,
    )
    if returned != destination:
        pytest.fail("SDK did not return the unchanged destination URI")
    assert len(uploaded) == 1
    assert not (tmp_path / "s3:").exists()
    downloaded = tmp_path / "downloaded.rrd"
    storage.download_file(destination, str(downloaded))
    payload = downloaded.read_bytes()
    assert payload and payload == uploaded[0]
    digest = hashlib.sha256(payload).hexdigest()
    assert digest == hashlib.sha256(uploaded[0]).hexdigest()

    chunks = list(load_recording(downloaded).chunks())
    roots = ["/world/skeleton"] + (["/world/predictions"] if mode == "predictions" else [])
    for root in roots:
        times = []
        for chunk in chunks:
            if str(chunk.entity_path) == f"{root}/joints" and not chunk.is_static:
                times.extend(chunk.to_record_batch().column("frame_time").cast(pa.int64()).to_pylist())
        assert sorted(times) == [0, 250_000_000, 500_000_000, 750_000_000]
    record_property("rrd_sha256", digest)
    record_property("rrd_bytes", len(payload))
    record_property("decoded_frames_per_entity", 4)
    record_property("decoded_joint_entities", len(roots))
