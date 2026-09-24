"""Verify an operator-selected warehouse capture through its durable NPA receipt.

Set NPA_INTEGRATION_E2E=1 and NPA_ANTIOCH_WAREHOUSE_REQUEST to the private JSON
request used for a completed, collected warehouse_spot_patrol. This test reads
existing evidence; it never dispatches a scenario or creates GPU resources.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path

import av
import numpy as np
import pytest

from npa.sdk.workbench.antioch import status
from npa.workbench.antioch.schemas import ResumeRequest
from npa.workbench.antioch.storage_config import resolve_storage_client

pytestmark = [pytest.mark.e2e, pytest.mark.gpu]


def _read(storage, uri):
    result = storage.read_bytes_with_etag(uri)
    assert result is not None, "Expected retained evidence is missing"
    return result[0]


def _artifact(storage, manifest, suffix):
    matches = [a for a in manifest["artifacts"] if a["name"].endswith(suffix)]
    assert len(matches) == 1
    artifact = matches[0]
    data = _read(storage, artifact["uri"])
    assert len(data) == artifact["size_bytes"]
    assert hashlib.sha256(data).hexdigest() == artifact["sha256"]
    return data


def _verify_motion(evidence, parameters):
    metrics = evidence["metrics"]
    assert metrics["finite_state"] is True
    assert metrics["minimum_base_height_m"] > 0.35
    assert metrics["minimum_body_up_z"] > 0.7
    assert metrics["maximum_joint_range_rad"] > 0.2
    assert metrics["displacement_m"] > 1.0
    assert metrics["policy_steps"] > 0
    assert metrics["physics_dt_s"] == pytest.approx(metrics["policy_physics_dt_s"])
    assert metrics["policy_decimation"] > 0
    assert metrics["minimum_frame_std"] > 10
    expected = round(parameters["seconds"] * 25)
    assert metrics["frames"] == metrics["expected_frames"] == expected
    assert len(evidence["states"]) == expected
    positions = np.asarray([s["position"] for s in evidence["states"]])
    assert np.isfinite(positions).all()
    assert np.linalg.norm(positions[-1, :2] - positions[0, :2]) == pytest.approx(
        metrics["displacement_m"]
    )
    clocks = np.asarray(evidence["camera_timestamps"])
    assert clocks.shape == (expected, 2)
    assert (np.diff(clocks, axis=0) > 0).all()
    assert np.diff(clocks, axis=0) == pytest.approx(
        np.full((expected - 1, 2), 1 / 25), abs=1e-5
    )


def _verify_video(data, parameters):
    with av.open(io.BytesIO(data)) as container:
        stream = container.streams.video[0]
        assert stream.width == parameters["width"]
        assert stream.height == parameters["width"] * 9 // 16
        assert stream.average_rate == 25
        times = [float(frame.time) for frame in container.decode(stream)]
    assert len(times) == round(parameters["seconds"] * 25)
    assert np.diff(times) == pytest.approx(np.full(len(times) - 1, 1 / 25))


def test_collected_warehouse_video_matches_successful_physics_run():
    selected = os.environ.get("NPA_ANTIOCH_WAREHOUSE_REQUEST", "")
    if not selected or os.environ.get("NPA_INTEGRATION_E2E") != "1":
        pytest.skip("Select an existing private warehouse request for live evidence")
    request = json.loads(Path(selected).expanduser().read_text())
    resume = ResumeRequest(
        **{key: request[key] for key in ("output_path", "workflow_run", "state_id")}
    )
    record = status(resume)
    assert record.status == "completed"
    assert record.remote_outcome == "passed"
    assert record.parameters == request["parameters"]
    storage = resolve_storage_client()
    receipt = json.loads(_read(storage, record.completion_uri))
    manifest_bytes = _read(storage, receipt["manifest_uri"])
    assert hashlib.sha256(manifest_bytes).hexdigest() == receipt["manifest_sha256"]
    manifest = json.loads(manifest_bytes)
    assert manifest["source"]["source_sha256"] == request["source_sha256"]
    assert manifest["remote"]["outcome"] == "passed"
    checks = manifest["remote"]["results"]["checks"]
    assert len(checks) == 7 and all(check["passed"] for check in checks)
    evidence = json.loads(_artifact(storage, manifest, "measurements.json"))
    _verify_motion(evidence, request["parameters"])
    _verify_video(
        _artifact(storage, manifest, "warehouse_spot.mp4"), request["parameters"]
    )
