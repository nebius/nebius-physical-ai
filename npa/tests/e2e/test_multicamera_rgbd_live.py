"""Opt-in S3 readback of real procedural rig capture; never substitute unit pixels.

Submit multicamera-rgbd-capture.yaml through the standard workflow runtime first.
Set NPA_INTEGRATION_E2E=1, NPA_RGBD_LIVE_MANIFEST_URI and NPA_RGBD_LIVE_REPORT_URI
to its capture manifest and a fresh validation report object. No resources are
provisioned by this test. The matrix separately covers workflow submission.
"""

from __future__ import annotations

import json
import hashlib
import os

import numpy as np
from PIL import Image
import pytest

from npa.workflows.isaac_rgbd.transport import validate_s3

pytestmark = [pytest.mark.e2e, pytest.mark.gpu]


def _qualified_fixture_geometry(root, manifest):
    frame = manifest["frames"][0]
    for view in frame["views"]:
        depth = np.load(root / view["artifacts"]["depth"], allow_pickle=False)
        with Image.open(root / view["artifacts"]["rgb"]) as image:
            rgb = np.asarray(image)
        assert np.ptp(rgb.astype(float)) > 0, "fixture RGB is flat"
        # Central rays hit the inner room walls: 5 - 0.1 - 0.15 = 4.75 meters.
        np.testing.assert_allclose(depth[119:121, 159:161], 4.75, atol=0.02, rtol=0)


def test_four_camera_capture_decodes_and_matches_known_metric_room(tmp_path):
    if os.environ.get("NPA_INTEGRATION_E2E") != "1":
        pytest.skip("set NPA_INTEGRATION_E2E=1 for live S3 readback")
    source = os.environ.get("NPA_RGBD_LIVE_MANIFEST_URI", "")
    output = os.environ.get("NPA_RGBD_LIVE_REPORT_URI", "")
    if not source or not output:
        pytest.fail(
            "supply NPA_RGBD_LIVE_MANIFEST_URI and a fresh NPA_RGBD_LIVE_REPORT_URI"
        )
    report = validate_s3(source, output, tmp_path)
    assert report["frames"] == 3 and report["cameras"] == 4 and report["views"] == 12
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["provenance"]["scope"] == "procedural-room"
    assert manifest["provenance"]["replicator_version"] != "unit-test"
    assert manifest["request"]["pointcloud"] is True
    _qualified_fixture_geometry(tmp_path, manifest)


def test_warehouse_run_publishes_full_validation_and_portable_preview(tmp_path):
    if os.environ.get("NPA_INTEGRATION_E2E") != "1":
        pytest.skip("set NPA_INTEGRATION_E2E=1 for live S3 readback")
    prefix = os.environ.get("NPA_RGBD_WAREHOUSE_RUN_URI", "").rstrip("/")
    if not prefix:
        pytest.skip("set NPA_RGBD_WAREHOUSE_RUN_URI to a completed warehouse run")
    from npa.clients.storage import StorageClient

    storage = StorageClient.from_environment()
    for name in ("validation.json", "capture/manifest.json", "reports/index.html"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        storage.download_file(prefix + "/" + name, str(target))
    validation = json.loads((tmp_path / "validation.json").read_text())
    manifest_path = tmp_path / "capture/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    html = (tmp_path / "reports/index.html").read_bytes()
    assert validation["validated"] is True
    assert (validation["frames"], validation["cameras"], validation["views"]) == (
        265,
        4,
        1060,
    )
    assert (
        validation["manifest_sha256"]
        == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    )
    assert validation["preview"]["sha256"] == hashlib.sha256(html).hexdigest()
    assert validation["preview"]["poses"] == 32
    assert b"data:image/jpeg;base64," in html and b"connect-src 'none'" in html
    assert validation["fused_points"] == validation["valid_depth_pixels"] > 0
    assert manifest["provenance"]["scope"] == "supplied-usd"
    assert manifest["provenance"]["replicator_version"] != "unit-test"
    assert all(
        (camera["width"], camera["height"]) == (1280, 720)
        for camera in manifest["request"]["cameras"]
    )
