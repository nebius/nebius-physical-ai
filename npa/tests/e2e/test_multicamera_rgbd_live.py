"""Opt-in S3 readback of real procedural rig capture; never substitute unit pixels.

Submit multicamera-rgbd-capture.yaml through the standard workflow runtime first.
Set NPA_INTEGRATION_E2E=1, NPA_RGBD_LIVE_MANIFEST_URI and NPA_RGBD_LIVE_REPORT_URI
to its capture manifest and a fresh validation report object. No resources are
provisioned by this test. The matrix separately covers workflow submission.
"""

from __future__ import annotations

import json
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
