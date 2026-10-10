"""Check native reconstruction provenance and the geometry shown by the offline viewer."""

import base64
import json

import numpy as np
import pytest

from npa.workflows.lyra_reconstruction_demo import (
    _encode_preview,
    _payload,
    _view_points,
)


def test_points_use_depth_intrinsics_and_camera_pose():
    depth = np.array([[2.0, 2.0], [0.0, np.nan]])
    intrinsic = np.diag([2.0, 2.0, 1.0])
    extrinsic = np.eye(4)
    extrinsic[0, 3] = -3
    colors = np.full((2, 2, 3), 127, dtype=np.uint8)
    points, sampled_colors, scores = _view_points(
        depth, np.ones((2, 2)), colors, intrinsic, extrinsic, 1
    )
    np.testing.assert_allclose(points, [[3, 0, 2], [4, 0, 2]])
    np.testing.assert_equal(sampled_colors, [[127] * 3] * 2)
    np.testing.assert_equal(scores, [1, 1])


def test_uniform_confidence_remains_finite_and_display_orientation_is_explicit():
    points = np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]])
    encoded = _encode_preview([(points, np.full((2, 3), 255), np.ones(2))], {})
    displayed = np.frombuffer(base64.b64decode(encoded["position"]), dtype="<f4")
    np.testing.assert_allclose(displayed.reshape(-1, 3), [[0, -1, 2], [1, -2, 3]])
    confidence = np.frombuffer(base64.b64decode(encoded["confidence"]), dtype="<f4")
    assert np.isfinite(confidence).all()
    assert "relative scale" in encoded["scope"]


def test_standalone_viewer_rejects_incomplete_native_provenance(tmp_path):
    (tmp_path / "reconstruction.json").write_text(json.dumps({"checksums": {}}))
    with pytest.raises(ValueError, match="provenance is incomplete"):
        _payload(tmp_path)
