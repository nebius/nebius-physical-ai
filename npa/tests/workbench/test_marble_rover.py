"""Check embodied geometry, clock synchronization, and motion evidence failures."""

import numpy as np
import pytest

from npa.workbench.marble.api import MarbleError
from npa.workbench.marble.rover_physics import _cameras, _validate_motion
from npa.workbench.marble.schemas import RoverRequest


def test_report_rejects_tampered_physics_evidence(tmp_path):
    from npa.workbench.marble.rover_report import _evidence

    (tmp_path / "trajectory.json").write_text("[]")
    with pytest.raises(MarbleError, match="trajectory hash mismatch"):
        _evidence(tmp_path, {"files": {"trajectory.json": {"sha256": "different"}}})


def test_sensor_camera_tracks_rigid_body_and_preserves_optical_axes():
    class Bullet:
        @staticmethod
        def getMatrixFromQuaternion(_):
            return np.eye(3).flatten()

    onboard, observer = _cameras(Bullet(), [2, 3, 0.24], [0, 0, 0, 1])
    assert np.allclose(onboard[:3, 3], [2.24, 0.8, -3])
    assert np.allclose(onboard[:3, 2], [1, 0, 0])
    assert np.allclose(onboard[:3, 1], [0, -1, 0])
    for pose in [onboard, observer]:
        assert np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6)
        assert np.linalg.det(pose[:3, :3]) == pytest.approx(1)


@pytest.mark.parametrize(
    "positions,contacts",
    [
        ([[0, 0, 0.25], [0, 0, 0.25]], 4),
        ([[0, 0, 0.25], [2, 0, 0.25]], 0),
        ([[0, 0, 0.25], [2, 0, -5]], 4),
    ],
)
def test_unsupported_or_motionless_rollout_is_rejected(positions, contacts):
    records = [
        {"position_bullet": position, "contacts": contacts} for position in positions
    ]
    with pytest.raises(MarbleError):
        _validate_motion(records)


def test_sensor_frequency_must_divide_physics_clock():
    with pytest.raises(ValueError, match="must divide"):
        RoverRequest(
            input_path="s3://example-bucket/world",
            output_path="s3://example-bucket/out",
            run_id="test",
            sensor_hz=11,
        )
