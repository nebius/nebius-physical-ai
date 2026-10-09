"""Preserve independent photographic camera poses in portable NuRec artifacts."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import struct
import tempfile
from zipfile import ZIP_STORED, ZipFile, ZipInfo

import numpy as np

from npa.workbench.nurec.nurec import NurecError, read_rig_sidecar

_MEMBER = "npa-capture-trajectory.json"
_SCHEMA = "npa.nurec.capture-trajectory.v1"


def attach_capture_trajectory(ncore_json: str, artifact: Path) -> None:
    """Attach exact photographic poses without changing native model data.

    Args:
        ncore_json: Verified NCore metadata with a derived-rig sidecar.
        artifact: Completed native USDZ to augment atomically.
    Returns:
        None. Native rigid-rig captures are left unchanged.
    Raises:
        NurecError: Camera poses cannot match the artifact's frame timestamps.
        OSError: Artifact or source data cannot be read or written.
    """
    if not read_rig_sidecar(ncore_json).get("reference_camera"):
        return
    from npa.workbench.nurec.ncore_rig import camera_world_trajectories

    with ZipFile(artifact) as archive:
        native = archive.read("rig_trajectories.json")
    trajectory = _photographic_trajectory(
        json.loads(native), camera_world_trajectories(ncore_json)
    )
    payload = {
        "schema": _SCHEMA,
        "native_rig_sha256": hashlib.sha256(native).hexdigest(),
        "source_meta_sha256": hashlib.sha256(Path(ncore_json).read_bytes()).hexdigest(),
        "offset_frame": "each photographic camera's local coordinate frame",
        "trajectory": trajectory,
    }
    _append_aligned_member(artifact, json.dumps(payload).encode("utf-8"))


def _photographic_trajectory(native: dict, poses: dict) -> dict:
    trajectory = copy.deepcopy(native)
    rigs = trajectory["rig_trajectories"]
    if len(rigs) != 1:
        raise NurecError("photographic rendering requires one capture sequence")
    rig = rigs[0]
    for key, calibration in trajectory["camera_calibrations"].items():
        camera = calibration["logical_sensor_name"]
        if camera not in poses:
            raise NurecError(f"capture trajectory is missing camera {camera!r}")
        requested = np.asarray(rig["cameras_frame_timestamps_us"][key])
        selected = _exact_frame_poses(camera, poses[camera], requested)
        rig["cameras_frame_T_rig_worlds"][key] = selected.tolist()
        calibration["T_sensor_rig"] = np.eye(4).tolist()
    return trajectory


def _exact_frame_poses(camera: str, trajectory: tuple, requested: np.ndarray):
    matrices, times = map(np.asarray, trajectory)
    if (
        times.ndim != 1
        or not len(times)
        or np.any(np.diff(times.astype("int64")) <= 0)
        or requested.ndim != 2
        or requested.shape[1] != 2
        or not len(requested)
        or requested.dtype.kind not in "iu"
        or times.dtype.kind not in "iu"
        or np.any(requested[:, 0] > requested[:, 1])
    ):
        raise NurecError(f"invalid photographic timestamps for {camera!r}")
    _validate_poses(matrices, (len(times), 4, 4))
    indices = np.searchsorted(times, requested)
    if np.any(indices >= len(times)) or not np.array_equal(times[indices], requested):
        raise NurecError(
            f"photographic poses do not cover exact frame times for {camera!r}"
        )
    return matrices[indices]


def _validate_poses(matrices: np.ndarray, shape: tuple) -> None:
    if matrices.shape != shape or not np.isfinite(matrices).all():
        raise NurecError("capture trajectory requires finite SE(3) poses")
    rotations = matrices[..., :3, :3]
    if (
        not np.allclose(matrices[..., 3, :], [0, 0, 0, 1], atol=1e-5)
        or not np.allclose(
            np.swapaxes(rotations, -1, -2) @ rotations, np.eye(3), atol=1e-4
        )
        or not np.allclose(np.linalg.det(rotations), 1, atol=1e-4)
    ):
        raise NurecError("capture trajectory contains a non-rigid camera transform")


def _append_aligned_member(artifact: Path, data: bytes) -> None:
    # USDZ entries must be uncompressed and start on a 64-byte boundary.
    with tempfile.NamedTemporaryFile(
        dir=artifact.parent, suffix=".usdz", delete=False
    ) as file:
        staged = Path(file.name)
    try:
        shutil.copyfile(artifact, staged)
        with ZipFile(staged, "a", compression=ZIP_STORED) as archive:
            if _MEMBER in archive.namelist():
                raise NurecError(
                    "capture trajectory is already attached to this artifact"
                )
            info = ZipInfo(_MEMBER)
            padding = (-(archive.start_dir + 30 + len(_MEMBER) + 4)) % 64
            info.extra = struct.pack("<HH", 0x1986, padding) + bytes(padding)
            archive.writestr(info, data)
        staged.replace(artifact)
    finally:
        staged.unlink(missing_ok=True)


def prepare_capture_render(
    artifact: str, output_dir: str, *, dry_run: bool = False
) -> str:
    """Materialize the artifact's photographic trajectory for native rendering.

    Args:
        artifact: USDZ carrying optional NPA photographic pose metadata.
        output_dir: Fresh render generation used for the trajectory JSON.
        dry_run: Validate and return the planned path without writing it.
    Returns:
        Native trajectory filename, or empty for an unaugmented artifact.
    Raises:
        NurecError: The embedded trajectory is invalid or belongs to another rig.
        OSError: The output cannot be written.
    """
    from zipfile import is_zipfile

    if not is_zipfile(artifact):
        return ""
    with ZipFile(artifact) as archive:
        if _MEMBER not in archive.namelist():
            return ""
        payload = json.loads(archive.read(_MEMBER))
        native = archive.read("rig_trajectories.json")
    _validate_payload(payload, native)
    target = Path(output_dir) / "capture-trajectory.json"
    if not dry_run:
        target.write_text(json.dumps(payload["trajectory"]), encoding="utf-8")
    return str(target)


def _validate_payload(payload: dict, native: bytes) -> None:
    if (
        payload.get("schema") != _SCHEMA
        or payload.get("native_rig_sha256") != hashlib.sha256(native).hexdigest()
    ):
        raise NurecError("capture trajectory does not match the artifact's native rig")
    original = json.loads(native)
    trajectory = payload["trajectory"]
    expected = copy.deepcopy(original)
    if len(trajectory["rig_trajectories"]) != 1:
        raise NurecError("capture trajectory requires one sequence")
    for key, calibration in expected["camera_calibrations"].items():
        calibration["T_sensor_rig"] = np.eye(4).tolist()
        frames = len(
            expected["rig_trajectories"][0]["cameras_frame_timestamps_us"][key]
        )
        poses = trajectory["rig_trajectories"][0]["cameras_frame_T_rig_worlds"][key]
        _validate_poses(np.asarray(poses), (frames, 2, 4, 4))
        expected["rig_trajectories"][0]["cameras_frame_T_rig_worlds"][key] = poses
    if trajectory != expected:
        raise NurecError(
            "capture trajectory changed intrinsics, frame times or world coordinates"
        )
