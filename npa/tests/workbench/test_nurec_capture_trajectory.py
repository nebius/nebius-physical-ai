"""Verify photographic poses survive USDZ export and native render invocation."""

import json
from pathlib import Path
import struct
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest

from npa.workbench.nurec import capture_trajectory as capture
from npa.workbench.nurec.nurec import NurecConfig, NurecError, render_novel_views


def _capture(tmp_path, monkeypatch):
    times = np.array([0, 1000000, 2000000], dtype=np.uint64)
    first = np.repeat(np.eye(4)[None], 3, axis=0)
    first[:, 0, 3] = [0, 1, 2]
    second = first.copy()
    second[:, 1, 3] = [5, 7, 9]
    poses = {"camera1": (first, times), "camera2": (second, times)}
    native = {
        "world_to_nre": {"matrix": np.eye(4).tolist()},
        "camera_calibrations": {
            camera: {
                "logical_sensor_name": camera,
                "T_sensor_rig": np.eye(4).tolist(),
                "camera_model_parameters": {"width": 2000, "height": 1500},
            }
            for camera in poses
        },
        "rig_trajectories": [
            {
                "cameras_frame_timestamps_us": {
                    camera: np.stack([times, times], 1).tolist() for camera in poses
                },
                "cameras_frame_T_rig_worlds": {
                    camera: np.stack([first, first], 1).tolist() for camera in poses
                },
            }
        ],
    }
    artifact = tmp_path / "last.usdz"
    with ZipFile(artifact, "w") as archive:
        archive.writestr("rig_trajectories.json", json.dumps(native))
        archive.writestr("checkpoint.ckpt", b"unchanged native model")
    meta = tmp_path / "sequence.json"
    meta.write_text("{}")
    monkeypatch.setattr(
        capture, "read_rig_sidecar", lambda path: {"reference_camera": "camera1"}
    )
    monkeypatch.setattr(
        "npa.workbench.nurec.ncore_rig.camera_world_trajectories", lambda path: poses
    )
    return artifact, meta, native, poses


def test_each_camera_preserves_its_actual_pose_and_native_model(tmp_path, monkeypatch):
    artifact, meta, native, poses = _capture(tmp_path, monkeypatch)
    capture.attach_capture_trajectory(str(meta), artifact)
    output = Path(capture.prepare_capture_render(str(artifact), str(tmp_path)))
    repaired = json.loads(output.read_text())
    for camera, (matrices, _) in poses.items():
        actual = np.asarray(
            repaired["rig_trajectories"][0]["cameras_frame_T_rig_worlds"][camera]
        )
        np.testing.assert_array_equal(actual[:, 0], matrices)
        np.testing.assert_array_equal(actual[:, 1], matrices)
    with ZipFile(artifact) as archive:
        assert archive.testzip() is None
        assert archive.read("checkpoint.ckpt") == b"unchanged native model"
        assert json.loads(archive.read("rig_trajectories.json")) == native
        info = archive.getinfo(capture._MEMBER)
    with artifact.open("rb") as stream:
        stream.seek(info.header_offset + 26)
        filename_length, extra_length = struct.unpack("<HH", stream.read(4))
    assert (info.header_offset + 30 + filename_length + extra_length) % 64 == 0
    with pytest.raises(NurecError, match="already attached"):
        capture.attach_capture_trajectory(str(meta), artifact)


def test_render_selects_per_frame_poses_and_keeps_nonzero_offset(tmp_path, monkeypatch):
    artifact, meta, _, _ = _capture(tmp_path, monkeypatch)
    capture.attach_capture_trajectory(str(meta), artifact)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        target = Path(command[command.index("--output-dir") + 1])
        (target / "000000.png").write_bytes(b"native render fixture")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    result = render_novel_views(
        NurecConfig(),
        artifact_path=str(artifact),
        output_dir=str(tmp_path / "render"),
        runner=run,
        export_video=False,
    )
    assert result.ok and result.novel_view
    command = commands[0]
    assert (
        command[command.index("--calib-source") + 1] == "training-rig-poses-per-frame"
    )
    assert command[
        command.index("--rig-translation-offset") + 1 : command.index(
            "--rig-translation-offset"
        )
        + 4
    ] == ["0.0", "0.25", "0.0"]
    assert "--no-replicate-training-views" in command
    assert Path(command[command.index("--custom-rig-trajectory") + 1]).is_file()


def test_dry_run_plans_photographic_poses_without_writing(tmp_path, monkeypatch):
    artifact, meta, _, _ = _capture(tmp_path, monkeypatch)
    capture.attach_capture_trajectory(str(meta), artifact)
    output = tmp_path / "not-created"
    result = render_novel_views(
        NurecConfig(),
        artifact_path=str(artifact),
        output_dir=str(output),
        dry_run=True,
        export_video=False,
    )
    assert "training-rig-poses-per-frame" in result.command
    assert not output.exists()


@pytest.mark.parametrize(
    "damage", ["intrinsics", "timestamp", "world", "rig", "nan", "scale"]
)
def test_rejects_changed_calibration_or_corrupt_pose_metadata(
    tmp_path, monkeypatch, damage
):
    artifact, meta, _, _ = _capture(tmp_path, monkeypatch)
    capture.attach_capture_trajectory(str(meta), artifact)
    with ZipFile(artifact) as archive:
        payload = json.loads(archive.read(capture._MEMBER))
        native = archive.read("rig_trajectories.json")
    trajectory = payload["trajectory"]
    if damage == "intrinsics":
        trajectory["camera_calibrations"]["camera2"]["camera_model_parameters"][
            "width"
        ] = 1000
    elif damage == "timestamp":
        trajectory["rig_trajectories"][0]["cameras_frame_timestamps_us"]["camera2"][0][
            0
        ] = 1
    elif damage == "world":
        trajectory["world_to_nre"]["matrix"][0][3] = 1
    elif damage == "rig":
        payload["native_rig_sha256"] = "wrong"
    else:
        trajectory["rig_trajectories"][0]["cameras_frame_T_rig_worlds"]["camera2"][0][
            0
        ][0][0] = float("nan") if damage == "nan" else 2
    with pytest.raises(NurecError):
        capture._validate_payload(payload, native)


def test_missing_photo_times_fail_instead_of_interpolating_another_camera():
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    with pytest.raises(NurecError, match="exact frame times"):
        capture._exact_frame_poses(
            "camera", (poses, np.array([0, 2])), np.array([[1, 1]])
        )


def test_native_rig_and_explicit_trajectory_keep_existing_behavior(
    tmp_path, monkeypatch
):
    artifact, meta, _, _ = _capture(tmp_path, monkeypatch)
    original = artifact.read_bytes()
    monkeypatch.setattr(capture, "read_rig_sidecar", lambda path: {})
    capture.attach_capture_trajectory(str(meta), artifact)
    assert artifact.read_bytes() == original
    result = render_novel_views(
        NurecConfig(),
        artifact_path=str(artifact),
        output_dir=str(tmp_path / "render"),
        custom_rig_trajectory="chosen.json",
        dry_run=True,
        export_video=False,
    )
    assert (
        result.command[result.command.index("--custom-rig-trajectory") + 1]
        == "chosen.json"
    )
    assert "--calib-source" not in result.command
