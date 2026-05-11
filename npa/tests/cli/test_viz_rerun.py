from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from npa.cli.viz.backends import rerun as rerun_backend


CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4)]


def _skeleton(frames: int = 3, joints: int = 5) -> np.ndarray:
    data = np.zeros((frames, joints, 3), dtype=np.float32)
    for frame in range(frames):
        data[frame, :, 0] = np.linspace(-0.2, 0.2, joints)
        data[frame, :, 1] = frame * 0.02
        data[frame, :, 2] = np.linspace(0.0, 1.0, joints) + frame * 0.01
    return data


def test_rerun_backend_requires_predictions_for_overlay(tmp_path: Path) -> None:
    with pytest.raises(rerun_backend.RerunRenderError, match="predictions_data is required"):
        rerun_backend.render(
            _skeleton(),
            None,
            "overlay",
            tmp_path / "overlay.mp4",
            (320, 180),
            4,
            1.0,
            "test render",
            CONNECTIONS,
        )


def test_rerun_backend_rejects_predictions_longer_than_input(tmp_path: Path) -> None:
    skeleton = _skeleton(frames=2)
    predictions = _skeleton(frames=3)

    with pytest.raises(rerun_backend.RerunRenderError, match="frame count cannot exceed"):
        rerun_backend.render(
            skeleton,
            predictions,
            "overlay",
            tmp_path / "overlay.mp4",
            (320, 180),
            4,
            1.0,
            "test render",
            CONNECTIONS,
        )


def test_rerun_backend_orchestrates_capture_and_encode(tmp_path: Path, mocker) -> None:
    skeleton = _skeleton()
    predictions = skeleton + np.array([0.05, 0.0, 0.0], dtype=np.float32)
    output = tmp_path / "out.mp4"
    mocker.patch.object(
        rerun_backend,
        "_ensure_runtime_tools",
        return_value=rerun_backend._RuntimeTools(
            rerun_cli="/bin/rerun",
            chrome="/bin/chrome",
            ffmpeg="/bin/ffmpeg",
        ),
    )
    prepare = mocker.patch.object(rerun_backend, "_prepare_web_viewer_assets")

    def write_recordings(*args, **kwargs):
        recordings_dir = args[3]
        recordings_dir.mkdir(parents=True)
        paths = []
        for frame_idx in range(skeleton.shape[0]):
            path = recordings_dir / f"frame_{frame_idx:06d}.rrd"
            path.write_bytes(b"rrd")
            paths.append(path)
        return paths

    write = mocker.patch.object(rerun_backend, "_write_frame_recordings", side_effect=write_recordings)
    capture = mocker.patch.object(rerun_backend, "_capture_rerun_frames")

    def encode(_ffmpeg, _frames_dir, encode_fps, encode_output):
        assert encode_fps == 12
        encode_output.write_bytes(b"mp4")

    encode_mock = mocker.patch.object(rerun_backend, "_encode_png_sequence", side_effect=encode)

    rerun_backend.render(
        skeleton,
        predictions,
        "side-by-side",
        output,
        (640, 360),
        12,
        0.25,
        "test render",
        CONNECTIONS,
    )

    assert output.read_bytes() == b"mp4"
    prepare.assert_called_once()
    write.assert_called_once()
    assert write.call_args.args[2] == "side-by-side"
    capture.assert_called_once()
    assert capture.call_args.args[4] == (640, 360)
    encode_mock.assert_called_once()


def test_rerun_backend_writes_frame_recordings_with_mocked_rerun(tmp_path: Path, mocker) -> None:
    skeleton = _skeleton(frames=2)
    predictions = skeleton + np.array([0.03, 0.0, 0.0], dtype=np.float32)
    fake_rr = _FakeRerun()
    fake_rrb = _fake_blueprint_module()
    mocker.patch.object(rerun_backend, "_import_rerun", return_value=(fake_rr, fake_rrb))

    recordings = rerun_backend._write_frame_recordings(
        skeleton,
        predictions,
        "overlay",
        tmp_path / "recordings",
        5,
        0.4,
        "test render",
        CONNECTIONS,
    )

    assert len(recordings) == 2
    assert all(path.exists() and path.stat().st_size > 0 for path in recordings)
    logged_paths = [entry["path"] for entry in fake_rr.logs]
    assert "world/input/joints" in logged_paths
    assert "world/input/bones" in logged_paths
    assert "world/predictions/joints" in logged_paths
    assert "world/predictions/bones" in logged_paths
    assert any(entry["static"] for entry in fake_rr.logs if entry["path"] == "world/input/joints")


def test_rerun_backend_logs_predictions_only_within_prediction_window(tmp_path: Path, mocker) -> None:
    skeleton = _skeleton(frames=3)
    predictions = _skeleton(frames=1) + np.array([0.03, 0.0, 0.0], dtype=np.float32)
    fake_rr = _FakeRerun()
    fake_rrb = _fake_blueprint_module()
    mocker.patch.object(rerun_backend, "_import_rerun", return_value=(fake_rr, fake_rrb))

    recordings = rerun_backend._write_frame_recordings(
        skeleton,
        predictions,
        "overlay",
        tmp_path / "recordings",
        5,
        0.6,
        "test render",
        CONNECTIONS,
    )

    assert len(recordings) == 3
    logged_paths = [entry["path"] for entry in fake_rr.logs]
    assert logged_paths.count("world/input/joints") == 3
    assert logged_paths.count("world/predictions/joints") == 1
    assert logged_paths.count("world/predictions/bones") == 1


def test_viewer_url_uses_same_origin_recording_url(tmp_path: Path) -> None:
    viewer_dir = tmp_path / "viewer"
    recording = viewer_dir / "recordings" / "frame_000000.rrd"
    recording.parent.mkdir(parents=True)
    recording.write_bytes(b"rrd")

    url = rerun_backend._viewer_url("http://127.0.0.1:12345", viewer_dir, recording)

    assert "url=http://127.0.0.1:12345/recordings/frame_000000.rrd" in url
    assert "renderer=webgl" in url


class _FakeRerun:
    def __init__(self) -> None:
        self.logs: list[dict[str, object]] = []
        self.times: list[float] = []

    class RecordingStream:
        def __init__(self, application_id: str) -> None:
            self.application_id = application_id

        def save(self, path: Path, default_blueprint=None) -> None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(b"rrd")

    def send_blueprint(self, blueprint, *, recording) -> None:
        return None

    def Points3D(self, positions, **kwargs):
        return ("Points3D", np.asarray(positions), kwargs)

    def LineStrips3D(self, strips, **kwargs):
        return ("LineStrips3D", np.asarray(strips), kwargs)

    def TextDocument(self, text, **kwargs):
        return ("TextDocument", text, kwargs)

    def Scalars(self, value):
        return ("Scalars", value)

    def SeriesLines(self, **kwargs):
        return ("SeriesLines", kwargs)

    def log(self, path: str, entity, *, static: bool = False, recording=None) -> None:
        self.logs.append({"path": path, "entity": entity, "static": static})

    def set_time(self, timeline: str, *, duration: float, recording=None) -> None:
        self.times.append(duration)

    def reset_time(self, *, recording=None) -> None:
        return None

    def disconnect(self, *, recording=None) -> None:
        return None


def _fake_blueprint_module():
    def make(name):
        return lambda *args, **kwargs: (name, args, kwargs)

    return SimpleNamespace(
        Blueprint=make("Blueprint"),
        Horizontal=make("Horizontal"),
        Spatial3DView=make("Spatial3DView"),
        TimeSeriesView=make("TimeSeriesView"),
        Background=make("Background"),
        EyeControls3D=make("EyeControls3D"),
        BlueprintPanel=make("BlueprintPanel"),
        SelectionPanel=make("SelectionPanel"),
        TimePanel=make("TimePanel"),
        PanelState=SimpleNamespace(Hidden="Hidden"),
        Eye3DKind=SimpleNamespace(Orbital="Orbital"),
    )
