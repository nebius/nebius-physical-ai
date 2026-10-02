"""Exercise CUDA refusal, pinned runtime integrity, and portable render evidence."""

from __future__ import annotations

import json
from types import SimpleNamespace

from PIL import Image
import pytest

from npa.workflows import digital_twin_render as render
from npa.workflows.digital_twin_scene import _cuda_devices


def _bundle(root):
    for index in range(2):
        Image.new("RGB", (16, 16), (index * 100, 80, 30)).save(
            root / f"frame-{index:03d}.png"
        )
    (root / "scene.glb").write_bytes(b"glTF-test-fixture")
    (root / "scene.usdc").write_bytes(b"PXR-USDC-test-fixture")
    native = {
        "backend": "Blender Cycles CUDA",
        "version": "4.5.3",
        "devices": [{"type": "CUDA", "name": "NVIDIA B200"}],
        "cpu_rendering": False,
        "samples": 64,
        "resolution": [1280, 720],
        "cameras": [{}, {}],
    }
    (root / "native-render.json").write_text(json.dumps(native))
    telemetry = {
        "sample_count": 2,
        "peak_utilization_percent": 90,
        "elapsed_seconds": 4.0,
    }
    return render._receipt(root, telemetry, 2)


def test_preview_binds_scene_and_frames_and_is_offline(tmp_path):
    record = _bundle(tmp_path)
    record["private_storage"] = "private-canary-never-display"
    (tmp_path / "render-evidence.json").write_text(json.dumps(record))
    render._preview(tmp_path)
    html = (tmp_path / "index.html").read_text()
    assert "NVIDIA B200" in html and "Blender Cycles" in html
    assert "private-canary" not in html
    assert "connect-src 'none'" in html
    assert "data:image/jpeg;base64," in html
    assert "not a scan of a physical facility" in html


@pytest.mark.parametrize(
    "name", ["scene.glb", "scene.usdc", "frame-000.png", "native-render.json"]
)
def test_modified_artifact_refuses_preview(tmp_path, name):
    _bundle(tmp_path)
    (tmp_path / name).write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        render._preview(tmp_path)
    assert not (tmp_path / "index.html").exists()


def test_extra_frame_cannot_inherit_gpu_evidence(tmp_path):
    _bundle(tmp_path)
    (tmp_path / "frame-002.png").write_bytes(b"unrelated")
    with pytest.raises(ValueError, match="inventory changed"):
        render.verify_render(tmp_path)


def _blender_context(device_types):
    devices = [
        SimpleNamespace(type=kind, name=f"device-{kind}", use=True)
        for kind in device_types
    ]
    preferences = SimpleNamespace(devices=devices, refresh_devices=lambda: None)
    scene = SimpleNamespace(render=SimpleNamespace(), cycles=SimpleNamespace())
    return SimpleNamespace(
        context=SimpleNamespace(
            preferences=SimpleNamespace(
                addons={"cycles": SimpleNamespace(preferences=preferences)}
            ),
            scene=scene,
        )
    ), devices


def test_native_renderer_disables_all_cpu_devices():
    bpy, devices = _blender_context(["CPU", "CUDA"])
    assert _cuda_devices(bpy) == [{"name": "device-CUDA", "type": "CUDA"}]
    assert [device.use for device in devices] == [False, True]
    assert bpy.context.scene.cycles.device == "GPU"


def test_native_renderer_refuses_cpu_fallback():
    bpy, devices = _blender_context(["CPU"])
    with pytest.raises(RuntimeError, match="CPU rendering is disabled"):
        _cuda_devices(bpy)
    assert not devices[0].use


def test_failed_native_renderer_publishes_nothing(tmp_path, monkeypatch):
    import subprocess

    monkeypatch.setattr(render, "_blender", lambda root: root / "blender")

    class Telemetry:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(render, "RenderTelemetry", Telemetry)

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["blender"])

    monkeypatch.setattr(render.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        render._run(str(tmp_path / "output"), 2, 1)
    assert not (tmp_path / "output").exists()


def test_corrupt_blender_archive_is_not_extracted(tmp_path, monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, size):
            return [b"untrusted archive"]

    monkeypatch.setattr(render.requests, "get", lambda *args, **kwargs: Response())
    with pytest.raises(ValueError, match="official pinned SHA-256"):
        render._blender(tmp_path)
    assert not (tmp_path / render.BLENDER_ARCHIVE).exists()


@pytest.mark.parametrize("field", ["gpu_models", "samples", "telemetry"])
def test_portable_report_refuses_private_strings_in_measurements(tmp_path, field):
    record = _bundle(tmp_path)
    record[field] = {"private-canary": "private-value"}
    (tmp_path / "render-evidence.json").write_text(json.dumps(record))
    with pytest.raises(ValueError):
        render._preview(tmp_path)
    assert not (tmp_path / "index.html").exists()
