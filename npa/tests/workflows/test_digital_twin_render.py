"""Exercise CUDA refusal, pinned runtime integrity, and portable render evidence."""

from __future__ import annotations

import json
from types import SimpleNamespace

from PIL import Image
import pytest

from npa.workflows import digital_twin_render as render
from npa.workflows.digital_twin_scene import _cuda_devices, _gpu_devices


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


def test_optix_selection_excludes_cpu_and_cuda_devices():
    bpy, devices = _blender_context(["CPU", "CUDA", "OPTIX"])
    assert _gpu_devices(bpy, "OPTIX") == [{"name": "device-OPTIX", "type": "OPTIX"}]
    assert [device.use for device in devices] == [False, False, True]


def test_optix_does_not_silently_fall_back_to_cuda():
    bpy, devices = _blender_context(["CPU", "CUDA"])
    with pytest.raises(RuntimeError, match="CPU rendering is disabled"):
        _gpu_devices(bpy, "OPTIX")
    assert not any(device.use for device in devices)


def test_every_campus_camera_count_covers_all_routes():
    from npa.workflows.digital_twin_campus import _camera

    for count in (4, 7, 32, 49):
        poses = [_camera(index, count) for index in range(count)]
        assert {pose[0] for pose in poses} == set(render.ROUTES)
        assert len({pose[1] for pose in poses}) == count


def test_private_strings_cannot_enter_source_evidence(tmp_path):
    record = _bundle(tmp_path)
    record["scene_sources_sha256"]["digital_twin_scene.py"] = "private-canary"
    (tmp_path / "render-evidence.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Scene source evidence"):
        render._preview(tmp_path)


def test_native_backend_cannot_be_relabelled_as_optix(tmp_path):
    record = _bundle(tmp_path)
    record["backend"] = "Blender Cycles OptiX"
    (tmp_path / "render-evidence.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="disagree with the native receipt"):
        render.verify_render(tmp_path)


def _campus_bundle(root):
    _bundle(root)
    for index in (2, 3):
        Image.new("RGB", (16, 16), (index * 60, 80, 30)).save(
            root / f"frame-{index:03d}.png"
        )
    native = json.loads((root / "native-render.json").read_text())
    native.update(
        backend="Blender Cycles OptiX",
        scene_id="industrial-campus",
        resolution=[2560, 1440],
    )
    native["devices"] = [{"type": "OPTIX", "name": "NVIDIA RTX PRO 6000 Blackwell"}]
    native["cameras"] = [
        {
            "frame": index,
            "route": route,
            "camera_to_world": [
                [float(row == column) for column in range(4)] for row in range(4)
            ],
        }
        for index, route in enumerate(render.ROUTES)
    ]
    native["scene_statistics"] = {
        "site_extent_m": [500, 360],
        "site_area_hectares": 18,
        "mesh_objects": 7000,
        "unique_meshes": 50,
        "instanced_triangles": 100000,
        "assets": {"robot": 32, "inventory_unit": 600},
        "private_note": "private-canary",
    }
    (root / "native-render.json").write_text(json.dumps(native))
    return render._receipt(
        root,
        {"sample_count": 2, "peak_utilization_percent": 90, "elapsed_seconds": 3},
        4,
    )


def test_campus_preview_uses_real_frame_inventory_and_allows_only_public_measurements(
    tmp_path,
):
    _campus_bundle(tmp_path)
    render._preview(tmp_path)
    html = (tmp_path / "index.html").read_text()
    assert "private-canary" not in html and "connect-src 'none'" in html
    encoded = html.split('<script id="campus-data" type="application/json">')[1].split(
        "</script>"
    )[0]
    payload = json.loads(encoded)
    assert {frame["route"] for frame in payload["frames"]} == set(render.ROUTES)
    assert len(payload["frames"]) == 4 and payload["backend"] == "Blender Cycles OptiX"
    assert payload["assets"]["robot"] == 32


def test_campus_cannot_publish_without_observed_gpu_activity(tmp_path):
    record = _campus_bundle(tmp_path)
    record["telemetry"]["peak_utilization_percent"] = 0
    (tmp_path / "render-evidence.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="requires observed GPU activity"):
        render._preview(tmp_path)
    assert not (tmp_path / "index.html").exists()


def test_campus_cannot_relabel_missing_routes_as_covered(tmp_path):
    _campus_bundle(tmp_path)
    native = json.loads((tmp_path / "native-render.json").read_text())
    native["cameras"][0]["route"] = render.ROUTES[1]
    (tmp_path / "native-render.json").write_text(json.dumps(native))
    with pytest.raises(ValueError, match="Every campus inspection route"):
        render._native_receipt(tmp_path, 4)


def test_campus_cannot_downgrade_to_legacy_evidence(tmp_path):
    record = _campus_bundle(tmp_path)
    record["schema"] = "npa.digital-twin.cuda-render.v1"
    record["scene_sources_sha256"]["digital_twin_campus.py"] = "private-canary"
    (tmp_path / "render-evidence.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Legacy evidence supports only"):
        render._preview(tmp_path)
    assert not (tmp_path / "index.html").exists()


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
