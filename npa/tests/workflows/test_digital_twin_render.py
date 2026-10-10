"""Exercise CUDA refusal, pinned runtime integrity, and portable render evidence."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Lock
from types import SimpleNamespace

from PIL import Image
import pytest

from npa.workflows import digital_twin_render as render
from npa.workflows import digital_twin_publication as publication
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


@pytest.mark.parametrize("value", [True, 2.0, "2", None])
def test_gpu_sample_count_requires_literal_integer_evidence(tmp_path, value):
    record = _bundle(tmp_path)
    record["telemetry"]["sample_count"] = value
    (tmp_path / "render-evidence.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="telemetry.sample_count"):
        render.verify_render(tmp_path)


def test_campus_resolution_rejects_numeric_coercion(tmp_path):
    _campus_bundle(tmp_path)
    native = json.loads((tmp_path / "native-render.json").read_text())
    native["resolution"] = [2560.0, 1440.0]
    (tmp_path / "native-render.json").write_text(json.dumps(native))
    with pytest.raises(ValueError, match="resolution must be a literal integer"):
        render._native_receipt(tmp_path, 4)


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


class _ConditionalStorage:
    def __init__(self):
        self.objects = {}
        self.lock = Lock()
        self.failure_suffix = ""
        self.fail_after_write = False
        self.upload_barrier = None

    def put_bytes_conditional(self, body, uri, *, if_none_match):
        assert if_none_match
        with self.lock:
            if uri in self.objects:
                raise publication.StoragePreconditionFailed("already claimed")
            fail = bool(self.failure_suffix) and uri.endswith(self.failure_suffix)
            if fail:
                self.failure_suffix = ""
            if fail and not self.fail_after_write:
                raise OSError("interrupted upload")
            self.objects[uri] = bytes(body)
            if fail:
                raise OSError("response lost after committed write")
        if self.upload_barrier and uri.endswith("/checksums.json"):
            self.upload_barrier.wait()
        return "fixture-etag"

    def read_bytes_with_etag(self, uri):
        with self.lock:
            value = self.objects.get(uri)
        return None if value is None else (value, "fixture-etag")

    def download_directory(self, uri, destination):
        root = Path(destination)
        root.mkdir()
        with self.lock:
            objects = dict(self.objects)
        prefix = uri.rstrip("/") + "/"
        for key, body in objects.items():
            if key.startswith(prefix):
                path = root / key[len(prefix) :]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(body)


@pytest.fixture
def render_publication(tmp_path, monkeypatch):
    root = tmp_path / "rendered"
    root.mkdir()
    _bundle(root)
    render._preview(root)
    storage = _ConditionalStorage()
    monkeypatch.setattr(publication.StorageClient, "from_environment", lambda: storage)
    request = render._render_request(2, 64, "factory-cell", "CUDA")
    return storage, root, "s3://fixture-bucket/run/rendered", request


@pytest.mark.parametrize(
    "suffix, after_write",
    [
        ("/render-request.json", True),
        ("/frame-001.png", False),
        ("/checksums.json", True),
        ("/claim.json", False),
        ("/claim.json", True),
        ("/completion.json", False),
        ("/completion.json", True),
    ],
)
def test_publication_recovers_each_interruption_without_replacing_bytes(
    render_publication, tmp_path, suffix, after_write
):
    storage, root, uri, request = render_publication
    storage.failure_suffix, storage.fail_after_write = suffix, after_write
    with pytest.raises(OSError):
        publication.publish(root, uri, request)
    original = dict(storage.objects)
    if not publication.recover(uri, request):
        publication.publish(root, uri, request)
    assert all(storage.objects[key] == value for key, value in original.items())
    assert (
        storage.objects[uri + "/claim.json"]
        == storage.objects[uri + "/completion.json"]
    )
    restored = publication.artifacts.materialize(uri, tmp_path / "restored")
    render.verify_render(restored)
    assert (restored / "index.html").read_bytes() == (root / "index.html").read_bytes()


@pytest.mark.parametrize("complete", [False, True])
def test_resumed_runner_reuses_verified_upload_without_starting_blender(
    render_publication, monkeypatch, complete
):
    storage, root, uri, request = render_publication
    if complete:
        publication.publish(root, uri, request)
    else:
        storage.failure_suffix = "/completion.json"
        with pytest.raises(OSError):
            publication.publish(root, uri, request)
    monkeypatch.setattr(render, "_blender", lambda _: pytest.fail("rerendered"))
    render._run(uri, 2, 64)
    assert publication.recover(uri, request)


@pytest.mark.parametrize(
    "field, value",
    [
        ("views", 3),
        ("views", 2.0),
        ("samples", 32),
        ("backend", "Blender Cycles OptiX"),
        ("scene_id", "industrial-campus"),
        ("blender_archive_sha256", "0" * 64),
        ("scene_sources_sha256", {"digital_twin_scene.py": "0" * 64}),
    ],
)
def test_request_change_cannot_reuse_or_replace_a_publication(
    render_publication, field, value
):
    storage, root, uri, request = render_publication
    publication.publish(root, uri, request)
    original = dict(storage.objects)
    changed = {**request, field: value}
    with pytest.raises(ValueError, match="conflicts"):
        publication.recover(uri, changed)
    assert storage.objects == original


def test_changed_runner_settings_fail_before_gpu_allocation(
    render_publication, monkeypatch
):
    _, root, uri, request = render_publication
    publication.publish(root, uri, request)
    monkeypatch.setattr(render, "_blender", lambda _: pytest.fail("rerendered"))
    with pytest.raises(ValueError, match="conflicts"):
        render._run(uri, 2, 65)


@pytest.mark.parametrize("filename", ["frame-000.png", "index.html", "checksums.json"])
def test_corrupt_completed_bytes_block_recovery(render_publication, filename):
    storage, root, uri, request = render_publication
    publication.publish(root, uri, request)
    claim = json.loads(storage.objects[uri + "/claim.json"])
    storage.objects[f"{uri}/{claim['attempt']}/{filename}"] = b"{}"
    original = dict(storage.objects)
    with pytest.raises(ValueError, match="differ"):
        publication.recover(uri, request)
    assert storage.objects == original


def test_concurrent_valid_attempts_select_one_complete_bundle(
    render_publication, tmp_path
):
    storage, first, uri, request = render_publication
    second = tmp_path / "second"
    second.mkdir()
    _bundle(second)
    render._preview(second)
    with (second / "index.html").open("a") as stream:
        stream.write("<!-- second complete attempt -->")
    storage.upload_barrier = Barrier(2)
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(
            workers.map(
                lambda root: publication.publish(root, uri, request), [first, second]
            )
        )
    assert results == [None, None]
    assert publication.recover(uri, request)
    attempts = {
        key.split("/attempts/")[1].split("/")[0]
        for key in storage.objects
        if "/attempts/" in key
    }
    assert len(attempts) == 2
    claim = json.loads(storage.objects[uri + "/completion.json"])
    assert (
        storage.objects[uri + "/completion.json"]
        == storage.objects[uri + "/claim.json"]
    )
    assert len(claim["files"]) == 7


def test_completed_legacy_publication_still_recovers(render_publication):
    _, root, uri, request = render_publication
    publication.artifacts.publish(root, uri)
    assert publication.recover(uri, request)


def test_wrong_request_cannot_poison_completed_legacy_publication(render_publication):
    storage, root, uri, request = render_publication
    publication.artifacts.publish(root, uri)
    original = dict(storage.objects)
    with pytest.raises(ValueError, match="differs from the requested"):
        publication.recover(uri, {**request, "samples": 32})
    assert storage.objects == original
    assert publication.recover(uri, request)


def test_storage_access_failure_cannot_start_a_new_render(
    render_publication, monkeypatch
):
    storage, _, uri, _ = render_publication

    def denied(_):
        raise PermissionError("storage access denied")

    monkeypatch.setattr(storage, "read_bytes_with_etag", denied)
    monkeypatch.setattr(
        render, "_blender", lambda _: pytest.fail("rendered without storage")
    )
    with pytest.raises(PermissionError):
        render._run(uri, 2, 64)


def test_incomplete_legacy_claim_fails_closed_without_overwrite(render_publication):
    storage, root, uri, request = render_publication
    storage.failure_suffix = "/frame-001.png"
    with pytest.raises(OSError):
        publication.artifacts.publish(root, uri)
    original = dict(storage.objects)
    with pytest.raises(FileNotFoundError):
        publication.recover(uri, request)
    assert all(storage.objects[key] == value for key, value in original.items())
    assert uri + "/completion.json" not in storage.objects


def test_local_completion_can_be_reused_without_rendering(
    render_publication, tmp_path, monkeypatch
):
    _, root, _, request = render_publication
    destination = str(tmp_path / "published")
    publication.publish(root, destination, request)
    monkeypatch.setattr(render, "_blender", lambda _: pytest.fail("rerendered"))
    render._run(destination, 2, 64)
