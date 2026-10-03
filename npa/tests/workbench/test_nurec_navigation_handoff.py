"""Reject substituted scan lineage and case overrides before native recipe construction."""

import json
from pathlib import Path

import pytest

from npa.workbench.nurec.navigation_assets import sha256
from npa.workbench.nurec.navigation_handoff import _cases, _qualified_scan
from npa.workbench.nurec.navigation_publication import publish_immutable

_PROBE = {
    "origin": [0, 0, 1],
    "direction": [0, 0, -1],
    "min_distance": 0.9,
    "max_distance": 1.1,
}


def _json(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def reports(tmp_path):
    # Byte fixtures exercise the handoff, never claim to simulate or reconstruct.
    scene, physics = tmp_path / "source", tmp_path / "report"
    scene.mkdir()
    physics.mkdir()
    (scene / "scene.usdz").write_bytes(b"scene byte fixture")
    _json(scene / "capture.json", {"schema": "npa.navigation.rgbd_capture.v1"})
    _json(
        scene / "reconstruction.json",
        {
            "schema": "npa.navigation.rgbd_reconstruction.v1",
            "capture_manifest_sha256": sha256(scene / "capture.json"),
        },
    )
    probes = [_PROBE]
    _json(
        scene / "provenance.json",
        {
            "schema": "npa.nurec.navigation_scene.v1",
            "scene_sha256": sha256(scene / "scene.usdz"),
            "capture_manifest_sha256": sha256(scene / "capture.json"),
            "reconstruction_report_sha256": sha256(scene / "reconstruction.json"),
            "ray_probes": probes,
        },
    )
    _json(
        physics / "physics_validation.json",
        {
            "schema": "npa.nurec.navigation_physics.v1",
            "scene_sha256": sha256(scene / "scene.usdz"),
            "assembly_provenance_sha256": sha256(scene / "provenance.json"),
            "physics_validated": True,
            "probes": [{"expected": probes[0]}],
        },
    )
    return scene, physics


def _sealed(tmp_path, reports):
    destinations = [tmp_path / "sealed-scene", tmp_path / "sealed-physics"]
    for source, destination in zip(reports, destinations, strict=True):
        publish_immutable(source, str(destination))
    return destinations


def test_handoff_retains_exact_scan_lineage(tmp_path, reports):
    scene, physics = _sealed(tmp_path, reports)
    hashes = _qualified_scan(scene, physics)
    assert hashes["capture.json"] == sha256(scene / "capture.json")
    assert hashes["physics_validation.json"] == sha256(
        physics / "physics_validation.json"
    )


@pytest.mark.parametrize("field", ["scene_sha256", "assembly_provenance_sha256"])
def test_handoff_rejects_report_for_another_scene(tmp_path, reports, field):
    path = reports[1] / "physics_validation.json"
    report = json.loads(path.read_text())
    report[field] = "f" * 64
    _json(path, report)
    with pytest.raises(ValueError, match="lineage disagree"):
        _qualified_scan(*_sealed(tmp_path, reports))


def test_handoff_rejects_replaced_capture_even_with_a_new_publication(
    tmp_path, reports
):
    _json(reports[0] / "capture.json", {"different": "capture"})
    with pytest.raises(ValueError, match="lineage disagree"):
        _qualified_scan(*_sealed(tmp_path, reports))


@pytest.mark.parametrize("change", ["unvalidated", "missing-probe"])
def test_handoff_requires_completed_matching_native_probe_set(
    tmp_path, reports, change
):
    path = reports[1] / "physics_validation.json"
    report = json.loads(path.read_text())
    if change == "unvalidated":
        report["physics_validated"] = False
    else:
        report["probes"][0]["expected"] = {"different": "probe"}
    _json(path, report)
    with pytest.raises(ValueError, match="native PhysX|assembled probes"):
        _qualified_scan(*_sealed(tmp_path, reports))


def test_handoff_rejects_changed_completed_artifact(tmp_path, reports):
    scene, physics = _sealed(tmp_path, reports)
    (scene / "scene.usdz").write_bytes(b"replaced")
    with pytest.raises(ValueError, match="differs from its seal"):
        _qualified_scan(scene, physics)


@pytest.mark.parametrize("field", ["image", "scene_sha256", "adapter_module"])
def test_case_input_cannot_override_native_execution_binding(tmp_path, field):
    source, target = tmp_path / "cases.json", tmp_path / "snapshot.json"
    _json(source, {"train_cases": [], "eval_cases": [], "probe": {}, field: "override"})
    with pytest.raises(ValueError, match="only train_cases"):
        _cases(source, target)
    assert not target.exists()


class _ObjectFixture:
    def __init__(self):
        self.objects = {}

    def put_bytes_conditional(self, data, uri, *, if_none_match):
        assert if_none_match and uri not in self.objects
        self.objects[uri] = data

    def read_bytes_with_etag(self, uri):
        data = self.objects.get(uri)
        return None if data is None else (data, "fixture-etag")

    def download_directory(self, uri, destination):
        root = Path(destination)
        root.mkdir()
        prefix = uri.rstrip("/") + "/"
        for key, data in self.objects.items():
            if key.startswith(prefix):
                path = root / key.removeprefix(prefix)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)


@pytest.fixture
def object_storage(monkeypatch):
    from npa.clients.storage import StorageClient

    objects = _ObjectFixture()
    monkeypatch.setattr(StorageClient, "from_environment", lambda: objects)
    return objects


@pytest.mark.parametrize("remote", [False, True])
def test_companion_native_preparation_consumes_handoff(
    tmp_path, reports, object_storage, remote
):
    stages = pytest.importorskip(
        "npa.workflows.navigation.stages",
        reason="requires companion navigation PR #805",
    )
    from npa.workbench.nurec.navigation_handoff import prepare_navigation_input
    from npa.workflows.navigation.reference_scene import cases

    scene, physics = _sealed(tmp_path, reports)
    route_file = tmp_path / "cases.json"
    _json(route_file, cases(2))
    destination = (
        "s3://unit-test-bucket/handoff" if remote else str(tmp_path / "handoff")
    )
    image = "registry.example.invalid/isaac@sha256:" + "a" * 64
    result = prepare_navigation_input(
        str(scene),
        str(physics),
        str(route_file),
        destination,
        image=image,
        num_envs=2,
        iterations=5,
        episode_steps=20,
    )
    # Real recipe construction and immutable publication, with object I/O in memory.
    prepared = stages.prepare(
        result["workflow_input_uri"], str(tmp_path / "prepared"), image
    )
    assert prepared["scene_sha256"] == sha256(scene / "scene.usdz")
    assert prepared["recipe_sha256"] == result["recipe_sha256"]
    assert (tmp_path / "prepared/scan/capture.json").read_bytes() == (
        scene / "capture.json"
    ).read_bytes()
    assert result["navigation_learning_verified"] is False
    if remote:
        assert "/attempts/" in result["workflow_input_uri"]
