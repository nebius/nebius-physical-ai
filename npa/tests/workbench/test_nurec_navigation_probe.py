"""Ensure native physics evidence cannot pass on missing or unrelated mesh hits."""

import hashlib
import json
import math
import sys
from types import SimpleNamespace

import pytest

from npa.workbench.nurec.navigation_probe import check_hit
from npa.workbench.nurec import navigation_probe as probe_module

PROBE = {
    "origin": [0, 0, 2],
    "direction": [0, 0, -1],
    "min_distance": 1.9,
    "max_distance": 2.1,
}
COLLIDER = "/World/Collision/Mesh_0"


def test_native_mesh_hit_preserves_measured_distance():
    hit = {"hit": True, "distance": 2.0, "collision": COLLIDER}
    result = check_hit(hit, PROBE, {COLLIDER})
    assert result["distance_m"] == 2.0
    assert result["collision"] == COLLIDER


@pytest.mark.parametrize(
    "distance", [math.nan, math.inf, -math.inf, "2.0", True, 0, 2.2, None]
)
def test_invalid_native_distances_cannot_claim_physics(distance):
    with pytest.raises(ValueError, match="PhysX probe"):
        check_hit(
            {"hit": True, "distance": distance, "collision": COLLIDER},
            PROBE,
            {COLLIDER},
        )


@pytest.mark.parametrize("hit", [False, True])
def test_unrelated_mesh_or_miss_is_not_scene_evidence(hit):
    result = {"hit": hit, "distance": 2.0, "collision": "/Stock/Ground"}
    with pytest.raises(ValueError, match="prepared collision mesh"):
        check_hit(result, PROBE, {COLLIDER})


IMAGE = "registry.example.invalid/isaac@sha256:" + "a" * 64


@pytest.mark.parametrize(
    "reference",
    [
        "",
        "tool://isaac-lab",
        "image:latest",
        "image@sha256:" + "a" * 64,
        "https://" + IMAGE,
        IMAGE + "\n",
        IMAGE[:-1],
        IMAGE.replace("sha256", "sha1"),
    ],
)
def test_image_declaration_requires_immutable_registry_reference(reference):
    """Reject unresolved, mutable, malformed, and protocol-bearing image references.

    Args:
        reference: Invalid operator declaration.
    Returns:
        None.
    Raises:
        AssertionError: An invalid identity is accepted.
    """
    with pytest.raises(ValueError, match="set --var isaac_image"):
        probe_module._declared_image(reference)


def test_unresolved_image_cli_fails_before_native_runtime_start(monkeypatch):
    """Refuse unresolved image declarations before importing or starting Isaac.

    Args:
        monkeypatch: Fixture providing the command-line argument vector.
    Returns:
        None.
    Raises:
        AssertionError: The unresolved image reaches the absent native runtime.
    """
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "navigation_probe",
            "--input-path",
            "unused",
            "--output-path",
            "unused-output",
            "--runtime-image",
            "tool://isaac-lab",
        ],
    )
    with pytest.raises(ValueError, match="set --var isaac_image"):
        probe_module.main()


def _runtime_modules(monkeypatch, version):
    manager = SimpleNamespace(
        get_enabled_extension_id=lambda name: "omni.physx-107.3.4",
        get_extension_dict=lambda name: SimpleNamespace(
            get_dict=lambda: {"package": {"version": "107.3.4"}}
        ),
    )
    app = SimpleNamespace(
        get_app=lambda: SimpleNamespace(get_extension_manager=lambda: manager)
    )
    kit = SimpleNamespace(app=app)
    for name, module in {
        "isaacsim.core.version": SimpleNamespace(get_version=lambda: version),
        "omni": SimpleNamespace(kit=kit),
        "omni.kit": kit,
        "omni.kit.app": app,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_runtime_versions_come_from_isaac_api_and_enabled_physx(monkeypatch):
    """Read native version metadata without using declared image names as versions.

    Args:
        monkeypatch: Fixture installing in-memory native API seams.
    Returns:
        None.
    Raises:
        AssertionError: Runtime metadata differs from the APIs.
    """
    _runtime_modules(monkeypatch, ("6.0.1", "build-fixture"))
    result = probe_module._runtime_versions()
    assert result["isaac_sim"]["version"] == "6.0.1"
    assert result["physx"] == {
        "available": True,
        "extension_id": "omni.physx-107.3.4",
        "version": "107.3.4",
    }
    _runtime_modules(monkeypatch, ("",))
    with pytest.raises(ValueError, match="Isaac Sim runtime version"):
        probe_module._runtime_versions()


@pytest.mark.parametrize("extension_id", ["", "omni.physx-unknown"])
def test_missing_physx_metadata_is_reported_without_invented_version(extension_id):
    """Keep missing extension-version metadata explicit in the report.

    Args:
        extension_id: Missing extension or extension with absent package metadata.
    Returns:
        None.
    Raises:
        AssertionError: Unknown metadata becomes a claimed version.
    """
    manager = SimpleNamespace(
        get_enabled_extension_id=lambda name: extension_id,
        get_extension_dict=lambda name: None,
    )
    result = probe_module._physx_version(manager)
    assert result["available"] is False
    assert "version" not in result
    assert "unavailable" in result["reason"]


def _write_bundle(root):
    scene = root / "scene.usdz"
    scene.write_bytes(b"synthetic scene bytes; native decoder explicitly mocked")
    provenance = {
        "schema": "npa.nurec.navigation_scene.v1",
        "scene_sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
        "colliders": [{"path": COLLIDER}],
        "ray_probes": [PROBE],
        "capture_manifest_sha256": "a" * 64,
    }
    (root / "provenance.json").write_text(json.dumps(provenance))


@pytest.fixture
def mocked_bundle(tmp_path, monkeypatch):
    """Provide file-backed bundle metadata with native scene/GPU seams mocked.

    Args:
        tmp_path: Isolated input directory.
        monkeypatch: Fixture replacing native validation and storage publication.
    Returns:
        Input directory and observed published report bodies.
    Raises:
        OSError: Fixture files cannot be written.
    """
    _write_bundle(tmp_path)
    monkeypatch.setattr(
        "npa.workbench.nurec.navigation_scene.verify_scene", lambda *args: None
    )
    monkeypatch.setattr(
        "npa.workbench.nurec.navigation_publication.verify_publication",
        lambda *args: None,
    )
    monkeypatch.setattr(probe_module, "materialize", lambda *args: tmp_path)
    native_hit = check_hit(
        {"hit": True, "distance": 2.0, "collision": COLLIDER}, PROBE, {COLLIDER}
    )
    monkeypatch.setattr(probe_module, "_query_scene", lambda *args: [native_hit])
    monkeypatch.setattr(
        probe_module,
        "_runtime_versions",
        lambda: {"isaac_sim": {"version": "6.0.1"}, "physx": {"available": False}},
    )
    reports = []
    monkeypatch.setattr(
        probe_module,
        "publish",
        lambda path, target: reports.append(
            json.loads((path / "physics_validation.json").read_text())
        ),
    )
    return tmp_path, reports


def test_report_binds_exact_provenance_and_operator_image_scope(mocked_bundle):
    """Bind changed capture lineage independently of an unchanged scene package.

    Args:
        mocked_bundle: File-backed inputs and captured publication seam.
    Returns:
        None.
    Raises:
        AssertionError: Evidence misses provenance bytes or overclaims image identity.
    """
    root, published = mocked_bundle
    path = root / "provenance.json"
    first = probe_module._verify_and_publish(str(root), "unused-output", IMAGE)
    assert (
        first["assembly_provenance_sha256"]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    provenance = json.loads(path.read_text())
    provenance["capture_manifest_sha256"] = "b" * 64
    path.write_text(json.dumps(provenance))
    second = probe_module._verify_and_publish(str(root), "unused-output", IMAGE)
    assert first["scene_sha256"] == second["scene_sha256"]
    assert first["assembly_provenance_sha256"] != second["assembly_provenance_sha256"]
    assert second["runtime_image"] == {
        "reference": IMAGE,
        "attestation_scope": "operator-declared workload image reference",
        "running_image_identity_verified": False,
    }
    assert published == [first, second]


def test_provenance_swap_during_query_prevents_publication(mocked_bundle, monkeypatch):
    """Reject provenance replaced between validation and report publication.

    Args:
        mocked_bundle: File-backed inputs and captured publication seam.
        monkeypatch: Fixture replacing the native query boundary.
    Returns:
        None.
    Raises:
        AssertionError: Swapped provenance produces a successful report.
    """
    root, published = mocked_bundle

    def swap_provenance(scene, provenance):
        provenance["capture_manifest_sha256"] = "c" * 64
        (root / "provenance.json").write_text(json.dumps(provenance))
        return []

    monkeypatch.setattr(probe_module, "_query_scene", swap_provenance)
    with pytest.raises(ValueError, match="provenance changed"):
        probe_module._verify_and_publish(str(root), "unused-output", IMAGE)
    assert published == []
