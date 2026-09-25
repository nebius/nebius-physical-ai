"""Validate public scene geometry, split generation and reference trust boundaries."""

import importlib
from types import SimpleNamespace

import pytest

from npa.workflows.navigation import native
from npa.workflows.navigation.reference_bundle import build_bundle
from npa.workflows.navigation.reference_scene import cases


def test_four_thousand_reference_resets_have_disjoint_physical_inputs():
    bundle = cases(4000)
    combined = bundle["train_cases"] + bundle["eval_cases"]
    assert len(combined) == len({case["id"] for case in combined}) == 8000
    assert (
        len({tuple(case["position_m"] + case["goal_m"]) for case in combined}) == 8000
    )
    assert len(bundle["probe"]["actions"]) == 30


def test_reference_bundle_contains_real_collision_scene(tmp_path, recipe):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics

    output = tmp_path / "reference"
    result = build_bundle(
        output, image=recipe.image, iterations=1000, episode_steps=150, num_envs=4000
    )
    native.validate_scene_package(output / "scene.usdz")
    scene = Usd.Stage.Open(str(output / "scene.usdz"))
    assert UsdGeom.GetStageMetersPerUnit(scene) == 1.0
    assert UsdGeom.GetStageUpAxis(scene) == "Z"
    meshes = [prim for prim in scene.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)]
    assert len(meshes) == 1
    assert len(UsdGeom.Mesh(meshes[0]).GetFaceVertexCountsAttr().Get()) == 84
    assert result["source_bundle_sha256"] == native.source_bundle_digest()
    assert native.task_adapter(SimpleNamespace(**result)).__name__.endswith(
        ".reference"
    )


def test_reference_inventory_rejected_before_import(recipe, monkeypatch):
    recipe.adapter_module = "npa.workflows.navigation.reference"
    recipe.source_bundle_sha256 = "0" * 64
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda *_: pytest.fail("unverified reference import"),
    )
    with pytest.raises(ValueError, match="source bundle"):
        native.task_adapter(recipe)


def test_external_geometry_requires_measured_cases(tmp_path, recipe):
    with pytest.raises(ValueError, match="explicit measured"):
        build_bundle(
            tmp_path / "external",
            image=recipe.image,
            iterations=500,
            episode_steps=150,
            num_envs=4000,
            scene_file=tmp_path / "scene.usdz",
        )


def test_static_ray_mesh_uses_transformed_collision_triangles(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom
    from npa.workflows.navigation.reference_geometry import _append_mesh, _sensor_mesh

    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/World/Warehouse")
    root.AddTranslateOp().Set((3, 4, 5))
    mesh = UsdGeom.Mesh.Define(stage, "/World/Warehouse/Collider")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    points, faces = [], []
    _append_mesh(mesh.GetPrim(), points, faces)
    assert tuple(points[0]) == (3.0, 4.0, 5.0)
    _sensor_mesh(stage, "/World/Warehouse", points, faces)
    sensor = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Warehouse/NpaRaycastMesh"))
    assert tuple(sensor.GetPointsAttr().Get()[0]) == (0.0, 0.0, 0.0)
    assert sensor.GetFaceVertexIndicesAttr().Get() == [0, 1, 2]


def test_nonmetric_source_frame_is_rejected(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom
    from npa.workflows.navigation.reference_geometry import validate_scene_frame

    source = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(source))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="meters and Z-up"):
        validate_scene_frame(str(source))


def test_imported_physics_is_scoped_to_native_scene_without_editing_asset(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics
    from npa.workflows.navigation.reference_geometry import (
        _collision_triangles,
        _physics_owner,
    )

    path = tmp_path / "scene.usda"
    source = Usd.Stage.CreateNew(str(path))
    source.SetDefaultPrim(UsdGeom.Xform.Define(source, "/World").GetPrim())
    UsdPhysics.Scene.Define(source, "/World/Physics")
    mesh = UsdGeom.Mesh.Define(source, "/World/Collider")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateSimulationOwnerRel().SetTargets(
        ["/World/Physics"]
    )
    source.GetRootLayer().Save()
    original = path.read_bytes()
    stage = Usd.Stage.CreateInMemory()
    UsdPhysics.Scene.Define(stage, "/physicsScene")
    root = UsdGeom.Xform.Define(stage, "/World/Warehouse").GetPrim()
    root.GetReferences().AddReference(str(path))
    owner = _physics_owner(root)
    points, faces = _collision_triangles(root, owner)
    assert len(points) == 3 and faces == [0, 1, 2]
    assert [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)] == [
        "/physicsScene"
    ]
    collider = stage.GetPrimAtPath("/World/Warehouse/Collider")
    assert UsdPhysics.CollisionAPI(collider).GetSimulationOwnerRel().GetTargets() == [
        owner
    ]
    assert path.read_bytes() == original


def test_physics_scene_with_geometry_cannot_be_deactivated():
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics
    from npa.workflows.navigation.reference_geometry import _physics_owner

    stage = Usd.Stage.CreateInMemory()
    UsdPhysics.Scene.Define(stage, "/physicsScene")
    root = UsdGeom.Xform.Define(stage, "/World/Warehouse").GetPrim()
    UsdPhysics.Scene.Define(stage, "/World/Warehouse/Physics")
    prim = UsdGeom.Mesh.Define(stage, "/World/Warehouse/Physics/Collider").GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    with pytest.raises(ValueError, match="contains collision geometry"):
        _physics_owner(root)
    assert prim.IsActive()


def test_contact_filters_broadcast_exact_enabled_scene_colliders():
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics
    from npa.workflows.navigation.reference_contacts import _scene_filters

    stage = Usd.Stage.CreateInMemory()
    root = "/World/Warehouse"
    for path, enabled in (
        (root + "/Floor", True),
        (root + "/Obstacles/Rack", True),
        (root + "/Disabled", False),
        ("/World/envs/env_0/Robot/base", True),
    ):
        prim = UsdGeom.Mesh.Define(stage, path).GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(enabled)
    UsdGeom.Mesh.Define(stage, root + "/NpaRaycastMesh")
    assert _scene_filters(stage, root) == [root + "/Floor", root + "/Obstacles/Rack"]
    with pytest.raises(RuntimeError, match="no enabled collision filters"):
        _scene_filters(stage, "/World/Absent")


def test_contact_reduction_excludes_support_and_detects_obstacles(monkeypatch):
    torch = pytest.importorskip("torch")
    from npa.workflows.navigation import reference_contacts

    monkeypatch.setattr(reference_contacts, "_tensor", lambda value: value)
    data = reference_contacts.ContactMeasurements.__new__(
        reference_contacts.ContactMeasurements
    )
    data.env = SimpleNamespace(num_envs=2, physics_dt=0.005)
    data.obstacle = torch.zeros(2)
    data.body_count, data.base_index = 2, 0
    forces = torch.tensor([20.0, 100.0, 5.0, 80.0])
    normals = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    data.view = SimpleNamespace(
        filter_count=1,
        get_contact_data=lambda **_: (
            forces,
            None,
            normals,
            None,
            torch.ones((4, 1)),
            torch.arange(4).reshape(4, 1),
        ),
    )
    assert data._obstacle_forces().tolist() == [20.0, 5.0]


def test_real_support_measurements_reject_holes_low_pose_tilt_and_flight():
    torch = pytest.importorskip("torch")
    from npa.workflows.navigation.reference_validity import validity_from_measurements

    clearance = torch.full((5, 5), 0.6)
    clearance[1, 2] = float("inf")
    clearance[2, :] = 0.15
    clearance[4, :] = 1.0
    upright = torch.tensor([1.0, 1.0, 1.0, 0.2, 1.0])
    result = validity_from_measurements(clearance, upright)
    assert result["physical_failure"].tolist() == [False, True, True, True, True]
    assert result["ground_support_fraction"][1].item() == pytest.approx(0.8)
    assert torch.isfinite(result["ground_clearance_m"]).all()
    assert result["upright_cosine"][3].item() == pytest.approx(0.2)
