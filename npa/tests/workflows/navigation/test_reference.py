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
