"""Check live USD sphere identity and signed contact witnesses without changing physics."""

import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from npa.workflows.navigation.contact_witness import _sphere_record, _terrain_witness

torch = pytest.importorskip("torch")

BODY = "/World/envs/env_0/Robot/LF_FOOT"


def _stage():
    stage = Usd.Stage.CreateInMemory()
    body = UsdGeom.Xform.Define(stage, BODY).GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    sphere = UsdGeom.Sphere.Define(stage, BODY + "/collisions/shape")
    sphere.CreateRadiusAttr(0.03)
    UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
    UsdGeom.Xformable(sphere).AddTranslateOp().Set((0, 0, 0.0225))
    return stage, body, sphere.GetPrim()


def test_actual_live_sphere_radius_pose_identity_is_read_only():
    stage, _, collider = _stage()
    before = stage.GetRootLayer().ExportToString()
    record = _sphere_record(stage, BODY, 1)
    assert record == {
        "native_path": BODY,
        "collider_path": str(collider.GetPath()),
        "geometry_type": "Sphere",
        "radius_m": 0.03,
        "scale": [1.0, 1.0, 1.0],
        "body_local_center_m": [0.0, 0.0, 0.0225],
        "status": "verified_live_sphere",
    }
    assert stage.GetRootLayer().ExportToString() == before


def test_instance_proxy_sphere_is_verified_in_its_actual_foot():
    stage, _, _ = _stage()
    stage.DefinePrim("/Prototype", "Xform")
    sphere = UsdGeom.Sphere.Define(stage, "/Prototype/shape")
    sphere.CreateRadiusAttr(0.04)
    UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
    stage.RemovePrim(BODY + "/collisions")
    instance = stage.DefinePrim(BODY + "/collisions", "Xform")
    instance.GetReferences().AddInternalReference("/Prototype")
    instance.SetInstanceable(True)
    record = _sphere_record(stage, BODY, 1)
    assert record["status"] == "verified_live_sphere"
    assert record["collider_path"] == BODY + "/collisions/shape"
    assert record["radius_m"] == 0.04


@pytest.mark.parametrize("kind", ["Cube", "Capsule", "Mesh"])
def test_non_sphere_geometry_cannot_use_sphere_witness(kind):
    stage, _, collider = _stage()
    collider.SetTypeName(kind)
    assert _sphere_record(stage, BODY, 1)["status"] == "unsupported_geometry"


@pytest.mark.parametrize(
    "condition", ["missing", "disabled", "multiple", "inactive", "nested"]
)
def test_unavailable_or_ambiguous_collider_identity_is_rejected(condition):
    stage, _, collider = _stage()
    if condition == "missing":
        stage.RemovePrim(str(collider.GetPath()))
    elif condition == "disabled":
        UsdPhysics.CollisionAPI(collider).CreateCollisionEnabledAttr(False)
    elif condition == "multiple":
        other = UsdGeom.Sphere.Define(stage, BODY + "/other")
        UsdPhysics.CollisionAPI.Apply(other.GetPrim())
    elif condition == "inactive":
        collider.SetActive(False)
    else:
        UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(BODY + "/collisions"))
    assert _sphere_record(stage, BODY, 1)["status"] != "verified_live_sphere"


def test_unloaded_payload_is_not_assumed_to_contain_one_sphere(tmp_path):
    payload = Usd.Stage.CreateNew(str(tmp_path / "payload.usda"))
    prim = UsdGeom.Xform.Define(payload, "/asset").GetPrim()
    payload.SetDefaultPrim(prim)
    payload.GetRootLayer().Save()
    stage, _, _ = _stage()
    child = stage.DefinePrim(BODY + "/unloaded", "Xform")
    child.GetPayloads().AddPayload(str(tmp_path / "payload.usda"))
    stage.Unload(child.GetPath())
    assert _sphere_record(stage, BODY, 1)["status"] == "inactive_or_unloaded_descendant"


@pytest.mark.parametrize("scale", [(1, 2, 1), (-1, 1, 1), (0, 1, 1)])
def test_nonuniform_reflected_or_singular_sphere_transform_is_rejected(scale):
    stage, _, collider = _stage()
    UsdGeom.Xformable(collider).AddScaleOp().Set(scale)
    assert _sphere_record(stage, BODY, 1)["status"] == "unsupported_sphere_transform"


def test_shear_invalid_radius_and_native_shape_count_are_rejected():
    stage, _, collider = _stage()
    matrix = Gf.Matrix4d(1)
    matrix[0, 1] = 0.2
    UsdGeom.Xformable(collider).AddTransformOp().Set(matrix)
    assert _sphere_record(stage, BODY, 1)["status"] == "unsupported_sphere_transform"
    stage, _, collider = _stage()
    UsdGeom.Sphere(collider).GetRadiusAttr().Set(-0.03)
    assert _sphere_record(stage, BODY, 1)["status"] == "invalid_sphere_radius"
    assert _sphere_record(stage, BODY, 2)["status"] == "unsupported_native_shape_count"


@pytest.mark.parametrize("force,normal", [(2.0, (0, 0, 1)), (-2.0, (0, 0, -1))])
def test_pair_orientation_recovers_same_terrain_witness_for_both_actor_orders(
    force, normal
):
    points = torch.tensor([[0, 0, -0.016]], dtype=torch.float32)
    normals = torch.tensor([normal], dtype=torch.float32)
    separation = torch.tensor([-0.016])
    forces = torch.tensor([force])
    before = [value.clone() for value in (points, normals, separation, forces)]
    witness, valid, reason = _terrain_witness(
        points, normals, separation, forces, torch.tensor([True])
    )
    assert witness.tolist() == [[0.0, 0.0, 0.0]] and valid.tolist() == [True]
    assert reason.tolist() == [0]
    for actual, saved in zip((points, normals, separation, forces), before):
        assert torch.equal(actual, saved)


@pytest.mark.parametrize(
    "normal,force,verified,expected",
    [
        ((0, 0, 0), 1, True, 13),
        ((0, 0, 1.01), 1, True, 13),
        ((float("nan"), 0, 1), 1, True, 13),
        ((0, 0, 1), 0, True, 13),
        ((0, 0, 1), float("inf"), True, 13),
        ((0, 0, 1), 1, False, 12),
    ],
)
def test_invalid_normal_force_or_unverified_shape_keeps_original_contact(
    normal, force, verified, expected
):
    witness, valid, reason = _terrain_witness(
        torch.zeros((1, 3)),
        torch.tensor([normal], dtype=torch.float32),
        torch.tensor([-0.016]),
        torch.tensor([force]),
        torch.tensor([verified]),
    )
    assert reason.tolist() == [expected] and valid.tolist() == [False]
    assert witness.tolist() == [[0, 0, 0]]


def test_witness_matches_explicit_float32_numpy_operation_order():
    points = np.array([[2.877, -0.759, -0.016]], dtype=np.float32)
    normals = np.array([[0.6, 0.0, 0.8]], dtype=np.float32)
    separation = np.array([-0.016558], dtype=np.float32)
    force = np.array([-2], dtype=np.float32)
    actual, _, _ = _terrain_witness(
        *map(torch.from_numpy, (points, normals, separation, force)),
        torch.tensor([True]),
    )
    assert np.array_equal(
        actual.numpy(), points - (separation * np.sign(force))[:, None] * normals
    )
