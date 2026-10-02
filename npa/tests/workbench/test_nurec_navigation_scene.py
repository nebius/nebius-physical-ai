"""Exercise real OpenUSD navigation assembly, portability, and hostile inputs."""

from __future__ import annotations

import base64
import io
import json
import shutil
import stat
import zipfile

import numpy as np
import pytest

from npa.workbench.nurec.navigation_assets import sha256
from npa.workbench.nurec.navigation_scene import prepare_scene

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils, UsdVol

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII="
)


def _matrix(translation=(0, 0, 0)):
    matrix = np.eye(4)
    matrix[3, :3] = translation
    return matrix.tolist()


def _layer(path, *, scale=1, translation=(0, 0, 0), points_only=False):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, scale)
    UsdGeom.SetStageUpAxis(stage, "Z")
    root = UsdGeom.Xform.Define(stage, "/Site")
    root.AddTranslateOp().Set(Gf.Vec3d(*translation))
    stage.SetDefaultPrim(root.GetPrim())
    geometry = UsdGeom.Points if points_only else UsdGeom.Mesh
    surface = geometry.Define(stage, "/Site/Surface")
    extent = 1 / scale
    surface.CreatePointsAttr(
        [
            (-extent, -extent, 0),
            (extent, -extent, 0),
            (extent, extent, 0),
            (-extent, extent, 0),
        ]
    )
    if not points_only:
        surface.CreateFaceVertexCountsAttr([3, 3])
        surface.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
        surface.CreateSubdivisionSchemeAttr("none")
    stage.GetRootLayer().Save()
    return stage


def _texture(stage, asset="texture.png"):
    material = UsdShade.Material.Define(stage, "/Site/Material")
    shader = UsdShade.Shader.Define(stage, "/Site/Material/Texture")
    shader.CreateIdAttr("UsdUVTexture")
    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(asset))
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath("/Site/Surface")).Bind(
        material
    )
    stage.GetRootLayer().Save()


def _write_contract(root, **updates):
    path = root / "scene.json"
    contract = json.loads(path.read_text()) if path.exists() else {}
    contract.update(updates)
    path.write_text(json.dumps(contract))
    return contract


@pytest.fixture
def scene_bundle(tmp_path):
    """Create separate appearance and collision inputs with known metric alignment."""
    root = tmp_path / "input"
    root.mkdir()
    visual = _layer(root / "visual.usda", scale=0.01, translation=(200, 0, 100))
    _texture(visual)
    (root / "texture.png").write_bytes(_PNG)
    _layer(root / "collision.usda", scale=0.001, translation=(2000, 0, 1000))
    _write_contract(
        root,
        schema="npa.nurec.navigation_input.v1",
        visual_asset="visual.usda",
        collision_asset="collision.usda",
        capture_provenance={"sha256": "a" * 64},
        visual_to_world=_matrix((3, 4, 0)),
        collision_to_world=_matrix((3, 4, 0)),
        ray_probes=[
            {
                "origin": [5, 4, 3],
                "direction": [0, 0, -1],
                "min_distance": 1.9,
                "max_distance": 2.1,
            }
        ],
    )
    return root


def _fails(root, output, pattern):
    with pytest.raises(ValueError, match=pattern):
        prepare_scene(str(root), str(output))
    assert not output.exists()


def _world_points(stage, path):
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(path))
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(mesh.GetPrim())
    return np.asarray(
        [transform.Transform(Gf.Vec3d(*point)) for point in mesh.GetPointsAttr().Get()]
    )


def test_assembly_preserves_metric_registration_and_static_collision(
    scene_bundle, tmp_path
):
    output = tmp_path / "output"
    report = prepare_scene(str(scene_bundle), str(output))
    stage = Usd.Stage.Open(str(output / "scene.usdz"))
    expected = [[4, 3, 1], [6, 3, 1], [6, 5, 1], [4, 5, 1]]
    np.testing.assert_allclose(
        _world_points(stage, "/World/Visual/Source/Surface"), expected
    )
    np.testing.assert_allclose(
        _world_points(stage, "/World/Collision/Mesh_0"), expected
    )
    collider = stage.GetPrimAtPath("/World/Collision/Mesh_0")
    assert UsdPhysics.CollisionAPI(collider).GetCollisionEnabledAttr().Get() is True
    assert UsdPhysics.MeshCollisionAPI(collider).GetApproximationAttr().Get() == "none"
    assert all(not prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in stage.Traverse())
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1
    assert UsdGeom.GetStageUpAxis(stage) == "Z"
    assert report["collider_count"] == 1
    assert report["triangle_count"] == 2
    assert report["colliders"][0]["bounds_m"] == [[4, 3, 1], [6, 5, 1]]


def test_output_provenance_hashes_actual_inputs_and_disclaims_native_validation(
    scene_bundle, tmp_path
):
    output = tmp_path / "output"
    contract = json.loads((scene_bundle / "scene.json").read_text())
    report = prepare_scene(str(scene_bundle), str(output))
    assert json.loads((output / "provenance.json").read_text()) == report
    assert report["schema"] == "npa.nurec.navigation_scene.v1"
    assert report["scene_sha256"] == sha256(output / "scene.usdz")
    assert report["capture_manifest_sha256"] == "a" * 64
    assert report["input_contract_sha256"] == sha256(scene_bundle / "scene.json")
    assert report["inputs"] == {
        "visual": sha256(scene_bundle / "visual.usda"),
        "collision": sha256(scene_bundle / "collision.usda"),
    }
    assert sha256(scene_bundle / "texture.png") in report["dependency_sha256"]
    assert report["source_units"] == {"visual": 0.01, "collision": 0.001}
    assert report["registration"]["visual_to_world"] == contract["visual_to_world"]
    assert report["ray_probes"] == contract["ray_probes"]
    assert report["physics_validated"] is False
    assert report["visual_render_validated"] is False


def test_export_reopens_after_inputs_deleted_and_archive_relocated(
    scene_bundle, tmp_path
):
    output = tmp_path / "output"
    prepare_scene(str(scene_bundle), str(output))
    relocated = tmp_path / "portable.usdz"
    shutil.move(output / "scene.usdz", relocated)
    shutil.rmtree(scene_bundle)
    shutil.rmtree(output)
    stage = Usd.Stage.Open(str(relocated))
    assert stage and not stage.GetCompositionErrors()
    shader = UsdShade.Shader(
        stage.GetPrimAtPath("/World/Visual/Source/Material/Texture")
    )
    texture = shader.GetInput("file").Get()
    assert str(relocated) in texture.resolvedPath
    with zipfile.ZipFile(relocated) as archive:
        textures = [name for name in archive.namelist() if name.endswith("texture.png")]
        assert len(textures) == 1
        assert archive.read(textures[0]) == _PNG
    _, _, unresolved = UsdUtils.ComputeAllDependencies(str(relocated))
    assert not unresolved


def test_neural_volume_usdz_preserves_opaque_payload_inside_portable_export(
    scene_bundle, tmp_path
):
    path = scene_bundle / "neural.usda"
    stage = _layer(path)
    stage.RemovePrim("/Site/Surface")
    volume = UsdVol.Volume.Define(stage, "/Site/Volume").GetPrim()
    volume.CreateAttribute("fixture:neuralAsset", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("neural.ply")
    )
    payload = b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n"
    (scene_bundle / "neural.ply").write_bytes(payload)
    stage.GetRootLayer().Save()
    archive = scene_bundle / "neural.usdz"
    assert UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(path)), str(archive))
    _write_contract(scene_bundle, visual_asset="neural.usdz", visual_to_world=_matrix())
    output = tmp_path / "output"
    report = prepare_scene(str(scene_bundle), str(output))
    relocated = tmp_path / "portable.usdz"
    shutil.move(output / "scene.usdz", relocated)
    shutil.rmtree(scene_bundle)
    shutil.rmtree(output)
    assembled = Usd.Stage.Open(str(relocated))
    prim = assembled.GetPrimAtPath("/World/Visual/Source/Volume")
    assert prim.IsA(UsdVol.Volume)
    asset = prim.GetAttribute("fixture:neuralAsset").Get()
    assert str(relocated) in asset.resolvedPath
    _, _, unresolved = UsdUtils.ComputeAllDependencies(str(relocated))
    assert not unresolved
    assert report["visual_render_validated"] is False
    with zipfile.ZipFile(relocated) as bundle:
        nested = [name for name in bundle.namelist() if name.endswith("neural.usdz")]
        assert len(nested) == 1
        with zipfile.ZipFile(io.BytesIO(bundle.read(nested[0]))) as inner:
            members = [name for name in inner.namelist() if name.endswith("neural.ply")]
            assert len(members) == 1
            assert inner.read(members[0]) == payload


@pytest.mark.parametrize("role", ["visual", "collision"])
def test_y_up_sources_use_explicit_rotation_into_z_up_world(
    scene_bundle, tmp_path, role
):
    path = scene_bundle / f"{role}.usda"
    stage = Usd.Stage.Open(str(path))
    UsdGeom.SetStageUpAxis(stage, "Y")
    stage.GetRootLayer().Save()
    rotation = np.eye(4)
    rotation[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    _write_contract(scene_bundle, **{f"{role}_to_world": rotation.tolist()})
    report = prepare_scene(str(scene_bundle), str(tmp_path / "output"))
    assert report["source_up_axis"][role] == "Y"
    path = (
        "/World/Visual/Source/Surface"
        if role == "visual"
        else "/World/Collision/Mesh_0"
    )
    result = Usd.Stage.Open(str(tmp_path / "output/scene.usdz"))
    np.testing.assert_allclose(
        _world_points(result, path), [[1, -1, -1], [3, -1, -1], [3, -1, 1], [1, -1, 1]]
    )


def test_missing_collision_contract_fails_without_publishing(scene_bundle, tmp_path):
    contract = json.loads((scene_bundle / "scene.json").read_text())
    del contract["collision_asset"]
    (scene_bundle / "scene.json").write_text(json.dumps(contract))
    _fails(scene_bundle, tmp_path / "output", "collision|asset")


def test_point_cloud_cannot_substitute_for_collision_surface(scene_bundle, tmp_path):
    _layer(scene_bundle / "points.usda", points_only=True)
    _write_contract(scene_bundle, collision_asset="points.usda")
    _fails(
        scene_bundle, tmp_path / "output", "Gaussian splats are not collision geometry"
    )


def test_empty_visual_root_does_not_count_as_reconstructed_appearance(
    scene_bundle, tmp_path
):
    stage = Usd.Stage.Open(str(scene_bundle / "visual.usda"))
    stage.RemovePrim("/Site/Surface")
    stage.GetRootLayer().Save()
    _fails(scene_bundle, tmp_path / "output", "visual|drawable|appearance")


def test_missing_referenced_texture_cannot_produce_success(scene_bundle, tmp_path):
    (scene_bundle / "texture.png").unlink()
    _fails(scene_bundle, tmp_path / "output", "missing|empty")


@pytest.mark.parametrize("role", ["visual_to_world", "collision_to_world"])
@pytest.mark.parametrize(
    "invalid",
    ["nan", "infinity", "singular", "reflection", "scale", "column", "missing"],
)
def test_registration_rejects_nonrigid_or_nonfinite_matrices(
    scene_bundle, tmp_path, role, invalid
):
    matrix = np.eye(4)
    if invalid == "column":
        matrix[0, 3] = 1
    else:
        matrix[0, 0] = {
            "nan": np.nan,
            "infinity": np.inf,
            "singular": 0,
            "reflection": -1,
            "scale": 2,
            "missing": 1,
        }[invalid]
    _write_contract(
        scene_bundle, **{role: None if invalid == "missing" else matrix.tolist()}
    )
    _fails(scene_bundle, tmp_path / "output", "registration")


@pytest.mark.parametrize("field", ["metersPerUnit", "upAxis", "defaultPrim"])
@pytest.mark.parametrize("role", ["visual", "collision"])
def test_missing_usd_metadata_is_not_silently_defaulted(
    scene_bundle, tmp_path, role, field
):
    stage = Usd.Stage.Open(str(scene_bundle / f"{role}.usda"))
    stage.ClearMetadata(field)
    stage.GetRootLayer().Save()
    _fails(scene_bundle, tmp_path / "output", "explicit|default prim")


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf")])
def test_invalid_metric_scale_fails_before_assembly(scene_bundle, tmp_path, scale):
    stage = Usd.Stage.Open(str(scene_bundle / "collision.usda"))
    UsdGeom.SetStageMetersPerUnit(stage, scale)
    stage.GetRootLayer().Save()
    _fails(scene_bundle, tmp_path / "output", "metersPerUnit")


@pytest.mark.parametrize("invalid", ["nan", "singular"])
def test_invalid_source_transform_cannot_evade_contract_validation(
    scene_bundle, tmp_path, invalid
):
    stage = Usd.Stage.Open(str(scene_bundle / "collision.usda"))
    root = UsdGeom.Xformable(stage.GetDefaultPrim())
    root.ClearXformOpOrder()
    matrix = np.eye(4)
    matrix[0, 0] = float("nan") if invalid == "nan" else 0
    root.AddTransformOp().Set(Gf.Matrix4d(matrix.tolist()))
    stage.GetRootLayer().Save()
    _fails(scene_bundle, tmp_path / "output", "source transform")


def test_collision_subdivision_control_mesh_is_not_treated_as_surface(
    scene_bundle, tmp_path
):
    stage = Usd.Stage.Open(str(scene_bundle / "collision.usda"))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Site/Surface"))
    mesh.GetSubdivisionSchemeAttr().Set("catmullClark")
    stage.GetRootLayer().Save()
    _fails(scene_bundle, tmp_path / "output", "subdivision|triangle")


@pytest.mark.parametrize(
    "case",
    ["nan", "infinity", "degenerate", "index", "count", "quad", "empty", "holes"],
)
def test_invalid_collision_geometry_fails_closed(scene_bundle, tmp_path, case):
    stage = Usd.Stage.Open(str(scene_bundle / "collision.usda"))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Site/Surface"))
    if case in {"nan", "infinity"}:
        mesh.GetPointsAttr().Set(
            [(float("nan" if case == "nan" else "inf"), 0, 0), (1, 0, 0), (0, 1, 0)]
        )
    elif case == "degenerate":
        mesh.GetPointsAttr().Set([(0, 0, 0)] * 4)
    elif case == "index":
        mesh.GetFaceVertexIndicesAttr().Set([0, 1, 8, 0, 2, 3])
    elif case == "count":
        mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2])
    elif case == "quad":
        mesh.GetFaceVertexCountsAttr().Set([4])
        mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
    elif case == "empty":
        mesh.GetFaceVertexCountsAttr().Set([])
    else:
        mesh.CreateHoleIndicesAttr([0])
    stage.GetRootLayer().Save()
    _fails(scene_bundle, tmp_path / "output", "collision mesh")


@pytest.mark.parametrize(
    "kind", ["reference", "payload", "sublayer", "texture", "inactive_variant"]
)
def test_external_usd_assets_are_rejected_before_composition(
    scene_bundle, tmp_path, kind
):
    stage = Usd.Stage.Open(str(scene_bundle / "visual.usda"))
    prim = stage.GetDefaultPrim()
    external = "https://example.invalid/unavailable.usda"
    if kind == "reference":
        prim.GetReferences().AddReference(external)
    elif kind == "payload":
        prim.GetPayloads().AddPayload(external)
    elif kind == "sublayer":
        stage.GetRootLayer().subLayerPaths.append(external)
    elif kind == "texture":
        _texture(stage, "https://example.invalid/unavailable.png")
    else:
        variants = prim.GetVariantSets().AddVariantSet("look")
        for name in ("clean", "external"):
            variants.AddVariant(name)
        variants.SetVariantSelection("external")
        with variants.GetVariantEditContext():
            prim.GetReferences().AddReference(external)
        variants.SetVariantSelection("clean")
    stage.GetRootLayer().Save()
    _fails(scene_bundle, tmp_path / "output", "relative bundle path")


@pytest.mark.parametrize(
    "asset",
    ["../outside.usda", "/outside.usda", "folder/../../outside.usda", "missing.usda"],
)
def test_collision_paths_require_existing_contained_files(
    scene_bundle, tmp_path, asset
):
    _write_contract(scene_bundle, collision_asset=asset)
    _fails(scene_bundle, tmp_path / "output", "escapes|missing")


@pytest.mark.parametrize("kind", ["traversal", "symlink", "duplicate", "compressed"])
def test_hostile_usdz_cannot_publish_output(scene_bundle, tmp_path, kind):
    archive = scene_bundle / "collision.usdz"
    compression = zipfile.ZIP_DEFLATED if kind == "compressed" else zipfile.ZIP_STORED
    with zipfile.ZipFile(archive, "w", compression=compression) as bundle:
        bundle.writestr("root.usda", (scene_bundle / "collision.usda").read_bytes())
        if kind == "traversal":
            bundle.writestr("../outside.txt", b"escape")
        elif kind == "symlink":
            member = zipfile.ZipInfo("link")
            member.external_attr = (stat.S_IFLNK | 0o777) << 16
            bundle.writestr(member, "../outside.txt")
        elif kind == "duplicate":
            with pytest.warns(UserWarning, match="Duplicate name"):
                bundle.writestr("root.usda", b"duplicate")
    _write_contract(scene_bundle, collision_asset="collision.usdz")
    _fails(scene_bundle, tmp_path / "output", "USDZ|unsafe|traversal|relative|escapes")
    assert not (tmp_path / "outside.txt").exists()


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_local_bundle_rejects_symlink_members(scene_bundle, tmp_path, kind):
    target = scene_bundle / "visual.usda" if kind == "file" else tmp_path
    (scene_bundle / "link").symlink_to(target, target_is_directory=kind == "directory")
    _fails(scene_bundle, tmp_path / "output", "symbolic")


@pytest.mark.parametrize("referenced", [True, False])
def test_nested_archive_external_dependency_is_rejected(
    scene_bundle, tmp_path, referenced
):
    bad = '#usda 1.0\ndef Xform "External" {\n asset source = @https://example.invalid/escape.ply@\n}\n'
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as archive:
        archive.writestr("root.usda", (scene_bundle / "collision.usda").read_bytes())
        archive.writestr("unselected.usda", bad)
    outer = scene_bundle / "nested.usdz"
    with zipfile.ZipFile(outer, "w") as archive:
        root = (scene_bundle / "collision.usda").read_text()
        if referenced:
            root = root.replace(
                'def Xform "Site"',
                'def Xform "Site" (prepend references = @inner.usdz@)',
            )
        archive.writestr("root.usda", root)
        archive.writestr("inner.usdz", inner.getvalue())
    _write_contract(scene_bundle, collision_asset="nested.usdz")
    _fails(scene_bundle, tmp_path / "output", "relative bundle path")


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "unsupported"),
        ("capture_provenance", {}),
        ("capture_provenance", {"sha256": "not-a-digest"}),
        ("ray_probes", []),
    ],
)
def test_incomplete_scene_contracts_fail_closed(scene_bundle, tmp_path, field, value):
    _write_contract(scene_bundle, **{field: value})
    _fails(scene_bundle, tmp_path / "output", "schema|capture_provenance|ray_probes")


@pytest.mark.parametrize(
    "field,value",
    [
        ("direction", [0, 0, 0]),
        ("origin", [float("nan"), 0, 0]),
        ("min_distance", -1),
        ("max_distance", float("inf")),
    ],
)
def test_invalid_physics_probe_contracts_fail_closed(
    scene_bundle, tmp_path, field, value
):
    contract = json.loads((scene_bundle / "scene.json").read_text())
    contract["ray_probes"][0][field] = value
    _write_contract(scene_bundle, **contract)
    _fails(scene_bundle, tmp_path / "output", "ray probe")


@pytest.mark.parametrize(
    "case", ["rigid_body", "collider", "physics_scene", "reset_stack", "animated"]
)
def test_unsupported_source_physics_and_motion_fail_closed(
    scene_bundle, tmp_path, case
):
    stage = Usd.Stage.Open(str(scene_bundle / "collision.usda"))
    prim = stage.GetDefaultPrim()
    if case == "rigid_body":
        UsdPhysics.RigidBodyAPI.Apply(prim)
    elif case == "collider":
        UsdPhysics.CollisionAPI.Apply(prim)
    elif case == "physics_scene":
        UsdPhysics.Scene.Define(stage, "/Site/Physics")
    elif case == "reset_stack":
        UsdGeom.Xformable(prim).SetResetXformStack(True)
    else:
        UsdGeom.Xformable(prim).GetOrderedXformOps()[0].Set(Gf.Vec3d(1, 0, 0), 1)
    stage.GetRootLayer().Save()
    _fails(scene_bundle, tmp_path / "output", "physics|registration|static")


def test_existing_output_is_never_overwritten(scene_bundle, tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    existing = output / "keep.txt"
    existing.write_text("operator data")
    with pytest.raises(FileExistsError):
        prepare_scene(str(scene_bundle), str(output))
    assert existing.read_text() == "operator data"
    assert sorted(path.name for path in output.iterdir()) == ["keep.txt"]


def test_perspective_source_transform_is_rejected(scene_bundle, tmp_path):
    stage = Usd.Stage.Open(str(scene_bundle / "collision.usda"))
    root = UsdGeom.Xformable(stage.GetDefaultPrim())
    root.ClearXformOpOrder()
    matrix = Gf.Matrix4d(1)
    matrix[0, 3] = 0.1
    root.AddTransformOp().Set(matrix)
    stage.GetRootLayer().Save()
    _fails(scene_bundle, tmp_path / "output", "must be affine")


def test_empty_neural_volume_without_payload_is_rejected(scene_bundle, tmp_path):
    stage = Usd.Stage.Open(str(scene_bundle / "visual.usda"))
    stage.RemovePrim("/Site/Surface")
    UsdVol.Volume.Define(stage, "/Site/EmptyVolume")
    stage.GetRootLayer().Save()
    _fails(scene_bundle, tmp_path / "output", "no geometry or neural Volume")


def test_unregistered_physx_api_cannot_hide_in_visual_source(scene_bundle, tmp_path):
    stage = Usd.Stage.Open(str(scene_bundle / "visual.usda"))
    stage.GetDefaultPrim().AddAppliedSchema("PhysxRigidBodyAPI")
    stage.GetRootLayer().Save()
    _fails(scene_bundle, tmp_path / "output", "preexisting physics")


def _dynamic_spec(layer, path, kind):
    prim = Sdf.CreatePrimInLayer(layer, path)
    prim.specifier = Sdf.SpecifierDef
    prim.typeName = "Xform"
    if kind in {"script", "script_api"}:
        prim.SetInfo(
            "apiSchemas", Sdf.TokenListOp.Create(prependedItems=["OmniScriptingAPI"])
        )
    if kind in {"script", "script_property"}:
        Sdf.AttributeSpec(
            prim, "omni:scripting:scripts", Sdf.ValueTypeNames.AssetArray
        ).default = [Sdf.AssetPath("behavior.py")]
    if kind in {"graph", "graph_node"}:
        prim.typeName = "OmniGraph" if kind == "graph" else "OmniGraphNode"
    if kind == "graph_property":
        Sdf.AttributeSpec(
            prim, "node:type", Sdf.ValueTypeNames.Token
        ).default = "omni.graph.nodes.Noop"
    if kind == "time_samples":
        attr = Sdf.AttributeSpec(prim, "xformOp:translate", Sdf.ValueTypeNames.Double3)
        layer.SetTimeSample(attr.path, 1, Gf.Vec3d(0, 0, 1))
    if kind == "clips":
        prim.SetInfo(
            "clips", {"default": {"assetPaths": Sdf.AssetPathArray(["clip.usda"])}}
        )
    return prim


def _hidden_layer(stage, location, root):
    layer, path = stage.GetRootLayer(), "/Site/Behavior"
    if location == "variant":
        variants = stage.GetDefaultPrim().GetVariantSets().AddVariantSet("behavior")
        for name in ("clean", "hidden"):
            variants.AddVariant(name)
        variants.SetVariantSelection("clean")
        path = "/Site{behavior=hidden}Behavior"
    if location == "overridden":
        layer = Sdf.Layer.CreateNew(str(root / "weak.usda"))
        stage.GetRootLayer().subLayerPaths.append("weak.usda")
        prim = stage.DefinePrim(path, "Xform")
        prim.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit([]))
        for name in ("omni:scripting:scripts", "node:type"):
            prim.CreateAttribute(name, Sdf.ValueTypeNames.Token).Block()
    return layer, path


@pytest.mark.parametrize("location", ["active", "inactive", "variant", "overridden"])
@pytest.mark.parametrize(
    "kind",
    [
        "script",
        "script_api",
        "script_property",
        "graph",
        "graph_node",
        "graph_property",
        "time_samples",
        "clips",
    ],
)
def test_raw_dynamic_content_is_rejected_before_stage_composition(
    scene_bundle, tmp_path, monkeypatch, location, kind
):
    stage = Usd.Stage.Open(str(scene_bundle / "visual.usda"))
    layer, path = _hidden_layer(stage, location, scene_bundle)
    prim = _dynamic_spec(layer, path, kind)
    if location == "inactive":
        prim.active = False
    (scene_bundle / "behavior.py").write_text("# inert attachment; never evaluated\n")
    layer.Save()
    stage.GetRootLayer().Save()
    monkeypatch.setattr(
        "npa.workbench.nurec.navigation_scene.open_static_stage",
        lambda *args: pytest.fail("dynamic content reached stage composition"),
    )
    _fails(scene_bundle, tmp_path / "output", "executable|static scene|static, without")


@pytest.mark.parametrize("kind", ["script", "graph", "time_samples", "clips"])
def test_unused_layers_inside_nested_usdz_reject_dynamic_content(
    scene_bundle, tmp_path, monkeypatch, kind
):
    hidden = Sdf.Layer.CreateNew(str(scene_bundle / "hidden.usda"))
    _dynamic_spec(hidden, "/Hidden", kind)
    hidden.Save()
    (scene_bundle / "behavior.py").write_text("# inert attachment\n")
    inner = scene_bundle / "inner.usdz"
    with Sdf.ZipFileWriter.CreateNew(str(inner)) as writer:
        writer.AddFile(str(scene_bundle / "collision.usda"), "root.usda")
        writer.AddFile(str(scene_bundle / "hidden.usda"), "hidden.usda")
        writer.AddFile(str(scene_bundle / "behavior.py"), "behavior.py")
    outer = scene_bundle / "outer.usdz"
    with Sdf.ZipFileWriter.CreateNew(str(outer)) as writer:
        writer.AddFile(str(scene_bundle / "collision.usda"), "root.usda")
        writer.AddFile(str(inner), "inner.usdz")
    _write_contract(scene_bundle, visual_asset="outer.usdz")
    monkeypatch.setattr(
        "npa.workbench.nurec.navigation_scene.open_static_stage",
        lambda *args: pytest.fail("nested dynamic content reached composition"),
    )
    _fails(scene_bundle, tmp_path / "output", "executable|static scene|static, without")


def test_shader_nodegraph_and_script_named_texture_remain_supported(
    scene_bundle, tmp_path
):
    stage = Usd.Stage.Open(str(scene_bundle / "visual.usda"))
    graph = UsdShade.NodeGraph.Define(stage, "/Site/Material/Graph")
    graph.GetPrim().SetMetadata(
        "apiSchemas", Sdf.TokenListOp.Create(prependedItems=["NodeGraphNodeAPI"])
    )
    texture = "ordinary_script_texture.png"
    (scene_bundle / texture).write_bytes(_PNG)
    _texture(stage, texture)
    output = tmp_path / "output"
    prepare_scene(str(scene_bundle), str(output))
    assembled = Usd.Stage.Open(str(output / "scene.usdz"))
    assert assembled.GetPrimAtPath("/World/Visual/Source/Material/Graph").IsA(
        UsdShade.NodeGraph
    )
    shader = assembled.GetPrimAtPath("/World/Visual/Source/Material/Texture")
    assert shader.GetAttribute("info:id").Get() == "UsdUVTexture"
    assert shader.GetAttribute("inputs:file").Get().resolvedPath


def test_custom_composition_format_cannot_bypass_raw_layer_audit(
    scene_bundle, tmp_path, monkeypatch
):
    path = scene_bundle / "visual.usda"
    layer = Sdf.Layer.FindOrOpen(str(path))
    layer.GetPrimAtPath("/Site").referenceList.prependedItems = [
        Sdf.Reference("hidden.sdf")
    ]
    layer.Save()
    (scene_bundle / "hidden.sdf").write_text('#usda 1.0\ndef OmniGraph "Hidden" {}\n')
    monkeypatch.setattr(
        "npa.workbench.nurec.navigation_scene.open_static_stage",
        lambda *args: pytest.fail("unsupported format reached stage composition"),
    )
    _fails(scene_bundle, tmp_path / "output", "composition dependencies must be USD")
