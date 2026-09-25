"""Parse actual OpenUSD files to reject scripts, variants, and partial composition."""

from __future__ import annotations

import sys
import shutil
from types import ModuleType, SimpleNamespace

from PIL import Image
from pxr import Sdf, Usd, UsdGeom, UsdUtils
import pytest

from npa.workflows.isaac_rgbd import runtime
from npa.workflows.isaac_rgbd.contract import _sha256
from npa.workflows.isaac_rgbd.contract import validate_request
from npa.workflows.isaac_rgbd.fixture import write_fixture
from npa.workflows.isaac_rgbd.scene import _check_scene_files


def _kit_context(monkeypatch, context):
    omni, usd = ModuleType("omni"), ModuleType("omni.usd")
    usd.get_context = lambda: context
    omni.usd = usd
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.usd", usd)


def _forbidden_content(layer, prim, kind):
    if kind in {"api", "graph_api"}:
        prim.SetInfo(
            "apiSchemas",
            Sdf.TokenListOp.Create(
                prependedItems=["OmniScriptingAPI" if kind == "api" else "OmniGraphAPI"]
            ),
        )
    elif kind == "attribute":
        prop = Sdf.AttributeSpec(
            prim, "omni:scripting:scripts", Sdf.ValueTypeNames.AssetArray
        )
        prop.default = Sdf.AssetPathArray([Sdf.AssetPath("behavior.py")])
    elif kind == "graph_type":
        prim.typeName = "OmniGraph"
    elif kind == "node_type":
        prop = Sdf.AttributeSpec(prim, "node:type", Sdf.ValueTypeNames.Token)
        prop.default = "omni.graph.scriptnode.ScriptNode"
    elif kind == "animation":
        prop = Sdf.AttributeSpec(prim, "xformOp:translate", Sdf.ValueTypeNames.Double3)
        layer.SetTimeSample(prop.path, 0, (0.0, 0.0, 0.0))
        layer.SetTimeSample(prop.path, 1, (1.0, 0.0, 0.0))
    else:
        prim.SetInfo(
            "clips",
            {
                "default": {
                    "assetPaths": Sdf.AssetPathArray([Sdf.AssetPath("scene.usda")])
                }
            },
        )


def _modified_fixture(tmp_path, location, kind):
    root = tmp_path / "bundle"
    request = write_fixture(root)
    layer = Sdf.Layer.FindOrOpen(str(root / "scene.usda"))
    paths = {
        "active": "/World/Behavior",
        "inactive": "/World/Inactive/Behavior",
        "variant_child": "/World{mode=unsafe}Behavior",
        "variant_root": "/World{mode=unsafe}",
    }
    prim = Sdf.CreatePrimInLayer(layer, paths[location])
    if location == "inactive":
        layer.GetPrimAtPath("/World/Inactive").active = False
    if location.startswith("variant"):
        Sdf.CreatePrimInLayer(layer, "/World{mode=safe}Cube")
        layer.GetPrimAtPath("/World").variantSelections["mode"] = "safe"
    _forbidden_content(layer, prim, kind)
    layer.Save()
    (root / "behavior.py").write_text("raise RuntimeError('must never run')\n")
    request["files"] = {
        name: _sha256(root / name) for name in ("scene.usda", "behavior.py")
    }
    return root, request


@pytest.mark.parametrize(
    "location,kind",
    [
        (location, kind)
        for location in ("active", "inactive", "variant_child", "variant_root")
        for kind in (
            "api",
            "graph_api",
            "attribute",
            "graph_type",
            "node_type",
            "animation",
            "clips",
        )
        # USDA cannot author a typeName on a variant root; graph API metadata can.
        if (location, kind) != ("variant_root", "graph_type")
    ],
)
def test_real_sdf_rejects_executable_or_animated_content_before_kit_open(
    tmp_path, monkeypatch, location, kind
):
    root, request = _modified_fixture(tmp_path, location, kind)
    opened = []
    context = SimpleNamespace(open_stage=lambda *args: opened.append(args))
    _kit_context(monkeypatch, context)
    with pytest.raises(ValueError, match="forbids|animated"):
        runtime._open_scene(root, request)
    assert opened == [], "no Kit stage-open side effects are permitted before preflight"


def test_valid_child_layer_missing_referenced_prim_fails_before_render(
    tmp_path, monkeypatch
):
    root = tmp_path / "bundle"
    request = write_fixture(root)
    child = Usd.Stage.CreateNew(str(root / "child.usda"))
    child.DefinePrim("/Present", "Xform")
    child.GetRootLayer().Save()
    layer = Sdf.Layer.FindOrOpen(str(root / "scene.usda"))
    prim = Sdf.CreatePrimInLayer(layer, "/World/Child")
    prim.referenceList.prependedItems = [Sdf.Reference("child.usda", "/Missing")]
    layer.Save()
    request["files"] = {
        name: _sha256(root / name) for name in ("scene.usda", "child.usda")
    }
    _check_scene_files(root, request)
    stage = Usd.Stage.Open(str(root / "scene.usda"))
    assert stage.GetCompositionErrors(), "fixture must have a real composition error"
    context = SimpleNamespace(open_stage=lambda path: True, get_stage=lambda: stage)
    _kit_context(monkeypatch, context)
    with pytest.raises(ValueError, match="composition errors"):
        runtime._open_scene(root, request)


def test_plain_fixture_is_valid_real_usd(tmp_path, monkeypatch):
    root = tmp_path / "bundle"
    request = write_fixture(root)
    stage = Usd.Stage.Open(str(root / "scene.usda"))
    assert not stage.GetCompositionErrors()
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1
    assert len(list(stage.Traverse())) == 8
    context = SimpleNamespace(open_stage=lambda path: True, get_stage=lambda: stage)
    _kit_context(monkeypatch, context)
    assert runtime._open_scene(root, request) == stage


@pytest.mark.parametrize(
    "asset",
    [
        "../escape.usda",
        "https://host/scene.usda",
        pytest.param(
            lambda directory: str(directory / "asset.usda"), id="absolute-path"
        ),
        "missing.png",
    ],
)
def test_real_usd_asset_dependencies_must_be_contained_and_declared(tmp_path, asset):
    if callable(asset):
        asset = asset(tmp_path)
    root = tmp_path / "bundle"
    request = write_fixture(root)
    layer = Sdf.Layer.FindOrOpen(str(root / "scene.usda"))
    prim = Sdf.CreatePrimInLayer(layer, "/World/Texture")
    prop = Sdf.AttributeSpec(prim, "inputs:file", Sdf.ValueTypeNames.Asset)
    prop.default = Sdf.AssetPath(asset)
    layer.Save()
    request["files"]["scene.usda"] = _sha256(root / "scene.usda")
    with pytest.raises(ValueError):
        _check_scene_files(root, request)


def test_real_texture_dependency_is_included_by_usd_extraction(tmp_path):
    root = tmp_path / "bundle"
    request = write_fixture(root)
    Image.new("RGB", (2, 2), "white").save(root / "texture.png")
    layer = Sdf.Layer.FindOrOpen(str(root / "scene.usda"))
    prim = Sdf.CreatePrimInLayer(layer, "/World/Texture")
    prim.typeName = "Shader"
    prop = Sdf.AttributeSpec(prim, "inputs:file", Sdf.ValueTypeNames.Asset)
    prop.default = Sdf.AssetPath("texture.png")
    layer.Save()
    assert (
        "texture.png" in UsdUtils.ExtractExternalReferences(str(root / "scene.usda"))[1]
    )
    request["files"]["scene.usda"] = _sha256(root / "scene.usda")
    with pytest.raises(ValueError, match="absent"):
        _check_scene_files(root, request)
    request["files"]["texture.png"] = _sha256(root / "texture.png")
    _check_scene_files(root, request)


def _scripted_package(root, suffix):
    layer = Sdf.Layer.CreateNew(str(root / "packaged-source.usda"))
    prim = Sdf.CreatePrimInLayer(layer, "/Nested")
    prim.specifier, prim.typeName = Sdf.SpecifierDef, "Xform"
    prim.SetInfo(
        "apiSchemas", Sdf.TokenListOp.Create(prependedItems=["OmniScriptingAPI"])
    )
    layer.defaultPrim = "Nested"
    layer.Save()
    package = root / "nested.usdz"
    assert UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(layer.identifier), str(package))
    if suffix != ".usdz":
        package = root / ("nested" + suffix)
        shutil.copyfile(root / "nested.usdz", package)
    assert Sdf.ZipFile.Open(str(package)).GetFileNames(), (
        "fixture must contain actual package bytes"
    )
    return package


@pytest.mark.parametrize("suffix", [".usdz", ".bin", ".png", ".usd", ".usda"])
def test_real_nested_usdz_and_renamed_packages_rejected_before_kit_open(
    tmp_path, monkeypatch, suffix
):
    root = tmp_path / "bundle"
    request = write_fixture(root)
    package = _scripted_package(root, suffix)
    layer = Sdf.Layer.FindOrOpen(str(root / "scene.usda"))
    prim = Sdf.CreatePrimInLayer(layer, "/World/Nested")
    prim.referenceList.prependedItems = [Sdf.Reference(package.name)]
    layer.Save()
    request["files"] = {
        name: _sha256(root / name) for name in ("scene.usda", package.name)
    }
    validate_request(request)
    if suffix == ".usdz":
        composed = Usd.Stage.Open(layer)
        assert not composed.GetCompositionErrors()
        assert (
            "OmniScriptingAPI"
            in composed.GetPrimAtPath("/World/Nested")
            .GetMetadata("apiSchemas")
            .GetAppliedItems()
        )
    opened = []
    _kit_context(
        monkeypatch, SimpleNamespace(open_stage=lambda *args: opened.append(args))
    )
    with pytest.raises(ValueError, match="packaged USD"):
        runtime._open_scene(root, request)
    assert opened == []


def test_package_relative_reference_is_rejected_before_kit_open(tmp_path, monkeypatch):
    root = tmp_path / "bundle"
    request = write_fixture(root)
    package = _scripted_package(root, ".usdz")
    layer = Sdf.Layer.FindOrOpen(str(root / "scene.usda"))
    prim = Sdf.CreatePrimInLayer(layer, "/World/Nested")
    prim.referenceList.prependedItems = [
        Sdf.Reference("nested.usdz[packaged-source.usda]")
    ]
    layer.Save()
    request["files"] = {
        name: _sha256(root / name) for name in ("scene.usda", package.name)
    }
    opened = []
    _kit_context(
        monkeypatch, SimpleNamespace(open_stage=lambda *args: opened.append(args))
    )
    with pytest.raises(ValueError, match="package-relative"):
        runtime._open_scene(root, request)
    assert opened == []
