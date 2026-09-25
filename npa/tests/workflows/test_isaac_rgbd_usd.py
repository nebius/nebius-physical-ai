"""Parse actual OpenUSD files to reject scripts, variants, and partial composition."""

from __future__ import annotations

import sys
import shutil
import stat
import zipfile
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
    with pytest.raises(ValueError, match="packaged USD|forbids"):
        runtime._open_scene(root, request)
    assert opened == []


def _package(path, members):
    with Sdf.ZipFileWriter.CreateNew(str(path)) as archive:
        for name, source in members:
            archive.AddFile(str(source), name)


def _package_request(root, request, scene):
    request["scene"] = scene
    request["files"] = {scene: _sha256(root / scene)}
    return validate_request(request)


def test_real_portable_usdz_opens_with_a_writable_session_layer(tmp_path, monkeypatch):
    root = tmp_path / "bundle"
    request = write_fixture(root)
    package = root / "scene.usdz"
    assert UsdUtils.CreateNewUsdzPackage(
        Sdf.AssetPath(str(root / "scene.usda")), str(package)
    )
    _package_request(root, request, package.name)
    original_hash = _sha256(package)
    (root / "scene.usda").unlink()
    stage = Usd.Stage.Open(str(package))
    _kit_context(
        monkeypatch,
        SimpleNamespace(open_stage=lambda path: True, get_stage=lambda: stage),
    )
    opened = runtime._open_scene(root, request)
    UsdGeom.Camera.Define(opened, "/NpaRgbdCapture/front")
    assert opened.GetSessionLayer().GetPrimAtPath("/NpaRgbdCapture/front")
    assert not opened.GetRootLayer().GetPrimAtPath("/NpaRgbdCapture/front")
    assert _sha256(package) == original_hash


def test_nested_usdz_with_archive_root_texture_fallback_is_portable(tmp_path):
    root = tmp_path / "bundle"
    request = write_fixture(root)
    inner_layer = root / "visual.usda"
    inner_layer.write_text("""#usda 1.0
(defaultPrim = "Visual")
def Xform "Visual" {
    def Shader "Texture" {
        asset inputs:file = @texture.png@
    }
}
""")
    Image.new("RGB", (2, 2), "white").save(root / "texture.png")
    nested = root / "nested.usdz"
    _package(
        nested,
        [("layers/visual.usda", inner_layer), ("texture.png", root / "texture.png")],
    )
    layer = Sdf.Layer.FindOrOpen(str(root / "scene.usda"))
    child = Sdf.CreatePrimInLayer(layer, "/World/Visual")
    child.referenceList.prependedItems = [Sdf.Reference("nested.usdz")]
    layer.Save()
    outer = root / "scene.usdz"
    _package(outer, [("scene.usda", root / "scene.usda"), ("nested.usdz", nested)])
    _package_request(root, request, outer.name)
    _check_scene_files(root, request)
    stage = Usd.Stage.Open(str(outer))
    assert not stage.GetCompositionErrors()
    texture = (
        stage.GetPrimAtPath("/World/Visual/Texture").GetAttribute("inputs:file").Get()
    )
    assert texture.resolvedPath


@pytest.mark.parametrize(
    "kind", ["api", "attribute", "graph_api", "node_type", "animation", "clips"]
)
@pytest.mark.parametrize("location", ["inactive", "variant_child", "variant_root"])
def test_unused_unsafe_layers_in_nested_usdz_never_reach_kit(
    tmp_path, monkeypatch, kind, location
):
    unsafe_root, _ = _modified_fixture(tmp_path, location, kind)
    clean_root = tmp_path / "clean"
    request = write_fixture(clean_root)
    inner = clean_root / "inner.usdz"
    _package(
        inner,
        [
            ("scene.usda", clean_root / "scene.usda"),
            ("unused.usda", unsafe_root / "scene.usda"),
        ],
    )
    outer = clean_root / "outer.usdz"
    _package(outer, [("scene.usda", clean_root / "scene.usda"), ("inner.usdz", inner)])
    _package_request(clean_root, request, outer.name)
    opened = []
    _kit_context(
        monkeypatch, SimpleNamespace(open_stage=lambda *args: opened.append(args))
    )
    with pytest.raises(ValueError, match="forbids|animated"):
        runtime._open_scene(clean_root, request)
    assert opened == []


@pytest.mark.parametrize(
    "kind",
    [
        "traversal",
        "absolute",
        "duplicate",
        "case",
        "prefix",
        "symlink",
        "compressed",
        "root",
        "external",
    ],
)
def test_usdz_members_and_dependencies_remain_contained(tmp_path, monkeypatch, kind):
    root = tmp_path / "bundle"
    request = write_fixture(root)
    content = (root / "scene.usda").read_bytes()
    if kind == "external":
        content += b'\ndef Shader "Texture" {\n asset inputs:file = @https://host/texture.png@\n }\n'
    path = root / "unsafe.usdz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("root.png" if kind == "root" else "scene.usda", content)
        names = {
            "traversal": ["../escape"],
            "absolute": ["/escape"],
            "duplicate": ["x", "x"],
            "case": ["x", "X"],
            "prefix": ["x", "x/a"],
        }.get(kind, [])
        for name in names:
            archive.writestr(name, b"unsafe")
        if kind == "symlink":
            member = zipfile.ZipInfo("link")
            member.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(member, b"/outside")
        if kind == "compressed":
            archive.writestr(
                "compressed", b"contents", compress_type=zipfile.ZIP_DEFLATED
            )
    _package_request(root, request, path.name)
    opened = []
    _kit_context(
        monkeypatch, SimpleNamespace(open_stage=lambda *args: opened.append(args))
    )
    with pytest.raises(ValueError):
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


def test_contained_parent_relative_usd_dependencies_are_supported(tmp_path):
    root = tmp_path / "bundle"
    request = write_fixture(root)
    (root / "layers").mkdir()
    child = Sdf.Layer.CreateNew(str(root / "layers/child.usda"))
    prim = Sdf.CreatePrimInLayer(child, "/Texture")
    prim.typeName = "Shader"
    Sdf.AttributeSpec(
        prim, "inputs:file", Sdf.ValueTypeNames.Asset
    ).default = Sdf.AssetPath("../texture.png")
    child.Save()
    Image.new("RGB", (2, 2), "white").save(root / "texture.png")
    request["files"].update(
        {name: _sha256(root / name) for name in ["layers/child.usda", "texture.png"]}
    )
    _check_scene_files(root, request)
    prim.attributes["inputs:file"].default = Sdf.AssetPath("../../escape.png")
    child.Save()
    request["files"]["layers/child.usda"] = _sha256(root / "layers/child.usda")
    with pytest.raises(ValueError, match="contained"):
        _check_scene_files(root, request)
