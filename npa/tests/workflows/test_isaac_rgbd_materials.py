"""Exercise native material identity and strict scene containment with real Sdf."""

from types import SimpleNamespace

import pytest

from npa.workflows.isaac_rgbd import materials
from npa.workflows.isaac_rgbd.contract import (
    _runtime_materials,
    _sha256,
    validate_request,
)
from npa.workflows.isaac_rgbd.fixture import write_fixture
from npa.workflows.isaac_rgbd.scene import _check_scene_files


@pytest.fixture
def native_library(tmp_path, monkeypatch):
    root = tmp_path / "kit-mdl"
    module = root / "core/Base/OmniPBR.mdl"
    module.parent.mkdir(parents=True)
    module.write_text("// synthetic native-module identity fixture\n")
    helper = root / "core/Base/helper.mdl"
    helper.write_text("// synthetic imported library file\n")
    resolver = SimpleNamespace(
        Resolve=lambda name: str(module) if name == module.name else ""
    )
    monkeypatch.setattr(materials, "_native_library", lambda: (root, resolver))
    monkeypatch.setattr(materials, "version", lambda _name: "6.0.1.0")
    monkeypatch.setenv("NPA_TASK_IMAGE", "registry.example/isaac@sha256:" + "a" * 64)
    return root, module, helper, resolver


def test_native_identity_measures_library_module_runtime_and_image(native_library):
    _, module, _, _ = native_library
    declared = materials._observe_materials(["OmniPBR.mdl"])
    assert declared["modules"] == {
        "OmniPBR.mdl": {
            "path": "core/Base/OmniPBR.mdl",
            "sha256": _sha256(module),
        }
    }
    assert materials._verify_materials({"runtime_materials": declared}) == declared


@pytest.mark.parametrize(
    "mutation", ["module", "helper", "image", "version", "shadow", "missing", "symlink"]
)
def test_changed_native_dependency_is_rejected(
    native_library, monkeypatch, tmp_path, mutation
):
    root, module, helper, resolver = native_library
    declared = materials._observe_materials(["OmniPBR.mdl"])
    if mutation in {"module", "helper"}:
        (module if mutation == "module" else helper).write_text("changed")
    elif mutation == "image":
        monkeypatch.setenv(
            "NPA_TASK_IMAGE", "registry.example/isaac@sha256:" + "b" * 64
        )
    elif mutation == "version":
        monkeypatch.setattr(materials, "version", lambda _name: "6.1.0")
    elif mutation == "shadow":
        impostor = tmp_path / module.name
        impostor.write_bytes(module.read_bytes())
        resolver.Resolve = lambda _name: str(impostor)
    elif mutation == "missing":
        resolver.Resolve = lambda _name: ""
    else:
        (root / "alias.mdl").symlink_to(module)
    with pytest.raises(ValueError):
        materials._verify_materials({"runtime_materials": declared})


@pytest.mark.parametrize(
    "field,value",
    [
        ("image", "registry.example/isaac:latest"),
        ("library_sha256", "missing"),
        ("isaac_sim_version", "6.1.0"),
        ("modules", {}),
        (
            "modules",
            {"../OmniPBR.mdl": {"path": "core/Base/OmniPBR.mdl", "sha256": "a" * 64}},
        ),
        (
            "modules",
            {"OmniPBR.mdl": {"path": "custom/OmniPBR.mdl", "sha256": "a" * 64}},
        ),
    ],
)
def test_native_contract_rejects_unbound_or_arbitrary_assets(
    native_library, field, value
):
    declared = materials._observe_materials(["OmniPBR.mdl"])
    declared[field] = value
    with pytest.raises(ValueError):
        _runtime_materials(declared)


def _scene(tmp_path):
    root = tmp_path / "scene"
    request = write_fixture(root)
    layer = root / "material.usda"
    layer.write_text(
        '#usda 1.0\ndef Shader "Material" {\n asset info:mdl:sourceAsset = @OmniPBR.mdl@\n}\n'
    )
    request["files"][layer.name] = _sha256(layer)
    return root, request


def test_native_material_declaration_is_explicit_and_does_not_allow_other_missing_assets(
    tmp_path, native_library
):
    root, request = _scene(tmp_path)
    with pytest.raises(ValueError, match="absent from the hashed"):
        _check_scene_files(root, request)
    declared = materials._reference_materials(root)
    request["runtime_materials"] = declared
    validate_request(request)
    _check_scene_files(root, request)
    layer = root / "material.usda"
    layer.write_text(layer.read_text().replace("OmniPBR.mdl", "missing.png"))
    request["files"][layer.name] = _sha256(layer)
    with pytest.raises(ValueError, match="absent from the hashed"):
        _check_scene_files(root, request)


@pytest.mark.parametrize("name", ["OmniPBR.mdl", "omnipbr.mdl"])
def test_scene_cannot_shadow_declared_native_module(tmp_path, native_library, name):
    root, request = _scene(tmp_path)
    request["runtime_materials"] = materials._reference_materials(root)
    shadow = root / "nested" / name
    shadow.parent.mkdir()
    shadow.write_text("// synthetic shadow")
    request["files"][shadow.relative_to(root).as_posix()] = _sha256(shadow)
    with pytest.raises(ValueError, match="cannot shadow"):
        _check_scene_files(root, request)


def test_reference_does_not_treat_unknown_mdl_as_native(tmp_path, native_library):
    root, request = _scene(tmp_path)
    layer = root / "material.usda"
    layer.write_text(layer.read_text().replace("OmniPBR.mdl", "Unknown.mdl"))
    with pytest.raises(ValueError, match="cannot be resolved"):
        materials._reference_materials(root)


def test_native_verification_precedes_kit_stage_open(
    tmp_path, native_library, monkeypatch
):
    import sys
    from npa.workflows.isaac_rgbd import runtime

    root, request = _scene(tmp_path)
    request["runtime_materials"] = materials._reference_materials(root)
    native_library[2].write_text("changed imported library")
    opened = []
    context = SimpleNamespace(open_stage=lambda path: opened.append(path))
    usd = SimpleNamespace(get_context=lambda: context)
    monkeypatch.setitem(sys.modules, "omni", SimpleNamespace(usd=usd))
    monkeypatch.setitem(sys.modules, "omni.usd", usd)
    with pytest.raises(ValueError, match="differs from input"):
        runtime._open_scene(root, request)
    assert not opened


def test_nested_package_cannot_shadow_declared_native_module(tmp_path, native_library):
    import zipfile

    root, request = _scene(tmp_path)
    request["runtime_materials"] = materials._reference_materials(root)
    inner = root / "inner.usdz"
    with zipfile.ZipFile(inner, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(root / "material.usda", "material.usda")
        archive.writestr("unused/OmniPBR.mdl", "// synthetic shadow")
    outer = root / "outer.usdz"
    with zipfile.ZipFile(outer, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(root / "scene.usda", "scene.usda")
        archive.write(inner, "inner.usdz")
    request["scene"] = outer.name
    request["files"] = {outer.name: _sha256(outer)}
    with pytest.raises(ValueError, match="cannot shadow"):
        _check_scene_files(root, request)
