"""Inspect raw USD layers, including inactive variants, before opening a Kit stage."""

from __future__ import annotations

from pathlib import Path
import posixpath
import re
import tempfile
import zipfile

from .contract import _contained, _sha256
from .packages import _extract_package


def _executable_schema(name):
    return str(name).startswith(("OmniGraph", "OmniScripting"))


def _check_prim_spec(spec):
    if _executable_schema(spec.typeName):
        raise ValueError(
            "static scene forbids executable OmniGraph/scripting prim types"
        )
    schemas = spec.GetInfo("apiSchemas") if spec.HasInfo("apiSchemas") else None
    for operation in (
        "explicitItems",
        "addedItems",
        "prependedItems",
        "appendedItems",
        "deletedItems",
        "orderedItems",
    ):
        if any(_executable_schema(name) for name in getattr(schemas, operation, ())):
            raise ValueError(
                "static scene forbids OmniScriptingAPI and OmniGraph API schemas"
            )
    if spec.HasInfo("clips") or spec.HasInfo("clipSets"):
        raise ValueError("static scene forbids animated value clips")


def _check_layer(layer):
    from pxr import Sdf

    if layer.ListAllTimeSamples():
        raise ValueError(
            "animated USD is unsupported, including inactive/unselected variants"
        )

    def inspect(path):
        spec = layer.GetObjectAtPath(path)
        if isinstance(spec, Sdf.PrimSpec):
            _check_prim_spec(spec)
        if isinstance(spec, Sdf.VariantSpec):
            _check_prim_spec(spec.primSpec)
        if isinstance(spec, Sdf.PropertySpec) and spec.name.startswith(
            ("omni:scripting:", "omni:graph:", "node:")
        ):
            raise ValueError(
                "static scene forbids scripting and executable graph properties"
            )

    # Raw Sdf traversal visits specs that composed Usd.Stage.Traverse hides.
    layer.Traverse(Sdf.Path.absoluteRootPath, inspect)


def _asset_file_format(path):
    from pxr import Ar, Sdf

    if Ar.IsPackageRelativePath(str(path)):
        raise ValueError(
            "explicit package-relative paths are unsupported; reference the USDZ root"
        )
    file_format = Sdf.FileFormat.FindByExtension(str(path))
    if zipfile.is_zipfile(path):
        if path.suffix != ".usdz":
            raise ValueError("packaged USD must use the USDZ filename extension")
        return "package"
    if file_format and file_format.IsPackage():
        raise ValueError("USDZ dependency is not a readable package")
    if file_format and file_format.formatId not in {"usd", "usda", "usdc"}:
        raise ValueError("scene dependencies use an unsupported USD file-format plugin")
    return file_format


def _dependency_name(root, parent, asset, names, package):
    from pxr import Ar

    if Ar.IsPackageRelativePath(asset):
        raise ValueError("explicit package-relative USD dependencies are unsupported")
    if not re.fullmatch(r"[A-Za-z0-9_./-]+", asset) or asset.startswith("/"):
        raise ValueError("USD dependency must be a contained relative asset path")
    if any(
        part.startswith(".") and part not in {".", ".."} for part in asset.split("/")
    ):
        raise ValueError("USD dependencies cannot contain hidden components")
    relative_parent = parent.resolve().relative_to(root.resolve()).as_posix()
    name = posixpath.normpath(posixpath.join(relative_parent, asset))
    _contained(root, name)
    if name in names:
        return name
    # USDZ resolves relative to the authored layer first, then the archive root.
    if package:
        name = posixpath.normpath(asset)
        _contained(root, name)
        if name in names:
            return name
    raise ValueError(
        "USD dependency is absent from the hashed input bundle: "
        f"{asset!r} authored below {relative_parent!r}"
    )


def _check_tree(root, names, *, package=False, runtime_modules=()):
    from pxr import Sdf, UsdUtils

    native_names = {name.casefold() for name in runtime_modules}
    if any(Path(name).name.casefold() in native_names for name in names):
        raise ValueError("scene bundle cannot shadow a declared native MDL module")
    for name in names:
        path = _contained(root, name)
        file_format = _asset_file_format(path)
        if file_format == "package":
            _check_package(path, runtime_modules=runtime_modules)
            continue
        if file_format is None:
            continue
        layer = Sdf.Layer.FindOrOpen(str(path))
        if not layer:
            raise ValueError("scene bundle contains an unreadable USD layer")
        _check_layer(layer)
        for group in UsdUtils.ExtractExternalReferences(str(path)):
            for asset in group:
                if asset in runtime_modules:
                    continue
                try:
                    _dependency_name(root, path.parent, asset, names, package)
                except ValueError as exc:
                    raise ValueError(f"USD layer {name!r}: {exc}") from exc


def _check_package(path, *, runtime_modules=()):
    with tempfile.TemporaryDirectory(prefix="npa-rgbd-usdz-audit-") as directory:
        root = Path(directory)
        names = _extract_package(path, root)
        # Inspect every member, even unused layers and nested archives, before Kit.
        _check_tree(root, names, package=True, runtime_modules=runtime_modules)


def _check_scene_files(root, request):
    from .contract import validate_request

    validate_request(request)
    for name, digest in request["files"].items():
        path = _contained(root, name)
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError("scene bundle contains a missing or hash-mismatched asset")
    _check_tree(
        root,
        request["files"],
        runtime_modules=request.get("runtime_materials", {}).get("modules", {}),
    )
