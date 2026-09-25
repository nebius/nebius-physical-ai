"""Inspect raw USD layers, including inactive variants, before opening a Kit stage."""

from __future__ import annotations

from .contract import _contained, _sha256


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
            "packaged USD dependencies and package-relative paths are unsupported"
        )
    file_format = Sdf.FileFormat.FindByExtension(str(path))
    if file_format and file_format.IsPackage():
        raise ValueError("packaged USD dependencies, including USDZ, are unsupported")
    # CanRead for USDZ is extension-sensitive. USD's ZIP reader also detects
    # valid packages renamed to texture or generic asset suffixes, without extraction.
    if Sdf.ZipFile.Open(str(path)).GetFileNames():
        raise ValueError(
            "packaged USD dependencies are unsupported regardless of filename"
        )
    if file_format and file_format.formatId not in {"usd", "usda", "usdc"}:
        raise ValueError("scene dependencies use an unsupported USD file-format plugin")
    return file_format


def _check_scene_files(root, request):
    from pxr import Ar, Sdf, UsdUtils

    for name, digest in request["files"].items():
        path = _contained(root, name)
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError("scene bundle contains a missing or hash-mismatched asset")
        if _asset_file_format(path) is None:
            continue
        layer = Sdf.Layer.FindOrOpen(str(path))
        if not layer:
            raise ValueError("scene bundle contains an unreadable USD layer")
        _check_layer(layer)
        for group in UsdUtils.ExtractExternalReferences(str(path)):
            for asset in group:
                if Ar.IsPackageRelativePath(asset):
                    raise ValueError(
                        "package-relative USD dependencies are unsupported"
                    )
                target = _contained(path.parent, asset)
                relative = target.relative_to(root.resolve()).as_posix()
                if relative not in request["files"]:
                    raise ValueError(
                        "USD dependency is absent from the hashed input bundle"
                    )
