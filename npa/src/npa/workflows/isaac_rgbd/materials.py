"""Bind canonical Kit MDL modules to measured installed library bytes before use."""

from __future__ import annotations

import hashlib
from importlib.metadata import version
import os
from pathlib import Path
import re

from .contract import _json_bytes, _runtime_materials, _sha256


def _native_library():
    import carb.tokens
    from pxr import Ar

    # NVIDIA's documented token locates the installed Kit library, not scene files.
    root = Path(carb.tokens.get_tokens_interface().resolve("${kit}/mdl")).resolve()
    if not root.is_dir():
        raise ValueError("installed Kit MDL library is unavailable")
    return root, Ar.GetResolver()


def _observe_materials(names):
    root, resolver = _native_library()
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("installed Kit MDL library cannot contain symlinks")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = _sha256(path)
    modules = {}
    for name in sorted(names):
        resolved = str(resolver.Resolve(name))
        path = Path(resolved)
        if not resolved or not path.is_absolute() or not path.is_file():
            raise ValueError(f"native MDL module cannot be resolved: {name!r}")
        if not path.resolve().is_relative_to(root / "core"):
            raise ValueError(
                "native MDL resolved outside the installed Kit core library"
            )
        relative = path.resolve().relative_to(root).as_posix()
        modules[name] = {"path": relative, "sha256": files[relative]}
    observed = {
        "isaac_sim_version": version("isaacsim"),
        "image": os.environ.get("NPA_TASK_IMAGE", ""),
        "library_sha256": hashlib.sha256(_json_bytes(files)).hexdigest(),
        "modules": modules,
    }
    _runtime_materials(observed)
    return observed


def _verify_materials(request):
    declared = request.get("runtime_materials")
    if declared is None:
        return None
    _runtime_materials(declared)
    observed = _observe_materials(declared["modules"])
    if observed != declared:
        raise ValueError(
            "installed native MDL library, image or version differs from input"
        )
    return observed


def _reference_materials(root):
    from pxr import UsdUtils

    names = set()
    for path in root.rglob("*"):
        if path.suffix not in {".usd", ".usda", ".usdc"}:
            continue
        for group in UsdUtils.ExtractExternalReferences(str(path)):
            for asset in group:
                if (
                    re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*\.mdl", asset)
                    and not (path.parent / asset).is_file()
                ):
                    names.add(asset)
    return _observe_materials(names) if names else None
