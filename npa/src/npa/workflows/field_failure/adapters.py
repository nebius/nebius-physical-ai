"""Verify trusted immutable-image adapter identity before importing its source."""

from __future__ import annotations

import importlib
import sys
from importlib.machinery import PathFinder, SourceFileLoader
from pathlib import Path

from npa.workflows.field_failure.artifacts import _digest


def _configured_identity(expected, entrypoint, runtime_image):
    if not entrypoint:
        raise ValueError(
            "operator adapter required: provide the sealed module:function"
        )
    if entrypoint != expected.entrypoint:
        raise ValueError("configured callable differs from sealed adapter identity")
    if runtime_image != expected.runtime_image:
        raise ValueError("configured runtime image differs from sealed immutable image")
    return expected


def _source_path(module_name):
    search_path = sys.path
    components = module_name.split(".")
    spec = None
    for index in range(len(components)):
        spec = PathFinder.find_spec(".".join(components[: index + 1]), search_path)
        if spec is None:
            raise ImportError(
                "operator adapter source is not installed in the runtime image"
            )
        search_path = spec.submodule_search_locations
        if index < len(components) - 1 and search_path is None:
            raise ImportError("operator adapter parent is not a package")
    if not isinstance(spec.loader, SourceFileLoader) or not spec.origin:
        raise ValueError("adapter must be an installed Python source module")
    return Path(spec.origin)


def _verify_source(identity):
    path = _source_path(identity.entrypoint.split(":")[0])
    if _digest(path.read_bytes()) != identity.source_sha256:
        raise ValueError("installed adapter source SHA-256 differs from sealed bundle")
    return path


def _invoke(identity, request):
    path = _verify_source(identity)
    module_name, function_name = identity.entrypoint.split(":")
    # Parent initializers and dependencies belong to the trusted immutable image.
    # PathFinder above locates the leaf without executing any of those modules.
    module = importlib.import_module(module_name)
    if Path(module.__file__).resolve() != path.resolve():
        raise ValueError("imported adapter differs from verified source location")
    function = getattr(module, function_name, None)
    if not callable(function):
        raise ValueError("configured operator adapter is not callable")
    result = function(request)
    if not isinstance(result, dict):
        raise ValueError("operator adapter must return a report object")
    return result
