"""npa - Nebius Physical AI CLI/SDK.

This package exposes a Python SDK surface that mirrors supported npa CLI
namespaces. The SDK is currently v0: pin the npa version for integrations until
the public API reaches v1 stability.
"""

from __future__ import annotations

import os as _os
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _package_version

try:
    __version__ = _package_version("npa")
except _PackageNotFoundError:  # pragma: no cover - source tree without install
    __version__ = "0.0.0.dev0"

__all__ = [
    "__version__",
    "convert",
    "demo",
    "errors",
    "network",
    "rerun",
    "workflow",
    "workbench",
]

# Opt-in light import for minimal interpreters (e.g. the Isaac Lab held-out
# eval image, which lacks the full npa dependency set such as pyarrow / lancedb
# / fiftyone). When NPA_SKIP_EAGER_IMPORTS is set, the SDK convenience
# submodules are not eagerly imported; ``import npa.<submodule>`` still works on
# demand. Default behavior (flag unset) eagerly imports the SDK surface.
if _os.environ.get("NPA_SKIP_EAGER_IMPORTS", "").strip().lower() not in (
    "1",
    "true",
    "yes",
):
    from npa import convert, demo, errors, network, rerun, workflow, workbench
