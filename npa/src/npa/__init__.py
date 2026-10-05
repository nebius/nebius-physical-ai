"""npa - Nebius Physical AI CLI/SDK.

This package exposes a Python SDK surface that mirrors supported npa CLI
namespaces. The SDK is currently v0: pin the npa version for integrations until
the public API reaches v1 stability.

Everything in :data:`__all__` is supported; anything else is internal and may
move without notice. ``npa.sdk`` is the namespaced entrypoint
(``npa.sdk.workbench.<tool>``, plus ``fleet`` / ``provisioning`` / ``soperator``);
the remaining names are shortcuts for surfaces that have no per-tool client.
See `docs/sdk/README.md` for the surface map.
"""

from __future__ import annotations

import importlib as _importlib
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _package_version
from typing import TYPE_CHECKING

try:
    __version__ = _package_version("npa")
except _PackageNotFoundError:  # pragma: no cover - source tree without install
    __version__ = "0.0.0.dev0"

# SDK convenience submodules. These are imported lazily (PEP 562) so that a bare
# ``import npa`` — which every ``npa`` CLI invocation triggers via its console
# script — does not eagerly pull in the whole SDK surface (pyarrow / lancedb /
# fiftyone / rerun / boto3 …). Attribute access such as ``npa.convert`` and
# ``from npa import convert`` still work and import on demand; this just defers
# the cost until something is actually used. It also lets minimal interpreters
# (e.g. the Isaac Lab held-out eval image) import ``npa`` without the full
# dependency set. The historical ``NPA_SKIP_EAGER_IMPORTS`` flag is now a no-op:
# imports are always lazy.
_LAZY_SUBMODULES = (
    "convert",
    "demo",
    "errors",
    "network",
    "rerun",
    "sdk",
    "workflow",
    "workbench",
)

__all__ = ["__version__", *sorted(_LAZY_SUBMODULES)]

if TYPE_CHECKING:  # pragma: no cover - type-checker visibility only
    from npa import (  # noqa: F401
        convert,
        demo,
        errors,
        network,
        rerun,
        sdk,
        workbench,
        workflow,
    )


def __getattr__(name: str):
    """Import an SDK submodule on first access (PEP 562)."""
    if name in _LAZY_SUBMODULES:
        module = _importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(__all__)
