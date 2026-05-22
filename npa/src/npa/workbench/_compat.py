"""Compatibility helpers for deprecated npa.workbench shims."""

from __future__ import annotations

import importlib
import sys
import types
import warnings
from typing import Any

_PRESERVED_NAMES = {
    "__builtins__",
    "__cached__",
    "__file__",
    "__loader__",
    "__name__",
    "__package__",
    "__path__",
    "__spec__",
}


class _WorkbenchShim(types.ModuleType):
    """Module proxy that forwards monkeypatches to the canonical module."""

    _npa_target_module: types.ModuleType

    def __getattr__(self, name: str) -> Any:
        try:
            return getattr(self._npa_target_module, name)
        except AttributeError as exc:
            raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}") from exc

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if name.startswith("_npa_") or name in _PRESERVED_NAMES:
            return
        target = self.__dict__.get("_npa_target_module")
        if target is not None:
            setattr(target, name, value)


def install_shim(module_name: str, target_name: str, namespace: dict[str, Any]) -> None:
    """Populate a deprecated npa.workbench module from its canonical target."""
    warnings.warn(
        f"{module_name} is deprecated, use {target_name}",
        DeprecationWarning,
        stacklevel=3,
    )
    target = importlib.import_module(target_name)
    module = sys.modules[module_name]
    for name, value in vars(target).items():
        if name in _PRESERVED_NAMES or name in {"__all__", "__doc__"}:
            continue
        namespace[name] = value
    exported = getattr(target, "__all__", None)
    if exported is None:
        exported = [name for name in vars(target) if not name.startswith("_")]
    namespace["__all__"] = list(exported)
    namespace["__doc__"] = target.__doc__
    module.__class__ = _WorkbenchShim
    module._npa_target_module = target
