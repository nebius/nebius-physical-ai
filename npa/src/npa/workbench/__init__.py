"""npa.workbench - deployable Physical AI workbench tools."""

from __future__ import annotations

__all__ = [
    "cosmos",
    "data",
    "detection_training",
    "fiftyone",
    "genesis",
    "groot",
    "isaac_lab",
    "lancedb",
    "lerobot",
    "mjlab",
    "retargeting",
    "sonic",
    "training_config",
    "trigger",
    "vlm_eval",
]

_LAZY_SUBMODULES = frozenset(__all__)


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        import importlib

        module = importlib.import_module(f"npa.workbench.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
