"""Backend registry for npa viz renderers."""

from __future__ import annotations

from types import ModuleType


class BackendUnavailable(Exception):
    """Raised when a requested visualization backend is not available."""


def get_backend(name: str) -> ModuleType:
    if name == "matplotlib":
        from npa.viz.backends import matplotlib

        return matplotlib
    if name == "rerun":
        try:
            from npa.viz.backends import rerun  # type: ignore[attr-defined]
        except ImportError as exc:
            raise BackendUnavailable(
                "Rerun backend is not implemented yet. Use renderer 'matplotlib'."
            ) from exc
        if not hasattr(rerun, "render"):
            raise BackendUnavailable(
                "Rerun backend is not implemented yet. Use renderer 'matplotlib'."
            )
        return rerun
    raise BackendUnavailable(f"Unsupported backend '{name}'. Expected matplotlib or rerun.")
