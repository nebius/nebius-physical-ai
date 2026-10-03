"""Read one exact Arena success value across HDF5 scalar representations."""

from __future__ import annotations

from typing import Any

from .errors import IsaacArenaError


def success_flag(value: Any, *, source: str) -> bool:
    """Accept a boolean or binary integer, including a singleton HDF5 array."""
    import numpy as np

    scalar = np.asarray(value)
    if scalar.size != 1 or not (
        np.issubdtype(scalar.dtype, np.bool_) or np.issubdtype(scalar.dtype, np.integer)
    ):
        raise IsaacArenaError(f"{source} must contain one boolean or 0/1")
    success = scalar.reshape(()).item()
    if success not in (0, 1):
        raise IsaacArenaError(f"{source} must contain one boolean or 0/1")
    return bool(success)
