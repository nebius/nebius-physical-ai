"""Environment protocol shared by the evaluation harness and task backends."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ManipEnv(Protocol):
    """Minimal interface a task environment must implement.

    ``step`` returns ``(obs, reward, done, info)`` with ``info["success"]`` a
    bool recording whether the episode reached the task goal.  Backends (for
    example :mod:`npa.workbench.mujoco_manip`) implement this structurally;
    they do not need to import this module.
    """

    @property
    def action_space(self) -> Mapping[str, Any]: ...

    @property
    def observation_space(self) -> Mapping[str, Any]: ...

    def reset(self, seed: int | None = None) -> np.ndarray: ...

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, dict[str, Any]]: ...


__all__ = ["ManipEnv"]
