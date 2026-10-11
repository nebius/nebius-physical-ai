"""Scripted policies for the MuJoCo manipulation tasks.

Each policy is a callable ``policy(obs) -> action`` over the
:class:`MujocoManipEnv` observation layout ``[qpos, qvel, tip_xyz, goal_xyz]``,
with the tip slice derived from the observation length
(``nq = (len(obs) - 6) // 2``).

* ``expert`` — a waypoint PD controller that steers the tool tip through the
  task's waypoints (and spins the yaw joint while driving the screw).
* ``noisy`` — the expert plus seeded Gaussian action noise.
* ``random`` — seeded uniform actions; a lower-bound baseline.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from npa.workbench.mujoco_manip.envs import MujocoManipError
from npa.workbench.mujoco_manip.scenes import (
    SCREW_DRIVING,
    TASKS,
    TASK_SPECS,
)

Policy = Callable[[np.ndarray], np.ndarray]

EXPERT = "expert"
NOISY = "noisy"
RANDOM = "random"
POLICIES: tuple[str, ...] = (EXPERT, NOISY, RANDOM)

_NOISE_STD = 0.25
_WAYPOINT_TOL = 0.008
_CARTESIAN_GAIN = 30.0


def _tip(obs: np.ndarray) -> np.ndarray:
    nq = (len(obs) - 6) // 2
    return np.asarray(obs[2 * nq : 2 * nq + 3], dtype=np.float64)


def _waypoints(task: str) -> list[tuple[float, float, float]]:
    raw = TASK_SPECS[task]["waypoints"]
    assert isinstance(raw, tuple)
    return [(float(w[0]), float(w[1]), float(w[2])) for w in raw]


class _WaypointPolicy:
    """PD controller that walks the tool tip through task waypoints."""

    def __init__(self, task: str) -> None:
        self._task = task
        self._waypoints = _waypoints(task)
        self._index = 0
        spec = TASK_SPECS[task]
        self._yaw_rate = float(spec["yaw_rate"]) if task == SCREW_DRIVING else 0.0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        tip = _tip(np.asarray(obs))
        target = np.array(self._waypoints[self._index])
        delta = target - tip
        if float(np.linalg.norm(delta)) < _WAYPOINT_TOL:
            self._index = min(self._index + 1, len(self._waypoints) - 1)
            target = np.array(self._waypoints[self._index])
            delta = target - tip
        action = np.zeros(4, dtype=np.float64)
        action[:3] = np.clip(delta * _CARTESIAN_GAIN, -1.0, 1.0)
        action[3] = self._yaw_rate
        return action


class _NoisyPolicy:
    """Expert policy with seeded Gaussian action noise."""

    def __init__(self, task: str, seed: int | None) -> None:
        self._expert = _WaypointPolicy(task)
        self._rng = np.random.default_rng(seed if seed is not None else 0)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        action = self._expert(obs)
        noise = self._rng.normal(0.0, _NOISE_STD, size=4)
        return np.clip(action + noise, -1.0, 1.0)


class _RandomPolicy:
    """Seeded uniform random actions (lower-bound baseline)."""

    def __init__(self, seed: int | None) -> None:
        self._rng = np.random.default_rng(seed if seed is not None else 0)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        _ = obs
        return self._rng.uniform(-1.0, 1.0, size=4)


def get_policy(task: str, name: str, *, seed: int | None = None) -> Policy:
    """Return the scripted policy *name* for *task*.

    ``seed`` seeds the stochastic policies (``noisy``, ``random``); the
    expert is deterministic.  Raises :class:`MujocoManipError` for unknown
    tasks or policy names.
    """
    if task not in TASKS:
        raise MujocoManipError(
            f"unknown mujoco_manip task {task!r}; known tasks: {TASKS}"
        )
    if name == EXPERT:
        return _WaypointPolicy(task)
    if name == NOISY:
        return _NoisyPolicy(task, seed)
    if name == RANDOM:
        return _RandomPolicy(seed)
    raise MujocoManipError(
        f"unknown mujoco_manip policy {name!r}; known policies: {POLICIES}"
    )


__all__ = [
    "EXPERT",
    "NOISY",
    "RANDOM",
    "POLICIES",
    "Policy",
    "get_policy",
]
