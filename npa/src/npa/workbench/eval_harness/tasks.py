"""Task registry for the evaluation harness.

Built-in tasks map to environment factories; factories stay lazy so importing
this module never pulls in heavy simulation backends.  Third parties can
register their own tasks with :func:`register_task`.
"""

from __future__ import annotations

from typing import Callable

from npa.workbench.eval_harness.env import ManipEnv

EnvFactory = Callable[[], ManipEnv]

_TASK_REGISTRY: dict[str, EnvFactory] = {}


class EvalHarnessError(RuntimeError):
    """Raised when an evaluation-harness invariant is not met."""


def register_task(name: str, factory: EnvFactory) -> None:
    """Register an environment factory under *name* (overwrites existing)."""
    if not name:
        raise EvalHarnessError("task name must be non-empty")
    _TASK_REGISTRY[name] = factory


def get_task(name: str) -> EnvFactory:
    """Return the factory for *name*; raises listing known tasks."""
    try:
        return _TASK_REGISTRY[name]
    except KeyError:
        raise EvalHarnessError(
            f"unknown eval_harness task {name!r}; known tasks: {sorted(_TASK_REGISTRY)}"
        ) from None


def known_tasks() -> tuple[str, ...]:
    """Return the registered task names, sorted."""
    return tuple(sorted(_TASK_REGISTRY))


def _mujoco_manip_task(task: str) -> ManipEnv:
    from npa.workbench import mujoco_manip

    return mujoco_manip.make_env(task)


for _task in ("peg_insertion", "screw_driving", "reach"):
    register_task(
        f"mujoco_manip.{_task}",
        (lambda _t=_task: lambda: _mujoco_manip_task(_t))(),
    )
del _task


__all__ = [
    "EnvFactory",
    "EvalHarnessError",
    "get_task",
    "known_tasks",
    "register_task",
]
