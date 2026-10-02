"""Policy-spec parsing for the evaluation harness.

A policy spec is either

* ``scripted:<name>`` — a built-in scripted policy (``expert``, ``noisy``,
  ``random``) for MuJoCo manipulation tasks, or
* ``python:<module>:<callable>`` — a user callable with signature
  ``policy(obs) -> action``.  A class is instantiated fresh per episode; a
  plain function is shared across episodes.

:func:`make_policy_factory` returns a factory that builds a fresh policy per
episode from the episode seed, so stateful policies (like the waypoint
expert) start every episode clean.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Callable

import numpy as np

from npa.workbench.eval_harness.tasks import EvalHarnessError

Policy = Callable[[np.ndarray], np.ndarray]
PolicyFactory = Callable[[int | None], Policy]


def make_policy_factory(spec: str, task: str) -> PolicyFactory:
    """Parse *spec* and return a per-episode policy factory for *task*."""
    if not spec:
        raise EvalHarnessError("policy spec must be non-empty")
    kind, _, rest = spec.partition(":")
    if kind == "scripted":
        name = rest.strip()
        if not name:
            raise EvalHarnessError(
                "scripted policy spec must name a policy, e.g. 'scripted:expert'"
            )
        return _scripted_factory(task, name)
    if kind == "python":
        return _python_factory(spec)
    raise EvalHarnessError(
        f"unknown policy spec {spec!r}; expected 'scripted:<name>' or "
        "'python:<module>:<callable>'"
    )


def _scripted_factory(task: str, name: str) -> PolicyFactory:
    def factory(seed: int | None) -> Policy:
        from npa.workbench import mujoco_manip

        short = task.split(".", 1)[-1]
        return mujoco_manip.get_policy(short, name, seed=seed)

    return factory


def _python_factory(spec: str) -> PolicyFactory:
    parts = spec.split(":")
    if len(parts) != 3 or not parts[1] or not parts[2]:
        raise EvalHarnessError(
            f"malformed python policy spec {spec!r}; expected "
            "'python:<module>:<callable>'"
        )
    _, module_name, attr_name = parts
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise EvalHarnessError(
            f"cannot import policy module {module_name!r}: {exc}"
        ) from exc
    target = getattr(module, attr_name, None)
    if target is None:
        raise EvalHarnessError(
            f"policy module {module_name!r} has no attribute {attr_name!r}"
        )
    if inspect.isclass(target):
        return lambda _seed: target()
    if callable(target):
        return lambda _seed: target
    raise EvalHarnessError(
        f"policy {spec!r} is not a class or callable with signature "
        "policy(obs) -> action"
    )


__all__ = ["Policy", "PolicyFactory", "make_policy_factory"]
