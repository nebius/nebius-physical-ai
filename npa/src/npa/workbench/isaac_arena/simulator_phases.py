"""Record scalar execution phases without changing simulator or policy decisions."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
import inspect
import json
import os
from pathlib import Path
import time
from typing import Any, Iterator


_PHASES = {
    "policy_action",
    "env_step",
    "pink_ik",
    "capture",
    "render_call",
    "action_apply",
    "scene_write",
    "scene_update",
    "simulation_step",
    "simulation_render",
    "physics_wait",
    "physics_step",
}
_METHOD_PHASES = {
    "pink_ik",
    "action_apply",
    "scene_write",
    "scene_update",
    "simulation_step",
    "simulation_render",
    "physics_wait",
    "physics_step",
}
_READINESS_FIELDS = {"stage_ready", "annotator_ready", "nonblack_rgb"}
_CLASS_OWNERS: dict[type, Any] = {}


@dataclass(frozen=True)
class _ObservedMethod:
    target: Any
    name: str
    previous: Any
    installed: Any
    had_local: bool

    def restore(self) -> None:
        # Another owner may have deliberately replaced a method after setup.
        if vars(self.target).get(self.name) is not self.installed:
            return
        if self.had_local:
            setattr(self.target, self.name, self.previous)
        else:
            delattr(self.target, self.name)


class _PhaseJournal:
    def __init__(self, directory: Path, rank: int) -> None:
        self.path = directory / f"simulator-phases-rank{rank}.jsonl"
        self.rank = rank
        self.sequence = 0
        self.action_step = 0
        self.write_failed = False
        self.observers: list[_ObservedMethod] = []
        self.class_targets: set[type] = set()

    def emit(self, phase: str, event: str, action_step: int, **fields: Any) -> None:
        _validate_event(phase, event, action_step, fields)
        self.sequence += 1
        row = {
            "schema": "npa.isaac-arena.simulator-phase.v1",
            "sequence": self.sequence,
            "monotonic_ns": time.monotonic_ns(),
            "rank": self.rank,
            "action_step": action_step,
            "phase": phase,
            "event": event,
            **fields,
        }
        if self.write_failed:
            return
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW
            descriptor = os.open(self.path, flags, 0o600)
            with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
        except OSError:
            # Diagnostics must not replace a policy result or its exception.
            self.write_failed = True


def _journal(env: Any) -> _PhaseJournal | None:
    base = getattr(env, "unwrapped", env)
    return getattr(base, "_npa_phase_journal", None)


def _nonnegative_integer(value: Any) -> bool:
    return type(value) is int and value >= 0


def _render_ordinal(value: Any) -> bool:
    return type(value) is int and value > 0


def _validate_readiness(render_call: Any, readiness: dict) -> None:
    if (
        not _render_ordinal(render_call)
        or set(readiness) != _READINESS_FIELDS
        or any(
            value is not None and type(value) is not bool
            for value in readiness.values()
        )
    ):
        raise ValueError("invalid simulator readiness fields")


def _validate_event(phase: Any, event: Any, action_step: Any, fields: dict) -> None:
    if (
        type(phase) is not str
        or type(event) is not str
        or not _nonnegative_integer(action_step)
    ):
        raise ValueError("invalid simulator phase event")
    if phase == "capture_readiness":
        if event != "observed" or "render_call" not in fields:
            raise ValueError("invalid simulator readiness event")
        readiness = {
            key: value for key, value in fields.items() if key != "render_call"
        }
        _validate_readiness(fields["render_call"], readiness)
        return
    if event == "unavailable":
        if phase not in _METHOD_PHASES or fields:
            raise ValueError("invalid simulator observer availability event")
        return
    if phase not in _PHASES or event not in {"begin", "end", "failed"}:
        raise ValueError("invalid simulator phase event")
    if set(fields) - {"render_call"} or (
        "render_call" in fields and not _render_ordinal(fields["render_call"])
    ):
        raise ValueError("invalid simulator phase fields")


@contextmanager
def phase_scope(
    env: Any, phase: str, action_step: int, *, render_call: int | None = None
) -> Iterator[None]:
    """Mark entry and return/failure around one unchanged native operation.

    Args:
        env: Base or wrapped simulator environment.
        phase: One fixed phase name; arbitrary text is rejected.
        action_step: Zero for initial state, otherwise the one-based source step.
        render_call: Optional one-based native render ordinal within a capture.
    Returns:
        A context manager that never records arguments or exception contents.
    Raises:
        ValueError: A diagnostic label or counter violates the scalar contract.
        BaseException: The original operation's exception, unchanged.
    """
    if (
        type(phase) is not str
        or phase not in _PHASES
        or not _nonnegative_integer(action_step)
    ):
        raise ValueError("invalid simulator phase or action counter")
    if render_call is not None and not _render_ordinal(render_call):
        raise ValueError("invalid simulator render counter")
    journal = _journal(env)
    fields = {} if render_call is None else {"render_call": render_call}
    if journal is not None:
        journal.action_step = action_step
        journal.emit(phase, "begin", action_step, **fields)
    try:
        yield
    except BaseException:
        if journal is not None:
            journal.emit(phase, "failed", action_step, **fields)
        raise
    else:
        if journal is not None:
            journal.emit(phase, "end", action_step, **fields)


def record_readiness(env: Any, render_call: int, **readiness: bool | None) -> None:
    """Retain observed readiness flags without image, state, or asset contents.

    Args:
        env: Simulator environment with an optional phase journal.
        render_call: Native render ordinal within the current capture.
        readiness: Fixed readiness fields; unobserved flags are None.
    Returns:
        None.
    Raises:
        ValueError: A field name or value is outside the scalar contract.
    """
    _validate_readiness(render_call, readiness)
    journal = _journal(env)
    if journal is not None:
        journal.emit(
            "capture_readiness",
            "observed",
            journal.action_step,
            render_call=render_call,
            **readiness,
        )


def _observe_controller(controller: Any, env: Any) -> None:
    _observe_method(controller, "compute", "pink_ik", env)


def _wrapped_method(original: Any, phase: str, env: Any) -> Any:
    owner = _journal(env)

    @wraps(original)
    def observed(*args: Any, **kwargs: Any) -> Any:
        if _journal(env) is not owner:
            return original(*args, **kwargs)
        with phase_scope(env, phase, owner.action_step):
            return original(*args, **kwargs)

    return observed


def _wrapped_classmethod(descriptor: classmethod, phase: str, env: Any) -> classmethod:
    owner = _journal(env)

    @wraps(descriptor.__func__)
    def observed(cls: type, *args: Any, **kwargs: Any) -> Any:
        if _journal(env) is not owner:
            return descriptor.__get__(None, cls)(*args, **kwargs)
        with phase_scope(env, phase, owner.action_step):
            return descriptor.__get__(None, cls)(*args, **kwargs)

    return classmethod(observed)


def _claim_class(target: type, env: Any) -> None:
    journal = _journal(env)
    for observed, owner in _CLASS_OWNERS.items():
        if owner is not journal and (
            issubclass(target, observed) or issubclass(observed, target)
        ):
            raise RuntimeError("simulator phase class already has an active owner")
    _CLASS_OWNERS[target] = journal
    journal.class_targets.add(target)


def _observe_method(target: Any, name: str, phase: str, env: Any) -> None:
    original = getattr(target, name, None)
    if not callable(original):
        return
    local = getattr(target, "__dict__", None)
    if local is None:
        _journal(env).emit(phase, "unavailable", 0)
        return
    had_local, previous = name in local, local.get(name)
    if isinstance(target, type):
        descriptor = inspect.getattr_static(target, name)
        if not isinstance(descriptor, classmethod):
            raise RuntimeError("simulator phase observer requires a native classmethod")
        _claim_class(target, env)
        installed = _wrapped_classmethod(descriptor, phase, env)
    else:
        installed = _wrapped_method(original, phase, env)
    if not _install_observer(target, name, installed, phase, env):
        return
    _journal(env).observers.append(
        _ObservedMethod(target, name, previous, installed, had_local)
    )


def _install_observer(
    target: Any, name: str, installed: Any, phase: str, env: Any
) -> bool:
    try:
        setattr(target, name, installed)
    except AttributeError:
        if isinstance(target, type):
            raise
        # Some native bindings expose attributes but prohibit instance overrides.
        # Their enclosing simulator call remains observable without changing them.
        _journal(env).emit(phase, "unavailable", 0)
        return False
    return True


def _observe_environment(env: Any) -> None:
    sim = getattr(env, "sim", None)
    targets = (
        (getattr(env, "action_manager", None), "apply_action", "action_apply"),
        (getattr(env, "scene", None), "write_data_to_sim", "scene_write"),
        (getattr(env, "scene", None), "update", "scene_update"),
        (sim, "step", "simulation_step"),
        (sim, "render", "simulation_render"),
        (getattr(sim, "physics_manager", None), "wait_for_playing", "physics_wait"),
        (getattr(sim, "physics_manager", None), "step", "physics_step"),
    )
    for target, name, phase in targets:
        _observe_method(target, name, phase, env)


def finalize_phase_journal(env: Any) -> None:
    """Restore only native method bindings still owned by this environment.

    Args:
        env: Base or wrapped simulator environment whose rollout has finished.
    Returns:
        None; journals remain on disk. An unexpectedly unrestorable binding stays
        owned for a later retry instead of masking the rollout's native exception.
    """
    journal = _journal(env)
    if journal is None:
        return
    pending = []
    for observer in reversed(journal.observers):
        try:
            observer.restore()
        except Exception:
            pending.append(observer)
    journal.observers = list(reversed(pending))
    if pending:
        return
    for target in journal.class_targets:
        if _CLASS_OWNERS.get(target) is journal:
            del _CLASS_OWNERS[target]
    journal.class_targets.clear()
    base = getattr(env, "unwrapped", env)
    if getattr(base, "_npa_phase_journal", None) is journal:
        del base._npa_phase_journal


def _observe_pink_controllers(env: Any) -> None:
    manager = getattr(env, "action_manager", None)
    for name in getattr(manager, "active_terms", ()):
        term = manager.get_term(name)
        if type(term).__module__ != "isaaclab.envs.mdp.actions.pink_task_space_actions":
            continue
        for controller in term._ik_controllers:
            if type(controller).__module__ == "isaaclab.controllers.pink_ik.pink_ik":
                _observe_controller(controller, env)


def configure_phase_journal(env: Any, directory: str | Path, rank: int = 0) -> None:
    """Attach run-local scalar diagnostics to the pinned Arena environment.

    Args:
        env: Base simulator environment after construction, before rollout.
        directory: Existing upstream run output directory.
        rank: Upstream process rank, used to keep journals independent.
    Returns:
        None; observation is idempotent for an already configured environment.
    Raises:
        ValueError: The rank is not a nonnegative integer.
        RuntimeError: A native class binding is unsupported or already observed
            by a different active environment.
    """
    if not _nonnegative_integer(rank):
        raise ValueError("invalid simulator phase rank")
    if _journal(env) is not None:
        return
    env._npa_phase_journal = _PhaseJournal(Path(directory), rank)
    try:
        _observe_pink_controllers(env)
        _observe_environment(env)
    except BaseException:
        finalize_phase_journal(env)
        raise
