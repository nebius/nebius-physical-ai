"""Record scalar execution phases without changing simulator or policy decisions."""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import json
import os
from pathlib import Path
import time
from typing import Any, Iterator


_PHASES = {"policy_action", "env_step", "pink_ik", "capture", "render_call"}
_READINESS_FIELDS = {"stage_ready", "annotator_ready", "nonblack_rgb"}


class _PhaseJournal:
    def __init__(self, directory: Path, rank: int) -> None:
        self.path = directory / f"simulator-phases-rank{rank}.jsonl"
        self.rank = rank
        self.sequence = 0
        self.action_step = 0
        self.write_failed = False

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
        or any(value is not None and type(value) is not bool for value in readiness.values())
    ):
        raise ValueError("invalid simulator readiness fields")


def _validate_event(phase: Any, event: Any, action_step: Any, fields: dict) -> None:
    if type(phase) is not str or type(event) is not str or not _nonnegative_integer(action_step):
        raise ValueError("invalid simulator phase event")
    if phase == "capture_readiness":
        if event != "observed" or "render_call" not in fields:
            raise ValueError("invalid simulator readiness event")
        readiness = {key: value for key, value in fields.items() if key != "render_call"}
        _validate_readiness(fields["render_call"], readiness)
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
    if type(phase) is not str or phase not in _PHASES or not _nonnegative_integer(action_step):
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
            "capture_readiness", "observed", journal.action_step,
            render_call=render_call, **readiness,
        )


def _observe_controller(controller: Any, env: Any) -> None:
    original = controller.compute

    @wraps(original)
    def compute(*args: Any, **kwargs: Any) -> Any:
        with phase_scope(env, "pink_ik", _journal(env).action_step):
            return original(*args, **kwargs)

    controller.compute = compute


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
    """
    if not _nonnegative_integer(rank):
        raise ValueError("invalid simulator phase rank")
    if _journal(env) is not None:
        return
    env._npa_phase_journal = _PhaseJournal(Path(directory), rank)
    _observe_pink_controllers(env)
