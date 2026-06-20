"""Named transition predicates for NPA workflow state machines."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from npa.orchestration.npa_workflow.errors import NpaWorkflowError

PredicateFn = Callable[[Mapping[str, Any]], bool]

PREDICATES: dict[str, PredicateFn] = {}


def register_predicate(name: str, fn: PredicateFn) -> None:
    PREDICATES[name] = fn


def evaluate_predicate(name: str, context: Mapping[str, Any]) -> bool:
    fn = PREDICATES.get(name)
    if fn is None:
        raise NpaWorkflowError(f"unknown predicate: {name!r} (known: {sorted(PREDICATES)})")
    return bool(fn(context))


def _promote_checkpoint(context: Mapping[str, Any]) -> bool:
    decision = str(context.get("last_decision") or "")
    return decision == "promote_checkpoint"


def _loop_back(context: Mapping[str, Any]) -> bool:
    decision = str(context.get("last_decision") or "")
    return decision == "loop_back_to_inner_loop"


register_predicate("promote_checkpoint", _promote_checkpoint)
register_predicate("loop_back", _loop_back)
