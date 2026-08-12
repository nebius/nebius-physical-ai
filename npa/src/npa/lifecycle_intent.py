"""Explicit lifecycle capability boundary shared by CLI and SDK operations."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum
from functools import wraps
from contextlib import redirect_stdout
import inspect
import io
import json
import os
import sys
from typing import Any, Callable, Iterator, TypeVar, cast


class OperationIntent(str, Enum):
    OBSERVE = "observe"
    ENSURE_PRESENT = "ensure-present"
    MUTATE = "mutate"
    DESTROY = "destroy"


class OperationIntentError(RuntimeError):
    """A primitive was invoked from an incompatible lifecycle operation."""


_INTENT: ContextVar[OperationIntent | None] = ContextVar("npa_operation_intent", default=None)
F = TypeVar("F", bound=Callable[..., Any])


def current_intent() -> OperationIntent:
    value = _INTENT.get()
    if value is not None:
        return value
    raw = os.environ.get("NPA_OPERATION_INTENT", "").strip().lower()
    if raw:
        try:
            return OperationIntent(raw)
        except ValueError as exc:
            raise OperationIntentError(f"invalid NPA operation intent {raw!r}") from exc
    # Compatibility for SDK callers that predate the explicit boundary. CLI
    # entrypoints always install an intent before reaching lifecycle primitives.
    return OperationIntent.MUTATE


@contextmanager
def operation_intent(intent: OperationIntent) -> Iterator[None]:
    token = _INTENT.set(intent)
    previous = os.environ.get("NPA_OPERATION_INTENT")
    os.environ["NPA_OPERATION_INTENT"] = intent.value
    try:
        yield
    finally:
        _INTENT.reset(token)
        if previous is None:
            os.environ.pop("NPA_OPERATION_INTENT", None)
        else:
            os.environ["NPA_OPERATION_INTENT"] = previous


def require_intent(*allowed: OperationIntent, primitive: str) -> None:
    active = current_intent()
    if active not in allowed:
        choices = ", ".join(item.value for item in allowed)
        raise OperationIntentError(
            f"{primitive} requires lifecycle intent {choices}; active intent is {active.value}"
        )


def forbid_destructive_provisioning(primitive: str) -> None:
    active = current_intent()
    if active in {OperationIntent.DESTROY, OperationIntent.OBSERVE}:
        raise OperationIntentError(
            f"{active.value} lifecycle intent cannot invoke provisioning primitive {primitive}"
        )


def intent_boundary(intent: OperationIntent) -> Callable[[F], F]:
    """Decorate a CLI/SDK entrypoint with a process-inheritable intent."""

    def decorate(function: F) -> F:
        @wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            with operation_intent(intent):
                return function(*args, **kwargs)

        return cast(F, wrapped)

    return decorate


def json_stdout_contract(function: F) -> F:
    """Guarantee one JSON stdout document for commands exposing a JSON flag."""

    signature = inspect.signature(function)

    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        enabled = bool(
            bound.arguments.get("output_json")
            or bound.arguments.get("json_output")
            or str(bound.arguments.get("output_format") or "").lower() == "json"
        )
        if not enabled:
            return function(*args, **kwargs)
        capture = io.StringIO()
        failure: BaseException | None = None
        result: Any = None
        with redirect_stdout(capture):
            try:
                result = function(*args, **kwargs)
            except BaseException as exc:  # preserve Typer/system exit semantics
                failure = exc
        raw = capture.getvalue().strip()
        documents: list[Any] = []
        decoder = json.JSONDecoder()
        cursor = 0
        while cursor < len(raw):
            candidates = [index for index in (raw.find("{", cursor), raw.find("[", cursor)) if index >= 0]
            if not candidates:
                break
            start = min(candidates)
            try:
                document, end = decoder.raw_decode(raw[start:])
            except json.JSONDecodeError:
                cursor = start + 1
                continue
            documents.append(document)
            cursor = start + end
        if documents:
            document = documents[-1]
        else:
            document = {
                "result": "error" if failure is not None else "completed",
                "mutated": failure is None,
            }
            if failure is not None:
                document["error_type"] = type(failure).__name__
        if raw and (len(documents) != 1 or raw != json.dumps(documents[0], indent=2, sort_keys=True)):
            print("command diagnostics were separated from JSON stdout", file=sys.stderr)
        print(json.dumps(document, indent=2, sort_keys=True))
        if failure is not None:
            raise failure
        return result

    return cast(F, wrapped)
