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


_INTENT: ContextVar[OperationIntent | None] = ContextVar(
    "npa_operation_intent", default=None
)
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


def _reports_failure(document: Any) -> bool:
    if not isinstance(document, dict):
        return False
    status = str(document.get("status") or "").strip().lower()
    outcome = str(document.get("outcome") or "").strip().lower()
    successes = {"completed", "planned", "ready", "submitted", "succeeded", "success"}
    if status in successes:
        return False
    status_failed = status in {
        "blocked",
        "cancelled",
        "canceled",
        "error",
        "failed",
        "failure",
    }
    outcome_failed = outcome in {
        "confirmation_required",
        "error",
        "failed",
        "partial_cancellation",
        "verification_failed",
    }
    return (
        status_failed
        or outcome_failed
        or bool(
            document.get("error")
            or document.get("error_type")
            or str(document.get("result") or "").strip().lower() == "error"
        )
    )


def _json_mode_enabled(bound: inspect.BoundArguments) -> bool:
    output_format = bound.arguments.get("output_format")
    output_value = getattr(output_format, "value", output_format)
    return bool(
        bound.arguments.get("output_json")
        or bound.arguments.get("json_output")
        or str(output_value or "").lower() == "json"
    )


def _invoke_with_captured_stdout(
    target: F, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[Any, BaseException | None, str]:
    capture = io.StringIO()
    failure: BaseException | None = None
    result: Any = None
    with redirect_stdout(capture):
        try:
            result = target(*args, **kwargs)
        except BaseException as exc:  # preserve Typer/system exit semantics
            failure = exc
    return result, failure, capture.getvalue().strip()


def _decode_json_documents(raw: str) -> list[Any]:
    documents: list[Any] = []
    decoder = json.JSONDecoder()
    cursor = 0
    while cursor < len(raw):
        candidates = [
            index
            for index in (raw.find("{", cursor), raw.find("[", cursor))
            if index >= 0
        ]
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
    return documents


def _successful_exit(failure: BaseException | None) -> bool:
    if isinstance(failure, SystemExit):
        return failure.code in (None, 0)
    failure_type = type(failure)
    return (
        failure_type.__module__ in {"click.exceptions", "typer.exceptions"}
        and failure_type.__name__ == "Exit"
        and getattr(failure, "exit_code", None) == 0
    )


def _default_json_document(failure: BaseException | None) -> dict[str, Any]:
    effective_failure = None if _successful_exit(failure) else failure
    document: dict[str, Any] = {
        "result": "error" if effective_failure is not None else "completed",
        "mutated": effective_failure is None,
    }
    if effective_failure is not None:
        document["error_type"] = type(effective_failure).__name__
    return document


def _select_json_document(
    documents: list[Any],
    failure: BaseException | None,
    *,
    fail_closed_on_exception: bool,
) -> Any:
    document = documents[-1] if documents else _default_json_document(failure)
    if (
        not fail_closed_on_exception
        or failure is None
        or _successful_exit(failure)
        or (documents and _reports_failure(document))
    ):
        return document
    return {
        "error_type": type(failure).__name__,
        "mutation_state": "unknown",
        "result": "error",
    }


def _diagnostics_were_removed(
    raw: str, documents: list[Any], selected_document: Any
) -> bool:
    if not raw:
        return False
    return len(documents) != 1 or raw != json.dumps(
        selected_document, indent=2, sort_keys=True
    )


def _decorate_json_stdout_contract(target: F, *, fail_closed_on_exception: bool) -> F:
    signature = inspect.signature(target)

    @wraps(target)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        if not _json_mode_enabled(bound):
            return target(*args, **kwargs)
        result, failure, raw = _invoke_with_captured_stdout(target, args, kwargs)
        documents = _decode_json_documents(raw)
        document = _select_json_document(
            documents,
            failure,
            fail_closed_on_exception=fail_closed_on_exception,
        )
        if _diagnostics_were_removed(raw, documents, document):
            print("command diagnostics were removed from JSON stdout", file=sys.stderr)
        print(json.dumps(document, indent=2, sort_keys=True))
        if failure is not None:
            raise failure
        return result

    return cast(F, wrapped)


def json_stdout_contract(
    function: F | None = None,
    *,
    fail_closed_on_exception: bool = False,
) -> F | Callable[[F], F]:
    """Guarantee one JSON stdout document for commands exposing a JSON flag.

    Args:
        function: Decorated command when used without arguments.
        fail_closed_on_exception: Replace success-shaped captured JSON when the
            decorated command raises a nonzero or unexpected exception.

    Returns:
        The decorated command, or a decorator when called with options.

    Raises:
        BaseException: Re-raises the decorated command's original exception
            after emitting the selected JSON document.
    """

    if function is None:

        def decorate(target: F) -> F:
            return _decorate_json_stdout_contract(
                target, fail_closed_on_exception=fail_closed_on_exception
            )

        return decorate
    return _decorate_json_stdout_contract(
        function, fail_closed_on_exception=fail_closed_on_exception
    )
