"""Resolve Typer option defaults when a command is invoked as a plain function.

Typer commands declare their parameters as ``typer.Option(...)`` /
``typer.Argument(...)`` defaults. Click resolves those into concrete values
before calling the command, so the CLI path never sees them. But several NPA
commands are *also* called directly from Python — ``npa agent setup`` delegates
to ``fresh-setup``, which delegates to ``deploy`` — and a direct call leaves any
omitted parameter as a raw ``typer.models.OptionInfo`` object.

That is not a harmless type error. The values flow straight into Terraform vars
and shell commands, so a leaked default becomes a literal
``"<typer.models.OptionInfo object at 0x7f...>"`` (``server_port``, ``ssh_user``)
and boolean flags become *truthy* regardless of their declared default
(``no_public_https=OptionInfo(False)`` silently disabled HTTPS).

:func:`resolve_typer_defaults` makes the direct-call path behave exactly like
the CLI path: every unresolved option is replaced by its declared default.

Usage — apply it *under* ``@app.command`` so Typer registers the wrapper::

    @app.command("deploy")
    @resolve_typer_defaults
    def deploy_cmd(port: int = typer.Option(8088, "--port")) -> None:
        ...

``npa/tests/guardrails/test_typer_command_calls.py`` enforces that any Typer
command called as a function from another command carries this decorator.
"""

from __future__ import annotations

import copy
import functools
import inspect
from typing import Any, Callable, TypeVar

from typer.models import ArgumentInfo, OptionInfo

__all__ = [
    "DECORATOR_ATTR",
    "is_unresolved_option",
    "resolve_option_default",
    "resolve_typer_defaults",
]

#: Marker attribute set on wrapped commands (used by the guardrail test).
DECORATOR_ATTR = "__npa_resolves_typer_defaults__"

F = TypeVar("F", bound=Callable[..., Any])


def is_unresolved_option(value: Any) -> bool:
    """Return True when *value* is an unresolved Typer parameter default."""

    return isinstance(value, (OptionInfo, ArgumentInfo))


def _declared_default(value: Any) -> Any:
    """Return the concrete default carried by an ``OptionInfo``/``ArgumentInfo``.

    Returns :data:`inspect.Parameter.empty` when the parameter is required
    (``typer.Option(...)`` with ``Ellipsis``), which callers must treat as "the
    caller has to supply this".
    """

    default = getattr(value, "default", None)
    if default is ... or default is inspect.Parameter.empty:
        return inspect.Parameter.empty
    # Copy so a mutable default (``typer.Option([])``) is never shared between
    # calls — otherwise one caller appending to ``tf_var`` would leak into the
    # next.
    return copy.copy(default)


def resolve_option_default(value: Any, *, name: str = "") -> Any:
    """Return *value*, substituting a concrete default for an unresolved option.

    Raises ``TypeError`` when *value* is a required Typer parameter, since no
    sensible default exists.
    """

    if not is_unresolved_option(value):
        return value
    resolved = _declared_default(value)
    if resolved is inspect.Parameter.empty:
        label = f" {name!r}" if name else ""
        raise TypeError(f"missing required argument{label}")
    return resolved


def resolve_typer_defaults(func: F) -> F:
    """Decorator: resolve unresolved Typer defaults on direct Python calls.

    A no-op on the Click/Typer path (Click always passes concrete values), so
    the CLI keeps its exact behavior while direct callers stop leaking
    ``OptionInfo`` objects into downstream code.
    """

    signature = inspect.signature(func)
    positional_only = [
        name
        for name, param in signature.parameters.items()
        if param.kind is inspect.Parameter.POSITIONAL_ONLY
    ]
    if positional_only:  # pragma: no cover - Typer commands never use these
        raise TypeError(
            f"resolve_typer_defaults does not support positional-only parameters: "
            f"{', '.join(positional_only)}"
        )

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind_partial(*args, **kwargs)
        resolved: dict[str, Any] = dict(bound.arguments)
        missing: list[str] = []
        for name, param in signature.parameters.items():
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if name in resolved:
                value = resolved[name]
            elif is_unresolved_option(param.default):
                # Omitted by a direct caller: fill in the declared default
                # rather than letting Python bind the OptionInfo sentinel.
                value = param.default
            else:
                continue
            if not is_unresolved_option(value):
                continue
            declared = _declared_default(value)
            if declared is inspect.Parameter.empty:
                missing.append(name)
                continue
            resolved[name] = declared
        if missing:
            names = ", ".join(repr(name) for name in missing)
            raise TypeError(
                f"{func.__name__}() missing required argument(s): {names}. "
                "Pass them explicitly when calling this command as a function."
            )
        return func(**resolved)

    # Keep Typer's introspection (which reads ``inspect.signature``) pointed at
    # the original parameter declarations so the CLI is registered unchanged.
    wrapper.__signature__ = signature  # type: ignore[attr-defined]
    setattr(wrapper, DECORATOR_ATTR, True)
    return wrapper  # type: ignore[return-value]
