"""Internal helpers for the public SDK wrapper layer."""

from __future__ import annotations

from enum import Enum
from importlib import import_module
from inspect import Parameter, signature
from typing import Any, Callable


def call_cli_callback(callback: Callable[..., Any], /, **kwargs: Any) -> Any:
    """Call a Typer callback with plain Python defaults and keyword arguments."""
    sig = signature(callback)
    unexpected = sorted(set(kwargs) - set(sig.parameters))
    if unexpected:
        joined = ", ".join(unexpected)
        raise TypeError(f"unexpected keyword argument(s): {joined}")

    call_kwargs: dict[str, Any] = {}
    missing: list[str] = []
    for name, param in sig.parameters.items():
        if name in kwargs:
            value = kwargs[name]
        else:
            value = _plain_default(param)
            if value is _MISSING:
                missing.append(name)
                continue
        call_kwargs[name] = _coerce_value(value, param.annotation)

    if missing:
        joined = ", ".join(missing)
        raise TypeError(f"missing required argument(s): {joined}")
    return callback(**call_kwargs)


def make_cli_wrapper(module_name: str, callback_name: str, doc: str):
    """Create a lazy SDK wrapper for a CLI callback."""

    def wrapper(**kwargs: Any) -> Any:
        module = import_module(module_name)
        callback = getattr(module, callback_name)
        return call_cli_callback(callback, **kwargs)

    wrapper.__doc__ = doc
    wrapper.__name__ = callback_name.removesuffix("_cmd")
    wrapper.__npa_cli_module__ = module_name
    wrapper.__npa_cli_callback__ = callback_name
    return wrapper


class _Missing:
    pass


_MISSING = _Missing()


def _plain_default(param: Parameter) -> Any:
    if param.default is Parameter.empty:
        return _MISSING
    default = param.default
    if hasattr(default, "default"):
        value = default.default
        if value is ...:
            return _MISSING
        return value
    return default


def _coerce_value(value: Any, annotation: Any) -> Any:
    if value is None:
        return None
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        if isinstance(value, annotation):
            return value
        return annotation(value)
    return value
