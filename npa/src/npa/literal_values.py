"""Validate literal JSON scalar evidence without coercion or default substitution."""

from __future__ import annotations

import math
from typing import Any


def require_boolean(value: Any, *, field: str) -> bool:
    """Require an exact boolean.

    Args:
        value: Present input value to validate.
        field: Input field name for diagnostics.
    Returns:
        The unchanged boolean.
    Raises:
        ValueError: If the value is not a literal boolean.
    """
    if type(value) is not bool:
        raise ValueError(f"{field} must be a literal boolean")
    return value


def require_integer(
    value: Any, *, field: str, minimum: int | None = None, maximum: int | None = None
) -> int:
    """Require an exact integer within optional inclusive bounds.

    Args:
        value: Present input value to validate; booleans are not integers.
        field: Input field name for diagnostics.
        minimum: Inclusive lower bound, if supplied.
        maximum: Inclusive upper bound, if supplied.
    Returns:
        The unchanged integer.
    Raises:
        ValueError: If the type or range is invalid.
    """
    if type(value) is not int:
        raise ValueError(f"{field} must be a literal integer")
    _require_range(value, field=field, minimum=minimum, maximum=maximum)
    return value


def require_number(
    value: Any,
    *,
    field: str,
    minimum: int | float | None = None,
    maximum: int | float | None = None,
) -> int | float:
    """Require an exact finite number within optional inclusive bounds.

    Args:
        value: Present input value; booleans and numeric strings are rejected.
        field: Input field name for diagnostics.
        minimum: Inclusive lower bound, if supplied.
        maximum: Inclusive upper bound, if supplied.
    Returns:
        The unchanged integer or float.
    Raises:
        ValueError: If the type, finiteness, or range is invalid.
    """
    if type(value) not in (int, float):
        raise ValueError(f"{field} must be a literal number")
    if type(value) is float and not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    _require_range(value, field=field, minimum=minimum, maximum=maximum)
    return value


def _require_range(value, *, field, minimum, maximum):
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field} must be at most {maximum}")
