"""Describe the neutral preference-response types used by the prompt and parser."""

from __future__ import annotations

from typing import Any


PREFERENCE_LABELS = ("A", "B", "tie", "unresolved")
CONFIDENCE_LEVELS = ("high", "medium", "low")


def preference_response_schema() -> dict[str, Any]:
    """Build the strict preference-response schema for a neutral judge prompt.

    Args:
        None.
    Returns:
        A fresh JSON Schema describing exact fields, enums and nonempty text.
    Raises:
        None.
    """

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **_object_schema(
            {
                "preference": {"type": "string", "enum": list(PREFERENCE_LABELS)},
                "confidence": {"type": "string", "enum": list(CONFIDENCE_LEVELS)},
                "observable_support": _text_list_schema(),
                "critical_defects": _object_schema(
                    {"A": _text_list_schema(), "B": _text_list_schema()}
                ),
                "uncertainty": _text_schema(),
            }
        ),
    }


def _object_schema(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _text_schema() -> dict[str, Any]:
    return {"type": "string", "minLength": 1, "pattern": r"\S"}


def _text_list_schema() -> dict[str, Any]:
    return {"type": "array", "items": _text_schema(), "minItems": 1}
