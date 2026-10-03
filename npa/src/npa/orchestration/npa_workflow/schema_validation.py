"""Validate raw workflow documents against the shipped JSON Schema (stdlib checks)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from npa.orchestration.npa_workflow.errors import NpaWorkflowError

_SCHEMA_PATH = (
    Path(__file__).resolve().parent / "schema" / "npa.workflow.v0.0.1.schema.json"
)


def validate_document(data: dict[str, Any]) -> None:
    """Lightweight structural validation using the shipped schema constants."""

    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    _validate_against_schema(data, schema, path="$", root=schema)


def _validate_against_schema(
    value: Any,
    schema: dict[str, Any],
    *,
    path: str,
    root: dict[str, Any],
) -> None:
    reference = schema.get("$ref")
    if reference is not None:
        if not isinstance(reference, str) or not reference.startswith("#/"):
            raise NpaWorkflowError(
                f"{path}: unsupported schema reference {reference!r}"
            )
        target: Any = root
        for component in reference[2:].split("/"):
            component = component.replace("~1", "/").replace("~0", "~")
            if not isinstance(target, dict) or component not in target:
                raise NpaWorkflowError(
                    f"{path}: unresolved schema reference {reference!r}"
                )
            target = target[component]
        if not isinstance(target, dict):
            raise NpaWorkflowError(f"{path}: invalid schema reference {reference!r}")
        _validate_against_schema(value, target, path=path, root=root)
        return

    if "const" in schema and value != schema["const"]:
        raise NpaWorkflowError(f"{path}: expected {schema['const']!r}, got {value!r}")
    if "enum" in schema and value not in schema["enum"]:
        raise NpaWorkflowError(
            f"{path}: expected one of {schema['enum']!r}, got {value!r}"
        )

    schema_type = schema.get("type")
    if schema_type == "object":
        if not isinstance(value, dict):
            raise NpaWorkflowError(
                f"{path}: expected object, got {type(value).__name__}"
            )
        non_string_keys = [key for key in value if not isinstance(key, str)]
        if non_string_keys:
            raise NpaWorkflowError(
                f"{path}: object keys must be strings, got {non_string_keys[0]!r}"
            )
        minimum = schema.get("minProperties")
        if minimum is not None and len(value) < minimum:
            raise NpaWorkflowError(
                f"{path}: expected at least {minimum} properties, got {len(value)}"
            )
        for key in schema.get("required", []):
            if key not in value:
                raise NpaWorkflowError(f"{path}: missing required field {key!r}")
        properties = schema.get("properties", {})
        for key, subschema in properties.items():
            if key in value:
                _validate_against_schema(
                    value[key], subschema, path=f"{path}.{key}", root=root
                )
        extras = sorted(set(value) - set(properties))
        additional = schema.get("additionalProperties", True)
        if additional is False:
            if extras:
                raise NpaWorkflowError(
                    f"{path}: unknown field(s): {', '.join(repr(item) for item in extras)}"
                )
        elif isinstance(additional, dict):
            for key in extras:
                _validate_against_schema(
                    value[key], additional, path=f"{path}.{key}", root=root
                )
        return

    if schema_type == "array":
        if not isinstance(value, list):
            raise NpaWorkflowError(
                f"{path}: expected array, got {type(value).__name__}"
            )
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(value):
                _validate_against_schema(
                    item, item_schema, path=f"{path}[{index}]", root=root
                )
        return

    if schema_type == "string" and not isinstance(value, str):
        raise NpaWorkflowError(f"{path}: expected string, got {type(value).__name__}")

    if schema_type == "boolean" and not isinstance(value, bool):
        raise NpaWorkflowError(f"{path}: expected boolean, got {type(value).__name__}")
