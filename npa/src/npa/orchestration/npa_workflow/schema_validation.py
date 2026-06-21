"""Validate raw workflow documents against the shipped JSON Schema (stdlib checks)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from npa.orchestration.npa_workflow.errors import NpaWorkflowError

_SCHEMA_PATH = Path(__file__).resolve().parent / "schema" / "npa.workflow.v0.0.1.schema.json"


def validate_document(data: dict[str, Any]) -> None:
    """Lightweight structural validation using the shipped schema constants."""

    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    _validate_against_schema(data, schema, path="$")


def _validate_against_schema(value: Any, schema: dict[str, Any], *, path: str) -> None:
    if "const" in schema and value != schema["const"]:
        raise NpaWorkflowError(f"{path}: expected {schema['const']!r}, got {value!r}")

    schema_type = schema.get("type")
    if schema_type == "object":
        if not isinstance(value, dict):
            raise NpaWorkflowError(f"{path}: expected object, got {type(value).__name__}")
        for key in schema.get("required", []):
            if key not in value:
                raise NpaWorkflowError(f"{path}: missing required field {key!r}")
        properties = schema.get("properties", {})
        for key, subschema in properties.items():
            if key in value:
                _validate_against_schema(value[key], subschema, path=f"{path}.{key}")
        return

    if schema_type == "array":
        if not isinstance(value, list):
            raise NpaWorkflowError(f"{path}: expected array, got {type(value).__name__}")
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(value):
                _validate_against_schema(item, item_schema, path=f"{path}[{index}]")
        return

    if schema_type == "string" and not isinstance(value, str):
        raise NpaWorkflowError(f"{path}: expected string, got {type(value).__name__}")

    if schema_type == "boolean" and not isinstance(value, bool):
        raise NpaWorkflowError(f"{path}: expected boolean, got {type(value).__name__}")
