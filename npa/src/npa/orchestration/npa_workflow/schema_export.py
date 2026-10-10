"""Expose the authoritative bundled workflow JSON Schema to integrations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_SCHEMA_PATH = (
    Path(__file__).resolve().parent / "schema" / "npa.workflow.v0.0.1.schema.json"
)


def load_workflow_schema() -> dict[str, Any]:
    """Load a fresh copy of the authoritative workflow JSON Schema.

    Args:
        None.

    Returns:
        Parsed JSON Schema for supported ``npa.workflow`` documents.

    Raises:
        FileNotFoundError: The bundled schema is unavailable from the installation.
        ValueError: The bundled schema is not a JSON object.
    """
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    if not isinstance(schema, dict):
        raise ValueError("bundled workflow schema must be a JSON object")
    return schema


__all__ = ["load_workflow_schema"]
