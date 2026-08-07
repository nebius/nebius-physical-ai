"""Strict parsing for JSON emitted after harmless SkyPilot CLI diagnostics."""

from __future__ import annotations

from typing import Any

from npa.clients.json_output import parse_single_json_document


def queue_rows_from_output(output: str) -> list[dict[str, Any]] | None:
    """Parse a verified SkyPilot queue list from one unambiguous JSON payload."""

    payload = parse_single_json_document(output)
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("jobs"), list):
        rows = payload["jobs"]
    else:
        return None
    if not all(isinstance(row, dict) for row in rows):
        return None
    return rows
