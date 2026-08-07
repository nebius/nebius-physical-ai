"""Strict parsing for JSON emitted after harmless SkyPilot CLI diagnostics."""

from __future__ import annotations

import json
from typing import Any


def parse_single_json_document(output: str) -> Any | None:
    """Return one trailing JSON document, or ``None`` when output is ambiguous.

    SkyPilot occasionally prints a diagnostic before its JSON payload.  A
    diagnostic prefix is tolerated, but trailing text, multiple JSON documents,
    empty output, and malformed JSON are deliberately rejected.  Cleanup and
    status callers must never guess from an ambiguous queue response.
    """

    text = str(output or "")
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            payload, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
        if text[end:].strip():
            continue
        prefix = text[:index]
        if _contains_json_value(prefix, decoder):
            return None
        return payload
    return None


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


def _contains_json_value(prefix: str, decoder: json.JSONDecoder) -> bool:
    """Whether a prefix already contains a complete JSON object or array."""

    for index, character in enumerate(prefix):
        if character not in "[{":
            continue
        try:
            decoder.raw_decode(prefix, index)
        except json.JSONDecodeError:
            continue
        return True
    return False
