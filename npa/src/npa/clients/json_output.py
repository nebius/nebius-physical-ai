"""Strict parsing for CLI JSON with a bounded diagnostic preamble."""

from __future__ import annotations

import json
from typing import Any


def parse_single_json_document(output: str) -> Any | None:
    """Return one trailing JSON document, rejecting ambiguity and trailing text."""

    text = str(output or "")
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            payload, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
        if text[end:].strip() or _contains_json_value(text[:index], decoder):
            continue
        return payload
    return None


def _contains_json_value(prefix: str, decoder: json.JSONDecoder) -> bool:
    for index, character in enumerate(prefix):
        if character not in "[{":
            continue
        try:
            decoder.raw_decode(prefix, index)
        except json.JSONDecodeError:
            continue
        return True
    return False
