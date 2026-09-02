"""Strict parsing for CLI JSON with a bounded diagnostic preamble."""

from __future__ import annotations

import json
import re
from typing import Any

# ANSI CSI/OSC control sequences (colors, cursor movement, erase-line) emitted
# by rich status spinners; their introducer is ESC+"[", which must not be
# mistaken for the start of a JSON array.
_ANSI_SEQUENCE_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?")


def parse_single_json_document(output: str) -> Any | None:
    """Return one trailing JSON document, rejecting ambiguity and trailing text.

    Trailing output that cannot begin another JSON value (for example SkyPilot's
    rich status spinner occasionally flushing one final ``⠏ Checking managed
    jobs`` frame with ANSI control sequences after the JSON array) is not
    ambiguity and is ignored; any trailing ``[`` or ``{`` still rejects.
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
        trailing = _ANSI_SEQUENCE_RE.sub("", text[end:])
        if any(ch in "[{" for ch in trailing) or _contains_json_value(
            text[:index], decoder
        ):
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
