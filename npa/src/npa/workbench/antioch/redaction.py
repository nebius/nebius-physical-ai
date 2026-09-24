"""Fail-closed redaction for vendor CLI output and persisted diagnostics."""

from __future__ import annotations

import json
import re
from typing import Any


REDACTED = "<redacted>"
_SENSITIVE_KEYS = re.compile(
    r"(?:authorization|bearer|cookie|credential|email|organization|tenant|token|secret|password|signed_url|user_id)",
    re.IGNORECASE,
)
_TOKEN_PATTERNS = (
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+"),
    re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"(?i)(X-Amz-(?:Credential|Signature|Security-Token)=)[^&\s]+"),
    re.compile(r"(?i)([?&](?:token|access_token|signature|sig|key)=)[^&\s]+"),
    re.compile(
        r"(?i)(\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret|credential)\s*[=:]\s*)[^\s,;]+"
    ),
    re.compile(r"\b(?:hf|ghp|github_pat)_[A-Za-z0-9_=-]{16,}\b"),
    # Antioch run/container identities are opaque hexadecimal values. They are
    # operationally sensitive even when the vendor emits them in plain text
    # instead of structured identifier fields.
    re.compile(r"\b[0-9a-fA-F]{32,64}\b"),
    re.compile(
        r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
    ),
)


def redact_text(value: str, *, limit: int = 4096) -> str:
    """Remove secret-shaped values and bound external diagnostic text."""

    text = value[:limit]
    for pattern in _TOKEN_PATTERNS:
        replacement = r"\1" + REDACTED if pattern.groups else REDACTED
        text = pattern.sub(replacement, text)
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text
    return json.dumps(redact_payload(parsed), sort_keys=True)


def redact_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): REDACTED
            if _SENSITIVE_KEYS.search(str(key))
            else redact_payload(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_payload(item) for item in value]
    if isinstance(value, str):
        redacted = value
        for pattern in _TOKEN_PATTERNS:
            replacement = r"\1" + REDACTED if pattern.groups else REDACTED
            redacted = pattern.sub(replacement, redacted)
        return redacted
    return value
