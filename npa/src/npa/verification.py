"""Shared truthful live-verification envelopes for operator-facing state.

Persisted state answers "what was last observed?"; it is not proof that the
resource is healthy now.  Workflow, cluster, JSON, and agent callers use this
module so transport/provider failures cannot silently turn into healthy-looking
``RUNNING`` or ``UNKNOWN`` output.
"""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any, Mapping, Sequence

VERIFIED = "VERIFIED"
VERIFICATION_UNAVAILABLE = "VERIFICATION_UNAVAILABLE"
CACHED = "CACHED"

_HORIZONTAL_WHITESPACE = r"[^\S\r\n\v\f\x1c-\x1e\x85\u2028\u2029]"
_SECRET_ASSIGNMENT_PREFIX = re.compile(
    r"(?i)(?<![a-z0-9_-])"
    r"(?P<key>[\"']?[a-z0-9_-]+[\"']?)"
    rf"(?P<separator>{_HORIZONTAL_WHITESPACE}*[:=]{_HORIZONTAL_WHITESPACE}*)"
    rf"(?P<scheme>(?:bearer|basic){_HORIZONTAL_WHITESPACE}+)?"
)
_SECRET_KEY_MARKERS = (
    "token",
    "password",
    "secret",
    "api_key",
    "api-key",
    "apikey",
    "authorization",
)
_NON_SECRET_WORKFLOW_TOKEN_REFERENCES = tuple(
    (f"unknown {scope} ", f"{scope}.") for scope in ("config", "run", "state", "loop")
)
_NON_SECRET_ASSIGNMENT_CONTEXT_WIDTH = max(
    len(context) for context, _value_prefix in _NON_SECRET_WORKFLOW_TOKEN_REFERENCES
)
_BEARER_TOKEN = re.compile(
    rf"(?i)\bbearer{_HORIZONTAL_WHITESPACE}+"
    r"(?![a-z][a-z0-9+.-]*://)"
    r"[A-Za-z0-9._~+/=-]+"
)


def _quoted_secret_end(text: str, start: int) -> tuple[int, bool]:
    quote = text[start]
    index = start + 1
    while index < len(text) and text[index] != "\n":
        if text[index] == "\\":
            if index + 1 < len(text) and text[index + 1] == "\n":
                return index + 1, False
            index += 2
            continue
        if text[index] != quote:
            index += 1
            continue
        tail = index + 1
        while tail < len(text) and text[tail] in " \t":
            tail += 1
        if tail == len(text) or text[tail] in ",;}]\n":
            return index + 1, True
        index += 1
    return min(index, len(text)), False


def _secret_assignment_value(text: str, start: int) -> tuple[int, str, bool]:
    quote = text[start] if text[start] in {'"', "'"} else ""
    if quote:
        end, closed = _quoted_secret_end(text, start)
        return end, quote, closed
    end = start
    while end < len(text) and not text[end].isspace() and text[end] not in ",;}]\"'":
        end += 1
    return end, "", False


def _is_workflow_token_reference(text: str, start: int, raw_value: str) -> bool:
    prefix = text[max(0, start - _NON_SECRET_ASSIGNMENT_CONTEXT_WIDTH) : start].lower()
    if not all(
        character.isascii() and (character.isalnum() or character in "_.-")
        for character in raw_value
    ):
        return False
    return any(
        prefix.endswith(context)
        and raw_value.lower().startswith(value_prefix)
        and len(raw_value) > len(value_prefix)
        for context, value_prefix in _NON_SECRET_WORKFLOW_TOKEN_REFERENCES
    )


def _redact_secret_assignments(text: str) -> str:
    pieces: list[str] = []
    cursor = 0
    for match in _SECRET_ASSIGNMENT_PREFIX.finditer(text):
        if match.start() < cursor:
            continue
        key = match.group("key")
        normalized_key = key.strip("\"'").lower()
        if not any(marker in normalized_key for marker in _SECRET_KEY_MARKERS):
            continue
        value_start = match.end()
        if value_start >= len(text) or text[value_start] == "\n":
            continue
        value_end, quote, closed_quote = _secret_assignment_value(text, value_start)
        if value_end == value_start:
            continue
        if (
            normalized_key == "token"
            and not quote
            and _is_workflow_token_reference(
                text, match.start(), text[value_start:value_end]
            )
        ):
            continue
        pieces.append(text[cursor : match.start()])
        pieces.append(f"{key}{match.group('separator')}{quote}<redacted>")
        if closed_quote:
            pieces.append(quote)
        cursor = value_end
    pieces.append(text[cursor:])
    return "".join(pieces)


def _url_schemes(text: str, start: int, end: int) -> list[tuple[int, int]]:
    urls: list[tuple[int, int]] = []
    search_from = start
    while True:
        delimiter = text.find("://", search_from, end)
        if delimiter < 0:
            return urls
        scheme_start = delimiter
        while scheme_start > start:
            candidate = text[scheme_start - 1]
            if not (
                candidate.isascii() and (candidate.isalnum() or candidate in "+.-")
            ):
                break
            scheme_start -= 1
        while scheme_start < delimiter and not (
            text[scheme_start].isascii() and text[scheme_start].isalpha()
        ):
            scheme_start += 1
        if scheme_start < delimiter:
            urls.append((scheme_start, delimiter))
        search_from = delimiter + 3


def _url_secret_ranges(text: str, start: int, end: int) -> list[tuple[int, int]]:
    replacements: list[tuple[int, int]] = []
    urls = _url_schemes(text, start, end)
    for index, (_scheme_start, delimiter) in enumerate(urls):
        url_end = urls[index + 1][0] if index + 1 < len(urls) else end
        authority_start = delimiter + 3
        authority_end = url_end
        for boundary in "/?#":
            boundary_index = text.find(boundary, authority_start, url_end)
            if boundary_index >= 0:
                authority_end = min(authority_end, boundary_index)
        userinfo_end = text.find("@", authority_start, authority_end)
        if userinfo_end >= 0:
            replacements.append((authority_start, userinfo_end))
        query_start = text.find("?", authority_start, url_end)
        if 0 <= query_start < end - 1:
            replacements.append((query_start + 1, end))
    return replacements


def _redact_ranges(text: str, replacements: list[tuple[int, int]]) -> str:
    if not replacements:
        return text
    pieces: list[str] = []
    cursor = 0
    for start, end in sorted(replacements):
        if start < cursor:
            cursor = max(cursor, end)
            continue
        pieces.append(text[cursor:start])
        pieces.append("<redacted>")
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def _redact_url_credentials(text: str) -> str:
    replacements: list[tuple[int, int]] = []
    token_start = 0
    while token_start < len(text):
        while token_start < len(text) and text[token_start].isspace():
            token_start += 1
        token_end = token_start
        while token_end < len(text) and not text[token_end].isspace():
            token_end += 1
        if token_start == token_end:
            break
        replacements.extend(_url_secret_ranges(text, token_start, token_end))
        token_start = token_end
    return _redact_ranges(text, replacements)


def redact_failure_text(reason: object, *, secrets: Sequence[str]) -> str:
    """Redact failure text without changing its display structure.

    Args:
        reason: Exception or provider diagnostic to make safe for display.
        secrets: Resolved credential values known at the call boundary.
    Returns:
        Redacted text with its original whitespace and line structure.
    Raises:
        None.
    """

    from npa.orchestration.skypilot.workflow_state import redact_text

    text = redact_text(str(reason or ""), secrets)
    text = _redact_secret_assignments(text)
    text = _BEARER_TOKEN.sub("Bearer <redacted>", text)
    return _redact_url_credentials(text)


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def sanitize_reason(reason: object, *, limit: int = 600) -> str:
    """Return a concise diagnostic without secrets or presigned query strings."""

    text = " ".join(redact_failure_text(reason, secrets=()).split())
    return text[:limit]


def sanitize_failure_reason(
    reason: object,
    *,
    secrets: Sequence[str],
    limit: int = 600,
) -> str:
    """Redact explicit and patterned secrets from one failure diagnostic.

    Args:
        reason: Exception or provider diagnostic to make safe for persistence.
        secrets: Resolved credential values known at the call boundary.
        limit: Maximum number of returned characters.
    Returns:
        A single-line diagnostic safe for operator-visible state.
    Raises:
        None.
    """

    text = " ".join(redact_failure_text(reason, secrets=secrets).split())
    return text[:limit]


def classify_verification_failure(reason: object) -> tuple[str, str]:
    """Classify common live-state failures into stable automation codes."""

    text = str(reason or "").lower()
    if any(
        item in text
        for item in ("no such host", "name or service not known", "dns", "getaddrinfo")
    ):
        return "DNS_RESOLUTION_FAILED", "DNS"
    if any(item in text for item in ("timed out", "timeout", "deadline exceeded")):
        return "LIVE_QUERY_TIMEOUT", "TIMEOUT"
    if any(item in text for item in ("forbidden", "rbac", "permission denied")):
        return "LIVE_QUERY_FORBIDDEN", "RBAC"
    if any(
        item in text
        for item in ("unauthorized", "unauthenticated", "invalid credential", "401")
    ):
        return "LIVE_QUERY_AUTHENTICATION_FAILED", "AUTHENTICATION"
    if any(
        item in text for item in ("context", "cluster identity", "project mismatch")
    ) and any(item in text for item in ("stale", "mismatch", "not found", "unknown")):
        return "LIVE_CONTEXT_MISMATCH", "CONTEXT"
    if any(
        item in text
        for item in ("json", "yaml", "parse", "unparseable", "malformed response")
    ):
        return "LIVE_RESPONSE_UNPARSEABLE", "RESPONSE"
    if any(
        item in text
        for item in ("connection refused", "unreachable", "controller", "api server")
    ):
        return "LIVE_CONTROLLER_UNREACHABLE", "CONTROLLER"
    return "LIVE_VERIFICATION_FAILED", "PROVIDER"


def verification_envelope(
    *,
    status: str,
    target: str,
    last_known_state: str = "",
    last_known_at: str = "",
    last_known_source: str = "",
    reason: object = "",
    retry_command: str = "",
    attempted_at: str = "",
) -> dict[str, Any]:
    """Build the backwards-compatible shared verification payload."""

    normalized = str(status or "").upper()
    if normalized not in {VERIFIED, VERIFICATION_UNAVAILABLE, CACHED}:
        raise ValueError(f"unsupported verification status: {status}")
    live_verified = normalized == VERIFIED
    safe_reason = sanitize_reason(reason)
    error_code = ""
    category = ""
    if normalized == VERIFICATION_UNAVAILABLE:
        error_code, category = classify_verification_failure(safe_reason)
    legacy = (
        "found"
        if live_verified
        else "cached"
        if normalized == CACHED
        else "unavailable"
    )
    return {
        "verification": legacy,
        "verification_status": normalized,
        "live_verified": live_verified,
        "automation_may_trust_state": live_verified,
        "last_known": {
            "state": str(last_known_state or "UNKNOWN").upper(),
            "observed_at": str(last_known_at or ""),
            "source": str(last_known_source or "unknown"),
        },
        "live_verification": {
            "status": normalized,
            "live_verified": live_verified,
            "attempted_at": str(attempted_at or utc_now()),
            "target": sanitize_reason(target, limit=240),
            "error_code": error_code,
            "category": category,
            "reason": safe_reason,
            "retry_command": str(retry_command or ""),
            "automation_may_trust_state": live_verified,
        },
    }


def apply_verification(
    payload: Mapping[str, Any],
    *,
    status: str,
    target: str,
    last_known_state: str = "",
    last_known_at: str = "",
    last_known_source: str = "",
    reason: object = "",
    retry_command: str = "",
    state_key: str = "status",
    attempted_at: str = "",
) -> dict[str, Any]:
    """Add verification fields and prevent a contradictory healthy top-level state."""

    result = dict(payload)
    known = str(last_known_state or result.get(state_key) or "UNKNOWN").upper()
    result.update(
        verification_envelope(
            status=status,
            target=target,
            last_known_state=known,
            last_known_at=last_known_at,
            last_known_source=last_known_source,
            reason=reason,
            retry_command=retry_command,
            attempted_at=attempted_at,
        )
    )
    if status == VERIFICATION_UNAVAILABLE:
        result[state_key] = VERIFICATION_UNAVAILABLE
    elif status == CACHED:
        result[state_key] = CACHED
    return result
