"""Shared truthful live-verification envelopes for operator-facing state.

Persisted state answers "what was last observed?"; it is not proof that the
resource is healthy now.  Workflow, cluster, JSON, and agent callers use this
module so transport/provider failures cannot silently turn into healthy-looking
``RUNNING`` or ``UNKNOWN`` output.
"""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any, Mapping

VERIFIED = "VERIFIED"
VERIFICATION_UNAVAILABLE = "VERIFICATION_UNAVAILABLE"
CACHED = "CACHED"

_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b([a-z0-9_-]*(?:token|password|secret|api[_-]?key|authorization)[a-z0-9_-]*)"
    r"\s*[:=]\s*(?:bearer\s+)?([^\s,;]+)"
)
_PRESIGNED_QUERY = re.compile(r"(?i)(https?://[^\s?]+)\?[^\s]+")
_BEARER_TOKEN = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
_URL_USERINFO = re.compile(r"(?i)(https?://)[^/@\s]+@")


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def sanitize_reason(reason: object, *, limit: int = 600) -> str:
    """Return a concise diagnostic without secrets or presigned query strings."""

    text = " ".join(str(reason or "").split())
    text = _SECRET_ASSIGNMENT.sub(lambda match: f"{match.group(1)}=<redacted>", text)
    text = _PRESIGNED_QUERY.sub(r"\1?<redacted>", text)
    text = _BEARER_TOKEN.sub("Bearer <redacted>", text)
    text = _URL_USERINFO.sub(r"\1<redacted>@", text)
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
