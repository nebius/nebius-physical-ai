"""Configured-bucket allowlist for operator-supplied S3 URIs (exfil guard).

Embedded into the agent-VM backend. Discovery may still list across accessible
buckets; caller-supplied ``s3_uri`` values may only target the configured agent
bucket (and optional explicit ``NPA_AGENT_S3_BUCKETS`` extras) — never every
bucket ListBuckets returns.
"""

from __future__ import annotations

from urllib.parse import urlparse


def configured_agent_s3_buckets(
    primary: str,
    extras_csv: str = "",
) -> set[str]:
    """Return the explicitly configured agent bucket set (no ListBuckets)."""
    buckets: set[str] = set()
    primary_name = str(primary or "").strip()
    if primary_name:
        buckets.add(primary_name)
    for part in str(extras_csv or "").split(","):
        name = part.strip()
        if name:
            buckets.add(name)
    return buckets


def parse_s3_bucket_key(uri: str) -> tuple[str, str]:
    """Parse ``s3://bucket/key`` into ``(bucket, key)``. Raises ValueError if invalid."""
    raw = str(uri or "").strip()
    parsed = urlparse(raw)
    if parsed.scheme != "s3":
        raise ValueError("s3_uri must use the s3:// scheme")
    bucket = str(parsed.netloc or "").strip()
    key = str(parsed.path or "").lstrip("/")
    if not bucket:
        raise ValueError("s3_uri is missing a bucket")
    return bucket, key


def s3_uri_in_configured_buckets(
    uri: str,
    *,
    primary: str,
    extras_csv: str = "",
    prefix: str = "",
) -> tuple[bool, str]:
    """Return ``(allowed, reason)`` for an operator-supplied S3 URI.

    When ``prefix`` is non-empty, the object key must equal the prefix or live
    under ``prefix/``.
    """
    try:
        bucket, key = parse_s3_bucket_key(uri)
    except ValueError as exc:
        return False, str(exc)
    allowed = configured_agent_s3_buckets(primary, extras_csv)
    if bucket not in allowed:
        return False, "s3_uri bucket is not the configured agent bucket"
    base = str(prefix or "").strip().strip("/")
    if base:
        if key != base and not key.startswith(base + "/"):
            return False, "s3_uri key is outside the configured agent prefix"
    return True, ""
