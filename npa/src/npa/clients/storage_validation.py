"""Secret-safe validation for S3-compatible runtime storage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4


@dataclass(frozen=True)
class StorageProbeResult:
    ok: bool
    code: str
    summary: str
    probe_key: str = ""
    cleanup_attempted: bool = False
    cleanup_succeeded: bool = False


def bucket_name(value: str) -> str:
    """Return a bare bucket name from a name or ``s3://`` URI."""

    cleaned = str(value or "").strip()
    if not cleaned:
        return ""
    if cleaned.startswith("s3://"):
        return urlparse(cleaned).netloc
    return cleaned.split("/", 1)[0]


def _safe_failure(exc: BaseException, *, operation: str) -> tuple[str, str]:
    """Classify a provider failure without serializing its credential-bearing text."""

    response = getattr(exc, "response", None)
    error: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    if isinstance(response, dict):
        raw_error = response.get("Error")
        raw_metadata = response.get("ResponseMetadata")
        error = raw_error if isinstance(raw_error, dict) else {}
        metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
    provider_code = str(error.get("Code", "") or "").strip()
    status = metadata.get("HTTPStatusCode")
    normalized = provider_code.lower().replace("_", "")
    if status in {401, 403} or normalized in {
        "accessdenied",
        "forbidden",
        "invalidaccesskeyid",
        "signaturedoesnotmatch",
    }:
        return (
            "forbidden",
            f"S3 {operation} was forbidden; the configured access key lacks data-plane permission.",
        )
    if normalized in {"nosuchbucket", "notfound", "404"} or status == 404:
        return (
            "bucket_unreachable",
            f"S3 {operation} could not find the configured bucket.",
        )
    if type(exc).__name__ in {
        "ConnectTimeoutError",
        "ConnectionClosedError",
        "EndpointConnectionError",
        "HTTPClientError",
        "ReadTimeoutError",
    }:
        return (
            "endpoint_unreachable",
            f"S3 {operation} could not reach the configured endpoint.",
        )
    # Provider error codes are not a trusted diagnostic channel. Some SDKs put
    # request material in arbitrary code/message fields, so do not echo an
    # unrecognized value into CLI output or persisted transaction state.
    return (
        "write_failed",
        f"S3 {operation} failed; credentials were not saved as usable.",
    )


def probe_storage_write(
    *,
    bucket: str,
    endpoint_url: str,
    access_key_id: str,
    secret_access_key: str,
    region: str = "",
    prefix: str = "",
    client: Any | None = None,
) -> StorageProbeResult:
    """Put and delete one isolated object, returning a secret-free result.

    The probe is intentionally explicit and mutating: deploy/submit paths need
    proof that artifacts can be written, not merely that a bucket name exists.
    Read-only commands never call this function. If the delete fails, the probe
    fails too and reports the exact non-secret key so an operator can remove it.
    """

    name = bucket_name(bucket)
    endpoint = str(endpoint_url or "").strip()
    access = str(access_key_id or "").strip()
    secret = str(secret_access_key or "").strip()
    missing = [
        label
        for label, value in (
            ("bucket", name),
            ("endpoint", endpoint),
            ("AWS_ACCESS_KEY_ID", access),
            ("AWS_SECRET_ACCESS_KEY", secret),
        )
        if not value
    ]
    if missing:
        return StorageProbeResult(
            False,
            "missing_configuration",
            "Writable S3 is not configured; missing " + ", ".join(missing) + ".",
        )

    if client is None:
        try:
            import boto3

            client = boto3.client(
                "s3",
                endpoint_url=endpoint,
                aws_access_key_id=access,
                aws_secret_access_key=secret,
                region_name=str(region or "").strip() or None,
            )
        except Exception as exc:  # noqa: BLE001 - optional SDK/configuration failure
            code, summary = _safe_failure(exc, operation="client setup")
            return StorageProbeResult(False, code, summary)

    clean_prefix = str(prefix or "").strip().strip("/")
    key = "/".join(
        part
        for part in (clean_prefix, ".npa-probes", f"write-{uuid4().hex}.tmp")
        if part
    )
    try:
        client.put_object(Bucket=name, Key=key, Body=b"")
    except Exception as exc:  # noqa: BLE001 - boto/provider exceptions vary
        code, summary = _safe_failure(exc, operation="write probe")
        return StorageProbeResult(False, code, summary, probe_key=key)

    try:
        client.delete_object(Bucket=name, Key=key)
    except Exception as exc:  # noqa: BLE001 - cleanup failure is a failed probe
        _code, summary = _safe_failure(exc, operation="probe cleanup")
        return StorageProbeResult(
            False,
            "cleanup_failed",
            f"{summary} Temporary object remains at s3://{name}/{key}.",
            probe_key=key,
            cleanup_attempted=True,
            cleanup_succeeded=False,
        )
    return StorageProbeResult(
        True,
        "ok",
        "Writable S3 verified with a cleaned write/delete probe.",
        probe_key=key,
        cleanup_attempted=True,
        cleanup_succeeded=True,
    )
