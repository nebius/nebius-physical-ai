"""Secret-safe validation for S3-compatible runtime storage."""

from __future__ import annotations

import json
import hashlib
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


def terraform_state_key(project: str, name: str) -> str:
    """Return the one canonical backend object key used by agent Terraform."""

    return f"npa/terraform-state/{project}/{name}/terraform.tfstate"


def terraform_backend_fingerprint(
    *,
    bucket: str,
    state_key: str,
    endpoint_url: str,
    access_key_id: str,
    secret_access_key: str,
    session_token: str = "",
    region: str = "",
    addressing_style: str = "path",
) -> str:
    """Return a stable non-reversible configuration-generation fingerprint."""

    material = "\0".join(
        (
            bucket_name(bucket),
            str(state_key or "").lstrip("/"),
            str(endpoint_url or "").strip(),
            str(region or "").strip(),
            str(addressing_style or "path").strip(),
            str(access_key_id or ""),
            str(secret_access_key or ""),
            str(session_token or ""),
        )
    )
    return "sha256:" + hashlib.sha256(material.encode("utf-8")).hexdigest()


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


def _is_not_found(exc: BaseException) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    error = response.get("Error")
    metadata = response.get("ResponseMetadata")
    code = str((error if isinstance(error, dict) else {}).get("Code", "")).lower()
    status = (metadata if isinstance(metadata, dict) else {}).get("HTTPStatusCode")
    return status == 404 or code.replace("_", "") in {"404", "notfound", "nosuchkey"}


def _storage_client(
    *,
    endpoint: str,
    access: str,
    secret: str,
    session_token: str,
    region: str,
    addressing_style: str,
) -> Any:
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        aws_session_token=session_token or None,
        region_name=region or None,
        config=Config(s3={"addressing_style": addressing_style or "path"}),
    )


def probe_terraform_backend(
    *,
    bucket: str,
    state_key: str,
    endpoint_url: str,
    access_key_id: str,
    secret_access_key: str,
    session_token: str = "",
    region: str = "",
    addressing_style: str = "path",
    client: Any | None = None,
) -> StorageProbeResult:
    """Verify the exact backend contract without ever mutating its state object.

    An existing state object is read and JSON-validated.  A missing state object is
    distinguished from forbidden access by creating a conditional random sibling,
    listing the exact prefix, reading it back, deleting it in ``finally``, and
    verifying absence.  The exact ``state_key`` is never written or deleted.
    """

    name = bucket_name(bucket)
    key = str(state_key or "").strip().lstrip("/")
    endpoint = str(endpoint_url or "").strip()
    access = str(access_key_id or "").strip()
    secret = str(secret_access_key or "").strip()
    missing = [
        label
        for label, value in (
            ("bucket", name),
            ("state_key", key),
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
            "Terraform backend is not configured; missing " + ", ".join(missing) + ".",
        )
    if client is None:
        try:
            client = _storage_client(
                endpoint=endpoint,
                access=access,
                secret=secret,
                session_token=str(session_token or "").strip(),
                region=str(region or "").strip(),
                addressing_style=str(addressing_style or "path").strip(),
            )
        except Exception as exc:  # noqa: BLE001 - optional SDK/configuration failure
            code, summary = _safe_failure(exc, operation="backend client setup")
            return StorageProbeResult(False, code, summary)

    existing_state = False
    try:
        client.head_object(Bucket=name, Key=key)
    except Exception as exc:  # noqa: BLE001 - provider exception types vary
        if not _is_not_found(exc):
            code, summary = _safe_failure(exc, operation="backend object HEAD")
            return StorageProbeResult(False, code, summary)
    else:
        existing_state = True
        try:
            response = client.get_object(Bucket=name, Key=key)
            body = response.get("Body") if isinstance(response, dict) else None
            read_body = getattr(body, "read", None)
            raw = read_body() if callable(read_body) else body
            parsed = json.loads(raw or b"")
            if not isinstance(parsed, dict) or not isinstance(
                parsed.get("version"), int
            ):
                raise ValueError("state document has no integer version")
        except Exception as exc:  # noqa: BLE001 - data/provider failures share safe output
            if _is_not_found(exc):
                return StorageProbeResult(
                    False,
                    "state_changed_during_probe",
                    "Terraform backend state disappeared between HEAD and GET; retry safely.",
                )
            code, _summary = _safe_failure(exc, operation="backend state read")
            if isinstance(exc, (ValueError, TypeError, json.JSONDecodeError)):
                code = "invalid_state"
            return StorageProbeResult(
                False,
                code,
                "Terraform backend state exists but is not a readable Terraform JSON document.",
            )

    prefix = key.rsplit("/", 1)[0] if "/" in key else ""
    probe_key = "/".join(
        part for part in (prefix, ".npa-probes", f"backend-{uuid4().hex}.tmp") if part
    )
    cleanup_attempted = False
    cleanup_succeeded = False
    probe_created = False
    result: StorageProbeResult | None = None
    try:
        client.put_object(
            Bucket=name,
            Key=probe_key,
            Body=b"npa-backend-probe-v1",
            IfNoneMatch="*",
        )
        probe_created = True
        listing = client.list_objects_v2(Bucket=name, Prefix=probe_key, MaxKeys=1)
        listed = {
            str(item.get("Key") or "")
            for item in (listing.get("Contents") or [])
            if isinstance(item, dict)
        }
        if probe_key not in listed:
            result = StorageProbeResult(
                False,
                "list_failed",
                "Terraform backend prefix listing did not return the isolated probe object.",
                probe_key=probe_key,
            )
        else:
            response = client.get_object(Bucket=name, Key=probe_key)
            body = response.get("Body") if isinstance(response, dict) else None
            read_body = getattr(body, "read", None)
            raw = read_body() if callable(read_body) else body
            if raw != b"npa-backend-probe-v1":
                result = StorageProbeResult(
                    False,
                    "read_mismatch",
                    "Terraform backend isolated probe did not round-trip exactly.",
                    probe_key=probe_key,
                )
    except Exception as exc:  # noqa: BLE001 - boto/provider exceptions vary
        code, summary = _safe_failure(exc, operation="exact backend prefix probe")
        result = StorageProbeResult(False, code, summary, probe_key=probe_key)
    finally:
        if probe_created:
            cleanup_attempted = True
            try:
                client.delete_object(Bucket=name, Key=probe_key)
                try:
                    client.head_object(Bucket=name, Key=probe_key)
                except Exception as exc:  # noqa: BLE001 - expected 404
                    cleanup_succeeded = _is_not_found(exc)
                else:
                    cleanup_succeeded = False
            except Exception:  # noqa: BLE001 - reported without provider text
                cleanup_succeeded = False

    if probe_created and not cleanup_succeeded:
        return StorageProbeResult(
            False,
            "cleanup_failed",
            "Terraform backend probe cleanup could not verify the sibling object absent.",
            probe_key=probe_key,
            cleanup_attempted=cleanup_attempted,
            cleanup_succeeded=False,
        )
    if result is not None:
        return StorageProbeResult(
            result.ok,
            result.code,
            result.summary,
            probe_key=probe_key,
            cleanup_attempted=cleanup_attempted,
            cleanup_succeeded=cleanup_succeeded,
        )
    if existing_state:
        return StorageProbeResult(
            True,
            "existing_state_valid",
            "Exact Terraform backend state is readable and its sibling prefix supports create/list/read/delete.",
            probe_key=probe_key,
            cleanup_attempted=True,
            cleanup_succeeded=True,
        )
    return StorageProbeResult(
        True,
        "new_state_prefix_valid",
        "Exact Terraform backend prefix supports conditional create/list/read/delete.",
        probe_key=probe_key,
        cleanup_attempted=True,
        cleanup_succeeded=True,
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
