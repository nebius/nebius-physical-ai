"""Credential-scoped operation helpers."""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import TypeVar
from urllib.parse import urlparse

from botocore.exceptions import ClientError, NoCredentialsError

from npa.errors import ScopedCredentialError

T = TypeVar("T")

SCOPED_CREDENTIAL_ERROR_CODES = {"AccessDenied", "Forbidden", "NoSuchBucket"}


def bucket_from_s3_uri(uri: str) -> str:
    parsed = urlparse(uri)
    return parsed.netloc


def client_error_code(exc: ClientError) -> str:
    return str(exc.response.get("Error", {}).get("Code", ""))


def run_with_host_credential_fallback(
    scoped_operation: Callable[[], T],
    host_fallback: Callable[[], T],
    *,
    bucket: str,
    operation: str,
    allow_host_creds: bool,
    logger: logging.Logger | None = None,
    on_fallback: Callable[[BaseException], None] | None = None,
) -> T:
    """Run a scoped operation and optionally fall back to host credentials.

    Only explicit S3 authorization or credential failures are eligible for host
    fallback. Other errors propagate so operational bugs do not silently expand
    IAM scope.
    """
    try:
        return scoped_operation()
    except ClientError as exc:
        if client_error_code(exc) not in SCOPED_CREDENTIAL_ERROR_CODES:
            raise
        return _fallback_or_raise(
            exc,
            host_fallback,
            bucket=bucket,
            operation=operation,
            allow_host_creds=allow_host_creds,
            logger=logger,
            on_fallback=on_fallback,
        )
    except NoCredentialsError as exc:
        return _fallback_or_raise(
            exc,
            host_fallback,
            bucket=bucket,
            operation=operation,
            allow_host_creds=allow_host_creds,
            logger=logger,
            on_fallback=on_fallback,
        )


def _fallback_or_raise(
    exc: BaseException,
    host_fallback: Callable[[], T],
    *,
    bucket: str,
    operation: str,
    allow_host_creds: bool,
    logger: logging.Logger | None,
    on_fallback: Callable[[BaseException], None] | None,
) -> T:
    if not allow_host_creds:
        raise ScopedCredentialError(bucket, operation) from exc

    active_logger = logger or logging.getLogger(__name__)
    active_logger.warning(
        "Scoped credentials failed for %s on bucket %r; falling back to host "
        "credentials because --allow-host-creds was set.",
        operation,
        bucket,
    )
    if on_fallback is not None:
        on_fallback(exc)
    return host_fallback()
