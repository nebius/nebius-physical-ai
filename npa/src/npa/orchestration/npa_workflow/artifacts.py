"""Artifact URI checks for NPA workflow steps."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from npa.orchestration.npa_workflow.errors import NpaWorkflowError


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise NpaWorkflowError(f"expected s3:// URI, got {uri!r}")
    key = parsed.path.lstrip("/")
    if not key:
        raise NpaWorkflowError(f"S3 URI missing key: {uri!r}")
    return parsed.netloc, key


def s3_object_exists(uri: str, *, checker: Any | None = None) -> bool:
    bucket, key = parse_s3_uri(uri)
    if checker is not None:
        return bool(checker(bucket, key))
    from botocore.exceptions import ClientError

    from npa.clients.storage import StorageClient

    client = StorageClient.from_environment()
    try:
        client._s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def require_input_artifacts(uris: list[str], *, checker: Any | None = None) -> None:
    missing = [uri for uri in uris if uri and not s3_object_exists(uri, checker=checker)]
    if missing:
        joined = ", ".join(missing)
        raise NpaWorkflowError(f"missing required input artifact(s): {joined}")
