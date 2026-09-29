"""Read, fingerprint, and publish the video sweep's durable artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from urllib.parse import urlsplit

from botocore.exceptions import ClientError


def digest(payload: dict) -> str:
    """Hash a canonical JSON document.

    Args:
        payload: JSON document.
    Returns:
        SHA-256 hexadecimal digest.
    Raises:
        ValueError: Non-finite values occur.
    """
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def file_digest(path: Path) -> str:
    """Hash a file without loading it into memory.

    Args:
        path: Existing file.
    Returns:
        SHA-256 hexadecimal digest.
    Raises:
        OSError: The file cannot be read.
    """
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _json_bytes(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, allow_nan=False).encode()


def _object(uri: str) -> tuple[str, str]:
    parsed = urlsplit(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError("Expected an exact S3 object URI")
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise ValueError("Signed or credential-bearing object URIs are unsupported")
    return parsed.netloc, parsed.path.lstrip("/")


def _client():
    from npa.clients.storage import StorageClient

    return StorageClient.from_environment().s3


def read_json(uri: str) -> dict:
    """Read an exact JSON object from S3 or a local test path.

    Args:
        uri: Object URI or local path.
    Returns:
        Parsed object.
    Raises:
        ValueError: The document is invalid.
        OSError: Local reading fails.
    """
    if uri.startswith("s3:"):
        bucket, key = _object(uri)
        body = _client().get_object(Bucket=bucket, Key=key)["Body"]
        try:
            raw = body.read()
        finally:
            body.close()
    else:
        raw = Path(uri).read_bytes()
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object")
    return result


def write_json(uri: str, payload: dict) -> None:
    """Publish a JSON artifact after its dependencies have succeeded.

    Args:
        uri: Destination object URI or local test path.
        payload: JSON object.
    Returns:
        None.
    Raises:
        ValueError: The document or URI is invalid.
        OSError: Local writing fails.
    """
    raw = _json_bytes(payload)
    if uri.startswith("s3:"):
        _write_s3_json(uri, raw)
        return
    path = Path(uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(raw)
    except FileExistsError:
        if path.read_bytes() != raw:
            raise ValueError(
                "Artifact already belongs to a different stage result; use a new run"
            ) from None


def download(uri: str, path: Path, expected_sha256: str = "") -> None:
    """Materialize and optionally verify an exact media object.

    Args:
        uri: Object URI or local test path.
        path: Destination path.
        expected_sha256: Required digest when verifying a handoff.
    Returns:
        None.
    Raises:
        ValueError: A digest does not match.
        OSError: Local copying fails.
    """
    if uri.startswith("s3:"):
        bucket, key = _object(uri)
        _client().download_file(bucket, key, str(path))
    else:
        shutil.copyfile(uri, path)
    if expected_sha256 and file_digest(path) != expected_sha256:
        raise ValueError("Media digest does not match the recorded handoff")


def upload(path: Path, uri: str) -> None:
    """Upload media without logging its path or content.

    Args:
        path: Source file.
        uri: Object URI or local test path.
    Returns:
        None.
    Raises:
        ValueError: An S3 URI is invalid.
        OSError: Local copying fails.
    """
    if uri.startswith("s3:"):
        bucket, key = _object(uri)
        _client().upload_file(str(path), bucket, key)
        return
    target = Path(uri)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)


def _write_s3_json(uri: str, raw: bytes) -> None:
    bucket, key = _object(uri)
    try:
        _client().put_object(
            Bucket=bucket,
            Key=key,
            Body=raw,
            ContentType="application/json",
            IfNoneMatch="*",
        )
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") not in {
            "PreconditionFailed",
            "412",
            "KeyAlreadyExists",
        }:
            raise
        if _json_bytes(read_json(uri)) != raw:
            raise ValueError(
                "Artifact already belongs to a different stage result; use a new run"
            ) from None
