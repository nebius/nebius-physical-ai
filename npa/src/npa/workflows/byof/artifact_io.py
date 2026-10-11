"""Small, conditional artifact writers for BYOF workflow stages.

Public workbench commands hand artifacts across tool boundaries through S3.
The pipeline modules still accept local paths for hermetic developer tests, but
an ``s3://`` destination is written atomically with create-only semantics so a
retry cannot silently replace another run's evidence.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from npa.clients.storage import (
    StorageClient,
    StorageError,
    StoragePreconditionFailed,
)


class ArtifactPathError(ValueError):
    """Raised when a stage artifact destination is malformed."""


def _local_path(uri: str) -> Path | None:
    parsed = urlparse(uri)
    if parsed.scheme == "":
        return Path(uri)
    if (
        parsed.scheme == "file"
        and not parsed.netloc
        and not parsed.query
        and not parsed.fragment
    ):
        return Path(parsed.path)
    return None


def join_output_path(output_path: str, filename: str) -> str:
    """Return one artifact path below a directory-like output destination.

    Args:
        output_path: A local directory for developer use or an S3 prefix.
        filename: A single artifact filename, not a path traversal expression.

    Returns:
        The full local, ``file://``, or S3 artifact destination.

    Raises:
        ArtifactPathError: The destination or filename is not safe and explicit.
    """
    if not output_path:
        raise ArtifactPathError("output_path is required")
    if not filename or Path(filename).name != filename:
        raise ArtifactPathError(
            f"artifact filename must be a basename, got {filename!r}"
        )

    local = _local_path(output_path)
    if local is not None:
        return str(local / filename)

    parsed = urlparse(output_path)
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or not parsed.path.strip("/")
        or parsed.query
        or parsed.fragment
    ):
        raise ArtifactPathError(
            "output_path must be a local directory or explicit s3://bucket/prefix, "
            f"got {output_path!r}"
        )
    return f"s3://{parsed.netloc}/{parsed.path.strip('/')}/{filename}"


def write_bytes(
    output_path: str,
    payload: bytes,
    *,
    content_type: str,
    storage: StorageClient | None = None,
) -> str:
    """Write one exact artifact without overwriting an existing S3 object.

    Args:
        output_path: Exact local, ``file://``, or S3 object destination.
        payload: The bytes to persist.
        content_type: Media type recorded for an S3 object.
        storage: Optional already-configured storage client, chiefly for tests.

    Returns:
        The supplied destination after it has been written.

    Raises:
        ArtifactPathError: The destination is not an exact writable object path.
        StorageError: The S3 destination is malformed, unavailable, or already
            contains different bytes.
    """
    if not output_path:
        raise ArtifactPathError("output_path is required")
    local = _local_path(output_path)
    if local is not None:
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(payload)
        return output_path

    parsed = urlparse(output_path)
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or not parsed.path.strip("/")
        or parsed.path.endswith("/")
        or parsed.query
        or parsed.fragment
    ):
        raise ArtifactPathError(
            "output_path must be an exact local file or s3://bucket/key, "
            f"got {output_path!r}"
        )
    client = storage or StorageClient.from_environment()
    try:
        client.put_bytes_conditional(
            payload,
            output_path,
            if_none_match=True,
            content_type=content_type,
        )
    except StoragePreconditionFailed as exc:
        existing = client.read_bytes_with_etag(output_path)
        if existing is not None and existing[0] == payload:
            return output_path
        raise StorageError(
            f"refusing to overwrite non-identical artifact at {output_path}"
        ) from exc
    return output_path


__all__ = ["ArtifactPathError", "join_output_path", "write_bytes"]
