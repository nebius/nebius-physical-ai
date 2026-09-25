"""Deliver a completed local video to an explicit S3 object and verify its bytes."""

from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import urlsplit

from boto3.exceptions import S3UploadFailedError
from botocore.exceptions import BotoCoreError, ClientError
from yaml import YAMLError

from npa.cli.path_contract import validate_write_path
from npa.clients.config import ConfigError
from npa.clients.credentials import CredentialStoreError
from npa.clients.project_credential_store import ProjectCredentialStoreError
from npa.clients.project_credentials import s3_client_for_project
from npa.errors import NpaError
from npa.lifecycle_intent import OperationIntent, operation_intent

_DELIVERY_ERRORS = (
    BotoCoreError,
    ClientError,
    S3UploadFailedError,
    NpaError,
    OSError,
    ConfigError,
    CredentialStoreError,
    ProjectCredentialStoreError,
    YAMLError,
)


def validate_video_output(output_path: str) -> tuple[str, str]:
    """Require an explicit S3 MP4 object without embedded URL credentials.

    Args:
        output_path: Destination URI, including the MP4 filename.
    Returns:
        The bucket and literal object key.
    Raises:
        ValueError: The destination is not an S3 MP4 object.
    """
    uri = validate_write_path(output_path, tool="video", required=True)
    parsed = urlsplit(uri)
    if (
        "@" in parsed.netloc
        or ":" in parsed.netloc
        or parsed.query
        or parsed.fragment
        or "?" in uri
        or "#" in uri
        or not parsed.path.lower().endswith(".mp4")
        or any(character.isspace() for character in parsed.netloc)
        or any(ord(character) < 32 for character in uri)
    ):
        raise ValueError(
            "--output-path must be an s3://bucket/key.mp4 URI without URL credentials, query or fragment"
        )
    return parsed.netloc, parsed.path[1:]


def _digest(stream):
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _deliver(source, bucket, key, project):
    with operation_intent(OperationIntent.OBSERVE):
        client = s3_client_for_project(project)
    with source.open("rb") as stream:
        expected = _digest(stream)
    client.upload_file(
        str(source),
        bucket,
        key,
        ExtraArgs={"ContentType": "video/mp4", "Metadata": {"sha256": expected[0]}},
    )
    remote = client.get_object(Bucket=bucket, Key=key)
    with remote["Body"] as stream:
        actual = _digest(stream)
    if actual != expected:
        raise ValueError(
            "S3 video verification failed: uploaded bytes differ from the local video"
        )
    with source.open("rb") as stream:
        if _digest(stream) != expected:
            raise ValueError(
                "Local video changed during S3 delivery; retry after rendering finishes"
            )
    return {
        "output_path": f"s3://{bucket}/{key}",
        "sha256": expected[0],
        "bytes": expected[1],
        "verified": True,
        "version_id": remote.get("VersionId"),
    }


def write_video_output(
    local_path: str | Path, output_path: str, *, project: str | None = None
) -> dict:
    """Upload an MP4 using external project credentials and verify a full readback.

    Args:
        local_path: Completed local MP4, retained on success and failure.
        output_path: Exact S3 MP4 destination; an existing object is replaced.
        project: NPA storage project alias, or the configured default.
    Returns:
        Destination, SHA-256, byte count, verification status and object version.
    Raises:
        ValueError: Destination, configuration, upload or verification fails.
    """
    bucket, key = validate_video_output(output_path)
    source = Path(local_path).expanduser()
    if not source.is_file() or source.suffix.lower() != ".mp4":
        raise ValueError("Video output requires a completed local MP4 file")
    try:
        return _deliver(source, bucket, key, project)
    except _DELIVERY_ERRORS as error:
        raise ValueError(
            f"S3 video delivery failed ({type(error).__name__}); local video retained. "
            "Check the selected project's storage configuration and object read/write permissions."
        ) from None
