"""Fetch an explicitly selected BYOF input using external S3 configuration."""

import hashlib
import os
from pathlib import Path
import re
from urllib.parse import urlparse


def download_input(uri: str, expected_sha256: str, destination: Path) -> Path:
    """Download and hash one selected S3 object without overwriting local files.

    Args:
        uri: Explicit s3:// bucket and key, without embedded credentials.
        expected_sha256: Required complete-object checksum.
        destination: New local file inside the run's artifact directory.

    Returns:
        Verified local input path.

    Raises:
        ValueError: The URI or checksum is invalid.
        RuntimeError: Downloaded bytes do not match the declared checksum.
    """
    import boto3

    parsed = urlparse(uri)
    if (parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/")
            or parsed.query or parsed.fragment or parsed.username or parsed.password):
        raise ValueError("Input must be an explicit s3:// bucket/key URI")
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("Input requires a complete lowercase SHA256")
    client = boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL"))
    response = client.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))
    digest = hashlib.sha256()
    with response["Body"] as body, destination.open("xb") as output:
        for block in iter(lambda: body.read(1024 * 1024), b""):
            output.write(block)
            digest.update(block)
    if digest.hexdigest() != expected_sha256:
        destination.unlink()
        raise RuntimeError("BYOF input checksum mismatch")
    return destination
