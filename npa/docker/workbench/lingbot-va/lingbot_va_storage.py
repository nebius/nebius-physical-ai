"""Minimal S3 helper carried by the source-only LingBot-VA image.

The workflow image needs only these four object operations.  It deliberately
uses Boto3's normal provider chain instead of serialising explicit access-key
parameters into the image: the operator's runtime secret plumbing supplies the
standard credentials, while image bytes contain neither a credential value nor
an acceptance record.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import boto3
from botocore.config import Config as BotoConfig


class StorageError(RuntimeError):
    """Raised for an invalid object-store URI or unsafe returned key."""


def _split_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or parsed.query or parsed.fragment:
        raise StorageError("Expected a plain s3://bucket/key URI")
    key = parsed.path.lstrip("/")
    if not key:
        raise StorageError("Object-store URI must include a key")
    return parsed.netloc, key


def _relative_key(key: str, prefix: str) -> Path:
    if not key.startswith(prefix):
        raise StorageError("Object store returned a key outside the requested prefix")
    relative = key[len(prefix) :]
    candidate = PurePosixPath(relative)
    if (
        not relative
        or candidate.is_absolute()
        or "\\" in relative
        or "\x00" in relative
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise StorageError("Object store returned an unsafe relative key")
    return Path(*candidate.parts)


class StorageClient:
    """Small object client that defers credentials to Boto3 at runtime."""

    def __init__(self, s3) -> None:
        self._s3 = s3

    @classmethod
    def from_environment(cls) -> "StorageClient":
        endpoint = os.environ.get("AWS_ENDPOINT_URL", "") or os.environ.get(
            "NEBIUS_S3_ENDPOINT", ""
        )
        if not endpoint:
            raise StorageError("Object-store endpoint is not configured")
        return cls(
            boto3.client(
                "s3",
                endpoint_url=endpoint,
                config=BotoConfig(
                    signature_version="s3v4",
                    retries={"max_attempts": 3, "mode": "adaptive"},
                    max_pool_connections=24,
                ),
            )
        )

    def download_file(self, uri: str, destination: str) -> None:
        bucket, key = _split_uri(uri)
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._s3.download_file(bucket, key, str(target))

    def upload_file(self, source: str, uri: str) -> str:
        bucket, key = _split_uri(uri)
        self._s3.upload_file(source, bucket, key)
        return uri

    def download_directory(self, uri: str, destination: str) -> None:
        bucket, raw_prefix = _split_uri(uri)
        prefix = raw_prefix.rstrip("/") + "/"
        target_root = Path(destination)
        if target_root.exists():
            raise StorageError(
                "Refusing to merge an object-store tree into an existing path"
            )
        target_root.mkdir(parents=True)
        for page in self._s3.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=prefix
        ):
            for item in page.get("Contents", []) or []:
                key = item.get("Key")
                if not key or key.endswith("/"):
                    continue
                target = target_root / _relative_key(key, prefix)
                target.parent.mkdir(parents=True, exist_ok=True)
                self._s3.download_file(bucket, key, str(target))

    def upload_directory(self, source: str, uri: str) -> str:
        bucket, raw_prefix = _split_uri(uri)
        prefix = raw_prefix.rstrip("/") + "/"
        root = Path(source)
        if not root.is_dir():
            raise StorageError("Expected a directory to upload")
        for path in sorted(
            candidate for candidate in root.rglob("*") if candidate.is_file()
        ):
            key = prefix + path.relative_to(root).as_posix()
            self._s3.upload_file(str(path), bucket, key)
        return uri
