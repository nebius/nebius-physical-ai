"""S3-compatible object storage operations for checkpoint management."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.config import Config as BotoConfig


class StorageError(Exception):
    pass


def _parse_bucket_uri(uri: str) -> tuple[str, str]:
    """Parse s3://bucket/prefix into (bucket, prefix)."""
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise StorageError(f"Expected s3:// URI, got: {uri}")
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    return bucket, prefix


class StorageClient:
    def __init__(
        self,
        *,
        endpoint_url: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
    ) -> None:
        if not endpoint_url:
            raise StorageError(
                "Storage endpoint URL is not configured. "
                "Set AWS_ENDPOINT_URL or storage.endpoint_url in ~/.npa/config.yaml"
            )
        self._s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id or None,
            aws_secret_access_key=aws_secret_access_key or None,
            config=BotoConfig(
                signature_version="s3v4",
                retries={"max_attempts": 3, "mode": "adaptive"},
            ),
        )

    @classmethod
    def from_environment(
        cls,
        *,
        endpoint_url: str = "",
        aws_access_key_id: str = "",
        aws_secret_access_key: str = "",
    ) -> "StorageClient":
        """Build a client from explicit values with environment fallbacks."""
        return cls(
            endpoint_url=(
                endpoint_url
                or os.environ.get("AWS_ENDPOINT_URL", "")
                or os.environ.get("NEBIUS_S3_ENDPOINT", "")
            ),
            aws_access_key_id=aws_access_key_id or os.environ.get("AWS_ACCESS_KEY_ID", ""),
            aws_secret_access_key=aws_secret_access_key or os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
        )

    def list_checkpoints(self, bucket_uri: str) -> list[dict[str, str]]:
        """List checkpoint directories under the given S3 URI."""
        bucket, prefix = _parse_bucket_uri(bucket_uri)
        if prefix and not prefix.endswith("/"):
            prefix += "/"

        results: list[dict[str, str]] = []
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(
            Bucket=bucket, Prefix=prefix, Delimiter="/"
        ):
            for cp in page.get("CommonPrefixes", []):
                p = cp["Prefix"]
                name = p.rstrip("/").rsplit("/", 1)[-1]
                results.append({"name": name, "uri": f"s3://{bucket}/{p}"})
        return results

    def upload_directory(
        self, local_dir: str, bucket_uri: str, *, remote_prefix: str = ""
    ) -> str:
        """Upload a local directory to S3. Returns the destination URI."""
        import os

        bucket, base_prefix = _parse_bucket_uri(bucket_uri)
        if remote_prefix:
            base_prefix = base_prefix.rstrip("/") + "/" + remote_prefix.strip("/")
        base_prefix = base_prefix.rstrip("/") + "/"

        for root, _dirs, files in os.walk(local_dir):
            for fname in files:
                local_path = os.path.join(root, fname)
                rel_path = os.path.relpath(local_path, local_dir)
                s3_key = base_prefix + rel_path
                self._s3.upload_file(local_path, bucket, s3_key)

        return f"s3://{bucket}/{base_prefix}"

    def upload_file(self, local_file: str, bucket_uri: str) -> str:
        """Upload a local file to S3. Returns the destination URI."""
        bucket, key = _parse_bucket_uri(bucket_uri)
        local_path = Path(local_file)
        if not key or key.endswith("/"):
            key = key + local_path.name
        self._s3.upload_file(str(local_path), bucket, key)
        return f"s3://{bucket}/{key}"

    def upload_path(self, local_path: str, bucket_uri: str) -> str:
        """Upload a local file or directory to S3. Returns the destination URI."""
        if Path(local_path).is_dir():
            return self.upload_directory(local_path, bucket_uri)
        return self.upload_file(local_path, bucket_uri)

    def download_directory(self, bucket_uri: str, local_dir: str) -> str:
        """Download an S3 prefix to a local directory. Returns local path."""
        import os

        bucket, prefix = _parse_bucket_uri(bucket_uri)
        if prefix and not prefix.endswith("/"):
            prefix += "/"

        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                rel_path = key[len(prefix) :]
                if not rel_path:
                    continue
                local_path = os.path.join(local_dir, rel_path)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                self._s3.download_file(bucket, key, local_path)

        return local_dir

    def download_path(self, bucket_uri: str, local_path: str) -> str:
        """Download an S3 object or prefix to a local path. Returns local path."""
        bucket, prefix = _parse_bucket_uri(bucket_uri)
        dest = Path(local_path)

        paginator = self._s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
        keys = [
            obj["Key"]
            for page in pages
            for obj in page.get("Contents", [])
            if obj.get("Key")
        ]

        if prefix in keys:
            target = dest / Path(prefix).name if dest.exists() and dest.is_dir() else dest
            target.parent.mkdir(parents=True, exist_ok=True)
            self._s3.download_file(bucket, prefix, str(target))
            return str(target)

        prefix_dir = prefix.rstrip("/") + "/"
        for key in keys:
            if not key.startswith(prefix_dir):
                continue
            rel_path = key[len(prefix_dir):]
            if not rel_path:
                continue
            target = dest / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            self._s3.download_file(bucket, key, str(target))

        return str(dest)
