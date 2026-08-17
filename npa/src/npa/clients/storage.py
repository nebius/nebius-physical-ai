"""S3-compatible object storage operations for checkpoint management."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError


class StorageError(Exception):
    pass


class StoragePreconditionFailed(StorageError):
    """A conditional object write lost its compare-and-swap race."""


def _parse_bucket_uri(uri: str) -> tuple[str, str]:
    """Parse s3://bucket/prefix into (bucket, prefix)."""
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise StorageError(f"Expected s3:// URI, got: {uri}")
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    return bucket, prefix


class LazyStorageClient:
    """A :class:`StorageClient` stand-in that connects on first actual use.

    ``StorageClient.from_environment`` raises when no S3 endpoint is configured, so
    a tool that accepts either ``s3://...`` or a local path cannot build one up
    front: doing so breaks local runs on machines with no object-storage
    credentials. Hold this instead and the client is built only if a remote URI is
    really touched.
    """

    def __init__(self, **kwargs: object) -> None:
        self._kwargs = kwargs
        self._client: "StorageClient | None" = None

    def resolve(self) -> "StorageClient":
        # Unsynchronized on purpose. Two threads racing here build one redundant
        # client and discard it, which costs nothing; a lock would make this object
        # uncopyable, and copying it without connecting is the point.
        if self._client is None:
            self._client = StorageClient.from_environment(**self._kwargs)  # type: ignore[arg-type]
        return self._client

    def __getattr__(self, name: str) -> object:
        # Only reached for names this class does not define, which is every
        # StorageClient method plus the `s3` property.
        #
        # Dunders are excluded deliberately: copy, pickle, and several inspection
        # paths probe for __deepcopy__, __reduce__, __getstate__ and friends, and
        # forwarding those would open a real connection just because something
        # looked at the object — the opposite of what this class is for.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return getattr(self.resolve(), name)


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

    @property
    def s3(self):
        """The underlying boto3 S3 client (endpoint already validated in __init__)."""
        return self._s3

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
            aws_access_key_id=aws_access_key_id
            or os.environ.get("AWS_ACCESS_KEY_ID", ""),
            aws_secret_access_key=aws_secret_access_key
            or os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
        )

    def list_checkpoints(self, bucket_uri: str) -> list[dict[str, str]]:
        """List checkpoint directories under the given S3 URI."""
        bucket, prefix = _parse_bucket_uri(bucket_uri)
        if prefix and not prefix.endswith("/"):
            prefix += "/"

        results: list[dict[str, str]] = []
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
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

    def read_bytes_with_etag(self, bucket_uri: str) -> tuple[bytes, str] | None:
        """Read one object and its immutable version token, or ``None`` if absent."""

        bucket, key = _parse_bucket_uri(bucket_uri)
        if not key or key.endswith("/"):
            raise StorageError(f"Expected an exact S3 object URI, got: {bucket_uri}")
        try:
            response = self._s3.get_object(Bucket=bucket, Key=key)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in {"404", "NoSuchKey", "NotFound"}:
                return None
            raise
        body = response["Body"]
        try:
            payload = body.read()
        finally:
            body.close()
        etag = str(response.get("ETag") or "").strip()
        if not etag:
            raise StorageError(f"Object storage returned no ETag for {bucket_uri}")
        return bytes(payload), etag

    def put_bytes_conditional(
        self,
        payload: bytes,
        bucket_uri: str,
        *,
        if_match: str = "",
        if_none_match: bool = False,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Atomically create or replace an object and return its new ETag.

        Exactly one of ``if_match`` and ``if_none_match`` must be selected.  The
        method intentionally exposes S3's object-level compare-and-swap rather
        than emulating it with HEAD + upload, which would leave a late writer
        able to publish over a newer recovery attempt.
        """

        if bool(if_match) == bool(if_none_match):
            raise ValueError("choose exactly one conditional object-write guard")
        bucket, key = _parse_bucket_uri(bucket_uri)
        if not key or key.endswith("/"):
            raise StorageError(f"Expected an exact S3 object URI, got: {bucket_uri}")
        kwargs: dict[str, object] = {
            "Bucket": bucket,
            "Key": key,
            "Body": payload,
            "ContentType": content_type,
        }
        if if_match:
            kwargs["IfMatch"] = if_match
        else:
            kwargs["IfNoneMatch"] = "*"
        try:
            response = self._s3.put_object(**kwargs)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            status = int(exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0) or 0)
            if code in {"412", "PreconditionFailed", "ConditionalRequestConflict"} or status in {
                409,
                412,
            }:
                raise StoragePreconditionFailed(
                    f"conditional object write was superseded for {bucket_uri}"
                ) from exc
            raise
        etag = str(response.get("ETag") or "").strip()
        if not etag:
            # S3-compatible providers are required to return an ETag for a
            # successful PutObject. Failing closed keeps the caller from doing a
            # later unguarded finalization with an unknown version token.
            raise StorageError(f"Object storage returned no ETag for {bucket_uri}")
        return etag

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

    def download_file(self, bucket_uri: str, local_path: str) -> str:
        """Download one exact S3 object without requiring ListBucket or HEAD."""

        bucket, key = _parse_bucket_uri(bucket_uri)
        if not key or key.endswith("/"):
            raise StorageError(f"Expected an exact S3 object URI, got: {bucket_uri}")
        target = Path(local_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        response = self._s3.get_object(Bucket=bucket, Key=key)
        body = response["Body"]
        try:
            with target.open("wb") as stream:
                for chunk in body.iter_chunks(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        stream.write(chunk)
        finally:
            body.close()
        return str(target)

    def download_path(self, bucket_uri: str, local_path: str) -> str:
        """Download an S3 object or prefix to a local path. Returns local path."""
        bucket, prefix = _parse_bucket_uri(bucket_uri)
        dest = Path(local_path)

        # Prefer a direct object fetch for file keys. Listing can lag briefly after
        # sibling jobs upload their result JSON.
        if prefix and not prefix.endswith("/"):
            try:
                self._s3.head_object(Bucket=bucket, Key=prefix)
            except ClientError as exc:
                code = str(exc.response.get("Error", {}).get("Code", ""))
                if code not in {"404", "NoSuchKey", "NotFound", "403"}:
                    raise
            else:
                target = (
                    dest / Path(prefix).name
                    if dest.exists() and dest.is_dir()
                    else dest
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                self._s3.download_file(bucket, prefix, str(target))
                return str(target)

        paginator = self._s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
        keys = [
            obj["Key"]
            for page in pages
            for obj in page.get("Contents", [])
            if obj.get("Key")
        ]

        if prefix in keys:
            target = (
                dest / Path(prefix).name if dest.exists() and dest.is_dir() else dest
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            self._s3.download_file(bucket, prefix, str(target))
            return str(target)

        prefix_dir = prefix.rstrip("/") + "/"
        for key in keys:
            if not key.startswith(prefix_dir):
                continue
            rel_path = key[len(prefix_dir) :]
            if not rel_path:
                continue
            target = dest / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            self._s3.download_file(bucket, key, str(target))

        return str(dest)
