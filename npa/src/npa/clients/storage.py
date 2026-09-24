"""S3-compatible object storage operations for checkpoint management."""

from __future__ import annotations

import functools
import itertools
import os
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Iterator, TypeVar
from urllib.parse import urlparse

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

# Multi-file directory transfers run this many objects concurrently.
_DIRECTORY_TRANSFER_WORKERS = 8
# Total per-file multipart thread budget shared across whatever number of
# files are actually in flight at once (see ``_adaptive_transfer_config``).
# Bounds the worst case at _DIRECTORY_TRANSFER_WORKERS-many files in flight
# to _TOTAL_TRANSFER_THREAD_BUDGET total multipart threads (a deliberate,
# documented ceiling) rather than an unbounded 8 x 10 multiplication, while
# still letting a directory with very few files -- down to the common
# single-large-checkpoint case -- use close to boto3's own default per-file
# concurrency instead of being capped by an outer-pool assumption that does
# not apply when there is no other file competing for it.
_TOTAL_TRANSFER_THREAD_BUDGET = 16
# boto3's own s3transfer default; never size a single file above this.
_MAX_PER_FILE_TRANSFER_CONCURRENCY = 10

_T = TypeVar("_T")


def _adaptive_transfer_config(file_count: int) -> TransferConfig:
    """Size one directory transfer's per-file multipart thread count.

    Args:
        file_count: Total files this directory transfer will move.

    Returns:
        A :class:`TransferConfig` whose ``max_concurrency`` divides
        ``_TOTAL_TRANSFER_THREAD_BUDGET`` across however many files can
        actually run at once (``min(file_count, _DIRECTORY_TRANSFER_WORKERS)``),
        capped at boto3's own default. Files below the multipart threshold
        (most small files) are unaffected either way: only large files use
        more than one thread per transfer.
    """

    effective_workers = max(1, min(file_count, _DIRECTORY_TRANSFER_WORKERS))
    per_file = _TOTAL_TRANSFER_THREAD_BUDGET // effective_workers
    per_file = max(1, min(per_file, _MAX_PER_FILE_TRANSFER_CONCURRENCY))
    return TransferConfig(max_concurrency=per_file)


def _prepare_directory_transfers(
    entries: Iterable[_T],
) -> tuple[Iterator[_T], TransferConfig | None]:
    """Size multipart concurrency from at most one worker pool of entries."""
    remaining = iter(entries)
    initial = list(itertools.islice(remaining, _DIRECTORY_TRANSFER_WORKERS))
    transfer_config = (
        _adaptive_transfer_config(len(initial)) if len(initial) > 1 else None
    )
    return itertools.chain(initial, remaining), transfer_config


def _local_upload_entries(local_dir: str, prefix: str) -> Iterator[tuple[str, str]]:
    """Walk files without collecting the entire upload tree."""
    for root, _dirs, filenames in os.walk(local_dir):
        for filename in filenames:
            local_path = os.path.join(root, filename)
            relative_path = os.path.relpath(local_path, local_dir)
            yield local_path, prefix + Path(relative_path).as_posix()


def _directory_object_keys(client: Any, bucket: str, prefix: str) -> Iterator[str]:
    """Yield non-marker object keys one provider page at a time."""
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key != prefix and not key.endswith("/"):
                yield key


def _path_object_keys(client: Any, bucket: str, prefix: str) -> Iterator[str]:
    """Filter a broad object lookup to its exact directory boundary."""
    directory_prefix = prefix.rstrip("/") + "/" if prefix else ""
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            key = obj.get("Key")
            if not key or not key.startswith(directory_prefix):
                continue
            if key != directory_prefix and not key.endswith("/"):
                yield key


def _run_bounded(
    tasks: Iterable[Callable[[], _T]], *, max_workers: int
) -> Iterator[_T]:
    """Run zero-argument callables with a bounded number in flight at once.

    Consumes ``tasks`` lazily so a very large source (a directory walk or a
    paginated bucket listing) is never submitted to the pool in one shot.
    Yields results in completion order. The first exception raised by a
    completed task propagates from this generator; the executor still joins
    every already-running task (via its context manager) before control
    returns to the caller, so no submitted work continues in the background
    after this function returns or raises.
    """

    if max_workers <= 1:
        for task in tasks:
            yield task()
        return
    task_iter = iter(tasks)

    def _fill(in_flight: set[Future], executor: ThreadPoolExecutor) -> None:
        while len(in_flight) < max_workers:
            task = next(task_iter, None)
            if task is None:
                return
            in_flight.add(executor.submit(task))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        in_flight: set[Future] = set()
        _fill(in_flight, executor)
        while in_flight:
            done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                yield future.result()
            _fill(in_flight, executor)


class StorageError(Exception):
    pass


class StoragePreconditionFailed(StorageError):
    """A conditional object write lost its compare-and-swap race."""


def _parse_bucket_uri(uri: str) -> tuple[str, str]:
    """Parse s3://bucket/prefix into (bucket, prefix)."""
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or parsed.query or parsed.fragment:
        raise StorageError(f"Expected s3:// URI, got: {uri}")
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    return bucket, prefix


def safe_s3_tree_relative_path(key: str, prefix: str) -> Path:
    """Return a local-safe path for one object listed under an S3 prefix."""

    if not key.startswith(prefix):
        raise StorageError(
            f"Object storage returned a key outside the requested prefix: {key!r}"
        )
    relative = key[len(prefix) :]
    path = PurePosixPath(relative)
    if (
        not relative
        or path.is_absolute()
        or relative != path.as_posix()
        or "\\" in relative
        or "\x00" in relative
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise StorageError(f"Object storage returned an unsafe relative key: {key!r}")
    return Path(*path.parts)


def safe_s3_download_target(root: Path | str, key: str, prefix: str) -> Path:
    """Contain an object-tree download, including existing filesystem symlinks."""
    relative = safe_s3_tree_relative_path(key, prefix)
    base = Path(root).resolve()
    target = (base / relative).resolve()
    if not target.is_relative_to(base):
        raise StorageError("Object storage download escapes its destination")
    return target


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
                # Comfortably above the worst-case _TOTAL_TRANSFER_THREAD_BUDGET
                # (16) concurrent connections a directory transfer can open,
                # so they don't queue waiting for a free pooled connection
                # (botocore's own default is 10).
                max_pool_connections=24,
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

    def probe_list_access(self, bucket_uri: str) -> None:
        """Verify listing permission with one request, including for empty buckets.

        Args:
            bucket_uri: S3 bucket or directory URI whose listing permission to test.

        Returns:
            None. Object names and continuation tokens are discarded.

        Raises:
            StorageError: The destination is not an S3 URI.
            botocore.exceptions.BotoCoreError: The request could not complete.
            ClientError: The service rejected the listing request.
        """
        bucket, prefix = _parse_bucket_uri(bucket_uri)
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        self._s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/", MaxKeys=1)

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
        self,
        local_dir: str,
        bucket_uri: str,
        *,
        remote_prefix: str = "",
        require_empty: bool = False,
    ) -> str:
        """Upload a directory with bounded file and multipart concurrency.

        Args:
            local_dir: Directory whose files to upload.
            bucket_uri: Destination S3 directory URI.
            remote_prefix: Optional directory suffix under the destination.
            require_empty: Reject existing objects before uploading. This is a
                preflight check, not a lock against concurrent writers.

        Returns:
            The destination directory URI.

        Raises:
            StorageError: The URI is invalid or a required-empty prefix is occupied.
            botocore.exceptions.ClientError: A provider rejected an upload.
        """
        bucket, base_prefix = _parse_bucket_uri(bucket_uri)
        if remote_prefix:
            base_prefix = "/".join(
                part
                for part in (base_prefix.rstrip("/"), remote_prefix.strip("/"))
                if part
            )
        base_prefix = base_prefix.rstrip("/") + "/" if base_prefix else ""
        if require_empty:
            existing = self._s3.list_objects_v2(
                Bucket=bucket, Prefix=base_prefix, MaxKeys=1
            )
            if existing.get("Contents"):
                raise StorageError(f"Output S3 prefix must be empty: {bucket_uri}")

        files = _local_upload_entries(local_dir, base_prefix)
        self._upload_directory_entries(files, bucket)
        return f"s3://{bucket}/{base_prefix}"

    def _upload_directory_entries(
        self, files: Iterable[tuple[str, str]], bucket: str
    ) -> None:
        """Upload a stream of paths using bounded file and multipart workers."""
        files, transfer_config = _prepare_directory_transfers(files)
        options = {"Config": transfer_config} if transfer_config is not None else {}

        def _upload_one(local_path: str, s3_key: str) -> None:
            self._s3.upload_file(local_path, bucket, s3_key, **options)

        tasks = (functools.partial(_upload_one, lp, key) for lp, key in files)
        workers = _DIRECTORY_TRANSFER_WORKERS if transfer_config is not None else 1
        for _ in _run_bounded(tasks, max_workers=workers):
            pass

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
            status = int(
                exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0) or 0
            )
            if code in {
                "412",
                "PreconditionFailed",
                "ConditionalRequestConflict",
            } or status in {
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
        """Download every object under an S3 prefix into a local directory.

        Objects are downloaded concurrently (bounded by
        ``_DIRECTORY_TRANSFER_WORKERS``); each key is contained under
        ``local_dir`` by ``safe_s3_download_target`` before any bytes move.

        Args:
            bucket_uri: Source ``s3://bucket/prefix`` to download from.
            local_dir: Local directory to write into (created if absent).

        Returns:
            ``local_dir``, unchanged, for chaining.

        Raises:
            StorageError: ``bucket_uri`` is not an ``s3://`` URI, or an
                object's key would resolve outside ``local_dir``.
            botocore.exceptions.ClientError: A provider rejected a listing or
                download call.
        """

        bucket, prefix = _parse_bucket_uri(bucket_uri)
        if prefix and not prefix.endswith("/"):
            prefix += "/"

        keys = _directory_object_keys(self._s3, bucket, prefix)
        self._download_directory_keys(keys, bucket, prefix, local_dir)
        return local_dir

    def _download_directory_keys(
        self, keys: Iterable[str], bucket: str, prefix: str, local_dir: Path | str
    ) -> None:
        """Download streamed keys with adaptive, bounded multipart concurrency."""
        keys, transfer_config = _prepare_directory_transfers(keys)
        options = {"Config": transfer_config} if transfer_config is not None else {}

        def _download_one(key: str) -> None:
            target = safe_s3_download_target(local_dir, key, prefix)
            target.parent.mkdir(parents=True, exist_ok=True)
            self._s3.download_file(bucket, key, str(target), **options)

        tasks = (functools.partial(_download_one, key) for key in keys)
        workers = _DIRECTORY_TRANSFER_WORKERS if transfer_config is not None else 1
        for _ in _run_bounded(tasks, max_workers=workers):
            pass

    def download_file(self, bucket_uri: str, local_path: str) -> str:
        """Download one exact S3 object without requiring ListBucket or HEAD.

        A sibling staging file replaces the destination atomically after a
        complete transfer and declared-length check. Existing permissions
        apply before any bytes arrive; new files respect the process umask.

        Args:
            bucket_uri: Exact object URI, ``s3://bucket/key`` (no trailing slash).
            local_path: Local file path to write.

        Returns:
            The local destination path.

        Raises:
            StorageError: The URI is invalid or the declared length mismatches.
            botocore.exceptions.ClientError: The provider rejected the GET.
        """

        bucket, key = _parse_bucket_uri(bucket_uri)
        if not key or key.endswith("/"):
            raise StorageError(f"Expected an exact S3 object URI, got: {bucket_uri}")
        target = Path(local_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        response = self._s3.get_object(Bucket=bucket, Key=key)
        body = response["Body"]
        try:
            self._replace_download(
                body, target, response.get("ContentLength"), bucket_uri
            )
        finally:
            body.close()
        return str(target)

    def _replace_download(
        self, body: Any, target: Path, declared_length: object, bucket_uri: str
    ) -> None:
        """Preserve existing permissions and replace only a complete transfer."""
        try:
            existing_mode = target.stat().st_mode & 0o777
        except FileNotFoundError:
            existing_mode = None
        staging = target.with_name(f".{target.name}.{uuid.uuid4().hex}.part")
        try:
            self._stream_to_staging(
                body, staging, existing_mode, declared_length, bucket_uri
            )
            os.replace(staging, target)
        except BaseException:
            staging.unlink(missing_ok=True)
            raise

    @staticmethod
    def _stream_to_staging(
        body: Any,
        staging: Path,
        mode: int | None,
        declared_length: object,
        bucket_uri: str,
    ) -> None:
        """Write a provider response body to a new staging file.

        Args:
            body: The provider's streaming response body (already open).
            staging: Path to create; must not already exist.
            mode: Existing destination permissions, or ``None`` for a new
                file that should respect the process umask.
            declared_length: The provider's declared byte count, or ``None``
                when the provider omitted it (the check below is skipped).
            bucket_uri: Source URI, used only to name the error below.

        Returns:
            None.

        Raises:
            StorageError: The written byte count does not match
                ``declared_length``.
        """

        creation_mode = mode if mode is not None else 0o666
        fd = os.open(str(staging), os.O_WRONLY | os.O_CREAT | os.O_EXCL, creation_mode)
        written = 0
        with os.fdopen(fd, "wb") as stream:
            if mode is not None:
                os.fchmod(stream.fileno(), mode)
            for chunk in body.iter_chunks(chunk_size=8 * 1024 * 1024):
                if chunk:
                    stream.write(chunk)
                    written += len(chunk)
        if declared_length is not None and written != int(declared_length):
            raise StorageError(
                f"Short read downloading {bucket_uri}: expected "
                f"{declared_length} bytes, wrote {written}."
            )

    def download_path(self, bucket_uri: str, local_path: str) -> str:
        """Download an exact S3 object, or every object under an S3 prefix.

        Exact keys take precedence, including after an ambiguous HEAD response.
        A one-object listing preserves that fallback without scanning the tree.
        Directory transfers stream provider pages with bounded concurrency and
        a small lookahead to preserve single-file multipart throughput.

        Args:
            bucket_uri: Source ``s3://bucket/key-or-prefix``.
            local_path: Local file (exact-object case) or directory
                (tree case) to write into.

        Returns:
            The resolved single-file target, or *local_path* for a tree.

        Raises:
            StorageError: The URI or an object's destination is invalid.
            botocore.exceptions.ClientError: A required provider call failed.
        """

        bucket, prefix = _parse_bucket_uri(bucket_uri)
        dest = Path(local_path)

        target = self._download_matching_object(bucket, prefix, dest)
        if target is not None:
            return target

        prefix_dir = prefix.rstrip("/") + "/" if prefix else ""
        keys = _path_object_keys(self._s3, bucket, prefix)
        self._download_directory_keys(keys, bucket, prefix_dir, dest)
        return str(dest)

    def _download_matching_object(
        self, bucket: str, prefix: str, dest: Path
    ) -> str | None:
        """Resolve an exact key before interpreting it as a directory prefix."""
        target = None
        if prefix:
            if not prefix.endswith("/"):
                target = self._head_and_download_exact(bucket, prefix, dest)
            # HEAD is skipped for a "/"-ending prefix, and is ambiguous on a
            # 403 (forbidden HEAD, but ListBucket may still be authorized):
            # either way, fall back to the one-item listing check so a real
            # exact key is never displaced by a same-named tree.
            if target is None and self._exact_key_exists(bucket, prefix):
                target = self._download_exact(bucket, prefix, dest)
        return target

    def _download_exact(self, bucket: str, key: str, dest: Path) -> str:
        """Download one already-confirmed exact object key.

        Args:
            bucket: Bucket the object lives in.
            key: Exact, already-confirmed-to-exist object key.
            dest: Local file, or directory to place the object's basename
                into.

        Returns:
            The local path written.

        Raises:
            StorageError: *key* would resolve outside *dest*'s parent.
        """

        target = (
            safe_s3_download_target(dest, Path(key).name, "")
            if dest.exists() and dest.is_dir()
            else dest
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        self._s3.download_file(bucket, key, str(target))
        return str(target)

    def _head_and_download_exact(self, bucket: str, key: str, dest: Path) -> str | None:
        """HEAD-check one exact key and download it if present.

        Args:
            bucket: Bucket to check.
            key: Exact, non-empty object key to HEAD.
            dest: Local file, or directory to place the object's basename
                into.

        Returns:
            The local path written, or ``None`` when *key* does not exist as
            an exact object (a 404/403 HEAD response), so the caller should
            fall back to treating it as a tree prefix.

        Raises:
            botocore.exceptions.ClientError: The provider rejected the HEAD
                for a reason other than absence (404) or ambiguous access
                (403).
        """

        try:
            self._s3.head_object(Bucket=bucket, Key=key)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code not in {"404", "NoSuchKey", "NotFound", "403"}:
                raise
            return None
        return self._download_exact(bucket, key, dest)

    def _exact_key_exists(self, bucket: str, key: str) -> bool:
        """Check whether *key* itself exists as an object, in one bounded call.

        S3 lists keys in lexicographic order, so if *key* exists as an
        object it is always the first result of a listing with
        ``Prefix=key``: any other object sharing that prefix has a strictly
        longer key, which sorts after it. This answers "does the literal
        prefix also name an object" (the directory-marker case) with one
        bounded, single-item request instead of materializing the whole
        prefix's listing to check membership.

        Args:
            bucket: Bucket to check.
            key: Exact key to check for.

        Returns:
            Whether *key* exists as an object in *bucket*.
        """

        paginator = self._s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(
            Bucket=bucket, Prefix=key, PaginationConfig={"MaxItems": 1, "PageSize": 1}
        )
        for page in pages:
            contents = page.get("Contents") or []
            return bool(contents) and contents[0].get("Key") == key
        return False
