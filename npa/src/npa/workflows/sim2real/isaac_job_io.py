"""Fail-closed S3 and input-file operations for immutable Isaac Jobs."""

from __future__ import annotations

import argparse
import base64
import glob
import hashlib
import os
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from threading import Event
from typing import Callable, Iterator, TypeVar
from urllib.parse import urlparse

import boto3
from boto3.exceptions import S3UploadFailedError
from botocore.exceptions import (
    ClientError,
    ConnectionClosedError,
    ConnectTimeoutError,
    EndpointConnectionError,
    ReadTimeoutError,
)


_ResultT = TypeVar("_ResultT")
_TRANSPORT_EXCEPTIONS = (
    ConnectionClosedError,
    ConnectTimeoutError,
    EndpointConnectionError,
    ReadTimeoutError,
)
_RETRYABLE_HTTP_STATUSES = frozenset({408, 425, 429})
_RETRYABLE_ERROR_CODES = frozenset(
    {
        "InternalError",
        "RequestTimeout",
        "RequestTimeoutException",
        "ServiceUnavailable",
        "SlowDown",
        "Throttling",
        "ThrottlingException",
    }
)


def _s3():
    return boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL") or None)


def _retry_delay(attempt: int) -> float:
    base = float(os.environ.get("NPA_S3_IO_RETRY_BASE_SECONDS", "2"))
    ceiling = float(os.environ.get("NPA_S3_IO_RETRY_MAX_SECONDS", "30"))
    if base <= 0 or ceiling <= 0:
        raise ValueError("S3 retry delays must be positive")
    return min(ceiling, base * (2 ** min(attempt - 1, 8)))


def _structured_client_error(exc: ClientError) -> tuple[int, str]:
    response = exc.response if isinstance(exc.response, dict) else {}
    metadata = response.get("ResponseMetadata", {})
    error = response.get("Error", {})
    status = metadata.get("HTTPStatusCode", 0)
    code = error.get("Code", "")
    return int(status) if isinstance(status, int) else 0, str(code)


def _retryable_client_error(exc: ClientError) -> bool:
    status, code = _structured_client_error(exc)
    return (
        status in _RETRYABLE_HTTP_STATUSES
        or status >= 500
        or code in _RETRYABLE_ERROR_CODES
    )


def _retry_failure(exc: BaseException) -> tuple[str, int, str] | None:
    """Classify typed SDK failures, including the managed-transfer wrapper."""
    if isinstance(exc, S3UploadFailedError):
        cause = exc.__cause__ or exc.__context__
        if cause is None:
            return None
        exc = cause
    if isinstance(exc, _TRANSPORT_EXCEPTIONS):
        return type(exc).__name__, 0, ""
    if isinstance(exc, ClientError) and _retryable_client_error(exc):
        status, code = _structured_client_error(exc)
        return type(exc).__name__, status, code
    return None


def _with_transport_recovery(
    operation: str,
    action: Callable[[], _ResultT],
    cancelled: Event | None = None,
) -> _ResultT:
    """Retry only structured transport/service failures, without a run budget."""

    attempt = 1
    while True:
        if cancelled is not None and cancelled.is_set():
            raise CancelledError("another upload failed")
        try:
            return action()
        except (*_TRANSPORT_EXCEPTIONS, ClientError, S3UploadFailedError) as exc:
            failure = _retry_failure(exc)
            if failure is None:
                raise
            classification, status, code = failure
        delay = _retry_delay(attempt)
        print(
            "S3_IO_HEARTBEAT "
            f"operation={operation} state=retrying attempt={attempt} "
            f"classification={classification} http_status={status} "
            f"error_code={code or '-'} next_delay_seconds={delay:g}",
            flush=True,
        )
        if cancelled is None:
            time.sleep(delay)
        elif cancelled.wait(delay):
            raise CancelledError("another upload failed")
        attempt += 1


def _uri(value: str) -> tuple[str, str]:
    parsed = urlparse(value)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"expected exact s3:// object URI, got {value!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def download(uri: str, destination: Path, expected_sha256: str = "") -> None:
    bucket, key = _uri(uri)
    destination.parent.mkdir(parents=True, exist_ok=True)
    s3 = _s3()
    _with_transport_recovery(
        "download",
        lambda: s3.download_file(bucket, key, str(destination)),
    )
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise RuntimeError(
            f"download SHA mismatch: expected={expected_sha256} actual={digest}"
        )
    print(f"DOWNLOADED uri={uri} sha256={digest} bytes={destination.stat().st_size}")


def upload(source: Path, uri: str) -> None:
    if not source.is_file() or source.stat().st_size == 0:
        raise RuntimeError(f"upload source missing/empty: {source}")
    bucket, key = _uri(uri)
    s3 = _s3()
    _with_transport_recovery(
        "upload",
        lambda: s3.upload_file(str(source), bucket, key),
    )
    print(f"UPLOADED uri={uri} bytes={source.stat().st_size}")


def _upload_paths(
    paths: list[Path], action: Callable[[Path], None], max_workers: int
) -> Iterator[Path]:
    """Stop sibling retries when a file encounters a terminal failure."""
    if max_workers == 1:
        for path in paths:
            _with_transport_recovery("upload-tree", partial(action, path))
            yield path
        return
    cancelled = Event()

    def upload_one(path: Path) -> Path | None:
        try:
            _with_transport_recovery("upload-tree", partial(action, path), cancelled)
            return path
        except CancelledError:
            return None
        except BaseException:
            cancelled.set()
            raise

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(upload_one, path) for path in paths]
        try:
            for future in as_completed(futures):
                path = future.result()
                if path is not None:
                    yield path
        finally:
            cancelled.set()
            for future in futures:
                future.cancel()


def _upload_tree_inputs(
    root: Path, uri: str, max_workers: int
) -> tuple[list[Path], str, str]:
    """Validate the destination and enumerate the complete artifact tree."""
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"expected s3:// prefix, got {uri!r}")
    if max_workers < 1:
        raise ValueError("upload workers must be positive")
    paths = [path for path in sorted(root.rglob("*")) if path.is_file()]
    if not paths:
        raise RuntimeError(f"upload tree contains no files: {root}")
    return paths, parsed.netloc, parsed.path.lstrip("/").rstrip("/")


def upload_tree(root: Path, uri: str, *, max_workers: int = 1) -> None:
    """Upload every file, retrying transport failures and rejecting partial trees.

    Args:
        root: Directory containing the complete artifact tree.
        uri: Destination S3 prefix.
        max_workers: Concurrent file uploads; one preserves serial ordering.

    Returns:
        None.

    Raises:
        ValueError: The URI or worker count is invalid.
        RuntimeError: The tree contains no files.
        ClientError: Storage rejects an upload without a retryable response.
        S3UploadFailedError: A managed transfer rejects an upload.
    """
    paths, bucket, prefix = _upload_tree_inputs(root, uri, max_workers)
    count = 0
    byte_count = 0
    s3 = _s3()

    def upload_one(path: Path) -> None:
        key = "/".join(
            part for part in (prefix, path.relative_to(root).as_posix()) if part
        )
        s3.upload_file(str(path), bucket, key)

    for path in _upload_paths(paths, upload_one, max_workers):
        count += 1
        byte_count += path.stat().st_size
        if count == 1 or count % 100 == 0:
            print(
                "S3_IO_HEARTBEAT "
                f"operation=upload-tree state=progress files={count} bytes={byte_count}",
                flush=True,
            )
    print(f"UPLOADED_TREE uri={uri} files={count}")


def upload_capture(
    root: Path, tree_uri: str, metadata: Path, metadata_uri: str
) -> None:
    """Publish native camera files before their completion metadata.

    Args:
        root: Camera and point-cloud artifact directory.
        tree_uri: Destination prefix; empty disables camera publication.
        metadata: Completed native rollout or evaluation JSON.
        metadata_uri: Exact destination object URI.

    Returns:
        None.

    Raises:
        RuntimeError: A requested camera tree is empty or metadata is missing.
        ClientError: Storage rejects any artifact or metadata upload.
        S3UploadFailedError: A managed transfer rejects an upload.
    """
    if tree_uri:
        upload_tree(root, tree_uri, max_workers=16)
    upload(metadata, metadata_uri)


def upload_training(checkpoint: Path, output_dir: Path, uri: str) -> None:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"expected s3:// prefix, got {uri!r}")
    prefix = parsed.path.lstrip("/").rstrip("/") + "/"
    s3 = _s3()
    _with_transport_recovery(
        "upload-training",
        lambda: s3.upload_file(
            str(checkpoint), parsed.netloc, prefix + "model_latest.pt"
        ),
    )
    for path_text in sorted(
        glob.glob(str(output_dir / "**" / "model_*.pt"), recursive=True)
    ):
        path = Path(path_text)
        _with_transport_recovery(
            "upload-training",
            partial(
                s3.upload_file,
                str(path),
                parsed.netloc,
                prefix + "checkpoints/" + path.name,
            ),
        )
    optional = {
        Path("/tmp/train_full.log"): "train_full.log",
        output_dir / "applied-scenarios.json": "applied-scenarios.json",
    }
    for path, name in optional.items():
        if path.is_file():
            _with_transport_recovery(
                "upload-training",
                partial(s3.upload_file, str(path), parsed.netloc, prefix + name),
            )
    print(f"UPLOADED_TRAINING uri={uri} checkpoint={checkpoint.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    get = sub.add_parser("download")
    get.add_argument("--uri", required=True)
    get.add_argument("--destination", type=Path, required=True)
    get.add_argument("--sha256", default="")
    put = sub.add_parser("upload")
    put.add_argument("--source", type=Path, required=True)
    put.add_argument("--uri", required=True)
    tree = sub.add_parser("upload-tree")
    tree.add_argument("--root", type=Path, required=True)
    tree.add_argument("--uri", required=True)
    write = sub.add_parser("write-base64")
    write.add_argument("--payload", required=True)
    write.add_argument("--destination", type=Path, required=True)
    training = sub.add_parser("upload-training")
    training.add_argument("--checkpoint", type=Path, required=True)
    training.add_argument("--output-dir", type=Path, required=True)
    training.add_argument("--uri", required=True)
    args = parser.parse_args()
    if args.command == "download":
        download(args.uri, args.destination, args.sha256)
    elif args.command == "upload":
        upload(args.source, args.uri)
    elif args.command == "upload-tree":
        upload_tree(args.root, args.uri)
    elif args.command == "write-base64":
        args.destination.parent.mkdir(parents=True, exist_ok=True)
        args.destination.write_bytes(base64.b64decode(args.payload, validate=True))
    else:
        upload_training(args.checkpoint, args.output_dir, args.uri)


if __name__ == "__main__":
    main()
