"""Fail-closed S3 and input-file operations for immutable Isaac Jobs."""

from __future__ import annotations

import argparse
import base64
import glob
import hashlib
import os
import time
from functools import partial
from pathlib import Path
from typing import Callable, TypeVar
from urllib.parse import urlparse

import boto3
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


def _with_transport_recovery(
    operation: str,
    action: Callable[[], _ResultT],
) -> _ResultT:
    """Retry only structured transport/service failures, without a run budget."""

    attempt = 1
    while True:
        try:
            return action()
        except _TRANSPORT_EXCEPTIONS as exc:
            classification = type(exc).__name__
            status = 0
            code = ""
        except ClientError as exc:
            if not _retryable_client_error(exc):
                raise
            status, code = _structured_client_error(exc)
            classification = type(exc).__name__
        delay = _retry_delay(attempt)
        print(
            "S3_IO_HEARTBEAT "
            f"operation={operation} state=retrying attempt={attempt} "
            f"classification={classification} http_status={status} "
            f"error_code={code or '-'} next_delay_seconds={delay:g}",
            flush=True,
        )
        time.sleep(delay)
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


def upload_tree(root: Path, uri: str) -> None:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"expected s3:// prefix, got {uri!r}")
    prefix = parsed.path.lstrip("/").rstrip("/")
    count = 0
    byte_count = 0
    s3 = _s3()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            relative = path.relative_to(root)
            _with_transport_recovery(
                "upload-tree",
                partial(
                    s3.upload_file, str(path), parsed.netloc, f"{prefix}/{relative}"
                ),
            )
            count += 1
            byte_count += path.stat().st_size
            if count == 1 or count % 100 == 0:
                print(
                    "S3_IO_HEARTBEAT "
                    f"operation=upload-tree state=progress files={count} "
                    f"bytes={byte_count}",
                    flush=True,
                )
    if count == 0:
        raise RuntimeError(f"upload tree contains no files: {root}")
    print(f"UPLOADED_TREE uri={uri} files={count}")


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
