"""Fail-closed S3 and input-file operations for immutable Isaac Jobs."""

from __future__ import annotations

import argparse
import base64
import glob
import hashlib
import os
from pathlib import Path
from urllib.parse import urlparse

import boto3


def _s3():
    return boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL") or None)


def _uri(value: str) -> tuple[str, str]:
    parsed = urlparse(value)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"expected exact s3:// object URI, got {value!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def download(uri: str, destination: Path, expected_sha256: str = "") -> None:
    bucket, key = _uri(uri)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _s3().download_file(bucket, key, str(destination))
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
    _s3().upload_file(str(source), bucket, key)
    print(f"UPLOADED uri={uri} bytes={source.stat().st_size}")


def upload_tree(root: Path, uri: str) -> None:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"expected s3:// prefix, got {uri!r}")
    prefix = parsed.path.lstrip("/").rstrip("/")
    count = 0
    for path in sorted(root.rglob("*")):
        if path.is_file():
            _s3().upload_file(
                str(path), parsed.netloc, f"{prefix}/{path.relative_to(root)}"
            )
            count += 1
    if count == 0:
        raise RuntimeError(f"upload tree contains no files: {root}")
    print(f"UPLOADED_TREE uri={uri} files={count}")


def upload_training(checkpoint: Path, output_dir: Path, uri: str) -> None:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"expected s3:// prefix, got {uri!r}")
    prefix = parsed.path.lstrip("/").rstrip("/") + "/"
    s3 = _s3()
    s3.upload_file(str(checkpoint), parsed.netloc, prefix + "model_latest.pt")
    for path_text in sorted(
        glob.glob(str(output_dir / "**" / "model_*.pt"), recursive=True)
    ):
        path = Path(path_text)
        s3.upload_file(str(path), parsed.netloc, prefix + "checkpoints/" + path.name)
    optional = {
        Path("/tmp/train_full.log"): "train_full.log",
        output_dir / "applied-scenarios.json": "applied-scenarios.json",
    }
    for path, name in optional.items():
        if path.is_file():
            s3.upload_file(str(path), parsed.netloc, prefix + name)
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
