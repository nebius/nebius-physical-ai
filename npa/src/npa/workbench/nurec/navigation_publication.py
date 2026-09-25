"""Seal navigation artifacts in permanently claimed, immutable publication prefixes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from urllib.parse import urlparse

from botocore.exceptions import ClientError

from npa.clients.storage import StorageClient, StorageError, StoragePreconditionFailed

_CLAIM = ".npa-navigation-claim.json"
_SEAL = ".npa-navigation-complete.json"
_OCCUPIED = "navigation destination is occupied or claimed; use a fresh output prefix"


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(directory: Path) -> dict:
    members = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.is_symlink() or path.name in {_CLAIM, _SEAL}:
            raise ValueError("publication accepts fresh regular artifact files only")
        members[path.name] = {"sha256": _digest(path), "bytes": path.stat().st_size}
    if not members:
        raise ValueError("cannot publish an empty navigation result")
    return members


def _json_bytes(payload: dict) -> bytes:
    return (json.dumps(payload, sort_keys=True, allow_nan=False) + "\n").encode()


def _records(members: dict) -> tuple[bytes, bytes]:
    claim = _json_bytes({"schema": "npa.nurec.navigation_claim.v1", "members": members})
    seal = _json_bytes(
        {
            "schema": "npa.nurec.navigation_publication.v1",
            "claim_sha256": hashlib.sha256(claim).hexdigest(),
            "members": members,
        }
    )
    return claim, seal


def _s3_location(destination: str) -> tuple[str, str, str]:
    parsed = urlparse(destination)
    if parsed.scheme != "s3" or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("publication requires a plain s3:// bucket/prefix")
    prefix = parsed.path.removeprefix("/").rstrip("/")
    if not prefix or any(part in {"", ".", ".."} for part in prefix.split("/")):
        raise ValueError("publication requires a fresh non-root S3 prefix")
    return parsed.netloc, prefix + "/", destination.rstrip("/")


def _occupied(client, bucket: str, prefix: str, *, claimed=False) -> bool:
    pages = client.s3.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=prefix
    )
    return any(
        not claimed or item["Key"] != prefix + _CLAIM
        for page in pages
        for item in page.get("Contents", [])
    )


def _claim(client, destination: str, record: bytes) -> tuple[str, str]:
    bucket, prefix, base = _s3_location(destination)
    if _occupied(client, bucket, prefix):
        raise ValueError(_OCCUPIED)
    try:
        client.put_bytes_conditional(record, base + "/" + _CLAIM, if_none_match=True)
    except StoragePreconditionFailed as exc:
        raise ValueError(_OCCUPIED) from exc
    # Retain claims after ambiguous writes or failures; retries use a new prefix.
    if _occupied(client, bucket, prefix, claimed=True):
        raise ValueError(_OCCUPIED)
    return bucket, prefix


def _put_artifact(client, bucket: str, key: str, path: Path) -> None:
    try:
        with path.open("rb") as stream:
            result = client.s3.put_object(
                Bucket=bucket, Key=key, Body=stream, IfNoneMatch="*"
            )
    except ClientError as exc:
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        code = exc.response.get("Error", {}).get("Code")
        if status in {409, 412} or code in {
            "PreconditionFailed",
            "ConditionalRequestConflict",
        }:
            raise ValueError(_OCCUPIED) from exc
        raise
    if not result.get("ETag"):
        raise StorageError("artifact publication returned no ETag; use a fresh prefix")


def publish_immutable(directory: Path, destination: str) -> None:
    """Claim a fresh destination and seal artifacts after conditional publication.

    Args:
        directory: Private directory containing completed regular artifacts.
        destination: New local directory or immutable run-scoped S3 prefix.
    Returns:
        None; a completion seal binds all published artifact bytes.
    Raises:
        ValueError: Destination is occupied or publication inputs are invalid.
        StorageError: Storage cannot prove conditional publication.
        ClientError: The S3 provider rejects an artifact write.
        OSError: Local publication fails; its directory remains permanently claimed.
    """
    members = _inventory(directory)
    claim, seal = _records(members)
    if destination.startswith("s3://"):
        client = StorageClient.from_environment()
        bucket, prefix = _claim(client, destination, claim)
        for name in members:
            _put_artifact(client, bucket, prefix + name, directory / name)
        client.put_bytes_conditional(
            seal, destination.rstrip("/") + "/" + _SEAL, if_none_match=True
        )
    elif not destination or "://" in destination:
        raise ValueError("output-path must be a new local directory or S3 prefix")
    else:
        output = Path(destination)
        output.mkdir(parents=True, exist_ok=False)
        (output / _CLAIM).write_bytes(claim)
        for name in members:
            with (
                (directory / name).open("rb") as source,
                (output / name).open("xb") as target,
            ):
                shutil.copyfileobj(source, target)
        (output / _SEAL).write_bytes(seal)


def verify_publication(root: Path) -> None:
    """Require an intact completion seal before consuming navigation artifacts.

    Args:
        root: Contained snapshot of a published navigation result.
    Returns:
        None.
    Raises:
        ValueError: Publication is partial, changed, or has extra members.
        OSError: Artifact bytes cannot be read.
    """
    from npa.workbench.nurec.navigation_assets import contained_file

    claim_path = contained_file(root, _CLAIM)
    seal = json.loads(contained_file(root, _SEAL).read_text())
    claim = json.loads(claim_path.read_text())
    if not isinstance(seal, dict) or not isinstance(claim, dict):
        raise ValueError("navigation publication records must be JSON objects")
    if seal.get("schema") != "npa.nurec.navigation_publication.v1":
        raise ValueError("navigation publication seal schema is invalid")
    members = seal.get("members")
    if (
        not isinstance(members, dict)
        or not members
        or claim.get("schema") != "npa.nurec.navigation_claim.v1"
        or members != claim.get("members")
        or seal.get("claim_sha256") != _digest(claim_path)
    ):
        raise ValueError("navigation publication seal differs from its permanent claim")
    if set(p.name for p in root.iterdir()) != {*members, _CLAIM, _SEAL}:
        raise ValueError("navigation publication has missing or extra members")
    for name, record in members.items():
        path = contained_file(root, name)
        if path.stat().st_size != record["bytes"] or _digest(path) != record["sha256"]:
            raise ValueError("navigation publication artifact differs from its seal")
