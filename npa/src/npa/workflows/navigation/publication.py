"""Publish immutable navigation attempts using provider-conditional S3 records."""

from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath
import re
import uuid


def _bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _record(storage, uri):
    result = storage.read_bytes_with_etag(uri)
    if result is None:
        raise ValueError(
            "navigation stage is incomplete: required publication record missing"
        )
    return json.loads(result[0])


def _validate_record(record):
    if not re.fullmatch(r"attempts/[0-9a-f]{32}", record.get("attempt", "")):
        raise ValueError("invalid immutable stage-attempt path")
    expected = record.get("files")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("publication record has no artifact manifest")
    for name, digest in expected.items():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or str(path) != name:
            raise ValueError("publication manifest contains an escaping artifact path")
        if name == "checksums.json" or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(
                "publication manifest contains an invalid digest or reserved file"
            )
    if hashlib.sha256(_bytes(expected)).hexdigest() != record.get("manifest_sha256"):
        raise ValueError("publication manifest digest mismatch")
    return expected


def _readback(storage, uri, target, expected):
    from npa.workflows.navigation.artifacts import _files

    storage.download_directory(uri, str(target))
    remote = json.loads((target / "checksums.json").read_text())
    if remote != expected or _files(target) != expected:
        raise ValueError("published bytes differ from the original local manifest")


def publish_attempt(storage, root, destination, expected) -> None:
    """Claim a stage once, write immutable attempt bytes, and complete after readback.

    Args:
        storage: StorageClient supporting provider-conditional object writes.
        root: Private local output directory with checksums.json.
        destination: Logical run-scoped S3 stage prefix.
        expected: Original local artifact hashes, retained across publication.
    Returns:
        None; completion.json points to the accepted immutable attempt.
    Raises:
        ValueError: Readback or the original claim differs from local evidence.
        StoragePreconditionFailed: Another publication has already claimed the stage.
        OSError: Storage fails; the attempt remains incomplete and cannot be reused.
    """
    destination = destination.rstrip("/")
    expected = dict(expected)
    claim = {
        "schema": "npa.navigation.publication.v1",
        "attempt": f"attempts/{uuid.uuid4().hex}",
        "files": expected,
        "manifest_sha256": hashlib.sha256(_bytes(expected)).hexdigest(),
    }
    claim_bytes = _bytes(claim)
    storage.put_bytes_conditional(
        claim_bytes, destination + "/claim.json", if_none_match=True
    )
    attempt = destination + "/" + claim["attempt"]
    _write_attempt(storage, root, attempt, expected)
    _readback(storage, attempt, root.parent / "readback", expected)
    if _record(storage, destination + "/claim.json") != claim:
        raise ValueError("publication claim changed before completion")
    storage.put_bytes_conditional(
        claim_bytes, destination + "/completion.json", if_none_match=True
    )


def materialize_attempt(storage, source, target) -> None:
    """Resolve a completed stage and check attempt bytes against its original claim.

    Args:
        storage: StorageClient used for completed records and artifact download.
        source: Logical stage prefix.
        target: Fresh private staging directory.
    Returns:
        None; the exact completed artifact bundle is written to target.
    Raises:
        ValueError: Completion, claim, path containment or artifact hashes differ.
        OSError: Storage read fails.
    """
    source = source.rstrip("/")
    completion = _record(storage, source + "/completion.json")
    expected = _validate_record(completion)
    if completion != _record(storage, source + "/claim.json"):
        raise ValueError("completion does not match the immutable publication claim")
    _readback(storage, source + "/" + completion["attempt"], target, expected)


def _write_attempt(storage, root, attempt, expected):
    for name in [*expected, "checksums.json"]:
        body = (root / name).read_bytes()
        if (
            name != "checksums.json"
            and hashlib.sha256(body).hexdigest() != expected[name]
        ):
            raise ValueError("local artifact changed after the publication claim")
        storage.put_bytes_conditional(body, f"{attempt}/{name}", if_none_match=True)
