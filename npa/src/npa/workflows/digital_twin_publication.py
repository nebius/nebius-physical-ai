"""Recover digital-twin publication by selecting only complete, verified S3 attempts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import uuid

from npa.clients.storage import StorageClient, StoragePreconditionFailed
from npa.workflows.navigation import artifacts, publication


def _read(storage, uri):
    result = storage.read_bytes_with_etag(uri)
    if result is None:
        return None
    record = json.loads(result[0])
    if not isinstance(record, dict):
        raise ValueError("Digital-twin publication requires an object record")
    return record


def _write_identical(storage, uri, record):
    body = publication._bytes(record)
    try:
        storage.put_bytes_conditional(body, uri, if_none_match=True)
    except StoragePreconditionFailed:
        if publication._bytes(_read(storage, uri)) != body:
            raise ValueError(
                "Digital-twin publication conflicts with existing evidence"
            )


def _bind_request(storage, destination, request):
    if _read(storage, destination + "/render-request.json") is None:
        claim = _read(storage, destination + "/claim.json")
        if claim is not None:
            _verify_attempt(storage, destination, claim, request)
    _write_identical(storage, destination + "/render-request.json", request)


def _verify_render(root, request):
    from npa.workflows.digital_twin_render import verify_render

    record = verify_render(root)
    actual = {
        "schema": request["schema"],
        "scene_id": record.get("scene_id", "factory-cell"),
        "backend": record["backend"],
        "views": record["frame_count"],
        "samples": record["samples"],
        "blender_archive_sha256": record["blender_archive_sha256"],
        "scene_sources_sha256": record.get(
            "scene_sources_sha256",
            {"digital_twin_scene.py": record.get("scene_script_sha256")},
        ),
    }
    if publication._bytes(actual) != publication._bytes(request):
        raise ValueError(
            "Published render differs from the requested scene or settings"
        )
    if not (root / "index.html").is_file():
        raise ValueError("Published render is missing its HTML viewer")


def _verify_attempt(storage, destination, claim, request):
    if claim.get("schema") != "npa.navigation.publication.v1":
        raise ValueError("Unrecognized digital-twin publication schema")
    expected = publication._validate_record(claim)
    with tempfile.TemporaryDirectory(prefix="npa-twin-readback-") as temporary:
        target = Path(temporary) / "rendered"
        publication._readback(
            storage, destination + "/" + claim["attempt"], target, expected
        )
        _verify_render(target, request)


def _recover(storage, destination, request):
    claim = _read(storage, destination + "/claim.json")
    completion = _read(storage, destination + "/completion.json")
    if claim is None:
        if completion is not None:
            raise ValueError("Digital-twin completion has no publication claim")
        return False
    if completion is not None and completion != claim:
        raise ValueError("Digital-twin completion differs from its publication claim")
    _verify_attempt(storage, destination, claim, request)
    _complete(storage, destination, claim)
    return True


def _complete(storage, destination, claim):
    if _read(storage, destination + "/claim.json") != claim:
        raise ValueError("Digital-twin claim changed during verification")
    _write_identical(storage, destination + "/completion.json", claim)


def recover(destination: str, request: dict) -> bool:
    """Reuse a verified render or finish its interrupted completion record.

    Args:
        destination: Logical S3 output prefix or local output directory.
        request: Exact scene, backend, quality and source identity requested.
    Returns:
        True when matching completed artifacts exist; otherwise False.
    Raises:
        ValueError: Existing evidence is corrupt or belongs to another request.
        OSError: Retained artifacts cannot be read.
        StorageError: Object storage rejects the request.
    """
    if not destination.startswith("s3:"):
        if not destination or "://" in destination:
            raise ValueError("output must be a local directory or S3 prefix")
        if not Path(destination).exists():
            return False
        with tempfile.TemporaryDirectory(prefix="npa-twin-reuse-") as temporary:
            root = artifacts.materialize(destination, Path(temporary) / "rendered")
            _verify_render(root, request)
        return True
    artifacts._validate_uri(destination)
    destination = destination.rstrip("/")
    storage = StorageClient.from_environment()
    _bind_request(storage, destination, request)
    return _recover(storage, destination, request)


def publish(root: Path, destination: str, request: dict) -> None:
    """Select one verified immutable render attempt without claiming partial uploads.

    Args:
        root: Completed local render bundle.
        destination: Logical S3 output prefix or fresh local directory.
        request: Exact scene, backend, quality and source identity requested.
    Returns:
        None; the standard completion receipt resolves the selected attempt.
    Raises:
        ValueError: Existing evidence conflicts or readback fails verification.
        OSError: Local artifacts cannot be read or written.
        StorageError: Object storage rejects the request.
    """
    _verify_render(root, request)
    if not destination.startswith("s3:"):
        artifacts.publish(root, destination)
        return
    artifacts._validate_uri(destination)
    destination = destination.rstrip("/")
    storage = StorageClient.from_environment()
    _bind_request(storage, destination, request)
    if _recover(storage, destination, request):
        return
    claim = _upload_attempt(storage, root, destination)
    _verify_attempt(storage, destination, claim, request)
    _select_attempt(storage, destination, claim, request)


def _upload_attempt(storage, root, destination):
    expected = artifacts._files(root)
    artifacts.write_json(root / "checksums.json", expected)
    claim = {
        "schema": "npa.navigation.publication.v1",
        "attempt": f"attempts/{uuid.uuid4().hex}",
        "files": expected,
        "manifest_sha256": hashlib.sha256(publication._bytes(expected)).hexdigest(),
    }
    publication._write_attempt(
        storage, root, destination + "/" + claim["attempt"], expected
    )
    return claim


def _select_attempt(storage, destination, claim, request):
    # A stable claim must never point at a partial upload: another worker may
    # need to complete it after this process has lost its temporary directory.
    try:
        storage.put_bytes_conditional(
            publication._bytes(claim), destination + "/claim.json", if_none_match=True
        )
    except StoragePreconditionFailed:
        if not _recover(storage, destination, request):
            raise ValueError("Concurrent digital-twin publication has no claim")
        return
    _complete(storage, destination, claim)
