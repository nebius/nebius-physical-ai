"""Stage contained S3 inputs and atomically publish immutable RGB-D manifests."""

from __future__ import annotations

import copy
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from npa.clients.storage import StorageClient

from .contract import _contained, _json_bytes, _read_json, _relative, validate_request
from .dataset import _validate_header, _verify_files, validate_dataset
from .fixture import write_fixture


def _s3_uri(uri):
    parsed = urlsplit(uri)
    if parsed.scheme != "s3" or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError(
            "expected an S3 object or run-scoped prefix, without query or fragment"
        )
    if any(character in parsed.netloc for character in "@:%\\"):
        raise ValueError("invalid S3 bucket")
    _relative(parsed.path.removeprefix("/").removesuffix("/"))
    return uri.rstrip("/")


def _download_files(storage, prefix, root, files):
    for name in sorted(files):
        target = _contained(root, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        storage.download_file(prefix + "/" + name, str(target))
    _verify_files(root, files)


def _acquire(input_path, root, storage=None):
    root.mkdir(parents=True, exist_ok=True)
    if input_path == "procedural://four-camera-room":
        return write_fixture(root), "procedural-room"
    uri = _s3_uri(input_path)
    if not uri.endswith("/request.json"):
        raise ValueError("input-path must name an S3 request.json bundle manifest")
    storage = storage or StorageClient.from_environment()
    storage.download_file(uri, str(root / "request.json"))
    request = validate_request(_read_json(root / "request.json"))
    _download_files(storage, uri.rsplit("/", 1)[0], root, request["files"])
    return request, "supplied-usd"


def _publish(root, manifest, output_path, storage=None):
    destination = _s3_uri(output_path)
    validate_dataset(root, manifest)
    storage = storage or StorageClient.from_environment()
    commit_uri = destination + "/manifest.json"
    if storage.read_bytes_with_etag(commit_uri) is not None:
        raise ValueError(
            "output already contains a committed capture; use a fresh run prefix"
        )
    # Concurrent or retried writers cannot mutate bytes referenced by a committed manifest.
    attempt = "captures/" + uuid4().hex
    published = copy.deepcopy(manifest)
    published["files"] = {
        attempt + "/" + name: digest for name, digest in manifest["files"].items()
    }
    for frame in published["frames"]:
        for view in frame["views"]:
            view["artifacts"] = {
                kind: attempt + "/" + name for kind, name in view["artifacts"].items()
            }
    for name in sorted(manifest["files"]):
        storage.upload_file(
            str(_contained(root, name)), destination + "/" + attempt + "/" + name
        )
    storage.put_bytes_conditional(
        _json_bytes(published),
        commit_uri,
        if_none_match=True,
        content_type="application/json",
    )


def validate_s3(input_path, output_path, root, storage=None):
    """Download and decode committed capture bytes, then publish a validation report.

    Args:
        input_path: S3 URI of the capture manifest.json.
        output_path: S3 URI for a separate validation.json object.
        root: Private empty local staging directory.
        storage: Optional configured StorageClient for dependency injection.

    Returns:
        Measured validation report, bound to the downloaded manifest SHA256.

    Raises:
        ValueError: Any dataset or path contract fails.
        StorageError: Storage operations fail, including an existing report.
        OSError: Downloaded data cannot be decoded.
    """
    from .contract import _sha256

    source, destination = _s3_uri(input_path), _s3_uri(output_path)
    if source == destination or not source.endswith("/manifest.json"):
        raise ValueError("validation requires manifest.json and a separate report URI")
    storage = storage or StorageClient.from_environment()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    storage.download_file(source, str(root / "manifest.json"))
    manifest = _read_json(root / "manifest.json")
    _validate_header(manifest)
    _download_files(storage, source.rsplit("/", 1)[0], root, manifest["files"])
    report = validate_dataset(root, manifest)
    report["manifest_sha256"] = _sha256(root / "manifest.json")
    storage.put_bytes_conditional(
        _json_bytes(report),
        destination,
        if_none_match=True,
        content_type="application/json",
    )
    return report
