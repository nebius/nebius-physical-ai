"""Read hash-bound S3 evidence and publish non-overwritable stage records."""

from __future__ import annotations

import hashlib
import json
from urllib.parse import urlsplit

from botocore.exceptions import ClientError

from npa.clients.storage import StorageClient
from npa.workflows.field_failure.contracts import _Artifact, _object_uri


def _location(uri):
    parsed = urlsplit(_object_uri(uri))
    return parsed.netloc, parsed.path[1:]


def _storage():
    return StorageClient.from_environment()


def _client():
    return _storage().s3


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _encode(payload):
    return json.dumps(
        payload, sort_keys=True, allow_nan=False, separators=(",", ":")
    ).encode()


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key in evidence")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError("non-finite JSON number in evidence")


def _read(uri, expected=None):
    bucket, key = _location(uri)
    response = _client().get_object(Bucket=bucket, Key=key)
    with response["Body"] as stream:
        data = stream.read()
    if expected is not None and _digest(data) != expected:
        raise ValueError("artifact SHA-256 mismatch")
    return json.loads(
        data, object_pairs_hook=_unique_keys, parse_constant=_invalid_constant
    ), _digest(data)


def _verify(artifact: _Artifact, prefix=None):
    if prefix is not None and not artifact.uri.startswith(prefix.rstrip("/") + "/"):
        raise ValueError("output artifact is outside its run-scoped stage prefix")
    bucket, key = _location(artifact.uri)
    response = _client().get_object(Bucket=bucket, Key=key)
    digest = hashlib.sha256()
    size = 0
    with response["Body"] as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    if not size or digest.hexdigest() != artifact.sha256:
        raise ValueError("empty artifact or SHA-256 mismatch")


def _publish(uri, payload):
    _location(uri)
    _storage().put_bytes_conditional(
        _encode(payload),
        uri,
        if_none_match=True,
        content_type="application/json",
    )


def _optional(uri):
    try:
        return _read(uri)[0]
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") in {
            "NoSuchKey",
            "404",
            "NotFound",
        }:
            return None
        raise


def _root(root, run_id):
    _object_uri(root)
    if root.rsplit("/", 1)[-1] != run_id:
        raise ValueError("output root must end with the exact run ID")
    return root
