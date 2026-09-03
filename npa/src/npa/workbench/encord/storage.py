"""Shared S3 JSON reader/writer for the Encord tool's receipts, manifests, labels.

Every durable artifact lives in object storage: the CLI path contract rejects
local paths, and workflow stages hand artifacts to each other by s3:// URI.
There is deliberately no local-file branch here — tests inject a storage client
that captures uploads instead.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from npa.clients.storage import StorageError, parse_bucket_uri
from npa.workbench.encord.schemas import EncordToolError


def artifact_uri_for(output_path: str, filename: str) -> str:
    """The exact artifact URI a receipt-style --output-path resolves to.

    A ``.json`` output path is the artifact itself; anything else is a prefix
    the artifact lands under. (Pull's manifest helper is deliberately not this:
    its output path is always a directory prefix holding media/ and items/.)
    """

    if output_path.endswith(".json"):
        return output_path
    return output_path.rstrip("/") + f"/{filename}"


def error_text(run_error: Exception | None) -> str:
    """The durable artifact's ``error`` field for a mid-run exception."""

    return f"{type(run_error).__name__}: {run_error}" if run_error else ""


def finalize_artifact(
    model: Any,
    *,
    result_uri: str,
    filename: str,
    storage_client: Any,
    run_error: Exception | None,
    failure_prefix: str,
    artifact_noun: str = "Receipt",
) -> None:
    """Write the durable artifact, then re-raise a mid-run failure.

    This is the cross-verb contract SKILL.md advertises: the artifact lands
    before any failure exit, and a crash after Encord was mutated is recorded
    in the artifact's ``error`` field. Verb-specific post-write checks (unit
    errors, zero selections, ...) stay at the call sites.
    """

    write_json(
        model.model_dump(by_alias=True),
        result_uri=result_uri,
        filename=filename,
        storage_client=storage_client,
    )
    if run_error is not None:
        raise EncordToolError(
            f"{failure_prefix}: {model.error}. {artifact_noun} written to {result_uri}."
        ) from run_error


def object_location(
    uri: str,
    *,
    error_type: type[Exception] = EncordToolError,
    require_key: bool = False,
) -> tuple[str, str]:
    """(bucket, key) of an s3:// URI, or the caller's domain error.

    ``require_key`` rejects a bare bucket or prefix when the caller needs one
    exact object (a media file to copy), not a place to write under.
    """

    try:
        bucket, key = parse_bucket_uri(uri)
    except StorageError as exc:
        raise error_type(
            f"Encord artifacts live in object storage; expected an s3:// URI, got {uri!r}."
        ) from exc
    if require_key and not (bucket and key):
        raise error_type(f"expected an exact s3:// object URI, got: {uri}")
    return bucket, key


def read_json(uri: str, *, storage_client: Any) -> dict[str, Any]:
    """Read a JSON document from an s3:// URI."""

    bucket, key = object_location(uri)
    body = storage_client.s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return json.loads(body)


def write_json(
    payload: dict[str, Any],
    *,
    result_uri: str,
    filename: str,
    storage_client: Any,
) -> str:
    """Write ``payload`` to an s3:// URI; return the written URI."""

    object_location(result_uri)
    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.TemporaryDirectory(prefix="npa-encord-") as tmp:
        local_path = Path(tmp) / filename
        local_path.write_text(body, encoding="utf-8")
        return storage_client.upload_file(str(local_path), result_uri)
