"""URI storage helpers for the insights lineage + metrics store."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from npa.verification import sanitize_reason
from npa.workbench.storage_scope import authorize_uri


class InsightsStorageError(RuntimeError):
    """Raised when a storage operation fails for a reason other than absence.

    A denied, expired-credential, or transport-level S3 failure must not be
    reported the same way as a genuinely missing object: the former means the
    store's real contents are unknown, while the latter means the store (or
    object) does not exist yet. Collapsing both into "absent" turns an auth
    outage into a silently empty query/ingest result.
    """


def uri_join(base: str, *parts: str) -> str:
    """Join URI path fragments without losing the scheme."""
    prefix = base.rstrip("/")
    suffix = "/".join(part.strip("/") for part in parts if part.strip("/"))
    return f"{prefix}/{suffix}" if suffix else prefix


def write_bytes_uri(uri: str, payload: bytes) -> None:
    target = authorize_uri(uri, operation="write")
    if target.kind == "s3":
        from botocore.exceptions import BotoCoreError, ClientError

        try:
            _s3_client().put_object(Bucket=target.bucket, Key=target.key, Body=payload)
        except (ClientError, BotoCoreError) as exc:
            raise InsightsStorageError(
                f"cannot write s3://{target.bucket}/{target.key}: "
                f"{_client_error_detail(exc)}"
            ) from exc
        return
    assert target.local_path is not None
    path = target.local_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def read_bytes_uri(uri: str) -> bytes:
    target = authorize_uri(uri, operation="read")
    if target.kind == "s3":
        from botocore.exceptions import BotoCoreError, ClientError

        try:
            response = _s3_client().get_object(Bucket=target.bucket, Key=target.key)
        except (ClientError, BotoCoreError) as exc:
            raise InsightsStorageError(
                f"cannot read s3://{target.bucket}/{target.key}: "
                f"{_client_error_detail(exc)}"
            ) from exc
        body = response["Body"]
        try:
            return body.read()
        except (ClientError, BotoCoreError) as exc:
            raise InsightsStorageError(
                f"cannot read s3://{target.bucket}/{target.key}: "
                f"{_client_error_detail(exc)}"
            ) from exc
        finally:
            # botocore's StreamingBody only auto-releases its connection back to
            # the pool once fully consumed; an exception mid-read (or a caller
            # that changes to a partial read later) would otherwise leak it.
            # Closing here is deterministic on both the success and error paths.
            body.close()
    assert target.local_path is not None
    return target.local_path.read_bytes()


# Codes an S3-compatible provider uses to say "you may not ask this" rather
# than "there is nothing here". These must win even if a provider also stamps
# a 404 HTTP status on a denial response, so an outage or revoked credential is
# never misread as "the object does not exist yet".
_ACCESS_DENIED_CODES = frozenset(
    {
        "AccessDenied",
        "AllAccessDisabled",
        "AuthFailure",
        "ExpiredToken",
        "Forbidden",
        "InvalidAccessKeyId",
        "SignatureDoesNotMatch",
        "TokenRefreshRequired",
        "UnauthorizedAccess",
    }
)
_MISSING_OBJECT_CODES = frozenset({"404", "NoSuchKey", "NotFound", "NoSuchBucket"})


def _is_missing_s3_object(exc: Exception) -> bool:
    """True when S3 said "no such object/bucket", not "could not answer"."""
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    code = str(response.get("Error", {}).get("Code", ""))
    if code in _ACCESS_DENIED_CODES:
        return False
    if code in _MISSING_OBJECT_CODES:
        return True
    return response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404


def _client_error_detail(exc: Exception) -> str:
    """Sanitized provider error detail — never a raw response/credential dump.

    ``ClientError.response`` can carry request ids, headers, and other
    provider-internal detail alongside the error body, so only the typed
    ``Code``/``Message`` fields are used (falling back to ``str(exc)`` for
    transport-level ``BotoCoreError``s, which carry no ``.response``). The
    result still passes through :func:`sanitize_reason`: a provider ``Message``
    can itself echo back request content — e.g. a presigned URL query string
    included in the original request — so the code/message split alone is not
    enough to guarantee nothing sensitive reaches logs or error text.
    """
    response = getattr(exc, "response", None)
    error = response.get("Error", {}) if isinstance(response, dict) else {}
    code = error.get("Code")
    message = error.get("Message")
    if code or message:
        detail = f"{code or 'UnknownError'}: {message or ''}".rstrip(": ")
    else:
        detail = str(exc)
    return sanitize_reason(detail)


def _head_object_exists(client: Any, bucket: str, key: str) -> bool:
    """Check exact-object existence via HEAD, distinguishing absence from failure.

    Used only by :func:`uri_exists`, which already required ``HeadObject``
    before this fix (only its error handling changed). Listing helpers use
    :func:`_s3_key_exists_via_list` instead so directory discovery keeps
    working under a ``ListBucket``-only policy that never grants ``HeadObject``.
    A missing-object *code* is the only thing that means "absent" here — a
    transport failure (``BotoCoreError``) can never mean that, so it always
    raises.
    """
    from botocore.exceptions import BotoCoreError, ClientError

    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        if _is_missing_s3_object(exc):
            return False
        raise InsightsStorageError(
            f"cannot check s3://{bucket}/{key}: {_client_error_detail(exc)}"
        ) from exc
    except BotoCoreError as exc:
        raise InsightsStorageError(
            f"cannot check s3://{bucket}/{key}: {_client_error_detail(exc)}"
        ) from exc


def _s3_key_exists_via_list(client: Any, bucket: str, key: str) -> bool:
    """True iff ``key`` itself names an object, using only ``ListBucket``.

    A bounded ``list_objects_v2(Prefix=key, MaxKeys=1)`` needs the same
    permission directory discovery already requires, so exact-object detection
    never demands ``HeadObject`` on top of it. S3 returns matches in UTF-8
    binary order, and any exact match sorts first among keys sharing that
    prefix (``"run-1"`` precedes both ``"run-1/manifest.json"`` and
    ``"run-10/..."``, since a string always sorts before any string it is a
    strict prefix of) — so the first returned key equals ``key`` if and only
    if ``key`` exists as an object.

    Unlike a single-object HEAD/GET, ``ListObjectsV2`` never raises to say "no
    matches" — an empty, genuinely absent prefix is a normal 200 response with
    empty ``Contents``. Any exception here is therefore a real failure (a
    missing *bucket*, denied access, a transport fault), never "the key is
    absent", so every exception raises rather than degrading to ``False``.
    """
    from botocore.exceptions import BotoCoreError, ClientError

    if not key:
        return False
    try:
        response = client.list_objects_v2(Bucket=bucket, Prefix=key, MaxKeys=1)
    except (ClientError, BotoCoreError) as exc:
        raise InsightsStorageError(
            f"cannot list s3://{bucket}/{key}: {_client_error_detail(exc)}"
        ) from exc
    contents = response.get("Contents") or []
    return bool(contents) and contents[0]["Key"] == key


def uri_exists(uri: str) -> bool:
    target = authorize_uri(uri, operation="read")
    if target.kind == "s3":
        return _head_object_exists(_s3_client(), target.bucket, target.key)
    assert target.local_path is not None
    return target.local_path.exists()


def write_json_uri(uri: str, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    write_bytes_uri(uri, data)


def read_json_uri(uri: str) -> dict[str, Any]:
    return json.loads(read_bytes_uri(uri).decode("utf-8"))


def write_text_uri(uri: str, text: str) -> None:
    write_bytes_uri(uri, text.encode("utf-8"))


def read_jsonl_uri(uri: str) -> list[dict[str, Any]]:
    """Read a JSONL object into a list of records (empty when absent)."""
    if not uri_exists(uri):
        return []
    text = read_bytes_uri(uri).decode("utf-8")
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        rows.append(json.loads(stripped))
    return rows


def shard_prefix_for(uri: str) -> str:
    """Directory-style prefix holding the append shards for a JSONL object.

    ``.../records.jsonl`` -> ``.../records.d``
    """
    base = uri[: -len(".jsonl")] if uri.endswith(".jsonl") else uri
    return f"{base}.d"


def list_jsonl_uris(prefix: str) -> list[str]:
    """List ``*.jsonl`` object URIs under a prefix (S3 or local), sorted."""
    target = authorize_uri(prefix, operation="read")
    if target.kind == "s3":
        client = _s3_client()
        directory_prefix = target.key.rstrip("/") + "/"
        found = _list_s3_keys_with_suffix(
            client, target.bucket, directory_prefix, suffix=".jsonl"
        )
        return sorted(f"s3://{target.bucket}/{key}" for key in found)
    assert target.local_path is not None
    base = target.local_path
    if not base.exists():
        return []
    return sorted(str(path) for path in base.rglob("*.jsonl"))


def read_jsonl_store(uri: str) -> list[dict[str, Any]]:
    """Read every row of an append-only JSONL store (base object + append shards).

    Reads the legacy single object first so stores written before sharding keep
    working, then every shard under ``<name>.d/`` in sorted (write-time) order.
    """
    rows = read_jsonl_uri(uri)
    for shard_uri in list_jsonl_uris(shard_prefix_for(uri)):
        rows.extend(read_jsonl_uri(shard_uri))
    return rows


def append_jsonl_uri(
    uri: str, rows: list[dict[str, Any]], *, previous_total: int | None = None
) -> int:
    """Append rows to an append-only JSONL store; return the store's row count.

    Object storage has no native append. Rewriting one object read-modify-write
    loses data whenever two writers overlap: both read N rows and both write
    N + their own, so the last write silently drops the other's rows while both
    ingests report success. Instead, every append lands in its own immutable
    shard object under ``<name>.d/``; readers concatenate the base object and all
    shards. That keeps the store genuinely append-only (rows are never mutated or
    removed) and safe for concurrent writers with no database and no locking.

    ``previous_total`` lets a caller that already knows the pre-append count skip
    a full re-read of the store: the total is then arithmetic rather than another
    list + GET of every shard. The returned count is **best effort** under
    concurrent writers — a writer that overlaps this one may land rows this count
    does not include. It is telemetry (surfaced as ``total_records`` /
    ``total_edges``), never an input to a correctness decision.
    """
    new_rows = list(rows)
    if new_rows:
        shard_name = f"{utc_stamp()}-{uuid.uuid4().hex[:12]}.jsonl"
        payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in new_rows)
        write_bytes_uri(
            uri_join(shard_prefix_for(uri), shard_name), payload.encode("utf-8")
        )
    if previous_total is not None:
        return previous_total + len(new_rows)
    return len(read_jsonl_store(uri))


def utc_stamp() -> str:
    """Sortable UTC timestamp used to order append shards by write time."""
    return datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S%f")


def list_json_uris(prefix: str) -> list[str]:
    """List ``*.json`` object URIs named by an exact object or a directory.

    ``prefix`` may point at one exact object (mirroring a local ``--input-path``
    that names a file) or a run/directory. A bare lexical ``Prefix=`` scan on S3
    would also match unrelated siblings that merely share the same leading
    characters (``.../run-1`` prefixing ``.../run-10``), silently mixing one
    run's artifacts into another's ingestion. This resolves ``prefix`` as an
    exact object first, then falls back to a delimiter-bounded directory
    listing — the same file-vs-directory distinction the local branch already
    makes with ``is_file()``/``rglob()`` — using only ``ListBucket``, the
    permission the pre-fix, listing-only implementation already required.
    """
    # A trailing slash is an explicit directory marker. authorize_uri's
    # canonicalization strips it before ``target.key`` is available, so it
    # must be read from the raw input: without this, a directory that
    # coincidentally shares its name with an unrelated same-named object
    # (rare, but a real bucket state) would be treated as that object instead
    # of the directory the caller pointed at.
    explicit_directory = prefix.strip().endswith("/")
    target = authorize_uri(prefix, operation="read")
    if target.kind == "s3":
        client = _s3_client()
        if not explicit_directory and _s3_key_exists_via_list(
            client, target.bucket, target.key
        ):
            if target.key.endswith(".json"):
                return [f"s3://{target.bucket}/{target.key}"]
            return []
        directory_prefix = f"{target.key}/" if target.key else ""
        found = _list_s3_keys_with_suffix(
            client, target.bucket, directory_prefix, suffix=".json"
        )
        return sorted(f"s3://{target.bucket}/{key}" for key in found)
    assert target.local_path is not None
    base = target.local_path
    if base.is_file():
        return [str(base)] if base.suffix == ".json" else []
    if not base.exists():
        return []
    return sorted(str(path) for path in base.rglob("*.json"))


def _list_s3_keys_with_suffix(
    client: Any, bucket: str, prefix: str, *, suffix: str
) -> list[str]:
    """List every key under ``prefix`` ending in ``suffix``, all pages or none.

    A genuinely empty/absent prefix is a normal paginated response with no
    ``Contents`` on any page — never an exception — so any exception raised
    mid-pagination (missing bucket, denied access, a dropped connection) is a
    real failure, not "nothing here". Returning what was already collected from
    earlier pages in that case would silently understate the result as if the
    listing had completed normally, so the whole call raises instead of
    returning a partial list.
    """
    from botocore.exceptions import BotoCoreError, ClientError

    paginator = client.get_paginator("list_objects_v2")
    found: list[str] = []
    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []) or []:
                key = obj["Key"]
                if key.endswith(suffix):
                    found.append(key)
    except (ClientError, BotoCoreError) as exc:
        raise InsightsStorageError(
            f"cannot list s3://{bucket}/{prefix}: {_client_error_detail(exc)}"
        ) from exc
    return found


def _s3_client():
    import boto3
    from botocore.config import Config as BotoConfig

    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL")
        or os.environ.get("NEBIUS_S3_ENDPOINT")
        or None,
        config=BotoConfig(signature_version="s3v4"),
    )
