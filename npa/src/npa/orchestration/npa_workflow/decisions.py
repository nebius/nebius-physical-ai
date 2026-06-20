"""Read and write threshold/decision artifacts for dynamic workflow transitions."""

from __future__ import annotations

import json
from typing import Any, Mapping
from urllib.parse import urlparse

from npa.orchestration.npa_workflow.errors import NpaWorkflowError

DECISION_PROMOTE = "promote_checkpoint"
DECISION_LOOP_BACK = "loop_back_to_inner_loop"


def normalize_decision(raw: str) -> str:
    """Map decision file values to predicate names."""

    value = raw.strip()
    if value in {DECISION_PROMOTE, "promote", "promote_checkpoint"}:
        return DECISION_PROMOTE
    if value in {DECISION_LOOP_BACK, "loop_back", "loop_back_to_inner_loop"}:
        return DECISION_LOOP_BACK
    return value


def decision_from_payload(payload: Mapping[str, Any]) -> str:
    for key in ("decision", "last_decision", "action"):
        if key in payload:
            return normalize_decision(str(payload[key]))
    raise NpaWorkflowError(f"decision payload missing decision field: {payload!r}")


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise NpaWorkflowError(f"expected s3:// URI, got {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def load_decision(uri: str, *, reader: Any | None = None) -> str:
    """Load a decision string from ``s3://bucket/key`` (inject ``reader`` in tests)."""

    bucket, key = parse_s3_uri(uri)
    body = _read_object(bucket, key, reader=reader)
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise NpaWorkflowError(f"decision at {uri} must be a JSON object")
    return decision_from_payload(payload)


def write_decision(uri: str, decision: str, *, writer: Any | None = None) -> None:
    """Write a decision JSON object to S3 (inject ``writer`` in tests)."""

    bucket, key = parse_s3_uri(uri)
    payload = {"decision": normalize_decision(decision)}
    body = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _write_object(bucket, key, body, writer=writer)


def refresh_context_decision(
    context: Mapping[str, Any],
    *,
    reader: Any | None = None,
) -> str:
    """Return ``last_decision`` from context or load ``config.decision_uri`` when set."""

    config = dict(context.get("config") or {})
    uri = str(config.get("decision_uri") or "").strip()
    if uri:
        return load_decision(uri, reader=reader)
    return normalize_decision(str(context.get("last_decision") or ""))


def _read_object(bucket: str, key: str, *, reader: Any | None = None) -> str:
    if reader is not None:
        return str(reader(bucket, key))
    from botocore.exceptions import ClientError

    from npa.clients.storage import StorageClient

    client = StorageClient.from_environment()
    try:
        response = client._s3.get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code in {"404", "NoSuchKey", "NotFound"}:
            raise FileNotFoundError(f"s3://{bucket}/{key}") from exc
        raise
    return response["Body"].read().decode("utf-8")


def _write_object(bucket: str, key: str, body: bytes, *, writer: Any | None = None) -> None:
    if writer is not None:
        writer(bucket, key, body)
        return
    from npa.clients.storage import StorageClient

    client = StorageClient.from_environment()
    client._s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
