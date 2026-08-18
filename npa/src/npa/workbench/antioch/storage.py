"""S3 compare-and-swap state and immutable artifact helpers for Antioch."""

from __future__ import annotations

import hashlib
import json
import mimetypes
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError

from npa.clients.storage import (
    StorageClient,
    StoragePreconditionFailed,
    _parse_bucket_uri,
)

from .schemas import ArtifactRecord, OperationRecord, utc_now


class AntiochStorageError(RuntimeError):
    pass


def canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def join_uri(base: str, *parts: str) -> str:
    suffix = "/".join(part.strip("/") for part in parts if part.strip("/"))
    return base.rstrip("/") + ("/" + suffix if suffix else "")


class StateStore:
    """Durable operation records protected by object-level compare-and-swap."""

    def __init__(self, client: StorageClient | None = None) -> None:
        self.client = client or StorageClient.from_environment()

    @staticmethod
    def state_uri(output_path: str, idempotency_key: str) -> str:
        return join_uri(output_path, "_control", f"{idempotency_key}.json")

    def read(
        self, output_path: str, idempotency_key: str
    ) -> tuple[OperationRecord, str] | None:
        result = self.client.read_bytes_with_etag(
            self.state_uri(output_path, idempotency_key)
        )
        if result is None:
            return None
        payload, etag = result
        try:
            return OperationRecord.model_validate_json(payload), etag
        except Exception as exc:
            raise AntiochStorageError(
                "durable Antioch state is malformed; refusing recovery"
            ) from exc

    def claim(self, record: OperationRecord) -> OperationRecord:
        uri = self.state_uri(record.output_path, record.idempotency_key)
        try:
            self.client.put_bytes_conditional(
                canonical_json(record.model_dump(mode="json")),
                uri,
                if_none_match=True,
                content_type="application/json",
            )
            return record
        except StoragePreconditionFailed:
            existing = self.read(record.output_path, record.idempotency_key)
            if existing is None:
                raise AntiochStorageError(
                    "idempotency claim was lost but no state is readable"
                )
            current, _ = existing
            if current.request_sha256 != record.request_sha256:
                raise AntiochStorageError(
                    "idempotency key already belongs to a different request"
                )
            return current

    def update(self, record: OperationRecord, **changes: Any) -> OperationRecord:
        while True:
            current = self.read(record.output_path, record.idempotency_key)
            if current is None:
                raise AntiochStorageError(
                    "cannot update missing Antioch operation state"
                )
            latest, etag = current
            updated = latest.model_copy(
                update={
                    **changes,
                    "updated_at": utc_now(),
                    "revision": latest.revision + 1,
                }
            )
            try:
                self.client.put_bytes_conditional(
                    canonical_json(updated.model_dump(mode="json")),
                    self.state_uri(updated.output_path, updated.idempotency_key),
                    if_match=etag,
                    content_type="application/json",
                )
                return updated
            except StoragePreconditionFailed:
                continue

    def list(self, output_path: str) -> list[OperationRecord]:
        bucket, prefix = _parse_bucket_uri(join_uri(output_path, "_control") + "/")
        records: list[OperationRecord] = []
        for page in self.client.s3.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=prefix
        ):
            for item in page.get("Contents", []):
                key = str(item.get("Key") or "")
                if not key.endswith(".json"):
                    continue
                response = self.client.s3.get_object(Bucket=bucket, Key=key)
                body = response["Body"]
                try:
                    records.append(OperationRecord.model_validate_json(body.read()))
                finally:
                    body.close()
        return sorted(records, key=lambda item: item.created_at)

    def put_immutable_json(self, uri: str, payload: dict[str, Any]) -> str:
        raw = canonical_json(payload)
        try:
            return self.client.put_bytes_conditional(
                raw, uri, if_none_match=True, content_type="application/json"
            )
        except StoragePreconditionFailed:
            existing = self.client.read_bytes_with_etag(uri)
            if existing is None or existing[0] != raw:
                raise AntiochStorageError(
                    f"immutable object already exists with different content: {uri}"
                )
            return existing[1]

    def upload_artifact(
        self,
        path: Path,
        uri: str,
        *,
        name: str,
        scenario_run_id: str = "",
    ) -> ArtifactRecord:
        digest = sha256_file(path)
        bucket, key = _parse_bucket_uri(uri)
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.client.s3.upload_file(
            str(path),
            bucket,
            key,
            ExtraArgs={
                "ContentType": content_type,
                "Metadata": {"sha256": digest, "npa-role": "antioch-artifact"},
            },
        )
        head = self.client.s3.head_object(Bucket=bucket, Key=key)
        size = int(head.get("ContentLength") or 0)
        metadata_sha = str((head.get("Metadata") or {}).get("sha256") or "")
        if size != path.stat().st_size or metadata_sha != digest:
            raise AntiochStorageError(
                "uploaded artifact failed size/checksum verification"
            )
        return ArtifactRecord(
            name=name,
            uri=uri,
            size_bytes=size,
            sha256=digest,
            content_type=content_type,
            scenario_run_id=scenario_run_id,
        )


def object_exists(client: StorageClient, uri: str) -> bool:
    bucket, key = _parse_bucket_uri(uri)
    try:
        client.s3.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise
    return True
