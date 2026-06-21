"""Durable run manifest for ``npa.workflow`` executions on object storage."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

RUN_SCHEMA_VERSION = "npa.workflow.run.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class RunManifest:
    workflow: str
    run_id: str
    api_version: str
    run_prefix_uri: str = ""
    status: str = "planned"
    steps: list[dict[str, Any]] = field(default_factory=list)
    updated_at: str = field(default_factory=utc_now)
    schema_version: str = RUN_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "workflow": self.workflow,
            "run_id": self.run_id,
            "api_version": self.api_version,
            "run_prefix_uri": self.run_prefix_uri,
            "status": self.status,
            "updated_at": self.updated_at,
            "steps": list(self.steps),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RunManifest:
        return cls(
            workflow=str(payload.get("workflow") or ""),
            run_id=str(payload.get("run_id") or ""),
            api_version=str(payload.get("api_version") or ""),
            run_prefix_uri=str(payload.get("run_prefix_uri") or ""),
            status=str(payload.get("status") or "planned"),
            steps=[dict(item) for item in payload.get("steps") or [] if isinstance(item, dict)],
            updated_at=str(payload.get("updated_at") or utc_now()),
            schema_version=str(payload.get("schema_version") or RUN_SCHEMA_VERSION),
        )


def manifest_key(prefix: str) -> str:
    base = prefix.rstrip("/")
    return f"{base}/npa-workflow/manifest.json"


def status_key(prefix: str) -> str:
    base = prefix.rstrip("/")
    return f"{base}/npa-workflow/status.json"


class RunStateStore:
    """Persist workflow run manifests (mock ``reader``/``writer`` in unit tests)."""

    def __init__(
        self,
        *,
        bucket: str,
        prefix: str,
        reader: Any | None = None,
        writer: Any | None = None,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self._reader = reader
        self._writer = writer

    @property
    def run_prefix_uri(self) -> str:
        return f"s3://{self.bucket}/{self.prefix}"

    def read_manifest(self) -> RunManifest | None:
        key = manifest_key(self.prefix)
        try:
            body = self._read(key)
        except FileNotFoundError:
            return None
        payload = json.loads(body)
        if not isinstance(payload, dict):
            return None
        return RunManifest.from_dict(payload)

    def write_manifest(self, manifest: RunManifest) -> dict[str, Any]:
        manifest.updated_at = utc_now()
        manifest.run_prefix_uri = self.run_prefix_uri
        payload = manifest.to_dict()
        self._write(manifest_key(self.prefix), payload)
        self._write(
            status_key(self.prefix),
            {
                "schema_version": RUN_SCHEMA_VERSION,
                "run_id": manifest.run_id,
                "workflow": manifest.workflow,
                "status": manifest.status,
                "updated_at": manifest.updated_at,
                "step_count": len(manifest.steps),
            },
        )
        return payload

    def append_step(self, manifest: RunManifest, step_record: Mapping[str, Any]) -> dict[str, Any]:
        manifest.steps.append(dict(step_record))
        return self.write_manifest(manifest)

    def _read(self, key: str) -> str:
        if self._reader is not None:
            return str(self._reader(self.bucket, key))
        from npa.clients.storage import StorageClient

        client = StorageClient.from_environment()
        try:
            response = client._s3.get_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            raise FileNotFoundError(f"s3://{self.bucket}/{key}") from exc
        return response["Body"].read().decode("utf-8")

    def _write(self, key: str, payload: Mapping[str, Any]) -> None:
        body = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
        if self._writer is not None:
            self._writer(self.bucket, key, body)
            return
        from npa.clients.storage import StorageClient

        client = StorageClient.from_environment()
        client._s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
        )


def store_for_config(config: Mapping[str, Any], *, run_id: str) -> RunStateStore | None:
    bucket = str(config.get("bucket") or "").strip()
    prefix = str(config.get("prefix") or run_id).strip()
    if not bucket:
        return None
    return RunStateStore(bucket=bucket, prefix=prefix)
