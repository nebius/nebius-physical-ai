"""Content-addressed durable reconciliation state for Sim2Real.

Mutable ``workflow_state.json`` remains a status surface.  This module is the
execution checkpoint: immutable payloads are uploaded before a small latest
pointer, and every read verifies controller identity and both input/output
digests before reuse.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError

from npa.clients.storage import StorageClient
from npa.workflows.sim2real.models import Sim2RealLoopConfig, Sim2RealLoopError
from npa.workflows.sim2real.utils import (
    _artifact_root_uri,
    _utc_now,
    _write_json_artifact,
)


JOURNAL_SCHEMA = "npa.sim2real.controller-journal.v1"
_SAFE_UNIT = re.compile(r"[^a-zA-Z0-9_.-]+")


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ControllerIdentity:
    run_id: str
    source_sha: str
    runtime_image: str
    spec_digest: str

    @classmethod
    def from_environment(cls, config: Sim2RealLoopConfig) -> "ControllerIdentity":
        return cls(
            run_id=str(getattr(config, "run_id", "")),
            source_sha=os.environ.get("NPA_SIM2REAL_SOURCE_SHA", "").strip(),
            runtime_image=os.environ.get("NPA_SIM2REAL_RUNTIME_IMAGE", "").strip(),
            spec_digest=os.environ.get(
                "NPA_SIM2REAL_CONTROLLER_SPEC_DIGEST", ""
            ).strip(),
        )

    def payload(self) -> dict[str, str]:
        return {
            "run_id": self.run_id,
            "source_sha": self.source_sha,
            "runtime_image": self.runtime_image,
            "spec_digest": self.spec_digest,
        }


class DurableStateStore:
    """Immutable unit/checkpoint store backed by the run's isolated S3 prefix."""

    def __init__(
        self,
        config: Sim2RealLoopConfig,
        local_dir: Path,
        *,
        client: StorageClient | None = None,
        identity: ControllerIdentity | None = None,
    ) -> None:
        self.config = config
        self.local_dir = local_dir
        self.identity = identity or ControllerIdentity.from_environment(config)
        self.enabled = bool(
            getattr(config, "upload_artifacts", False)
            and getattr(config, "s3_bucket", "")
        )
        self._client = client
        self.root = (
            f"{_artifact_root_uri(config)}/state/controller" if self.enabled else ""
        )
        if (
            self.enabled
            and os.environ.get("NPA_SIM2REAL_REQUIRE_REAL_COMPONENTS", "").strip()
            == "1"
        ):
            values = self.identity.payload()
            if not re.fullmatch(r"[0-9a-f]{40}", values["source_sha"]):
                raise Sim2RealLoopError(
                    "durable real-tier controller requires an exact source SHA"
                )
            if "@sha256:" not in values["runtime_image"]:
                raise Sim2RealLoopError(
                    "durable real-tier controller requires an immutable runtime image"
                )
            if not re.fullmatch(r"[0-9a-f]{64}", values["spec_digest"]):
                raise Sim2RealLoopError(
                    "durable real-tier controller requires an exact spec digest"
                )

    @property
    def client(self) -> StorageClient:
        if self._client is None:
            self._client = StorageClient.from_environment(
                endpoint_url=self.config.s3_endpoint
            )
        return self._client

    @staticmethod
    def input_digest(payload: Any) -> str:
        return canonical_digest(payload)

    def _local_path(self, *parts: str) -> Path:
        return self.local_dir / "state" / "controller" / Path(*parts)

    def _download_json(self, uri: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        fd, temporary = tempfile.mkstemp(prefix="npa-resume-", suffix=".json")
        os.close(fd)
        try:
            try:
                self.client.download_file(uri, temporary)
            except ClientError as exc:
                code = str(exc.response.get("Error", {}).get("Code", ""))
                if code in {"404", "NoSuchKey", "NotFound"}:
                    return None
                raise
            payload = json.loads(Path(temporary).read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise Sim2RealLoopError(f"durable state is not an object: {uri}")
            return payload
        finally:
            Path(temporary).unlink(missing_ok=True)

    def _upload_json(self, local_path: Path, uri: str, payload: dict[str, Any]) -> str:
        _write_json_artifact(local_path, payload)
        if self.enabled:
            return self.client.upload_file(str(local_path), uri)
        return str(local_path)

    def _verify_envelope(
        self,
        envelope: dict[str, Any],
        *,
        unit: str = "",
        input_digest: str = "",
    ) -> dict[str, Any]:
        if envelope.get("schema") != JOURNAL_SCHEMA:
            raise Sim2RealLoopError("unsupported durable controller journal schema")
        if envelope.get("identity") != self.identity.payload():
            raise Sim2RealLoopError(
                "durable controller identity differs from this exact run/source/image"
            )
        if unit and envelope.get("unit") != unit:
            raise Sim2RealLoopError("durable unit name mismatch")
        if input_digest and envelope.get("input_digest") != input_digest:
            raise Sim2RealLoopError("durable unit input digest mismatch")
        payload = envelope.get("payload")
        if not isinstance(payload, dict):
            raise Sim2RealLoopError("durable unit payload must be an object")
        if canonical_digest(payload) != envelope.get("payload_digest"):
            raise Sim2RealLoopError("durable unit payload digest mismatch")
        return payload

    def load_unit(self, unit: str, input_payload: Any) -> dict[str, Any] | None:
        digest = self.input_digest(input_payload)
        safe = _SAFE_UNIT.sub("-", unit).strip("-")
        envelope = self._download_json(f"{self.root}/units/{safe}/{digest}.json")
        if envelope is None:
            return None
        return self._verify_envelope(envelope, unit=unit, input_digest=digest)

    def commit_unit(
        self, unit: str, input_payload: Any, payload: dict[str, Any]
    ) -> dict[str, Any]:
        digest = self.input_digest(input_payload)
        payload_digest = canonical_digest(payload)
        envelope = {
            "schema": JOURNAL_SCHEMA,
            "kind": "execution_unit",
            "identity": self.identity.payload(),
            "unit": unit,
            "input_digest": digest,
            "payload_digest": payload_digest,
            "payload": payload,
            "committed_at": _utc_now(),
        }
        safe = _SAFE_UNIT.sub("-", unit).strip("-")
        path = self._local_path("units", safe, f"{digest}.json")
        self._upload_json(path, f"{self.root}/units/{safe}/{digest}.json", envelope)
        return payload

    def persist_workflow_checkpoint(self, payload: dict[str, Any]) -> str:
        """Write immutable records/checkpoint, then atomically advance latest."""

        if not self.enabled:
            return ""
        payload_digest = canonical_digest(payload)
        envelope = {
            "schema": JOURNAL_SCHEMA,
            "kind": "workflow_checkpoint",
            "identity": self.identity.payload(),
            "payload_digest": payload_digest,
            "payload": payload,
            "committed_at": _utc_now(),
        }
        immutable_uri = f"{self.root}/checkpoints/sha256-{payload_digest}.json"
        immutable_path = self._local_path(
            "checkpoints", f"sha256-{payload_digest}.json"
        )
        self._upload_json(immutable_path, immutable_uri, envelope)
        for kind, records in (
            ("component", payload.get("components") or []),
            ("stage", payload.get("stage_records") or []),
        ):
            for record in records:
                if not isinstance(record, dict):
                    continue
                record_digest = canonical_digest(record)
                record_path = self._local_path(
                    "records", kind, f"sha256-{record_digest}.json"
                )
                self._upload_json(
                    record_path,
                    f"{self.root}/records/{kind}/sha256-{record_digest}.json",
                    record,
                )
        pointer = {
            "schema": JOURNAL_SCHEMA,
            "kind": "latest_pointer",
            "identity": self.identity.payload(),
            "checkpoint_uri": immutable_uri,
            "payload_digest": payload_digest,
            "committed_at": _utc_now(),
        }
        self._upload_json(
            self._local_path("latest.json"), f"{self.root}/latest.json", pointer
        )
        return immutable_uri

    def hydrate_workflow_state(self) -> dict[str, Any] | None:
        pointer = self._download_json(f"{self.root}/latest.json")
        if pointer is None:
            return None
        if pointer.get("identity") != self.identity.payload():
            raise Sim2RealLoopError("latest controller pointer identity mismatch")
        envelope = self._download_json(str(pointer.get("checkpoint_uri") or ""))
        if envelope is None:
            raise Sim2RealLoopError("latest durable checkpoint object is missing")
        payload = self._verify_envelope(envelope)
        if canonical_digest(payload) != pointer.get("payload_digest"):
            raise Sim2RealLoopError("latest durable checkpoint pointer digest mismatch")
        payload["local_artifact_dir"] = str(self.local_dir)
        _write_json_artifact(self.local_dir / "state" / "workflow_state.json", payload)
        return payload

    def heartbeat(self, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        envelope = {
            "schema": JOURNAL_SCHEMA,
            "kind": "heartbeat",
            "identity": self.identity.payload(),
            "payload": payload,
            "payload_digest": canonical_digest(payload),
            "updated_at": _utc_now(),
        }
        self._upload_json(
            self._local_path("heartbeat.json"),
            f"{self.root}/heartbeat.json",
            envelope,
        )
