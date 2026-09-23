"""Conditional S3 handoff for private NCore qualification evidence."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from npa.errors import NpaError


HANDOFF_FORMAT = "npa_ncore_qualification_evidence_handoff_v1"
KINDS = {"runtime-attestation", "workflow-status"}


class NcoreEvidenceHandoffError(NpaError):
    """A private qualification evidence object was not handed off exactly once."""


def _write_private(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _validated_payload(raw: bytes, kind: str, run_id: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NcoreEvidenceHandoffError("evidence handoff source is invalid") from exc
    if not isinstance(payload, dict):
        raise NcoreEvidenceHandoffError("evidence handoff source is not an object")
    if kind == "runtime-attestation":
        if (
            payload.get("format") != "npa_nurec_runtime_attestation_v4"
            or payload.get("status") != "pass"
            or payload.get("workflow_run_id_sha256")
            != hashlib.sha256(run_id.encode()).hexdigest()
        ):
            raise NcoreEvidenceHandoffError(
                "runtime attestation handoff binding differs"
            )
    elif (
        payload.get("run_id") != run_id
        or payload.get("status") != "SUCCEEDED"
        or not isinstance(payload.get("stages"), dict)
    ):
        raise NcoreEvidenceHandoffError("workflow status handoff binding differs")
    return payload


def publish_qualification_evidence(
    source_path: Path,
    output_uri: str,
    *,
    kind: str,
    run_id: str,
    receipt_path: Path,
    storage_client: Any = None,
) -> dict[str, Any]:
    """Conditionally create and independently read back one exact evidence object."""
    from npa.clients.storage import StorageClient, StoragePreconditionFailed

    if kind not in KINDS:
        raise NcoreEvidenceHandoffError("evidence handoff kind is invalid")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,252}", run_id) is None:
        raise NcoreEvidenceHandoffError("workflow run ID is invalid")
    if (
        source_path.is_symlink()
        or not source_path.is_file()
        or source_path.stat().st_uid != os.getuid()
        or source_path.stat().st_nlink != 1
        or source_path.stat().st_mode & 0o077
    ):
        raise NcoreEvidenceHandoffError("evidence handoff source is not private")
    raw = source_path.read_bytes()
    _validated_payload(raw, kind, run_id)
    client = storage_client or StorageClient.from_environment()
    try:
        etag = client.put_bytes_conditional(
            raw,
            output_uri,
            if_none_match=True,
            content_type="application/json",
        )
    except StoragePreconditionFailed as exc:
        raise NcoreEvidenceHandoffError(
            "evidence handoff destination already exists"
        ) from exc
    observed = client.read_bytes_with_etag(output_uri)
    if observed is None or observed[0] != raw or observed[1] != etag:
        raise NcoreEvidenceHandoffError("evidence handoff read-back differs")
    receipt = {
        "format": HANDOFF_FORMAT,
        "status": "pass",
        "kind": kind,
        "run_id_sha256": hashlib.sha256(run_id.encode()).hexdigest(),
        "output_uri_sha256": hashlib.sha256(output_uri.encode()).hexdigest(),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "etag_sha256": hashlib.sha256(etag.encode()).hexdigest(),
        "conditional_create": True,
        "readback_verified": True,
    }
    _write_private(receipt_path, receipt)
    return receipt
