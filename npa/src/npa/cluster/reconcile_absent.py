"""Close one failed legacy operation after original evidence and fresh absence."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid

from npa.cluster.absent_evidence import digest, load_legacy_evidence, require
from npa.cluster.absent_journal import (
    complete_absence,
    completed_receipt,
    recovery_lease,
)
from npa.cluster.absent_provider import AbsenceProvider, verify_absence
from npa.provisioning_journal import load_operation


def _write_private(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _audit(
    operation, evidence: dict, manifest_data: bytes, output: Path, records: list
) -> dict:
    original = operation.path.read_bytes()
    manifest = json.loads(manifest_data)
    require(
        digest(original) == manifest["original_journal"]["sha256"],
        "Journal generation changed before audit",
    )
    _write_private(output / "original-journal.json", original)
    _write_private(output / "evidence-manifest.json", manifest_data)
    payload = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "action": "verified_absence_reconciliation",
        "operation_id": operation.operation_id,
        "original_journal_sha256": digest(original),
        "manifest_sha256": digest(manifest_data),
        "producer_source": evidence["producer_source"],
        "producer_files": evidence["producer_files"],
        "state_sha256": evidence["state_sha256"],
        "resources": evidence["resources"],
        "provider_reads": records,
        "legacy_binding": evidence["legacy_binding"],
        "limits": "Fresh provider observations; local leases serialize cooperating NPA writers, not external provider actors. Original failed provision remains failed; no cloud mutation or workload qualification.",
    }
    data = (json.dumps(payload, indent=2) + "\n").encode()
    path = output / "reconciliation.json"
    _write_private(path, data)
    return {
        "path": str(path),
        "sha256": digest(data),
        "manifest_sha256": digest(manifest_data),
        "original_journal_sha256": digest(original),
    }


def _audit_directory(operation) -> Path:
    parent = operation.path.parent / "absence-recovery"
    require(not parent.is_symlink(), "Recovery audit directory is a symlink")
    parent.mkdir(mode=0o700, exist_ok=True)
    metadata = parent.stat()
    require(
        parent.is_dir()
        and metadata.st_uid == os.getuid()
        and metadata.st_mode & 0o077 == 0,
        "Existing recovery audit directory is not owner-private",
    )
    output = parent / uuid.uuid4().hex
    output.mkdir(mode=0o700)
    return output


def reconcile_absent(evidence_file: Path) -> dict:
    require(
        evidence_file.is_file() and not evidence_file.is_symlink(),
        "Recovery manifest must be a regular private file",
    )
    data = evidence_file.read_bytes()
    manifest = json.loads(data)
    evidence = load_legacy_evidence(manifest)
    operation = load_operation(manifest["operation_id"])
    completed = completed_receipt(operation, digest(data))
    if completed is not None:
        return completed
    expected = manifest["original_journal"]["sha256"]
    with recovery_lease(operation, expected) as lease:
        output = _audit_directory(operation)
        provider = AbsenceProvider(evidence["authority"], output)
        verify_absence(provider, evidence)
        require(evidence_file.read_bytes() == data, "Recovery manifest changed")
        require(load_legacy_evidence(manifest) == evidence, "Original evidence changed")
        receipt = _audit(operation, evidence, data, output, provider.records)
        complete_absence(operation, expected, receipt, lease)
    return {
        "status": "reconciled-destroyed",
        "operation_id": operation.operation_id,
        "fresh_provider_verification": True,
        "receipt_sha256": receipt["sha256"],
    }
