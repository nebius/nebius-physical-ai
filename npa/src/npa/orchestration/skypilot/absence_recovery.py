"""Release an original failed Sky operation only after fresh owned-resource absence."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from npa.cluster.absent_evidence import digest, require
from npa.cluster.absent_journal import (
    complete_absence,
    completed_receipt,
    recovery_lease,
)
from npa.cluster.reconcile_absent import (
    _audit_directory,
    _retain_originals,
    _write_private,
)
from npa.orchestration.skypilot.absence_evidence import load_evidence
from npa.orchestration.skypilot.absence_target import reader_environment
from npa.provisioning_journal import load_operation


def _verify_reads(output: Path, records: list) -> None:
    require(isinstance(records, list) and len(records) >= 4, "Incomplete reader audit")
    for record in records:
        require(
            Path(record["file"]).name == record["file"], "Invalid reader audit path"
        )
        path = output / record["file"]
        require(path.is_file() and not path.is_symlink(), "Reader body unavailable")
        require(digest(path.read_bytes()) == record["sha256"], "Reader audit changed")


def _read(manifest_path: Path, manifest: dict, output: Path) -> list[dict]:
    environment = reader_environment(manifest["authority"])
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "npa.orchestration.skypilot.absence_reader",
            str(manifest_path),
            str(output),
        ],
        env=environment,
        capture_output=True,
        check=False,
    )
    _write_private(output / "reader.stdout", result.stdout)
    _write_private(output / "reader.stderr", result.stderr)
    require(
        result.returncode == 0,
        "Fresh absence could not be verified; raw failure retained",
    )
    records = json.loads((output / "read-records.json").read_bytes())
    _verify_reads(output, records)
    return records


def _completed(operation, manifest_hash: str) -> dict | None:
    current = operation.read()
    if current.get("phase") != "destroyed":
        return None
    receipt = current.get("absence_recovery", {})
    require(receipt.get("manifest_sha256") == manifest_hash, "Other recovery evidence")
    path = Path(receipt["path"])
    require(path.is_file() and not path.is_symlink(), "Recovery audit unavailable")
    data = path.read_bytes()
    require(digest(data) == receipt["sha256"], "Recovery audit changed")
    audit = json.loads(data)
    require(
        audit.get("action") == "verified_sky_absence_reconciliation"
        and audit.get("historical_workload_outcome") == "unknown",
        "Different recovery contract",
    )
    _verify_reads(path.parent, audit["provider_reads"])
    return completed_receipt(operation, manifest_hash)


def _audit(operation, manifest_data, evidence, output, records, lease) -> dict:
    original = _retain_originals(operation, manifest_data, output, lease)
    payload = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "action": "verified_sky_absence_reconciliation",
        "operation_id": operation.operation_id,
        "original_journal_sha256": digest(original),
        "original_lease_sha256": lease["sha256"],
        "manifest_sha256": digest(manifest_data),
        "scope": evidence["scope"],
        "historical_workload_outcome": "unknown",
        "historical_read_limits": evidence.get("historical_read_limits", {}),
        "provider_reads": records,
        "limits": "Fresh, separately timed reads; local locks serialize cooperating NPA writers only. Original failure remains in history. No workload success, original Pod UID, cancellation or cloud deletion is inferred.",
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


def _fresh_reconciliation(record, data, manifest, evidence, operation, apply) -> dict:
    expected = manifest["original_journal"]["sha256"]
    with recovery_lease(operation, expected) as lease:
        output = _audit_directory(operation)
        records = _read(record, manifest, output)
        require(record.read_bytes() == data, "Recovery manifest changed")
        require(load_evidence(manifest) == evidence, "Original evidence changed")
        receipt = _audit(operation, data, evidence, output, records, lease)
        if apply:
            complete_absence(operation, expected, receipt, lease)
    return {
        "status": "reconciled-absent" if apply else "absence-verified-not-applied",
        "operation_id": operation.operation_id,
        "receipt_sha256": receipt["sha256"],
        "historical_workload_outcome": "unknown",
        "cloud_mutations": False,
    }


def reconcile_absent(record: Path, *, apply: bool = False) -> dict:
    """Check original Sky absence; optionally commit the bound local lease transition.

    Args:
        record: Private evidence manifest referring to original producer files.
        apply: Commit only after fresh reads under original project/operation locks.
    Returns:
        Sanitized absence result, without a historical workload success claim.
    Raises:
        ValueError: Evidence or actual absence is incomplete or inconsistent.
        OperationJournalError: Ownership, execution, or journal generation changed.
    """
    require(
        record.is_absolute() and record.is_file() and not record.is_symlink(),
        "Private absolute manifest required",
    )
    data = record.read_bytes()
    manifest = json.loads(data)
    evidence = load_evidence(manifest)
    operation = load_operation(manifest["operation_id"])
    if apply and (completed := _completed(operation, digest(data))) is not None:
        return {**completed, "historical_workload_outcome": "unknown"}
    return _fresh_reconciliation(record, data, manifest, evidence, operation, apply)
