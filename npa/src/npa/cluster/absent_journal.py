"""Exclusive, compare-and-swap terminal transition for reviewed absence proof."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path

from npa import provisioning_journal as journals
from npa.cluster.absent_evidence import digest, require


@contextmanager
def _exclusive_existing(path: Path):
    require(path.is_file() and not path.is_symlink(), "Original lock is unavailable")
    descriptor = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "r+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise journals.OperationJournalError(
                "Lifecycle execution is active"
            ) from exc
        yield


def _candidate(operation, expected: str) -> dict:
    require(not operation.path.is_symlink(), "Original journal is not regular")
    raw = operation.path.read_bytes()
    require(digest(raw) == expected, "Journal generation changed")
    payload = json.loads(raw)
    require(
        payload["phase"] in {"recovery-required", "rollback-incomplete"},
        "Operation is not a failed recoverable generation",
    )
    require(payload.get("lifecycle") != "running", "Operation still reports running")
    require(
        not journals._pid_is_alive(int(payload.get("owner_pid") or 0)),
        "Original owner process is still alive",
    )
    return payload


@contextmanager
def recovery_lease(operation, expected: str):
    require(
        not os.environ.get("NPA_PARENT_LIFECYCLE_OPERATION")
        and journals.current_operation() is None,
        "Nested recovery is not supported",
    )
    payload = _candidate(operation, expected)
    directory = journals._project_lease_directory(payload["project_id"])
    require(
        directory.is_dir() and not directory.is_symlink(),
        "Original project lease missing",
    )
    with _exclusive_existing(directory / ".lock"):
        lease = journals._read_project_lease(directory / "lease.json")
        require(
            lease.get("operation_id") == operation.operation_id,
            "Project lease belongs to another operation",
        )
        with _exclusive_existing(operation.path.parent / ".execution.lock"):
            _candidate(operation, expected)
            lease_path = directory / "lease.json"
            yield {"path": lease_path, "sha256": digest(lease_path.read_bytes())}


def complete_absence(
    operation, expected: str, receipt: dict, lease_binding: dict
) -> None:
    lease_path = lease_binding["path"]
    with journals._locked_operation(operation.operation_id):
        require(
            not lease_path.is_symlink()
            and digest(lease_path.read_bytes()) == lease_binding["sha256"],
            "Project lease generation changed",
        )
        payload = _candidate(operation, expected)
        now = journals.utc_now()
        payload["absence_recovery"] = receipt
        payload.update(
            phase="destroyed",
            lifecycle="succeeded",
            result="destroyed",
            updated_at=now,
            heartbeat_at=now,
            owner_pid=os.getpid(),
        )
        payload.setdefault("events", []).append(
            {
                "phase": "destroyed",
                "recorded_at": now,
                "details": {"action": "verified_absence_reconciliation", **receipt},
            }
        )
        journals._write_atomic(operation.path, payload)
    _release_bound_lease(operation, payload, receipt, lease_path)


def _audit_payload(receipt: dict) -> dict:
    path = Path(receipt["path"])
    require(path.is_file() and not path.is_symlink(), "Recovery audit is unavailable")
    data = path.read_bytes()
    require(digest(data) == receipt["sha256"], "Recovery audit changed")
    return json.loads(data)


def _original_lease(receipt: dict) -> tuple[dict, str]:
    audit = _audit_payload(receipt)
    path = Path(receipt["path"]).parent / "original-project-lease.json"
    require(path.is_file() and not path.is_symlink(), "Original lease is unavailable")
    data = path.read_bytes()
    require(digest(data) == audit["original_lease_sha256"], "Original lease changed")
    return journals._read_project_lease(path), digest(data)


def _release_bound_lease(operation, payload: dict, receipt: dict, path: Path) -> None:
    original, original_hash = _original_lease(receipt)
    require(
        original.get("operation_id") == operation.operation_id,
        "Original project lease belongs to another operation",
    )
    released = dict(original)
    released.update(
        phase="destroyed",
        lifecycle="succeeded",
        owner_pid=payload["owner_pid"],
        released_at=payload["updated_at"],
    )
    current = journals._read_project_lease(path)
    if current == released:
        return
    require(
        digest(path.read_bytes()) == original_hash, "Project lease generation changed"
    )
    journals._write_atomic(path, released)
    require(
        journals._read_project_lease(path) == released, "Lease release readback failed"
    )


@contextmanager
def _terminal_locks(operation, payload: dict, expected: str):
    require(
        not os.environ.get("NPA_PARENT_LIFECYCLE_OPERATION")
        and journals.current_operation() is None,
        "Nested recovery is not supported",
    )
    directory = journals._project_lease_directory(payload["project_id"])
    require(
        directory.is_dir() and not directory.is_symlink(),
        "Original project lease missing",
    )
    with _exclusive_existing(directory / ".lock"):
        with _exclusive_existing(operation.path.parent / ".execution.lock"):
            with journals._locked_operation(operation.operation_id):
                require(
                    not operation.path.is_symlink()
                    and digest(operation.path.read_bytes()) == expected,
                    "Terminal journal generation changed",
                )
                yield directory / "lease.json"


def completed_receipt(operation, manifest_hash: str) -> dict | None:
    require(not operation.path.is_symlink(), "Original journal is not regular")
    data = operation.path.read_bytes()
    payload = json.loads(data)
    if payload.get("phase") != "destroyed":
        return None
    receipt = payload.get("absence_recovery", {})
    require(
        receipt.get("manifest_sha256") == manifest_hash,
        "Terminal operation belongs to different recovery evidence",
    )
    with _terminal_locks(operation, payload, digest(data)) as path:
        _release_bound_lease(operation, payload, receipt, path)
    return {
        "status": "already-reconciled",
        "operation_id": operation.operation_id,
        "fresh_provider_verification": False,
        "receipt_sha256": receipt["sha256"],
    }
