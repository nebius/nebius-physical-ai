"""Keep one durable execution record per prescribed evaluation case using S3 CAS."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import re
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from npa.clients.storage import StorageClient

SCHEMA = "npa.behavior.case-execution.v1"


class CaseAlreadyStarted(RuntimeError):
    """A prescribed rollout started and cannot be automatically repeated."""


@dataclass(frozen=True)
class CaseVersion:
    """Hold a case record and the object version required for its next transition."""

    record: dict[str, Any]
    etag: str


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _valid_time(value) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return datetime.fromisoformat(value).utcoffset() is not None
    except ValueError:
        return False


def _valid_claim_fields(record: dict) -> bool:
    return (
        isinstance(record.get("claim_id"), str)
        and re.fullmatch(r"[0-9a-f]{32}", record["claim_id"]) is not None
        and isinstance(record.get("worker_id"), str)
        and bool(record["worker_id"].strip())
        and _valid_time(record.get("claimed_at"))
    )


def _valid_state_fields(record: dict) -> bool:
    fields = {
        "schema",
        "panel_id",
        "case",
        "state",
        "claim_id",
        "worker_id",
        "claimed_at",
        "previous_claims",
    }
    if record["state"] in {"started", "complete"}:
        fields.add("started_at")
        if not _valid_time(record.get("started_at")):
            return False
    if record["state"] == "complete":
        fields.update(("completed_at", "rollout"))
        if not _valid_time(record.get("completed_at")) or not isinstance(
            record.get("rollout"), dict
        ):
            return False
        if any(
            record["rollout"].get(key) != value for key, value in record["case"].items()
        ):
            return False
    history = record.get("previous_claims")
    return (
        set(record) == fields
        and isinstance(history, list)
        and all(
            isinstance(claim, dict)
            and set(claim) == {"claim_id", "worker_id", "claimed_at"}
            and _valid_claim_fields(claim)
            for claim in history
        )
    )


def _prefix(uri: str) -> str:
    parsed = urlparse(uri)
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or not parsed.path.strip("/")
        or parsed.query
        or parsed.fragment
        or parsed.username
        or ".." in parsed.path.split("/")
    ):
        raise ValueError("Evaluation state requires a scoped S3 prefix")
    return uri.rstrip("/")


def case_name(case: dict) -> str:
    """Build a safe key for a prescribed task instance and rollout.

    Args:
        case: Official task, instance_id, and rollout_id fields.
    Returns:
        A relative S3 key without path traversal.
    Raises:
        ValueError: Case identity has invalid names or numeric types.
    """
    task = case.get("task")
    if not isinstance(task, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", task):
        raise ValueError("Invalid official task name")
    for key in ("instance_id", "rollout_id"):
        if type(case.get(key)) is not int or case[key] < 0:
            raise ValueError(f"Invalid prescribed {key}")
    return f"{task}/{case['instance_id']}/{case['rollout_id']}"


class CaseStore:
    """Store atomic case transitions within one immutable policy panel.

    Args:
        storage: Configured storage client with conditional object writes.
        output_path: Stable campaign S3 prefix, reused across workflow resumes.
        panel_id: SHA-256 identity of the frozen panel.
    Returns:
        None.
    Raises:
        ValueError: State prefix or panel identity is invalid.
    """

    def __init__(self, storage: StorageClient, output_path: str, panel_id: str):
        if not re.fullmatch(r"[0-9a-f]{64}", panel_id):
            raise ValueError("Panel identity must be a SHA-256")
        self.storage = storage
        self.panel_id = panel_id
        self.prefix = f"{_prefix(output_path)}/panels/{panel_id}"

    def uri(self, case: dict) -> str:
        """Return the exact state object for a prescribed case.

        Args:
            case: Official case identity.
        Returns:
            S3 state URI.
        Raises:
            ValueError: Case identity is invalid.
        """
        return f"{self.prefix}/cases/{case_name(case)}/state.json"

    def read(self, case: dict) -> CaseVersion | None:
        """Read a case while rejecting a conflicting panel or case identity.

        Args:
            case: Exact prescribed case, including its public index.
        Returns:
            Record and ETag, or None only for authoritative absence.
        Raises:
            ValueError: Stored evidence has a conflicting identity or state.
        """
        response = self.storage.read_bytes_with_etag(self.uri(case))
        if response is None:
            return None
        payload, etag = response
        record = json.loads(payload)
        self._verify_identity(record, case)
        return CaseVersion(record, etag)

    def _verify_identity(self, record: dict, case: dict) -> None:
        if (
            not isinstance(record, dict)
            or record.get("schema") != SCHEMA
            or record.get("panel_id") != self.panel_id
            or record.get("case") != case
            or record.get("state") not in {"claimed", "started", "complete"}
            or not _valid_claim_fields(record)
            or not _valid_state_fields(record)
        ):
            raise ValueError("Stored evaluation case identity or state differs")

    def claim(self, case: dict, worker_id: str) -> CaseVersion:
        """Claim an unstarted case or return its existing completed record.

        A replaced pre-start owner cannot start: its ETag no longer matches.

        Args:
            case: Prescribed case from the frozen panel.
            worker_id: Non-secret worker/run identity for provenance.
        Returns:
            Claimed case version, or the already completed version.
        Raises:
            CaseAlreadyStarted: A rollout started without a completion receipt.
            StoragePreconditionFailed: Another worker won the state transition.
            ValueError: Worker or stored case identity is invalid.
        """
        if not isinstance(worker_id, str) or not worker_id.strip():
            raise ValueError("A worker identity is required")
        previous = self.read(case)
        if previous and previous.record["state"] == "complete":
            return previous
        if previous and previous.record["state"] == "started":
            raise CaseAlreadyStarted(f"Recover original evidence for {case_name(case)}")
        record = self._new_claim(case, worker_id, previous)
        etag = self._write(case, record, previous.etag if previous else "")
        return CaseVersion(record, etag)

    def _new_claim(self, case, worker_id, previous):
        history = list(previous.record.get("previous_claims", [])) if previous else []
        if previous:
            fields = ("claim_id", "worker_id", "claimed_at")
            history.append({key: previous.record[key] for key in fields})
        return {
            "schema": SCHEMA,
            "panel_id": self.panel_id,
            "case": dict(case),
            "state": "claimed",
            "claim_id": uuid4().hex,
            "worker_id": worker_id,
            "claimed_at": _now(),
            "previous_claims": history,
        }

    def _write(self, case: dict, record: dict, etag: str = "") -> str:
        return self.storage.put_bytes_conditional(
            _json_bytes(record),
            self.uri(case),
            if_match=etag,
            if_none_match=not bool(etag),
            content_type="application/json",
        )

    def start(self, version: CaseVersion) -> CaseVersion:
        """Commit the start marker before invoking the official evaluator.

        Args:
            version: Current pre-start claim returned by this store.
        Returns:
            Started record with its new ETag.
        Raises:
            ValueError: The provided version is not an unstarted claim.
            StoragePreconditionFailed: The worker lost ownership before starting.
        """
        record = dict(version.record)
        self._verify_identity(record, record["case"])
        if record["state"] != "claimed":
            raise ValueError("Only a claimed case can begin evaluation")
        record.update(state="started", started_at=_now())
        return CaseVersion(record, self._write(record["case"], record, version.etag))

    def complete(self, version: CaseVersion, validated_record: dict) -> CaseVersion:
        """Commit fully validated, durably uploaded original rollout evidence.

        Args:
            version: Started case version; never a new claim after a failure.
            validated_record: Direct inspect_rollout result after verified upload.
        Returns:
            Immutable completed record and its ETag.
        Raises:
            ValueError: Case identity or state differs from the original start.
            StoragePreconditionFailed: Another completion superseded this version.
        """
        record = dict(version.record)
        self._verify_identity(record, record["case"])
        if record["state"] != "started":
            raise ValueError("Completion requires the original started version")
        if any(
            validated_record.get(key) != value for key, value in record["case"].items()
        ):
            raise ValueError("Validated rollout differs from its prescribed case")
        record.update(state="complete", completed_at=_now(), rollout=validated_record)
        return CaseVersion(record, self._write(record["case"], record, version.etag))

    def artifact_prefix(self, version: CaseVersion) -> str:
        """Return the immutable original-artifact prefix for a case claim.

        Args:
            version: Claimed, started, or completed case version.
        Returns:
            Claim-specific S3 prefix under the stable panel.
        Raises:
            ValueError: Version has a conflicting identity.
        """
        record = version.record
        self._verify_identity(record, record["case"])
        return f"{self.prefix}/cases/{case_name(record['case'])}/artifacts/{record['claim_id']}"
