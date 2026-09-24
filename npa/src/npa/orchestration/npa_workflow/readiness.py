"""Validate workflow readiness records against their adjacent workflow bytes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any


SCHEMA_VERSION = "workflow-readiness/v1"
SUPPORTED_STATUSES = frozenset({"verified", "unverified", "blocked", "not_applicable"})
_TOP_LEVEL_FIELDS = frozenset(
    {"schema_version", "workflow_sha256", "planning", "prerequisites"}
)
_PLANNING_FIELDS = frozenset({"validation", "task_fidelity"})
_PREREQUISITE_FIELDS = frozenset(
    {"output_storage", "worker_input", "credentials", "source_image", "target_runtime"}
)
_ENTRY_FIELDS = frozenset({"status", "reason", "evidence"})
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ReadinessRecordError(ValueError):
    """Raised when a readiness record is malformed or bound to other bytes."""


def _reject_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for field, value in pairs:
        if field in record:
            raise ReadinessRecordError(f"duplicate field {field!r}")
        record[field] = value
    return record


def _require_object(value: object, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReadinessRecordError(f"{location} must be an object")
    return value


def _require_exact_fields(
    value: dict[str, Any], expected: frozenset[str], location: str
) -> None:
    missing = sorted(expected - value.keys())
    unexpected = sorted(value.keys() - expected)
    if missing:
        raise ReadinessRecordError(f"{location} is missing required fields: {missing}")
    if unexpected:
        raise ReadinessRecordError(f"{location} has unsupported fields: {unexpected}")


def _validate_entry(value: object, location: str) -> None:
    entry = _require_object(value, location)
    _require_exact_fields(entry, _ENTRY_FIELDS, location)
    status = entry["status"]
    if not isinstance(status, str) or status not in SUPPORTED_STATUSES:
        raise ReadinessRecordError(
            f"{location}.status must be one of {sorted(SUPPORTED_STATUSES)}, got {status!r}"
        )
    reason = entry["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise ReadinessRecordError(f"{location}.reason must be a nonempty string")
    evidence = entry["evidence"]
    if not isinstance(evidence, list):
        raise ReadinessRecordError(f"{location}.evidence must be an array")
    if any(not isinstance(item, str) or not item.strip() for item in evidence):
        raise ReadinessRecordError(
            f"{location}.evidence entries must be nonempty strings"
        )
    if status == "verified" and not evidence:
        raise ReadinessRecordError(
            f"{location}.evidence must be nonempty when status is 'verified'"
        )


def _validate_section(value: object, expected: frozenset[str], location: str) -> None:
    section = _require_object(value, location)
    _require_exact_fields(section, expected, location)
    for name in sorted(expected):
        _validate_entry(section[name], f"{location}.{name}")


def _validate_structure(record: object, record_path: Path) -> dict[str, Any]:
    root = _require_object(record, str(record_path))
    _require_exact_fields(root, _TOP_LEVEL_FIELDS, str(record_path))
    if root["schema_version"] != SCHEMA_VERSION:
        raise ReadinessRecordError(
            f"{record_path}: schema_version must be {SCHEMA_VERSION!r}"
        )
    digest = root["workflow_sha256"]
    if not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(digest):
        raise ReadinessRecordError(
            f"{record_path}: workflow_sha256 must be 64 lowercase hexadecimal characters"
        )
    _validate_section(root["planning"], _PLANNING_FIELDS, f"{record_path}.planning")
    _validate_section(
        root["prerequisites"],
        _PREREQUISITE_FIELDS,
        f"{record_path}.prerequisites",
    )
    return root


def adjacent_workflow_path(record_path: Path) -> Path:
    """Resolve the workflow YAML that a readiness record describes.

    Args:
        record_path: Path ending in ``.readiness.json``.

    Returns:
        The adjacent path with the readiness suffix replaced by ``.yaml``.

    Raises:
        ReadinessRecordError: If the record does not use the required filename.
    """
    suffix = ".readiness.json"
    if not record_path.name.endswith(suffix):
        raise ReadinessRecordError(
            f"{record_path}: readiness filename must end with {suffix}"
        )
    stem = record_path.name.removesuffix(suffix)
    return record_path.with_name(f"{stem}.yaml")


def load_readiness_record(record_path: Path) -> dict[str, Any]:
    """Load and locally validate one digest-bound workflow readiness record.

    Args:
        record_path: Readiness JSON beside the workflow YAML it describes.

    Returns:
        The validated readiness record.

    Raises:
        ReadinessRecordError: If JSON, structure, adjacency, or digest is invalid.
    """
    try:
        text = record_path.read_text(encoding="utf-8")
        payload = json.loads(text, object_pairs_hook=_reject_duplicate_fields)
    except ReadinessRecordError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReadinessRecordError(
            f"{record_path}: could not load JSON: {exc}"
        ) from exc
    record = _validate_structure(payload, record_path)
    workflow_path = adjacent_workflow_path(record_path)
    if not workflow_path.is_file():
        raise ReadinessRecordError(
            f"{record_path}: adjacent workflow is missing: {workflow_path}"
        )
    actual = hashlib.sha256(workflow_path.read_bytes()).hexdigest()
    if record["workflow_sha256"] != actual:
        raise ReadinessRecordError(
            f"{record_path}: workflow_sha256 does not match {workflow_path}; "
            f"expected {actual}, found {record['workflow_sha256']}"
        )
    return record


def validate_repository_readiness_records(workflows_root: Path) -> list[Path]:
    """Validate every readiness record discovered below a workflows directory.

    Args:
        workflows_root: Repository workflow catalog root.

    Returns:
        Sorted paths of all validated readiness records.

    Raises:
        ReadinessRecordError: If no records exist or any discovered record fails.
    """
    record_paths = sorted(workflows_root.rglob("*.readiness.json"))
    if not record_paths:
        raise ReadinessRecordError(
            f"{workflows_root}: no workflow readiness records were discovered"
        )
    for record_path in record_paths:
        load_readiness_record(record_path)
    return record_paths
