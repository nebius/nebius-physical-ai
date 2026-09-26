"""Tests for strict, local workflow readiness-record validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from npa.orchestration.npa_workflow.readiness import (
    ReadinessRecordError,
    load_readiness_record,
    validate_repository_readiness_records,
)


def _record(workflow: bytes) -> dict:
    entry = {"status": "unverified", "reason": "Not checked.", "evidence": []}
    verified = {
        "status": "verified",
        "reason": "The local validator passed.",
        "evidence": ["local validation"],
    }
    return {
        "schema_version": "workflow-readiness/v1",
        "workflow_sha256": hashlib.sha256(workflow).hexdigest(),
        "planning": {"validation": verified, "task_fidelity": entry},
        "prerequisites": {
            "output_storage": entry,
            "worker_input": entry,
            "credentials": entry,
            "source_image": entry,
            "target_runtime": entry,
        },
    }


def _write_record(root: Path, name: str = "sample") -> Path:
    workflow = b"apiVersion: npa.workflow/v0.0.1\nkind: Workflow\n"
    (root / f"{name}.yaml").write_bytes(workflow)
    path = root / f"{name}.readiness.json"
    path.write_text(json.dumps(_record(workflow)), encoding="utf-8")
    return path


def test_loads_record_and_verifies_adjacent_workflow_digest(tmp_path: Path) -> None:
    path = _write_record(tmp_path)

    record = load_readiness_record(path)

    assert record["schema_version"] == "workflow-readiness/v1"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda record: record.update(schema_version="v2"), "schema_version"),
        (lambda record: record.pop("planning"), "missing required fields"),
        (
            lambda record: record["planning"]["validation"].update(status="ready"),
            "status must be one of",
        ),
        (
            lambda record: record["planning"]["validation"].update(reason=" "),
            "reason must be a nonempty string",
        ),
        (
            lambda record: record["planning"]["validation"].update(evidence=[]),
            "evidence must be nonempty",
        ),
        (
            lambda record: record["planning"]["validation"].update(evidence=[""]),
            "evidence entries must be nonempty strings",
        ),
    ],
)
def test_rejects_malformed_record_fields(
    tmp_path: Path, mutation, message: str
) -> None:
    path = _write_record(tmp_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    mutation(record)
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ReadinessRecordError, match=message):
        load_readiness_record(path)


def test_rejects_duplicate_fields(tmp_path: Path) -> None:
    path = _write_record(tmp_path)
    text = path.read_text(encoding="utf-8").replace(
        '"schema_version": "workflow-readiness/v1",',
        '"schema_version": "workflow-readiness/v1", "schema_version": "other",',
    )
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ReadinessRecordError, match="duplicate field 'schema_version'"):
        load_readiness_record(path)


def test_rejects_digest_mismatch_with_actionable_paths(tmp_path: Path) -> None:
    path = _write_record(tmp_path)
    (tmp_path / "sample.yaml").write_text("changed\n", encoding="utf-8")

    with pytest.raises(ReadinessRecordError, match=r"sample\.yaml; expected"):
        load_readiness_record(path)


def test_repository_discovery_rejects_orphaned_record(tmp_path: Path) -> None:
    _write_record(tmp_path)
    (tmp_path / "sample.yaml").unlink()

    with pytest.raises(ReadinessRecordError, match="adjacent workflow is missing"):
        validate_repository_readiness_records(tmp_path)


def test_repository_discovery_validates_all_records_without_fixed_count(
    tmp_path: Path,
) -> None:
    first = _write_record(tmp_path, "first")
    second = _write_record(tmp_path, "second")

    assert validate_repository_readiness_records(tmp_path) == [first, second]
