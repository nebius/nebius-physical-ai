from __future__ import annotations

import concurrent.futures
from pathlib import Path

import pytest

from npa.orchestration.npa_workflow.submission_state import (
    audit_project_submissions,
    inspect_submission_state,
    load_submission_state,
    submission_lock,
    submission_state_path,
    update_submission_state,
)


def test_submission_state_is_owner_only_and_restart_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    update_submission_state("demo", "run-1", {"source_uri": "s3://bucket/src/"})
    state = load_submission_state("demo", "run-1")

    assert state["source_uri"] == "s3://bucket/src/"
    assert state["schema_version"] == "npa.workflow.submission.v1"
    assert submission_state_path("demo", "run-1").stat().st_mode & 0o777 == 0o600


def test_submission_state_rejects_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    with pytest.raises(ValueError, match="must not contain"):
        update_submission_state("demo", "run-1", {"aws_secret_access_key": "nope"})


def test_concurrent_updates_do_not_corrupt_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    def write(index: int) -> None:
        update_submission_state("demo", "run-1", {f"field_{index}": index})

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write, range(20)))

    state = load_submission_state("demo", "run-1")
    assert all(state[f"field_{index}"] == index for index in range(20))


def test_locked_update_supports_a_multi_step_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    with submission_lock("demo", "run-1"):
        update_submission_state("demo", "run-1", {"launch_state": "reserved"}, locked=True)

    assert load_submission_state("demo", "run-1")["launch_state"] == "reserved"


def test_inspection_distinguishes_absent_and_corrupt_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    assert inspect_submission_state("demo", "run-1").outcome == "absent"

    update_submission_state("demo", "run-1", {"launch": {"status": "launching"}})
    assert inspect_submission_state("demo", "run-1").outcome == "found"

    submission_state_path("demo", "run-1").write_text("not-json", encoding="utf-8")
    inspected = inspect_submission_state("demo", "run-1")
    assert inspected.outcome == "unavailable"
    assert "invalid receipt JSON" in inspected.error


def test_project_audit_requires_every_exact_ledger_to_prove_no_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    assert audit_project_submissions("demo").outcome == "absent"

    update_submission_state("demo", "reserved", {"launch_state": "reserved"})
    audit = audit_project_submissions("demo")
    assert audit.outcome == "not_submitted"
    assert audit.ledger_count == 1

    update_submission_state("demo", "launched", {"launch": {"status": "launching"}})
    assert audit_project_submissions("demo").outcome == "launch_evidence"


def test_project_audit_rejects_symlinks_and_unavailable_ledgers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    update_submission_state("demo", "reserved", {"launch_state": "reserved"})
    foreign = submission_state_path("other", "foreign")
    foreign.parent.mkdir(parents=True)
    foreign.write_text("{}", encoding="utf-8")
    link = submission_state_path("demo", "linked")
    link.symlink_to(foreign)
    assert audit_project_submissions("demo").outcome == "unavailable"

    link.unlink()
    submission_state_path("demo", "corrupt").write_text("not-json", encoding="utf-8")
    assert audit_project_submissions("demo").outcome == "unavailable"
