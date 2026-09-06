"""Separate operator runtimes must not share workflow receipt authority."""

from pathlib import Path

import pytest

from npa.orchestration.npa_workflow import submission_state as state


@pytest.fixture
def private_home(tmp_path, monkeypatch):
    root = tmp_path / "synthetic-home"
    root.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: root))
    return root


def test_independent_runtime_cannot_read_or_overwrite_another_launch(
    private_home, tmp_path, monkeypatch
):
    first = tmp_path / "first-runtime"
    second = tmp_path / "second-runtime"
    monkeypatch.setenv("NPA_CONFIG_DIR", str(first))
    state.update_submission_state("demo", "same-run", {"launch": {"sky_job_id": "41"}})
    first_path = state.submission_state_path("demo", "same-run")
    first_bytes = first_path.read_bytes()
    monkeypatch.setenv("NPA_CONFIG_DIR", str(second))
    assert state.inspect_submission_state("demo", "same-run").outcome == "absent"
    state.update_submission_state("demo", "same-run", {"launch": {"sky_job_id": "92"}})
    assert (
        state.load_submission_state("demo", "same-run")["launch"]["sky_job_id"] == "92"
    )
    assert first_path.read_bytes() == first_bytes
    monkeypatch.setenv("NPA_CONFIG_DIR", str(first))
    assert (
        state.load_submission_state("demo", "same-run")["launch"]["sky_job_id"] == "41"
    )
    assert not (private_home / ".npa").exists()


def test_cleanup_audit_excludes_launches_from_unrelated_runtime(
    private_home, tmp_path, monkeypatch
):
    monkeypatch.setenv("NPA_CONFIG_DIR", str(tmp_path / "first-runtime"))
    state.update_submission_state(
        "demo", "launched", {"launch": {"status": "submitted", "sky_job_id": "41"}}
    )
    monkeypatch.setenv("NPA_CONFIG_DIR", str(tmp_path / "second-runtime"))
    state.update_submission_state("demo", "planned", {"launch_state": "planned"})
    audit = state.audit_project_submissions("demo")
    assert audit.outcome == "not_submitted"
    assert audit.ledger_count == 1


def test_default_runtime_preserves_existing_receipt_location(private_home, monkeypatch):
    monkeypatch.delenv("NPA_CONFIG_DIR", raising=False)
    state.update_submission_state("demo", "same-run", {"launch_state": "reserved"})
    path = state.submission_state_path("demo", "same-run")
    assert (
        path
        == private_home / ".npa" / "workflow-submissions" / "demo" / "same-run.json"
    )
    assert state.inspect_submission_state("demo", "same-run").outcome == "found"
    assert path.stat().st_mode & 0o777 == 0o600
