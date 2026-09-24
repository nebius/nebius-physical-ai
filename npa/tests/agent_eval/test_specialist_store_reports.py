"""Keep coordinator repair evidence separate from failed workers and bind current edits."""

import hashlib
import json

import pytest

from npa.agent_backend.specialists.config import (
    Profile,
    Operation,
    TeamConfig,
    fingerprint,
)
from npa.agent_backend.specialists.reports import task_report_for_store
from npa.agent_backend.specialists.store import TaskStore
from npa.agent_backend.specialists.team import SpecialistTeam
from npa.agent_backend.specialists.tools import WorkbenchTools


@pytest.fixture
def stores(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "source.txt").write_text("broken")
    profile = Profile(
        name="repair",
        description="Repair one file",
        model="synthetic",
        workspace=workspace,
        write_paths=["source.txt"],
        required_operations=["verify"],
        operations={
            "verify": Operation(
                argv=[
                    "{python}",
                    "-c",
                    "from pathlib import Path; assert Path('source.txt').read_text() == 'fixed'",
                ],
                description="Verify source",
            ),
        },
    )
    config = TeamConfig(
        state_directory=tmp_path / "workers",
        profiles=[profile],
        default_profile="repair",
    )
    team = SpecialistTeam(config)
    team.submit("Repair", specialist="repair", task_id="worker")
    team.store._update("worker", "needs_attention", error="Original generation failed")
    coordinator = TaskStore(tmp_path / "coordinator")
    coordinator._submit(
        "coordinator-repair", "repair", fingerprint(profile), "coordinator", "", {}
    )
    coordinator._update("coordinator-repair", "completed")
    return config, team, coordinator


def _execute(config, store, name, arguments, identity):
    return WorkbenchTools(config.profiles[0], store, "coordinator-repair").execute(
        {
            "id": identity,
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }
    )


def _edit(config, store, before, after, identity):
    return _execute(
        config,
        store,
        "edit_file",
        {
            "path": "source.txt",
            "expected_sha256": hashlib.sha256(before.encode()).hexdigest(),
            "old": before,
            "new": after,
        },
        identity,
    )


def test_completed_coordinator_placeholder_does_not_claim_verification(stores):
    config, team, coordinator = stores
    report = task_report_for_store(config, coordinator, "coordinator-repair")
    assert report["status_completed"] is True
    assert "model_completed" not in report
    assert report["required_operations_passed"] is False
    assert team.task_report("worker")["model_completed"] is False


def test_takeover_receipts_preserve_original_worker_and_invalidate_changed_source(
    stores,
):
    config, team, coordinator = stores
    original = team.status("worker")
    _edit(config, coordinator, "broken", "fixed", "repair")
    _execute(config, coordinator, "run_operation", {"name": "verify"}, "verification")
    report = task_report_for_store(config, coordinator, "coordinator-repair")
    assert report["required_operations_passed"] is True
    assert report["required_operations"]["verify"]["call_id"] == "verification"
    assert report["changes"][0]["sha256"] == hashlib.sha256(b"fixed").hexdigest()
    assert "fixed" not in json.dumps(report)
    (config.profiles[0].workspace / "source.txt").write_text("changed externally")
    changed = task_report_for_store(config, coordinator, "coordinator-repair")
    assert changed["required_operations_passed"] is False
    assert changed["changes"][0]["reason"] == "source_changed"
    assert team.status("worker") == original


def test_later_edit_and_failed_verification_cannot_reuse_old_success(stores):
    config, team, coordinator = stores
    _edit(config, coordinator, "broken", "fixed", "repair")
    _execute(config, coordinator, "run_operation", {"name": "verify"}, "good")
    _edit(config, coordinator, "fixed", "broken again", "regression")
    report = task_report_for_store(config, coordinator, "coordinator-repair")
    assert report["required_operations_passed"] is False
    assert report["required_operations"]["verify"] is None
    _execute(config, coordinator, "run_operation", {"name": "verify"}, "bad")
    failed = task_report_for_store(config, coordinator, "coordinator-repair")
    assert failed["required_operations_passed"] is False
    assert failed["failure_count"] == 1
    assert failed["required_operations"]["verify"]["returncode"] != 0
    assert team.status("worker")["error"] == "Original generation failed"
