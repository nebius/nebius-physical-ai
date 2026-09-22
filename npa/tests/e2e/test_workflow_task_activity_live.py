"""Verify task activity against an operator-selected live failed workflow."""

import json
import os
from pathlib import Path

import pytest

from npa.cli.workbench.workflow import _durable_workflow_status

pytestmark = pytest.mark.e2e


def test_failed_workflow_retains_live_task_activity():
    config_path = os.environ.get("NPA_WORKFLOW_TASK_ACTIVITY_LIVE_CONFIG")
    if not config_path:
        pytest.skip("requires an operator-selected failed run and expected task states")
    settings = json.loads(Path(config_path).read_text())
    payload = _durable_workflow_status(
        settings["run_id"],
        project=settings["project"],
        isolated_config_dir=settings["isolated_config_dir"],
    )
    Path(settings["output_path"]).write_text(json.dumps(payload, indent=2) + "\n")
    assert payload["live_verified"] is True
    assert payload["status"] == settings["expected_workflow_status"]
    assert payload["raw_controller_state"] == "FAILED"
    activity = payload["scheduler_task_activity"]
    assert activity["active_stage_keys"] == settings["expected_active_stage_keys"]
    assert activity["unresolved_stage_keys"] == []
    assert (
        activity["all_stage_tasks_terminal"] is settings["expected_all_tasks_terminal"]
    )
    for key, state in settings["expected_task_states"].items():
        assert payload["stages"][key]["raw_task_scheduler_state"] == state
    conflicts = [
        key for key, stage in payload["stages"].items() if stage["outcome_conflict"]
    ]
    assert conflicts == settings["expected_outcome_conflicts"]
    for key in activity["active_stage_keys"]:
        assert payload["stages"][key]["raw_task_scheduler_state"] == "RUNNING"
