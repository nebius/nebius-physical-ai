"""Keep live sibling task activity visible after a parallel workflow fails."""

import pytest

from npa.orchestration.npa_workflow.run_state import (
    RunManifest,
    build_actionable_run_status,
)


def _failed_parallel_status(rows, members=4):
    names = [f"worker-{index}" for index in range(members)]
    manifest = RunManifest(
        "evaluation",
        "test-run",
        "npa.workflow/v0.0.1",
        status="FAILED",
        sky_job_id="11",
        steps=[{"state": name, "status": "FAILED"} for name in names],
    )
    return build_actionable_run_status(
        manifest,
        runtime_waves=[
            {
                "key": "001|workers|" + ",".join(f"workers:{name}:-" for name in names),
                "states": names,
                "group": "workers",
                "kind": "parallel",
                "job_id": "11",
                "attempt": 1,
                "status": "failed",
            }
        ],
        job_observations={"11": {"status": "FAILED", "task_rows": rows}},
    )


def _rows(states):
    return [
        {"task_id": index, "task_name": f"worker-{index}", "status": state}
        for index, state in enumerate(states)
    ]


@pytest.mark.parametrize(
    "active_state", ["PENDING", "STARTING", "RUNNING", "RECOVERING", "CANCELLING"]
)
def test_failed_parallel_outcome_preserves_observed_active_siblings(active_state):
    result = _failed_parallel_status(
        _rows(["FAILED", "FAILED", active_state, active_state])
    )
    assert result["status"] == "FAILED"
    assert result["scheduler_task_activity"] == {
        "active_stage_keys": ["worker-2", "worker-3"],
        "unresolved_stage_keys": [],
        "all_stage_tasks_terminal": False,
    }
    assert all(stage["state"] == "FAILED" for stage in result["stages"].values())


def test_missing_sibling_rows_cannot_establish_task_completion():
    result = _failed_parallel_status(_rows(["FAILED"]))
    assert result["scheduler_task_activity"] == {
        "active_stage_keys": [],
        "unresolved_stage_keys": ["worker-1", "worker-2", "worker-3"],
        "all_stage_tasks_terminal": False,
    }


def test_one_unnamed_parallel_row_is_not_evidence_for_every_member():
    result = _failed_parallel_status([{"task_id": 0, "status": "FAILED"}])
    assert result["scheduler_task_activity"]["unresolved_stage_keys"] == [
        "worker-0",
        "worker-1",
        "worker-2",
        "worker-3",
    ]
    assert result["scheduler_task_activity"]["all_stage_tasks_terminal"] is False


def test_one_unnamed_row_can_be_attributed_to_a_single_stage_job():
    result = _failed_parallel_status([{"task_id": 0, "status": "FAILED"}], members=1)
    assert result["scheduler_task_activity"] == {
        "active_stage_keys": [],
        "unresolved_stage_keys": [],
        "all_stage_tasks_terminal": True,
    }


def test_terminal_job_without_task_observations_is_unresolved():
    result = _failed_parallel_status([])
    assert result["scheduler_task_activity"]["unresolved_stage_keys"] == [
        "worker-0",
        "worker-1",
        "worker-2",
        "worker-3",
    ]
    assert result["scheduler_task_activity"]["all_stage_tasks_terminal"] is False


def test_every_task_requires_an_explicit_terminal_observation():
    result = _failed_parallel_status(
        _rows(["FAILED", "FAILED_SETUP", "CANCELLED", "SUCCEEDED"])
    )
    assert result["scheduler_task_activity"] == {
        "active_stage_keys": [],
        "unresolved_stage_keys": [],
        "all_stage_tasks_terminal": True,
    }


def test_unrecognized_task_state_remains_unresolved():
    result = _failed_parallel_status(
        _rows(["FAILED", "new-state", "RUNNING", "FAILED"])
    )
    assert result["scheduler_task_activity"] == {
        "active_stage_keys": ["worker-2"],
        "unresolved_stage_keys": ["worker-1"],
        "all_stage_tasks_terminal": False,
    }


def test_extra_scheduler_task_prevents_terminal_proof():
    rows = _rows(["FAILED", "FAILED"])
    rows.append({"task_id": 2, "task_name": "worker-extra", "status": "RUNNING"})
    activity = _failed_parallel_status(rows, members=2)["scheduler_task_activity"]
    assert activity["all_stage_tasks_terminal"] is False
    assert activity["unresolved_stage_keys"] == ["worker-0", "worker-1"]


@pytest.mark.parametrize("reverse", [False, True])
def test_duplicate_named_rows_preserve_activity_and_ambiguity(reverse):
    rows = _rows(["FAILED", "FAILED"])
    rows.append({"task_id": 2, "task_name": "worker-0", "status": "RUNNING"})
    if reverse:
        rows.reverse()
    activity = _failed_parallel_status(rows, members=2)["scheduler_task_activity"]
    assert activity == {
        "active_stage_keys": ["worker-0"],
        "unresolved_stage_keys": ["worker-0"],
        "all_stage_tasks_terminal": False,
    }


def test_duplicate_task_ids_prevent_terminal_proof():
    rows = _rows(["FAILED", "FAILED"])
    rows[1]["task_id"] = 0
    activity = _failed_parallel_status(rows, members=2)["scheduler_task_activity"]
    assert activity["unresolved_stage_keys"] == ["worker-0", "worker-1"]
    assert activity["all_stage_tasks_terminal"] is False


def test_malformed_extra_row_prevents_terminal_proof():
    activity = _failed_parallel_status([*_rows(["FAILED", "FAILED"]), None], members=2)[
        "scheduler_task_activity"
    ]
    assert activity["unresolved_stage_keys"] == ["worker-0", "worker-1"]
    assert activity["all_stage_tasks_terminal"] is False


def test_legacy_task_ids_prove_terminal_only_with_unique_complete_coverage():
    manifest = RunManifest(
        "evaluation",
        "test-run",
        "npa.workflow/v0.0.1",
        status="FAILED",
        steps=[{"state": "worker-0", "status": "FAILED"}],
    )
    result = build_actionable_run_status(manifest, task_rows=_rows(["FAILED"]))
    assert result["scheduler_task_activity"]["all_stage_tasks_terminal"] is True
    result = build_actionable_run_status(
        manifest, task_rows=[*_rows(["FAILED"]), *_rows(["RUNNING"])]
    )
    assert result["scheduler_task_activity"]["all_stage_tasks_terminal"] is False
    assert result["scheduler_task_activity"]["active_stage_keys"] == ["worker-0"]


def test_one_named_row_cannot_prove_two_same_name_stages_terminal():
    manifest = RunManifest(
        "evaluation",
        "test-run",
        "npa.workflow/v0.0.1",
        status="FAILED",
        sky_job_id="11",
        steps=[{"state": "worker", "status": "FAILED"}] * 2,
    )
    result = build_actionable_run_status(
        manifest,
        job_observations={
            "11": {
                "status": "FAILED",
                "task_rows": [
                    {"task_id": 0, "task_name": "worker", "status": "FAILED"},
                ],
            }
        },
    )
    assert result["scheduler_task_activity"] == {
        "active_stage_keys": [],
        "unresolved_stage_keys": ["worker", "worker@1"],
        "all_stage_tasks_terminal": False,
    }
