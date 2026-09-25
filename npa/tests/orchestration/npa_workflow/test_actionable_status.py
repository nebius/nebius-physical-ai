"""Actionable scheduler projection over the existing npa.workflow manifest."""

from __future__ import annotations

from datetime import datetime, timezone
import json

import pytest

from npa.orchestration.npa_workflow.run_state import (
    NORMALIZED_DELETED_RAY_NODE,
    RunManifest,
    build_actionable_run_status,
    reconcile_submitted_manifest,
)
from npa.orchestration.skypilot.workflow import (
    _status_from_queue_payload,
    parse_task_statuses,
)


NOW = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)


def _manifest() -> RunManifest:
    return RunManifest(
        workflow="physical-ai-data-factory",
        run_id="paidf-status-fixture",
        api_version="npa.workflow/v0.0.1",
        sky_job_id="42",
        status="submitted",
        updated_at="2026-08-05T11:55:00Z",
        steps=[
            {"state": "augment", "status": "SUCCEEDED"},
            {
                "state": "curate",
                "status": "SUBMITTED",
                "resources_profile": {"accelerators": "RTXPRO6000:1"},
            },
            {"state": "visualize", "status": "SUBMITTED"},
        ],
    )


def test_pending_stage_is_not_collapsed_to_running() -> None:
    result = build_actionable_run_status(
        _manifest(),
        live_status="RUNNING",
        task_rows=[
            {"task_id": 0, "status": "SUCCEEDED", "end_at": "2026-08-05T11:50:00Z"},
            {"task_id": 1, "status": "PENDING", "submitted_at": "2026-08-05T11:51:00Z"},
        ],
        project="demo",
        now=NOW,
    )

    assert result["status"] == "PENDING"
    assert result["active_stage_name"] == "curate"
    assert result["active_stage_index"] == 2
    curate = result["stages"]["curate"]
    assert curate["scheduler_state"] == "PENDING"
    # A scheduler poll is an observation, not a workload heartbeat.  With no
    # real progress event there is deliberately no heartbeat age to fabricate.
    assert curate["last_observed_at"] == "2026-08-05T11:55:00Z"
    assert curate["last_heartbeat_at"] == ""
    assert curate["staleness_seconds"] is None
    assert curate["log_command"].endswith("--stage curate --project demo")


def test_retry_backoff_is_exposed() -> None:
    result = build_actionable_run_status(
        _manifest(),
        live_status="RUNNING",
        task_rows=[{"task_id": 1, "status": "PENDING", "retry_count": 4}],
        now=NOW,
    )

    assert result["status"] == "RETRYING"
    assert result["stages"]["curate"]["retry_count"] == 4


def test_repeated_deleted_ray_node_terminalizes_startup_without_cancellation() -> None:
    manifest = _manifest()
    output = "\n".join(
        [
            'container not found ("ray-node")',
            "cannot exec in a deleted state",
            'container not found ("ray-node")',
        ]
    )

    result = build_actionable_run_status(
        manifest,
        live_status="RUNNING",
        task_rows=[{"task_id": 1, "status": "PENDING"}],
        controller_output=output,
        failure_threshold=3,
        now=NOW,
    )

    assert result["status"] == "FAILED_STARTUP"
    assert result["raw_controller_state"] == "RUNNING"
    assert result["stages"]["curate"]["last_normalized_startup_failure"] == (
        NORMALIZED_DELETED_RAY_NODE
    )
    assert result["stages"]["curate"]["startup_failure_evidence"] == 3
    assert manifest.steps[1]["status"] == "FAILED_STARTUP"


def test_persisted_startup_failure_does_not_regress_when_logs_are_unavailable() -> None:
    manifest = _manifest()
    build_actionable_run_status(
        manifest,
        live_status="RUNNING",
        task_rows=[{"task_id": 1, "status": "PENDING"}],
        controller_output="\n".join(['container not found ("ray-node")'] * 3),
        failure_threshold=3,
        now=NOW,
    )

    # A later poll can still see the raw controller RUNNING while the bounded
    # controller-log query is temporarily unavailable. The durable classifier
    # must remain terminal and actionable instead of oscillating back to RUNNING.
    reconcile_submitted_manifest(
        manifest,
        live_status="RUNNING",
        task_rows=[{"task_id": 1, "status": "PENDING"}],
    )
    result = build_actionable_run_status(
        manifest,
        live_status="RUNNING",
        task_rows=[{"task_id": 1, "status": "PENDING"}],
        controller_output="",
        now=NOW,
    )

    assert result["status"] == "FAILED_STARTUP"
    assert result["raw_controller_state"] == "RUNNING"
    assert result["active_stage_name"] == "curate"
    assert result["stages"]["curate"]["last_normalized_startup_failure"] == (
        NORMALIZED_DELETED_RAY_NODE
    )


@pytest.mark.parametrize(
    ("live", "rows", "expected"),
    [
        (
            "SUCCEEDED",
            [
                {"task_id": 0, "status": "SUCCEEDED"},
                {"task_id": 1, "status": "SUCCEEDED"},
                {"task_id": 2, "status": "SUCCEEDED"},
            ],
            "SUCCEEDED",
        ),
        ("CANCELLED", [{"task_id": 1, "status": "CANCELLED"}], "CANCELLED"),
        ("", [], "SUBMITTED"),
    ],
)
def test_terminal_and_unavailable_scheduler_fixtures(live, rows, expected) -> None:  # noqa: ANN001
    manifest = _manifest()
    if live == "SUCCEEDED":
        for step in manifest.steps:
            step["status"] = "SUCCEEDED"
    result = build_actionable_run_status(
        manifest,
        live_status=live,
        task_rows=rows,
        now=NOW,
    )

    assert result["status"] == expected


@pytest.mark.parametrize("failed_task", ["FAILED", "FAILED_SETUP", "CANCELLED"])
def test_terminal_job_with_successful_parallel_member_is_not_a_conflict(failed_task):
    manifest = RunManifest(
        "parallel",
        "run-test",
        "npa.workflow/v0.0.1",
        sky_job_id="11",
        steps=[
            {"state": "prepare", "status": "succeeded"},
            {"state": "generate", "status": failed_task},
        ],
    )
    wave = {
        "key": "001|parallel|parallel:prepare:-,parallel:generate:-",
        "states": ["prepare", "generate"],
        "group": "parallel",
        "kind": "parallel",
        "job_id": "11",
        "attempt": 1,
        "status": "failed" if failed_task.startswith("FAILED") else "cancelled",
    }
    result = build_actionable_run_status(
        manifest,
        runtime_waves=[wave],
        job_observations={
            "11": {
                "status": "FAILED" if failed_task.startswith("FAILED") else "CANCELLED",
                "task_rows": [
                    {"task_id": 0, "task_name": "prepare", "status": "SUCCEEDED"},
                    {"task_id": 1, "task_name": "generate", "status": failed_task},
                ],
            }
        },
    )
    assert result["status"] == (
        "FAILED" if failed_task.startswith("FAILED") else "CANCELLED"
    )
    assert result["stages"]["prepare"]["state"] == "SUCCEEDED"
    assert result["stages"]["generate"]["state"] == failed_task
    assert all(not stage["outcome_conflict"] for stage in result["stages"].values())


@pytest.mark.parametrize("cancelled_first", [False, True])
@pytest.mark.parametrize("reverse_task_ids", [False, True])
@pytest.mark.parametrize(
    "failed_task",
    [
        "FAILED",
        "FAILED_SETUP",
        "FAILED_PRECHECKS",
        "FAILED_CONTROLLER",
        "FAILED_STARTUP",
    ],
)
def test_mixed_failure_and_cancellation_matches_actual_queue_aggregate(
    cancelled_first, reverse_task_ids, failed_task
):
    states = (
        ["CANCELLED", failed_task] if cancelled_first else [failed_task, "CANCELLED"]
    )
    rows = [
        {
            "job_id": "11",
            "task_id": 1 - index if reverse_task_ids else index,
            "task_name": f"member-{index}",
            "status": state,
        }
        for index, state in enumerate(states)
    ]
    queue_payload = json.dumps(rows)
    aggregate = _status_from_queue_payload(queue_payload, "11")
    assert aggregate == states[0]
    manifest = RunManifest(
        "parallel",
        "run-test",
        "npa.workflow/v0.0.1",
        sky_job_id="11",
        steps=[{"state": row["task_name"], "status": row["status"]} for row in rows],
    )
    wave = {
        "key": "001|parallel|parallel:member-0:-,parallel:member-1:-",
        "states": ["member-0", "member-1"],
        "group": "parallel",
        "kind": "parallel",
        "job_id": "11",
        "attempt": 1,
        "status": aggregate.lower(),
    }
    result = build_actionable_run_status(
        manifest,
        runtime_waves=[wave],
        job_observations={
            "11": {
                "status": aggregate,
                "task_rows": parse_task_statuses(queue_payload, "11"),
            }
        },
    )
    assert result["status"] == "FAILED"
    assert all(not stage["outcome_conflict"] for stage in result["stages"].values())
    assert [
        stage["raw_task_scheduler_state"] for stage in result["stages"].values()
    ] == states


@pytest.mark.parametrize(
    ("job_state", "task_state"),
    [
        ("FAILED", "CANCELLED"),
        ("CANCELLED", "FAILED_SETUP"),
        ("SUCCEEDED", "FAILED_CONTROLLER"),
        ("SUCCEEDED", "CANCELLED"),
        ("FAILED", "SUCCEEDED"),
        ("CANCELLED", "SUCCEEDED"),
    ],
)
def test_single_task_terminal_disagreement_survives_actual_queue_parsers(
    job_state, task_state
):
    job_snapshot = json.dumps([{"job_id": "11", "task_id": 0, "status": job_state}])
    task_snapshot = json.dumps(
        [{"job_id": "11", "task_id": 0, "task_name": "prepare", "status": task_state}]
    )
    manifest = RunManifest(
        "serial",
        "run-test",
        "npa.workflow/v0.0.1",
        sky_job_id="11",
        steps=[{"state": "prepare", "status": task_state}],
    )
    result = build_actionable_run_status(
        manifest,
        job_observations={
            "11": {
                "status": _status_from_queue_payload(job_snapshot, "11"),
                "task_rows": parse_task_statuses(task_snapshot, "11"),
            }
        },
    )
    assert result["status"] == "UNKNOWN"
    stage = result["stages"]["prepare"]
    assert stage["outcome_conflict"] is True
    assert stage["raw_job_scheduler_state"] == job_state
    assert stage["raw_task_scheduler_state"] == task_state
