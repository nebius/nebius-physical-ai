"""Actionable scheduler projection over the existing npa.workflow manifest."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from npa.orchestration.npa_workflow.run_state import (
    NORMALIZED_DELETED_RAY_NODE,
    RunManifest,
    build_actionable_run_status,
    reconcile_submitted_manifest,
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
