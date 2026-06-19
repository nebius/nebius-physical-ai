"""Workflow status must route sim2real runs before durable S3 monitor."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from npa.cli.main import app

runner = CliRunner()


def test_workflow_status_sim2real_preempts_s3_bucket_monitor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "run_id": "sim2real-staged-run-1",
        "status": "RUNNING",
        "current_stage": "stage_10_eval_heldout",
        "run_prefix_uri": "s3://demo-bucket/sim2real-b/sim2real-staged-run-1/",
        "stages": {"stage_01_trigger": {"state": "SUCCEEDED", "tier": ""}},
    }

    def fake_get_status(run_id: str, **kwargs: object) -> dict:
        del kwargs
        assert run_id == "sim2real-staged-run-1"
        return payload

    def fail_durable(*args: object, **kwargs: object) -> dict:
        del args, kwargs
        raise AssertionError("durable workflow monitor should not run for sim2real runs")

    monkeypatch.setattr(
        "npa.workflows.sim2real.monitor.sim2real_run_exists",
        lambda run_id, **kwargs: run_id == "sim2real-staged-run-1",
    )
    monkeypatch.setattr(
        "npa.workflows.sim2real.monitor.get_sim2real_workflow_status",
        fake_get_status,
    )
    monkeypatch.setattr(
        "npa.cli.workbench.workflow._durable_workflow_status",
        fail_durable,
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "status",
            "sim2real-staged-run-1",
            "--s3-bucket",
            "demo-bucket",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    body = json.loads(result.stdout)
    assert body["run_id"] == "sim2real-staged-run-1"
    assert body["current_stage"] == "stage_10_eval_heldout"
