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
        raise AssertionError(
            "durable workflow monitor should not run for sim2real runs"
        )

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


def test_workflow_status_explicit_generic_prefix_skips_sim2real_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit generic prefix must not consult unrelated legacy context."""

    def fail_legacy_probe(*args: object, **kwargs: object) -> bool:
        del args, kwargs
        raise AssertionError("explicit generic workflow prefix must skip sim2real")

    captured: dict[str, object] = {}

    def durable_status(run_id: str, **kwargs: object) -> dict[str, object]:
        captured["run_id"] = run_id
        captured.update(kwargs)
        return {"run_id": run_id, "status": "RUNNING", "stages": {}}

    monkeypatch.setattr(
        "npa.workflows.sim2real.monitor.sim2real_run_exists", fail_legacy_probe
    )
    monkeypatch.setattr(
        "npa.cli.workbench.workflow._durable_workflow_status", durable_status
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "status",
            "dm05-opendm-device-check",
            "--workflow-s3-prefix",
            "onboarding/dm05-opendm",
            "--s3-bucket",
            "operator-bucket",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["status"] == "RUNNING"
    assert captured["run_id"] == "dm05-opendm-device-check"
    assert captured["workflow_s3_prefix"] == "onboarding/dm05-opendm"
    assert captured["s3_bucket"] == "operator-bucket"
