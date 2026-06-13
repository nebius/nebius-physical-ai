"""Tests for Sim2Real workflow status monitor."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from npa.workflows.sim2real.monitor import (
    OperatorConfig,
    get_sim2real_workflow_status,
    orchestrator_job_name,
)


def test_orchestrator_job_name() -> None:
    assert orchestrator_job_name("demo-run") == "sim2real-demo-run"


def test_sim2real_workflow_status_marks_failed_image_pull(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = OperatorConfig(
        bucket="demo-bucket",
        endpoint_url="https://storage.example",
        registry="cr.example/registry",
        k8s_context="demo-context",
    )
    monkeypatch.setattr(
        "npa.workflows.sim2real.monitor.load_operator_config",
        lambda: config,
    )
    monkeypatch.setattr(
        "npa.workflows.sim2real.monitor.resolve_kubeconfig",
        lambda context: tmp_path / "kubeconfig",
    )

    job_payload = {"status": {"active": 1, "succeeded": 0, "failed": 0}}
    pod_payload = {
        "items": [
            {
                "status": {
                    "phase": "Pending",
                    "containerStatuses": [
                        {"state": {"waiting": {"reason": "ImagePullBackOff"}}}
                    ],
                }
            }
        ]
    }

    def fake_kubectl_json(args, *, kubeconfig):
        del kubeconfig
        if args[-1] == orchestrator_job_name("run-1"):
            return job_payload
        return pod_payload

    monkeypatch.setattr(
        "npa.workflows.sim2real.monitor._kubectl_json",
        fake_kubectl_json,
    )
    monkeypatch.setattr(
        "npa.workflows.sim2real.monitor._k8s_sibling_summary",
        lambda **kwargs: [],
    )

    fake_client = MagicMock()
    fake_client._s3.head_object.side_effect = Exception("missing")
    fake_client._s3.list_objects_v2.return_value = {"KeyCount": 0}
    monkeypatch.setattr(
        "npa.workflows.sim2real.monitor.StorageClient.from_environment",
        lambda **kwargs: fake_client,
    )

    result = get_sim2real_workflow_status("run-1")
    assert result["status"] == "FAILED"
    assert result["pod_reason"] == "ImagePullBackOff"
    assert result["stages"]["stage_01_trigger"]["state"] == "PENDING"


def test_workflow_cli_status_sim2real_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner

    from npa.cli.main import app

    payload = {
        "run_id": "run-1",
        "status": "RUNNING",
        "current_stage": "stage_03_augment",
        "run_prefix_uri": "s3://demo-bucket/sim2real-b/run-1/",
        "stages": {"stage_01_trigger": {"state": "SUCCEEDED", "tier": ""}},
    }
    monkeypatch.setattr(
        "npa.workflows.sim2real.monitor.get_sim2real_workflow_status",
        lambda run_id, **kwargs: payload,
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["workbench", "workflow", "status", "run-1", "--tool", "sim2real", "--json"],
    )
    assert result.exit_code == 0
    body = json.loads(result.stdout)
    assert body["run_id"] == "run-1"
    assert body["status"] == "RUNNING"
