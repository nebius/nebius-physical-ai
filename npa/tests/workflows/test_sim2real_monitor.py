"""Tests for Sim2Real workflow status monitor."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from npa.workflows.sim2real.monitor import (
    OperatorConfig,
    get_sim2real_workflow_status,
    normalize_staged_run_id,
    orchestrator_job_name,
    parse_submit_run_id,
)


def test_orchestrator_job_name() -> None:
    assert orchestrator_job_name("demo-run") == "sim2real-demo-run"


def test_normalize_staged_run_id_strips_polluted_submit_line() -> None:
    polluted = "sim2real-staged-20260615t120000z job=sim2real-staged-20260615t120000z"
    assert normalize_staged_run_id(polluted) == "sim2real-staged-20260615t120000z"


def test_parse_submit_run_id_from_combined_line() -> None:
    output = "run_id=sim2real-staged-20260615t120000z job=sim2real-staged-20260615t120000z"
    assert parse_submit_run_id(output) == "sim2real-staged-20260615t120000z"


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


def test_sim2real_module_status_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "run_id": "run-1",
        "status": "RUNNING",
        "current_stage": "stage_03_augment",
        "run_prefix_uri": "s3://demo-bucket/sim2real-b/run-1/",
        "stages": {"stage_01_trigger": {"state": "SUCCEEDED", "tier": ""}},
    }
    monkeypatch.setattr(
        "npa.workflows.sim2real.monitor.watch_sim2real_status",
        lambda run_id, **kwargs: payload,
    )
    from npa.workflows.sim2real.cli import main

    assert main(["status", "run-1", "--json"]) == 0


def test_workflow_cli_status_sim2real_run(monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner

    from npa.cli.main import app

    payload = {
        "run_id": "sim2real-staged-run-1",
        "status": "RUNNING",
        "current_stage": "stage_03_augment",
        "run_prefix_uri": "s3://demo-bucket/sim2real-b/sim2real-staged-run-1/",
        "stages": {"stage_01_trigger": {"state": "SUCCEEDED", "tier": ""}},
    }
    monkeypatch.setattr(
        "npa.workflows.sim2real.monitor.sim2real_run_exists",
        lambda run_id, **kwargs: True,
    )
    monkeypatch.setattr(
        "npa.workflows.sim2real.monitor.get_sim2real_workflow_status",
        lambda run_id, **kwargs: payload,
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["workbench", "workflow", "status", "sim2real-staged-run-1", "--json"],
    )
    assert result.exit_code == 0
    body = json.loads(result.stdout)
    assert body["run_id"] == "sim2real-staged-run-1"
    assert body["status"] == "RUNNING"


def test_is_sim2real_runbook() -> None:
    from npa.workflows.sim2real.k8s_submit import is_sim2real_runbook

    root = Path(__file__).resolve().parents[2]
    runbook = root / "workflows" / "workbench" / "sim2real" / "runbook.yaml"
    assert is_sim2real_runbook(runbook)
    assert not is_sim2real_runbook(root / "workflows" / "workbench" / "skypilot" / "vlm-eval.yaml")
