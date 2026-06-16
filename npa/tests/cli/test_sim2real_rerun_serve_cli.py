from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from npa.cli.main import app


runner = CliRunner()


def test_sim2real_rerun_serve_help_lists_run_id() -> None:
    result = runner.invoke(app, ["workbench", "sim2real", "rerun", "serve", "--help"])
    assert result.exit_code == 0
    assert "--run-id" in result.output


def test_sim2real_status_command(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "run_id": "sim2real-staged-run-1",
        "status": "RUNNING",
        "current_stage": "stage_03_augment",
        "run_prefix_uri": "s3://demo-bucket/sim2real-b/sim2real-staged-run-1/",
        "stages": {"stage_01_trigger": {"state": "SUCCEEDED", "tier": ""}},
    }
    def fake_watch(run_id: str, **kwargs: object) -> dict:
        del run_id, kwargs
        print(json.dumps(payload))
        return payload

    monkeypatch.setattr(
        "npa.workflows.sim2real.monitor.watch_sim2real_status",
        fake_watch,
    )
    result = runner.invoke(
        app,
        [
            "workbench",
            "sim2real",
            "status",
            "--run-id",
            "sim2real-staged-run-1",
            "--json",
        ],
    )
    assert result.exit_code == 0
    body = json.loads(result.stdout)
    assert body["run_id"] == "sim2real-staged-run-1"
    assert body["status"] == "RUNNING"


def test_sim2real_rerun_serve_dry_run_prints_manifest(mocker) -> None:
    mocker.patch(
        "npa.cli.workbench.sim2real._rerun_serve_credentials",
        return_value=("ak", "sk"),
    )
    mocker.patch(
        "npa.cli.workbench.sim2real.build_rerun_serve_config",
        return_value=mocker.Mock(
            deployment_name="npa-sim2real-rerun-demo",
            secret_name="npa-sim2real-rerun-demo-s3",
        ),
    )
    manifest = {
        "kind": "List",
        "items": [
            {"kind": "Secret", "data": {"S3_URI": "redacted"}},
            {"kind": "Deployment"},
            {"kind": "Service"},
        ],
    }
    mocker.patch("npa.cli.workbench.sim2real.build_rerun_serve_manifest", return_value=manifest)
    mocker.patch(
        "npa.cli.workbench.sim2real.redact_rerun_serve_manifest",
        return_value=manifest,
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "sim2real",
            "rerun",
            "serve",
            "--run-id",
            "demo-run",
            "--dry-run",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["kind"] == "List"
    assert [item["kind"] for item in payload["items"]] == ["Secret", "Deployment", "Service"]


def test_sim2real_rerun_serve_deploy_emits_public_url(mocker) -> None:
    mocker.patch(
        "npa.cli.workbench.sim2real._rerun_serve_credentials",
        return_value=("ak", "sk"),
    )
    config = mocker.Mock()
    mocker.patch("npa.cli.workbench.sim2real.build_rerun_serve_config", return_value=config)
    mocker.patch(
        "npa.cli.workbench.sim2real.require_kubeconfig",
        return_value="/tmp/kubeconfig",
    )
    mocker.patch(
        "npa.cli.workbench.sim2real.apply_rerun_serve",
        return_value=mocker.Mock(
            to_dict=lambda: {
                "status": "deployed",
                "run_id": "demo-run",
                "rrd_s3_uri": "s3://bucket/sim2real-b/demo-run/reports/sim2real.rrd",
                "public_url": "http://203.0.113.10:9090/?url=rerun%2Bhttp%3A%2F%2F203.0.113.10%3A9876%2Fproxy",
                "local_url": "http://127.0.0.1:9090/?url=rerun%2Bhttp%3A%2F%2F127.0.0.1%3A9876%2Fproxy",
                "cluster_url": "http://svc.default.svc.cluster.local:9090",
                "port_forward_command": "kubectl port-forward -n default deployment/x 9090:9090 9876:9876",
            }
        ),
    )

    result = runner.invoke(
        app,
        ["workbench", "sim2real", "rerun", "serve", "--run-id", "demo-run"],
    )

    assert result.exit_code == 0
    assert "public_url: http://203.0.113.10:9090/" in result.output
    assert "local_url: http://127.0.0.1:9090/" in result.output
    assert "port_forward:" in result.output


def test_sim2real_rerun_serve_deploy_emits_local_url_when_public_pending(mocker) -> None:
    mocker.patch(
        "npa.cli.workbench.sim2real._rerun_serve_credentials",
        return_value=("ak", "sk"),
    )
    config = mocker.Mock()
    mocker.patch("npa.cli.workbench.sim2real.build_rerun_serve_config", return_value=config)
    mocker.patch(
        "npa.cli.workbench.sim2real.require_kubeconfig",
        return_value="/tmp/kubeconfig",
    )
    mocker.patch(
        "npa.cli.workbench.sim2real.apply_rerun_serve",
        return_value=mocker.Mock(
            to_dict=lambda: {
                "status": "deployed",
                "run_id": "demo-run",
                "rrd_s3_uri": "s3://bucket/sim2real-b/demo-run/reports/sim2real.rrd",
                "public_url": "",
                "local_url": "http://127.0.0.1:9090/?url=rerun%2Bhttp%3A%2F%2F127.0.0.1%3A9876%2Fproxy",
                "service_type": "LoadBalancer",
                "deployment_name": "npa-sim2real-rerun-npa-rtxpro-mk8s",
                "namespace": "default",
                "cluster_url": "http://svc.default.svc.cluster.local:9090",
                "port_forward_command": "kubectl port-forward -n default deployment/x 9090:9090 9876:9876",
            }
        ),
    )

    result = runner.invoke(
        app,
        ["workbench", "sim2real", "rerun", "serve", "--run-id", "demo-run"],
    )

    assert result.exit_code == 0
    assert "public_url: pending" in result.output
    assert "local_url: http://127.0.0.1:9090/" in result.output
    assert "port_forward:" in result.output


def test_sim2real_hidden_from_workbench_help() -> None:
    result = runner.invoke(app, ["workbench", "--help"])
    assert result.exit_code == 0
    assert "sim2real" not in result.output
