from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from npa.cli.main import app
from npa.cli.workbench.antioch import _deployment


def test_run_propagates_explicit_non_cartpole_metadata(monkeypatch) -> None:  # noqa: ANN001
    captured = {}

    def run(request, **kwargs):  # noqa: ANN001, ANN003, ANN202
        captured["request"] = request
        return {"status": "completed"}

    monkeypatch.setattr("npa.sdk.workbench.antioch.run", run)
    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "antioch",
            "run",
            "--input-path",
            "s3://safe/input",
            "--output-path",
            "s3://safe/output",
            "--workflow-run",
            "run-1",
            "--state-id",
            "simulate",
            "--robot-type",
            "warehouse-arm",
            "--task",
            "Place the blue component in the inspection tray",
            "--suite",
            "stable-suite",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    request = captured["request"]
    assert request.robot_type == "warehouse-arm"
    assert request.task == "Place the blue component in the inspection tray"


def test_deployment_uses_terms_secret_and_workload_identity_storage() -> None:
    manifest = _deployment(
        "registry.invalid/npa-antioch:test",
        "npa-antioch",
        "workbench",
        "antioch-config",
        "service-token",
        "terms-acceptance",
    )
    container = manifest["items"][0]["spec"]["template"]["spec"]["containers"][0]
    env = {item["name"]: item for item in container["env"]}
    assert env["NPA_ANTIOCH_ACCEPT_TERMS"]["valueFrom"]["secretKeyRef"] == {
        "name": "terms-acceptance",
        "key": "accepted",
    }
    assert env["AWS_ENDPOINT_URL"]["value"].startswith("https://storage.")
    assert "envFrom" not in container
    rendered = json.dumps(manifest)
    assert "AWS_ACCESS_KEY_ID" not in rendered
    assert "AWS_SECRET_ACCESS_KEY" not in rendered


def test_antioch_image_contains_self_contained_storage_resolver() -> None:
    npa_root = Path(__file__).resolve().parents[2]
    dockerfile = (npa_root / "docker/workbench/antioch/Dockerfile").read_text(
        encoding="utf-8"
    )
    assert "COPY src/npa/workbench/antioch" in dockerfile
    assert "storage_config import resolve_storage_client" in dockerfile
    assert "COPY src/npa/clients/config.py" not in dockerfile


def test_live_start_uses_sdk_without_putting_credentials_in_options(
    monkeypatch, tmp_path: Path
) -> None:  # noqa: ANN001
    source = tmp_path / "source"
    bundle = tmp_path / "bundle"
    source.mkdir()
    bundle.mkdir()
    captured = {}

    def start(**kwargs):  # noqa: ANN003, ANN202
        captured.update(kwargs)
        return {"status": "running", "session": "npa-live-test"}

    monkeypatch.setattr("npa.sdk.workbench.antioch.live_start", start)
    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "antioch",
            "live-start",
            "--source",
            str(source),
            "--project-id",
            "assigned-project",
            "--client-bundle",
            str(bundle),
            "--scenario-timeout-seconds",
            "14400",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured == {
        "source": source,
        "project_id": "assigned-project",
        "client_bundle": bundle,
        "scenario_timeout_seconds": 14_400,
    }
    assert "api-key" not in result.output


def test_live_stop_orders_through_sdk(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        "npa.sdk.workbench.antioch.live_stop",
        lambda **kwargs: {"status": "stopped", **kwargs},
    )
    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "antioch",
            "live-stop",
            "--project-id",
            "assigned-project",
            "--timeout-seconds",
            "30",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "stopped"


def test_antioch_cluster_live_cli_uses_private_runtime_config(
    tmp_path: Path, monkeypatch
) -> None:  # noqa: ANN001
    runtime = tmp_path / "runtime.json"
    runtime.write_text("{}", encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        "npa.sdk.workbench.antioch.live_k8s_deploy",
        lambda **kwargs: calls.append(("deploy", kwargs)) or {"status": "ok"},
    )
    monkeypatch.setattr(
        "npa.sdk.workbench.antioch.live_k8s_status",
        lambda **kwargs: calls.append(("status", kwargs)) or {"status": "ready"},
    )
    runner = CliRunner()
    for command in ("live-k8s-deploy", "live-k8s-status"):
        result = runner.invoke(
            app,
            [
                "workbench",
                "antioch",
                command,
                "--runtime-config",
                str(runtime),
                "--output",
                "json",
            ],
        )
        assert result.exit_code == 0, result.output
    assert calls == [
        ("deploy", {"runtime_config": runtime}),
        ("status", {"runtime_config": runtime}),
    ]
