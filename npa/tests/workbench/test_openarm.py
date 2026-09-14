"""OpenArm CLI, SDK, service, workflow, and packaging contracts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from npa.cli.main import app
from npa.orchestration.npa_workflow import build_plan, load_spec, validate_spec
from npa.orchestration.npa_workflow.catalog import argv_for_tool
from npa.sdk.workbench import openarm as sdk
from npa.workbench.openarm.schemas import OpenArmRunRequest, OpenArmStatusResponse
from npa.workbench.openarm.service import RunRegistry, create_app

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / "workflows/testing/openarm-simulators.yaml"


def test_cli_registered() -> None:
    result = CliRunner().invoke(app, ["workbench", "openarm", "--help"])
    assert result.exit_code == 0, result.output
    for command in ("run", "deploy", "delete", "status", "system-info"):
        assert command in result.output


def test_light_image_cli_imports_only_openarm_sdk() -> None:
    env = dict(os.environ)
    env.update(NPA_SKIP_EAGER_IMPORTS="1", NPA_LIGHT_WORKBENCH_TOOL="openarm")
    result = subprocess.run(
        [sys.executable, "-m", "npa", "workbench", "openarm", "--help"],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Enactic OpenArm" in result.stdout


def test_request_rejects_non_s3_output() -> None:
    with pytest.raises(ValueError, match="S3"):
        OpenArmRunRequest(simulator="mujoco", output_uri="/tmp/output")


def test_sdk_service_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_request(method, endpoint, path, **kwargs):
        captured.update(method=method, endpoint=endpoint, path=path, **kwargs)
        return {
            "run_id": "run-1",
            "status": "running",
            "simulator": "isaac-lab",
            "output_uri": "s3://bucket/openarm/",
            "manifest_sha256": "a" * 64,
        }

    monkeypatch.setattr(sdk, "_request", fake_request)
    response = sdk.run(
        simulator="isaac-lab",
        output_path="s3://bucket/openarm/",
        mode="service",
        endpoint="https://openarm.example",
        task="Isaac-Reach-OpenArm-v0",
    )
    assert response.run_id == "run-1"
    assert captured["payload"]["simulator"] == "isaac-lab"
    assert captured["payload"]["task"] == "Isaac-Reach-OpenArm-v0"


def test_service_auth_and_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENARM_AUTH_MODE", "token")
    monkeypatch.setenv("OPENARM_TOKEN", "test-token")
    registry = RunRegistry()
    registry.put(
        OpenArmStatusResponse(
            run_id="known",
            status="completed",
            simulator="mujoco",
            output_uri="s3://bucket/result/",
            result={"ok": True},
        )
    )
    client = TestClient(create_app(registry=registry))
    assert client.get("/health").status_code == 200
    assert client.get("/runs").status_code == 401
    response = client.get(
        "/status",
        params={"run_id": "known"},
        headers={"Authorization": "Bearer test-token"},
    )
    assert response.status_code == 200
    assert response.json()["result"] == {"ok": True}


def test_workflow_and_toolrefs_are_real_and_routed() -> None:
    spec = load_spec(WORKFLOW)
    validate_spec(spec)
    plan = build_plan(spec, run_id="test")
    assert [step.state for step in plan.steps] == [
        "mujoco-rollout",
        "isaac-rollout",
        "isaac-training",
    ]
    for ref in (
        "workbench.openarm.mujoco_rollout",
        "workbench.openarm.isaac_rollout",
        "workbench.openarm.isaac_train",
    ):
        argv = argv_for_tool(ref)
        assert argv[:4] == ["npa", "workbench", "openarm", "run"]
        assert "--output-path" in argv
    assert "--render" in argv_for_tool("workbench.openarm.mujoco_rollout")
    assert "--max-iterations" in argv_for_tool("workbench.openarm.isaac_train")


def test_packaging_pins_and_excludes_isaac_payload() -> None:
    dockerfile = (ROOT / "npa/docker/workbench/openarm/Dockerfile").read_text(
        encoding="utf-8"
    )
    assert "a8c979629f2591ad035d99d338ce114969e6cddc" in dockerfile
    assert "bad82e23716e6941c2de78ccb978f57c78b37734" in dockerfile
    assert "install_isaac_runtime_base.sh" in dockerfile
    assert "isaac-bootstrap ensure" not in dockerfile
    assert "nvcr.io/nvidia/isaac" not in dockerfile
    assert "--require-hashes" in dockerfile
    assert (ROOT / "npa/docker/workbench/openarm/THIRD_PARTY_NOTICES.md").is_file()
    components = json.loads(
        (ROOT / "npa/docker/workbench/openarm/components.json").read_text(
            encoding="utf-8"
        )
    )
    baked = {row["name"] for row in components["components"] if row["baked"]}
    assert "enactic/openarm_mujoco" in baked
    assert "enactic/openarm_isaac_lab" in baked
    assert "NVIDIA Isaac Sim and Isaac Lab" not in baked


def test_deploy_dry_run_redacts_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENARM_TOKEN", "do-not-print")
    monkeypatch.setattr(
        "npa.cli.workbench.openarm.load_credentials", lambda: type("C", (), {})()
    )
    monkeypatch.setattr(
        "npa.cli.workbench.openarm.apply_shared_credential_env",
        lambda env, credentials: None,
    )
    result = CliRunner().invoke(
        app,
        ["workbench", "openarm", "deploy", "--dry-run", "--output-format", "json"],
    )
    assert result.exit_code == 0, result.output
    assert "do-not-print" not in result.output
    payload = json.loads(result.output)
    assert payload["items"][0]["data"]["OPENARM_TOKEN"] == "<redacted>"
    pod_spec = payload["items"][1]["spec"]["template"]["spec"]
    assert pod_spec["nodeSelector"] == {
        "node.kubernetes.io/instance-type": "gpu-rtx6000"
    }
