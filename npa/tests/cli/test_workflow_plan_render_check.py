"""CLI coverage for the local production-render check on workflow plans."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner
import yaml

from npa.cli.main import app
from npa.cli.workbench import workflow as workflow_cli

RUNNER = CliRunner()
REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE = REPO_ROOT / "workflows" / "testing" / "vlm-eval-single.yaml"


def _workflow(tmp_path: Path, *, run_as_root: bool) -> Path:
    document = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    init_container = {
        "name": "initialize-output",
        "command": [
            "/bin/sh",
            "-ceu",
            "sudo -n install -d -o 1000 -g 1000 -m 0770 /owned/output",
        ],
        "volumeMounts": [{"name": "owned", "mountPath": "/owned"}],
    }
    if run_as_root:
        init_container["securityContext"] = {"runAsUser": 0}
    document["resources"]["gpu"]["kubernetes"] = {
        "pod_config": {
            "spec": {
                "volumes": [{"name": "owned", "emptyDir": {}}],
                "initContainers": [init_container],
                "containers": [{"name": "ray-node"}],
            }
        }
    }
    path = tmp_path / "workflow.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def test_plan_spec_render_check_reaches_production_security_guard(
    tmp_path: Path,
) -> None:
    workflow = _workflow(tmp_path, run_as_root=True)

    ordinary = RUNNER.invoke(app, ["workbench", "workflow", "plan-spec", str(workflow)])
    checked = RUNNER.invoke(
        app,
        ["workbench", "workflow", "plan-spec", str(workflow), "--check-render"],
    )

    assert ordinary.exit_code == 0, ordinary.output
    assert checked.exit_code == 1
    assert "runAsUser: 0 overrides are forbidden" in checked.output


def test_plan_spec_render_check_rejects_oversized_shell_argument(
    tmp_path: Path,
) -> None:
    workflow = _workflow(tmp_path, run_as_root=False)
    document = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    initial = document["initial"]
    document["states"][initial] = {
        "run": {"shell": "x" * 131_072},
        "resources": "gpu",
        "terminal": True,
    }
    workflow.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    ordinary = RUNNER.invoke(app, ["workbench", "workflow", "plan-spec", str(workflow)])
    checked = RUNNER.invoke(
        app,
        ["workbench", "workflow", "plan-spec", str(workflow), "--check-render"],
    )

    assert ordinary.exit_code == 0, ordinary.output
    assert checked.exit_code == 1
    assert "131072 UTF-8 bytes" in checked.output
    assert "declared workflow inputs" in checked.output


def test_plan_spec_render_check_is_secret_free_and_provider_free(
    tmp_path: Path,
    monkeypatch,
    mocker,
) -> None:
    workflow = _workflow(tmp_path, run_as_root=False)
    secret = "must-not-appear-in-render-check"
    monkeypatch.setenv("SKYPILOT_DOCKER_SERVER", "ghcr.io")
    monkeypatch.setenv("SKYPILOT_DOCKER_USERNAME", "operator")
    monkeypatch.setenv("SKYPILOT_DOCKER_PASSWORD", secret)
    preflight = mocker.patch("npa.cli.workbench.workflow._execution_target_preflight")
    submit = mocker.patch("npa.orchestration.skypilot.workflow.submit_workflow")

    result = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "plan-spec",
            str(workflow),
            "--check-render",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["render_check"] == {
        "provider_actions": False,
        "registry_secrets_materialized": False,
        "sha256": payload["render_check"]["sha256"],
        "status": "valid",
        "tasks": 1,
    }
    assert len(payload["render_check"]["sha256"]) == 64
    assert secret not in result.output
    preflight.assert_not_called()
    submit.assert_not_called()


def test_plan_spec_direct_call_keeps_check_render_disabled_by_default(
    tmp_path: Path,
    mocker,
) -> None:
    workflow = _workflow(tmp_path, run_as_root=True)
    render_check = mocker.patch.object(workflow_cli, "_check_static_plan_render")

    workflow_cli.plan_spec_cmd(
        workflow,
        run_id="",
        assume_decision="",
        var=[],
        preset="",
        waves=False,
        json_output=False,
    )

    render_check.assert_not_called()


def test_plan_spec_waves_json_includes_render_check(tmp_path: Path) -> None:
    workflow = _workflow(tmp_path, run_as_root=False)

    result = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "plan-spec",
            str(workflow),
            "--waves",
            "--check-render",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["render_check"]["status"] == "valid"
    assert payload["render_check"]["tasks"] == 1
