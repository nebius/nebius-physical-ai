from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from npa.cli.main import app

RUNNER = CliRunner()
SPEC = (
    Path(__file__).resolve().parents[2]
    / "workflows"
    / "workbench"
    / "npa-workflows"
    / "nurec-reconstruct.yaml"
)


def test_plan_reports_exact_toolref_dependency_without_provider_calls(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "npa.clients.credentials.load_credentials",
        lambda: (_ for _ in ()).throw(
            AssertionError("planning must not load credentials")
        ),
    )
    result = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "plan-spec",
            str(SPEC),
            "--run-id",
            "approval-plan",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    requirements = json.loads(result.stdout)["access_requirements"]
    assert requirements["hf"] == 0
    assert requirements["ngc"] == 1
    assert requirements["artifacts"][0]["artifact"] == "nvcr.io/nvidia/nre/nre-ga:26.04"
    assert requirements["artifacts"][0]["revision"] == "26.04"


def test_execute_json_blocks_before_runtime_without_prompt_or_browser(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("NPA_ACCESS_APPROVAL_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setattr(
        "npa.clients.credentials.load_credentials",
        lambda: SimpleNamespace(hf_token="", ngc_api_key=""),
    )
    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open_new_tab", opened.append)
    result = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "run-spec",
            str(SPEC),
            "--run-id",
            "approval-run",
            "--execute",
            "--json",
        ],
    )
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "blocked"
    assert payload["providers"]["ngc"][0]["status"] == "Pending"
    assert payload["resume_command"].startswith("npa workbench workflow run-spec")
    assert payload["legal_assent_performed"] is False
    assert opened == []
    assert "NGC_API_KEY" not in result.stdout


def test_public_unrelated_workflow_has_no_approval_gate() -> None:
    from npa.cli.workbench.workflow import _workflow_access_requirement_payload

    spec = SimpleNamespace(
        states={"view": SimpleNamespace(tool_ref="workbench.foxglove.convert")}
    )
    assert _workflow_access_requirement_payload(spec) == {
        "hf": 0,
        "ngc": 0,
        "artifacts": [],
    }


def test_enforcement_uses_explicit_project_scoped_credentials(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.cli.workbench.workflow import _enforce_workflow_access

    monkeypatch.setenv("NPA_ACCESS_APPROVAL_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setattr(
        "npa.clients.credentials.load_credentials",
        lambda: (_ for _ in ()).throw(AssertionError("must not load default credentials")),
    )
    monkeypatch.setattr(
        "npa.clients.huggingface.validate_hf_access",
        lambda token, *_args: SimpleNamespace(
            ok=token == "hf-project-scoped", status_code=200, error=""
        ),
    )
    spec = SimpleNamespace(
        states={"train": SimpleNamespace(tool_ref="workbench.groot.finetune")}
    )

    plan = _enforce_workflow_access(
        spec,
        json_output=True,
        resume_command="npa workbench workflow submit workflow.yaml",
        hf_token="hf-project-scoped",
        ngc_key="",
    )

    assert plan["status"] == "ready"
