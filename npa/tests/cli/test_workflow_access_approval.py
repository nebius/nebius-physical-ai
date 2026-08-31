from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

from npa.cli.main import app
from npa.workbench.model_access import GatedAsset, HF, NGC

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


def _blocked_hf_ngc_requirements() -> tuple[GatedAsset, ...]:
    return (
        GatedAsset(
            "nvidia/Cosmos-Reason2-2B",
            HF,
            ("groot",),
            True,
            revision="revision-hf",
            probe_path="weights/model.safetensors",
            official_url="https://huggingface.co/nvidia/Cosmos-Reason2-2B",
            terms_revision="terms-hf",
        ),
        GatedAsset(
            "nvcr.io/nvidia/nre/nre-ga:26.04",
            NGC,
            ("nurec",),
            True,
            repo_type="container",
            revision="26.04",
            official_url=(
                "https://catalog.ngc.nvidia.com/orgs/nvidia/nre/containers/nre-ga"
            ),
            terms_revision="terms-ngc",
        ),
    )


def test_interactive_workflow_gate_opens_exact_pages_only_after_affirmative_consent(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    from npa.cli.workbench.workflow import _enforce_workflow_access

    monkeypatch.setenv("NPA_ACCESS_APPROVAL_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setattr(
        "npa.cli.workbench.workflow._workflow_access_requirements",
        lambda _spec: _blocked_hf_ngc_requirements(),
    )
    monkeypatch.setattr("npa.cli.workbench.workflow.sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("npa.cli.workbench.workflow.typer.confirm", lambda *_a, **_k: True)
    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open_new_tab", opened.append)
    spec = SimpleNamespace(states={})

    with pytest.raises(typer.Exit):
        _enforce_workflow_access(
            spec,
            json_output=False,
            resume_command="npa workbench workflow run-spec workflow.yaml --execute",
            hf_token="",
            ngc_key="",
        )

    assert opened == [item.official_url for item in _blocked_hf_ngc_requirements()]
    output = capsys.readouterr().err
    assert "NPA did not accept any terms or start provisioning" in output
    assert "npa workbench workflow run-spec" in output


def test_interactive_workflow_gate_negative_answer_opens_nothing(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    from npa.cli.workbench.workflow import _enforce_workflow_access

    monkeypatch.setenv("NPA_ACCESS_APPROVAL_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setattr(
        "npa.cli.workbench.workflow._workflow_access_requirements",
        lambda _spec: _blocked_hf_ngc_requirements(),
    )
    monkeypatch.setattr("npa.cli.workbench.workflow.sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("npa.cli.workbench.workflow.typer.confirm", lambda *_a, **_k: False)
    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open_new_tab", opened.append)
    spec = SimpleNamespace(states={})

    with pytest.raises(typer.Exit):
        _enforce_workflow_access(
            spec,
            json_output=False,
            resume_command="npa workbench workflow run-spec workflow.yaml --execute",
            hf_token="",
            ngc_key="",
        )

    assert opened == []
    assert "npa workbench workflow run-spec" in capsys.readouterr().err


def test_nurec_workflow_gate_accepts_provider_validated_registry_credential(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.cli.workbench.workflow import _enforce_workflow_access

    secret = "registry-credential"
    observed: list[tuple[str, str]] = []
    monkeypatch.setenv("NPA_ACCESS_APPROVAL_STATE_PATH", str(tmp_path / "state.json"))

    def validate(key: str, *, image: str) -> str:
        observed.append((key, image))
        return "reachable"

    monkeypatch.setattr(
        "npa.workbench.nurec.nurec.check_ngc_image_access", validate
    )
    spec = SimpleNamespace(
        states={"reconstruct": SimpleNamespace(tool_ref="workbench.nurec.reconstruct")}
    )

    plan = _enforce_workflow_access(
        spec,
        json_output=True,
        resume_command="npa workbench workflow run-spec workflow.yaml --execute",
        hf_token="",
        ngc_key=secret,
    )

    assert plan["status"] == "ready"
    assert observed == [(secret, "nvcr.io/nvidia/nre/nre-ga:26.04")]
    assert secret not in json.dumps(plan, sort_keys=True)


def test_hf_workflow_gate_rechecks_pending_then_resumes_after_byte_entitlement(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.cli.workbench.workflow import _enforce_workflow_access

    monkeypatch.setenv("NPA_ACCESS_APPROVAL_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setattr(
        "npa.cli.workbench.workflow._workflow_access_requirements",
        lambda _spec: (_blocked_hf_ngc_requirements()[0],),
    )
    entitled = False
    observed: list[tuple[str, str]] = []

    def validate(_token, _repo, _repo_type, revision, probe_path):
        observed.append((revision, probe_path))
        return SimpleNamespace(
            ok=entitled,
            status_code=200 if entitled else 403,
            error="pending" if not entitled else "",
        )

    monkeypatch.setattr("npa.clients.huggingface.validate_hf_access", validate)
    spec = SimpleNamespace(states={})

    with pytest.raises(typer.Exit):
        _enforce_workflow_access(
            spec,
            json_output=True,
            resume_command="npa workbench workflow run-spec workflow.yaml --execute",
            hf_token="hf-synthetic",
            ngc_key="",
        )

    entitled = True
    plan = _enforce_workflow_access(
        spec,
        json_output=True,
        resume_command="npa workbench workflow run-spec workflow.yaml --execute",
        hf_token="hf-synthetic",
        ngc_key="",
    )

    assert plan["status"] == "ready"
    assert observed == [
        ("revision-hf", "weights/model.safetensors"),
        ("revision-hf", "weights/model.safetensors"),
    ]
