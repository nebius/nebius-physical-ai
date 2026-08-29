from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from npa.cli.main import app

runner = CliRunner()


def _credentials(*, hf: str = "", ngc: str = "") -> SimpleNamespace:
    return SimpleNamespace(hf_token=hf, ngc_api_key=ngc)


def test_health_prepare_json_is_prompt_and_browser_free(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("NPA_ACCESS_APPROVAL_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setattr(
        "npa.cli.workbench.health.load_credentials", lambda: _credentials()
    )
    opened: list[str] = []
    monkeypatch.setattr(
        "npa.cli.workbench.health.webbrowser.open_new_tab", opened.append
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "health",
            "access",
            "--capability",
            "groot",
            "--prepare",
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["schema_version"] == "npa.workbench.access-approval.v1"
    assert payload["status"] == "blocked"
    assert payload["counts"] == {"hf": 1, "ngc": 0}
    assert payload["legal_assent_performed"] is False
    assert payload["resume_command"].startswith("npa workbench health access")
    assert opened == []
    assert "HF_TOKEN" not in result.output


def test_health_prepare_open_pages_requires_affirmative_flag(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("NPA_ACCESS_APPROVAL_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setattr(
        "npa.cli.workbench.health.load_credentials", lambda: _credentials()
    )
    opened: list[str] = []
    monkeypatch.setattr(
        "npa.cli.workbench.health.webbrowser.open_new_tab", opened.append
    )

    declined = runner.invoke(
        app,
        ["workbench", "health", "access", "--capability", "groot", "--prepare"],
    )
    assert declined.exit_code == 1
    assert opened == []

    accepted = runner.invoke(
        app,
        [
            "workbench",
            "health",
            "access",
            "--capability",
            "groot",
            "--prepare",
            "--open-pages",
        ],
    )
    assert accepted.exit_code == 1
    assert opened == ["https://huggingface.co/nvidia/Cosmos-Reason2-2B"]
    assert "did not accept any terms" in accepted.output


def test_health_prepare_recheck_resumes_to_ready(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NPA_ACCESS_APPROVAL_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setattr(
        "npa.cli.workbench.health.load_credentials",
        lambda: _credentials(hf="hf-synthetic"),
    )
    monkeypatch.setattr(
        "npa.cli.workbench.health.validate_hf_access",
        lambda token, repo, repo_type: SimpleNamespace(
            ok=True, status_code=200, error=""
        ),
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "health",
            "access",
            "--capability",
            "groot",
            "--prepare",
            "--recheck",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "ready"
    assert payload["counts"] == {"hf": 0, "ngc": 0}
    assert "hf-synthetic" not in result.output


def test_configure_catalog_audit_is_optional_and_non_blocking(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("NPA_ACCESS_APPROVAL_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setattr(
        "npa.clients.credentials.load_credentials", lambda: _credentials()
    )
    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open_new_tab", opened.append)

    result = runner.invoke(app, ["configure", "--prepare-catalog-access"])

    assert result.exit_code == 0, result.output
    assert "Full Workbench catalog access audit" in result.output
    assert "Other Workbench capabilities remain usable" in result.output
    assert "did not accept any terms" in result.output
    assert opened == []
