"""Verify demo launch delegates to the guarded runtime with exact resume semantics."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from npa.cli.workbench.workflow import app
from npa.cli.workbench.workflow import demo


def test_demo_catalog_is_available_without_credentials():
    result = CliRunner().invoke(app, ["demo", "list", "--output-format", "json"])
    assert result.exit_code == 0, result.output
    assert len(json.loads(result.stdout)["demos"]) == 4


@pytest.mark.parametrize("resume", [False, True])
def test_demo_uses_standard_submit_with_source_runtime_and_no_deadline(
    monkeypatch, resume
):
    selection = {
        "name": "nurec",
        "yaml_path": Path("nurec-reconstruct.yaml"),
        "run_id": "existing-run" if resume else "fresh-run",
        "project": "example",
        "report_uri": "s3://example-bucket/result/reports/index.html",
        "secret_env": ["NGC_API_KEY"],
        "var": ["source_overlay=true"],
    }
    monkeypatch.setattr(demo, "prepare_demo", lambda *a, **kw: selection)
    calls = []
    monkeypatch.setattr(
        "npa.cli.workbench.workflow.submit_cmd", lambda **kw: calls.append(kw)
    )
    arguments = ["demo", "run", "nurec", "--project", "example"]
    if resume:
        arguments += ["--resume-run", "existing-run"]
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    call = calls[0]
    assert call["runtime"] is True and call["stage_src"] is True
    assert call["durable_s3"] is True and call["max_wait_seconds"] == 0
    assert call["resume_run"] == ("existing-run" if resume else "")
    assert call["run_id"] == ("" if resume else "fresh-run")
    assert "report_uri" not in call
    assert "demo view nurec" in result.output


def test_run_and_resume_conflict_before_preparation(monkeypatch):
    monkeypatch.setattr(
        demo, "prepare_demo", lambda *a, **kw: pytest.fail("must not prepare")
    )
    result = CliRunner().invoke(
        app, ["demo", "run", "nurec", "--run-id", "new", "--resume-run", "old"]
    )
    assert result.exit_code == 1
    assert "either" in result.output


def test_json_launch_separates_submit_diagnostics(monkeypatch):
    monkeypatch.setattr(
        demo,
        "prepare_demo",
        lambda *a, **kw: {"name": "nurec", "run_id": "sample", "project": "example"},
    )

    def submit(**kwargs):
        print("Source staging completed")
        print(json.dumps({"run_id": "sample", "result": "submitted"}))

    monkeypatch.setattr("npa.cli.workbench.workflow.submit_cmd", submit)
    result = CliRunner().invoke(
        app, ["demo", "run", "nurec", "--output-format", "json"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"run_id": "sample", "result": "submitted"}
    assert "demo view nurec sample" in result.stderr


def test_plan_only_retains_standard_preflights_and_does_not_offer_completed_report(
    monkeypatch,
):
    monkeypatch.setattr(
        demo,
        "prepare_demo",
        lambda *a, **kw: {"name": "nurec", "run_id": "preview", "project": "example"},
    )
    calls = []
    monkeypatch.setattr(
        "npa.cli.workbench.workflow.submit_cmd", lambda **kw: calls.append(kw)
    )
    result = CliRunner().invoke(app, ["demo", "run", "nurec", "--plan-only"])
    assert result.exit_code == 0
    assert calls[0]["plan_only"] is True
    assert "skip_preflight" not in calls[0] and "preflight_images" not in calls[0]
    assert "View the measured results" not in result.output
