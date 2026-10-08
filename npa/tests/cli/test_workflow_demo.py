"""Verify demo launch delegates to the guarded runtime with exact resume semantics."""

import json
import os
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from npa.cli.workbench.workflow import app
from npa.cli.workbench.workflow import demo
from npa.clients.config import StorageConfig
from npa.orchestration.npa_workflow import demos

_REAL_VALIDATE = demo._validate_demo_plan
_ROOT = Path(__file__).resolve().parents[3]
_ISAAC = "registry.example.invalid/npa-isaac-lab@sha256:" + "a" * 64
_CPU = "registry.example.invalid/npa-sonic@sha256:" + "b" * 64


@pytest.fixture(autouse=True)
def isolated_demo_storage(monkeypatch):
    monkeypatch.setattr(demo, "demo_storage_environment", lambda selection: {})
    monkeypatch.setattr(demo, "_validate_demo_plan", lambda *args: None)


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


def test_run_identity_and_view_command_survive_quality_failure(monkeypatch):
    monkeypatch.setattr(
        demo,
        "prepare_demo",
        lambda *a, **kw: {
            "name": "real-to-sim",
            "run_id": "sample",
            "project": "example",
        },
    )

    def failed_submission(**kwargs):
        raise typer.Exit(1)

    monkeypatch.setattr("npa.cli.workbench.workflow.submit_cmd", failed_submission)
    result = CliRunner().invoke(app, ["demo", "run", "real-to-sim"])
    assert result.exit_code == 1
    assert "Run ID: sample" in result.stderr
    assert "demo view real-to-sim sample" in result.stderr


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


@pytest.mark.parametrize("fail", [False, True])
def test_demo_launch_uses_saved_project_pair_and_restores_caller_environment(
    monkeypatch, tmp_path, fail
):
    from npa.orchestration.npa_workflow.submit_credentials import (
        resolve_submit_credentials,
    )

    storage = StorageConfig(
        "example-bucket", "https://example.invalid", "saved-key", "saved-secret"
    )
    monkeypatch.setattr(
        demos, "_storage", lambda project: ("example", storage, "example-bucket", "")
    )
    monkeypatch.setattr(
        demos, "resolve_npa_workflow_spec", lambda name: tmp_path / name
    )
    monkeypatch.setattr(
        demo, "demo_storage_environment", demos.demo_storage_environment
    )
    for name in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_ENDPOINT_URL_S3",
        "NPA_SRC_S3_URI",
        "NPA_SKYPILOT_PROJECT",
    ):
        monkeypatch.setenv(name, "unrelated-value")
    monkeypatch.setenv("NGC_API_KEY", "retained-ngc")
    before = dict(os.environ)

    def submit(**kwargs):
        context = resolve_submit_credentials(
            project="example",
            explicit_endpoint=kwargs["s3_endpoint"],
            requested=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
        )
        assert (
            context.access_key_id,
            context.secret_access_key,
            context.session_token,
        ) == ("saved-key", "saved-secret", "")
        assert context.endpoint_url == "https://example.invalid"
        assert os.environ["NPA_SRC_S3_URI"] == ""
        assert os.environ["NPA_SKYPILOT_PROJECT"] == "example"
        assert os.environ["NGC_API_KEY"] == "retained-ngc"
        if fail:
            raise typer.Exit(1)

    monkeypatch.setattr("npa.cli.workbench.workflow.submit_cmd", submit)
    result = CliRunner().invoke(app, ["demo", "run", "nurec", "--project", "example"])
    assert result.exit_code == int(fail), result.output
    assert dict(os.environ) == before
    assert "saved-key" not in result.output and "saved-secret" not in result.output


def test_partial_project_pair_fails_before_submit(monkeypatch, tmp_path):
    storage = StorageConfig(
        "example-bucket", "https://example.invalid", "saved-key", ""
    )
    monkeypatch.setattr(
        demos, "_storage", lambda project: ("example", storage, "example-bucket", "")
    )
    monkeypatch.setattr(
        demos, "resolve_npa_workflow_spec", lambda name: tmp_path / name
    )
    monkeypatch.setattr(
        demo, "demo_storage_environment", demos.demo_storage_environment
    )
    monkeypatch.setattr(
        "npa.cli.workbench.workflow.submit_cmd",
        lambda **kwargs: pytest.fail("must fail before submission"),
    )
    result = CliRunner().invoke(app, ["demo", "run", "nurec", "--project", "example"])
    assert result.exit_code == 1
    assert "complete S3 credential pair" in result.output


def _real_plan_selection(monkeypatch, name):
    definition = demos.select_demo(name)
    selection = {
        "name": name,
        "yaml_path": _ROOT / "workflows/testing" / definition.workflow,
        "run_id": "operator-images",
        "project": "example",
        "var": [
            "bucket=example-bucket",
            "prefix=demos/explicit",
            "source_overlay=true",
        ],
        "report_uri": "s3://example-bucket/demos/explicit/reports/index.html",
    }
    monkeypatch.setattr(demo, "prepare_demo", lambda *args, **kwargs: selection)
    monkeypatch.setattr(demo, "_validate_demo_plan", _REAL_VALIDATE)
    return selection


@pytest.mark.parametrize("name", ["real-to-sim", "rl-improvement"])
@pytest.mark.parametrize("plan_only", [False, True])
def test_demo_missing_images_fails_before_storage_environment_or_submit(
    monkeypatch, name, plan_only
):
    _real_plan_selection(monkeypatch, name)
    monkeypatch.setattr(
        demo,
        "demo_storage_environment",
        lambda selection: pytest.fail("must not bind credentials"),
    )
    monkeypatch.setattr(
        "npa.cli.workbench.workflow.submit_cmd",
        lambda **kwargs: pytest.fail("must not contact provider"),
    )
    before = dict(os.environ)
    arguments = ["demo", "run", name] + (["--plan-only"] if plan_only else [])
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 1 and "Error:" in result.output
    assert "requires an explicit" in result.output and "--var" in result.output
    assert "Traceback" not in result.output and dict(os.environ) == before


@pytest.mark.parametrize(
    "name,images",
    [
        ("real-to-sim", {"isaac_image": _ISAAC, "assembly_image": _CPU}),
        ("rl-improvement", {"navigation_image": _ISAAC, "reconstruction_image": _CPU}),
        ("synthetic-data", {"isaac_image": _ISAAC}),
    ],
)
def test_demo_vars_reach_actual_plan_then_standard_submit(monkeypatch, name, images):
    _real_plan_selection(monkeypatch, name)
    calls = []
    monkeypatch.setattr(
        "npa.cli.workbench.workflow.submit_cmd", lambda **kwargs: calls.append(kwargs)
    )
    arguments = ["demo", "run", name, "--plan-only"]
    for key, value in images.items():
        arguments.extend(["--var", f"{key}={value}"])
    arguments.extend(["--var", "source_overlay=true"])
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 0, result.output
    assert len(calls) == 1 and calls[0]["plan_only"] is True
    for key, value in images.items():
        assert f"{key}={value}" in calls[0]["var"]
    assert "bucket=example-bucket" in calls[0]["var"]


@pytest.mark.parametrize(
    "variable,message",
    [
        ("missing-equals", "KEY=VALUE"),
        ("=no-key", "KEY=VALUE"),
        ("bucket=other-bucket", "Demo storage"),
        ("prefix=elsewhere", "Demo storage"),
    ],
)
def test_demo_vars_preserve_standard_parser_and_report_storage(
    monkeypatch, variable, message
):
    _real_plan_selection(monkeypatch, "synthetic-data")
    monkeypatch.setattr(demo, "_submit", lambda *args: pytest.fail("must not submit"))
    result = CliRunner().invoke(
        app, ["demo", "run", "synthetic-data", "--var", variable]
    )
    assert result.exit_code == 1 and message in result.output


@pytest.mark.parametrize("variable", ["missing-equals", "=no-key"])
def test_demo_invalid_vars_emit_actionable_json_errors(monkeypatch, variable):
    _real_plan_selection(monkeypatch, "synthetic-data")
    monkeypatch.setattr(demo, "_submit", lambda *args: pytest.fail("must not submit"))
    result = CliRunner().invoke(
        app,
        [
            "demo",
            "run",
            "synthetic-data",
            "--output-format",
            "json",
            "--var",
            variable,
        ],
    )
    assert result.exit_code == 1
    assert json.loads(result.stdout) == {
        "error": "Invalid --var format. Use KEY=VALUE."
    }
