"""Pre-submit prerequisite checks, `--var` on plan/run-spec, and `stage-src`.

A first `npa workbench workflow submit` used to fail one prerequisite at a time
(no npa source, then no SkyPilot CLI, then a placeholder bucket), each as a
separate run, and there was no command to produce the npa source copy at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from npa.cli.main import app

runner = CliRunner()

SPEC = (
    Path(__file__).resolve().parents[3]
    / "npa"
    / "workflows"
    / "physical-ai-data-factory.yaml"
)


@pytest.fixture(autouse=True)
def _no_ambient_src(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NPA_SRC_S3_URI", raising=False)
    monkeypatch.delenv("NPA_E2E_NPA_SRC_S3_URI", raising=False)
    monkeypatch.delenv("NPA_SKYPILOT_BIN", raising=False)


def _submit(*args: str):
    return runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(SPEC),
            "--run-id",
            "preflight-demo",
            "--assume-decision",
            "promote_checkpoint",
            *args,
        ],
    )


def test_submit_lists_every_missing_prerequisite_at_once() -> None:
    result = _submit()

    assert result.exit_code == 1, result.output
    assert "missing prerequisites" in result.output
    # SkyPilot CLI, npa source and the placeholder bucket, all in one report.
    assert "SkyPilot CLI is not usable" in result.output
    assert "npa skypilot bootstrap" in result.output
    assert "NPA_SRC_S3_URI is unset" in result.output
    assert "--stage-src" in result.output
    assert "example-bucket" in result.output
    assert "--var bucket=<your-bucket>" in result.output
    assert "--skip-preflight" in result.output


def test_submit_preflight_does_not_reach_skypilot(mocker) -> None:
    submit_workflow = mocker.patch(
        "npa.orchestration.skypilot.workflow.submit_workflow"
    )

    result = _submit()

    assert result.exit_code == 1
    submit_workflow.assert_not_called()


def test_submit_preflight_clears_as_prerequisites_are_met(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Each satisfied prerequisite drops out of the report."""
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://real-bucket/npa-src/npa")
    result = _submit("--var", "bucket=real-bucket")
    assert result.exit_code == 1
    assert "NPA_SRC_S3_URI is unset" not in result.output
    assert "example-bucket" not in result.output
    assert "SkyPilot CLI is not usable" in result.output

    # ... and with a resolvable sky binary the preflight passes entirely.
    sky = tmp_path / "sky"
    sky.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    sky.chmod(0o755)
    monkeypatch.setenv("NPA_SKYPILOT_BIN", str(sky))
    result = _submit("--var", "bucket=real-bucket", "--plan-only")
    assert result.exit_code == 0, result.output
    assert "missing prerequisites" not in result.output


def test_plan_only_skips_runtime_only_prerequisites(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--plan-only` renders locally, so it must not demand a SkyPilot CLI."""
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://real-bucket/npa-src/npa")

    result = _submit("--plan-only")

    assert result.exit_code == 0, result.output
    assert "status: PLANNED" in result.output
    # The placeholder bucket is still surfaced, as a warning not a blocker.
    assert "example-bucket" in result.output


def test_skip_preflight_bypasses_the_checks(mocker) -> None:
    mocker.patch(
        "npa.orchestration.skypilot.workflow.submit_workflow",
        side_effect=AssertionError("submit reached"),
    )

    result = _submit("--skip-preflight")

    # Not the preflight error: the run got past it (and then failed later).
    assert "missing prerequisites" not in result.output


def test_image_override_satisfies_the_npa_source_requirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _submit("--image", "cr.example.invalid/reg/npa-tool:v1", "--plan-only")

    assert "NPA_SRC_S3_URI is unset" not in result.output


def test_image_none_still_requires_npa_source() -> None:
    """`--image none` pins every task to the default image, which has no npa."""
    result = _submit("--image", "none")

    assert result.exit_code == 1
    assert "NPA_SRC_S3_URI is unset" in result.output


def test_plan_spec_var_overrides_the_placeholder_bucket() -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "plan-spec",
            str(SPEC),
            "--run-id",
            "plan-demo",
            "--assume-decision",
            "promote_checkpoint",
            "--var",
            "bucket=my-real-bucket",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    rendered = json.dumps(payload)
    assert "my-real-bucket" in rendered
    assert "example-bucket" not in rendered


def test_plan_spec_without_var_warns_about_the_placeholder() -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "plan-spec",
            str(SPEC),
            "--run-id",
            "plan-demo",
            "--assume-decision",
            "promote_checkpoint",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "config.bucket is 'example-bucket'" in result.output
    assert "--var bucket=<your-bucket>" in result.output


def test_plan_spec_json_output_stays_machine_readable() -> None:
    """`--json` must emit a clean document, not the placeholder warning."""
    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "plan-spec",
            str(SPEC),
            "--run-id",
            "plan-demo",
            "--assume-decision",
            "promote_checkpoint",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "config.bucket is" not in result.output
    json.loads(result.output)  # parses even with stderr mixed in


def test_run_spec_accepts_var_overrides() -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "run-spec",
            str(SPEC),
            "--run-id",
            "run-demo",
            "--assume-decision",
            "promote_checkpoint",
            "--var",
            "bucket=my-real-bucket",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "my-real-bucket" in result.stdout
    assert "example-bucket" not in result.stdout
