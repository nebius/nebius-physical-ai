"""Live validation of NPA workflow specs against real infrastructure (optional)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.orchestration.npa_workflow import build_plan, load_spec, validate_spec

pytestmark = pytest.mark.skipif(
    os.environ.get("NPA_INTEGRATION_E2E") != "1",
    reason="Set NPA_INTEGRATION_E2E=1 to run live NPA workflow spec checks.",
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SPECS = REPO_ROOT / "npa" / "workflows" / "workbench" / "npa-workflows"
RUNNER = CliRunner()


@pytest.mark.parametrize(
    "name",
    ["vlm-eval-single.yaml", "tokenfactory-rollout-judge.yaml", "sim2real-vlm-rl.yaml"],
)
def test_live_npa_workflow_specs_plan(name: str) -> None:
    """Ensure golden specs load and expand on the operator machine."""

    spec = load_spec(SPECS / name)
    validate_spec(spec)
    plan = build_plan(spec, run_id="live-spec-check")
    assert plan.steps, name


@pytest.mark.parametrize(
    "name",
    ["vlm-eval-single.yaml", "tokenfactory-rollout-judge.yaml", "sim2real-vlm-rl.yaml"],
)
def test_live_npa_workflow_cli_validate_and_plan(name: str) -> None:
    """Exercise validate-spec / plan-spec / run-spec --plan-only on live creds."""

    path = SPECS / name
    validate = RUNNER.invoke(app, ["workbench", "workflow", "validate-spec", str(path), "--json"])
    assert validate.exit_code == 0, validate.output
    payload = json.loads(validate.output)
    assert payload["status"] == "valid"

    plan = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "plan-spec",
            str(path),
            "--run-id",
            "live-cli-check",
            "--assume-decision",
            "promote_checkpoint",
            "--json",
        ],
    )
    assert plan.exit_code == 0, plan.output
    plan_payload = json.loads(plan.output)
    assert plan_payload["steps"], name

    run = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "run-spec",
            str(path),
            "--run-id",
            "live-cli-check",
            "--plan-only",
            "--assume-decision",
            "promote_checkpoint",
            "--json",
        ],
    )
    assert run.exit_code == 0, run.output
    run_payload = json.loads(run.output)
    assert run_payload["steps"], name
