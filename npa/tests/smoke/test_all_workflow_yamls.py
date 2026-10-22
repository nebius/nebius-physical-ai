"""Smoke validation for every checked-in workflow YAML."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.orchestration.npa_workflow import API_VERSION, API_VERSION_BETA
from npa.orchestration.npa_workflow import build_plan, load_spec, validate_spec
from npa.orchestration.npa_workflow.blueprints import iter_npa_workflow_specs
from npa.orchestration.npa_workflow.submit import merge_config_overrides

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = CliRunner()

# Specs with dynamic transitions / decision-gated loops need an assumed decision
# to plan deterministically (CLI plan-spec / run-spec require --assume-decision).
ASSUME_DECISION_SPECS = {
    "sim2real.yaml",
    "tokenfactory-cosmos-gate.yaml",
    "rl-policy-training-sim-success.yaml",
    "physical-ai-data-factory.yaml",
}


def _npa_yaml_paths() -> list[Path]:
    return iter_npa_workflow_specs()


def _operator_image_fixtures(spec):
    # Schema/argv smoke coverage uses inert exact references, never runnable
    # acceptance evidence. The separate default audit exercises missing inputs.
    return {
        key: "registry.example.invalid/workflow-smoke@sha256:" + "0" * 64
        for key in spec.config.get("required_immutable_images", [])
    }


def _assumed_decision(path):
    return (
        "loop_back"
        if path.name
        in {
            "sim2real.yaml",
            "tokenfactory-cosmos-gate.yaml",
            "rl-policy-training-sim-success.yaml",
        }
        else "promote_checkpoint"
    )


def _cli_plan(path, variables, assume, *, scheduler=False):
    arguments = [
        "workbench",
        "workflow",
        "run-spec" if scheduler else "plan-spec",
        str(path),
        "--run-id",
        f"smoke-{path.stem}",
        "--json",
        *variables,
    ]
    if path.name in ASSUME_DECISION_SPECS:
        arguments.extend(["--assume-decision", assume])
    if scheduler:
        arguments.extend(["--plan-only", "--scheduler-plan"])
    return RUNNER.invoke(app, arguments)


@pytest.mark.parametrize("path", _npa_yaml_paths(), ids=lambda p: p.name)
def test_npa_workflow_yaml_validates(path: Path) -> None:
    spec = load_spec(path)
    validate_spec(spec)
    assert spec.api_version in {API_VERSION, API_VERSION_BETA}


@pytest.mark.parametrize("path", _npa_yaml_paths(), ids=lambda p: p.name)
def test_npa_workflow_cli_validate_and_plan(path: Path) -> None:
    validate = RUNNER.invoke(
        app,
        ["workbench", "workflow", "validate-spec", str(path), "--json"],
    )
    assert validate.exit_code == 0, validate.output
    payload = json.loads(validate.output)
    assert payload["status"] == "valid"
    spec = load_spec(path)
    overrides = _operator_image_fixtures(spec)
    variables = [
        argument
        for key, value in overrides.items()
        for argument in ("--var", f"{key}={value}")
    ]

    assume = _assumed_decision(path)
    plan = _cli_plan(path, variables, assume)
    assert plan.exit_code == 0, plan.output
    # Read stdout, not the mixed stream: plan-spec writes diagnostics (e.g. the
    # placeholder-bucket warning for specs that ship `bucket: example-bucket`)
    # to stderr, which CliRunner interleaves into `.output`.
    plan_payload = json.loads(plan.stdout)
    assert plan_payload["steps"], path.name

    spec = merge_config_overrides(spec, overrides)
    built = build_plan(spec, run_id=f"smoke-{path.stem}", assume_decision=assume)
    assert built.steps

    scheduler = _cli_plan(path, variables, assume, scheduler=True)
    assert scheduler.exit_code == 0, scheduler.output
    scheduler_payload = json.loads(scheduler.output)
    assert scheduler_payload.get("scheduler", {}).get("tasks"), path.name
