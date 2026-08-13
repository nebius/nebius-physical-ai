from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.orchestration.npa_workflow.blueprints import resolve_npa_workflow_spec


def _spec(name: str) -> Path:
    path = resolve_npa_workflow_spec(name)
    assert path is not None, f"{name} not found in any npa.workflow spec root"
    return path


@pytest.mark.parametrize(
    "name",
    [
        "vlm-eval-single.yaml",
        "tokenfactory-rollout-judge.yaml",
        "sim2real.yaml",
        "bdd100k-pipeline.yaml",
        "tokenfactory-cosmos-gate.yaml",
        "av-night-scene-hardening.yaml",
        "cosmos-synth-fanout-curation.yaml",
        "physical-ai-data-factory.yaml",
    ],
)
def test_cli_validate_spec(name: str) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["workbench", "workflow", "validate-spec", str(_spec(name)), "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "valid"


def test_cli_plan_spec_json() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "plan-spec",
            str(_spec("tokenfactory-rollout-judge.yaml")),
            "--run-id",
            "smoke-1",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["workflow"] == "tokenfactory-rollout-judge"
    assert len(payload["steps"]) == 2


def test_cli_plan_spec_omits_assume_decision_for_loop_free_spec() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "plan-spec",
            str(_spec("vlm-eval-single.yaml")),
            "--run-id",
            "smoke-loopfree",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "assume_decision:" not in result.output


def test_cli_plan_spec_shows_assume_decision_for_loop_spec() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "plan-spec",
            str(_spec("tokenfactory-cosmos-gate.yaml")),
            "--run-id",
            "smoke-loop",
            "--assume-decision",
            "loop_back",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "assume_decision: loop_back_to_inner_loop" in result.output
