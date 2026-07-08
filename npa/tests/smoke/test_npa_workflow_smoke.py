from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from npa.cli.main import app

REPO_ROOT = Path(__file__).resolve().parents[3]
SPECS = REPO_ROOT / "npa" / "workflows" / "workbench" / "npa-workflows"


@pytest.mark.parametrize(
    "name",
    [
        "vlm-eval-single.yaml",
        "tokenfactory-rollout-judge.yaml",
        "sim2real-vlm-rl.yaml",
        "bdd100k-pipeline.yaml",
        "tokenfactory-cosmos-gate.yaml",
        "av-night-scene-hardening.yaml",
        "cosmos-synth-fanout-curation.yaml",
    ],
)
def test_cli_validate_spec(name: str) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["workbench", "workflow", "validate-spec", str(SPECS / name), "--json"],
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
            str(SPECS / "tokenfactory-rollout-judge.yaml"),
            "--run-id",
            "smoke-1",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["workflow"] == "tokenfactory-rollout-judge"
    assert len(payload["steps"]) == 2
