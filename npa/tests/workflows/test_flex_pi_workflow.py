"""Validate shipped Flex-Pi plans against the live CLI and conservative defaults."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.orchestration.npa_workflow.interpreter import build_plan
from npa.orchestration.npa_workflow.spec import load_spec

ROOT = Path(__file__).resolve().parents[3]
TRAINING = (
    "flex-pi-b200-public-training.yaml",
    "flex-pi-b300-multinode-public-training.yaml",
)
INFERENCE = (
    "flex-pi-b200-inference.yaml",
    "flex-pi-b300-inference.yaml",
    "flex-pi-rtxpro-inference.yaml",
)


@pytest.mark.parametrize("name", TRAINING + INFERENCE)
def test_shipped_plan_argv_executes_cli_dry_run_without_launch(name, monkeypatch):
    def unexpected_launch(*args, **kwargs):
        pytest.fail("workflow dry run launched a worker")

    monkeypatch.setattr("subprocess.run", unexpected_launch)
    spec = load_spec(ROOT / "workflows/testing" / name)
    manifest = ROOT / "npa/docker/workbench/flex-pi/public_robotwin_sample.json"
    monkeypatch.setattr(
        "npa.workbench.flex_pi.runtime._materialize_input",
        lambda *args: manifest.read_bytes(),
    )
    monkeypatch.setattr("npa.workbench.flex_pi.runtime._execute", unexpected_launch)
    plan = build_plan(spec, run_id="flex-pi-workflow-contract")
    assert len(plan.steps) == 1
    step = plan.steps[0]
    result = CliRunner().invoke(app, [*step.argv[1:], "--dry-run"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    if name in TRAINING:
        assert payload["execution"]["dry_run"] is True
        assert payload["effective_batch"] == 96
    else:
        assert payload["status"] == "dry_run"
    output = step.argv[step.argv.index("--output-path") + 1]
    assert output + "result.json" in {row["uri"] for row in step.outputs}


@pytest.mark.parametrize("name", TRAINING)
def test_training_defaults_preserve_recomputation_and_eager_execution(name):
    spec = load_spec(ROOT / "workflows/testing" / name)
    argv = build_plan(spec).steps[0].argv
    for option, expected in (
        ("--activation-checkpointing", "on"),
        ("--cuda-graphs", "off"),
        ("--memory-fill", "on"),
        ("--microbatch-per-rank", "1"),
        ("--optimizer", "default"),
        ("--mode", "train"),
    ):
        assert argv[argv.index(option) + 1] == expected
