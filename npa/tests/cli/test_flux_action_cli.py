"""Check registration, S3 handoffs, and the shared fine-tuning CLI boundary."""

import json

from typer.testing import CliRunner

from npa.cli.main import app
from npa.workbench.flux_action import runner


def test_help_exposes_native_training_contract():
    result = CliRunner().invoke(app, ["workbench", "flux-action", "finetune", "--help"])
    assert result.exit_code == 0
    assert "--recipe-uri" in result.output
    assert "--processes" in result.output


def test_cli_calls_shared_runner_and_emits_one_json(monkeypatch):
    seen = []

    def run(request, *, dry_run):
        seen.append((request, dry_run))
        return {"status": "planned"}

    monkeypatch.setattr(runner, "finetune", run)
    args = [
        "workbench",
        "flux-action",
        "finetune",
        "--input-path",
        "s3://test-bucket/data",
        "--recipe-uri",
        "s3://test-bucket/recipe.json",
        "--output-path",
        "s3://test-bucket/run",
        "--processes",
        "3",
        "--dry-run",
    ]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"status": "planned"}
    assert seen[0][0].processes == 3 and seen[0][1]
    args[args.index("--input-path") + 1] = "/tmp/local"
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1
    assert len(seen) == 1
