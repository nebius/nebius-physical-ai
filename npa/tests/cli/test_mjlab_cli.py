"""Exercise MJLab's thin CLI and reject the historical score override."""

import json

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.workbench.mjlab import runtime

runner = CliRunner()


@pytest.mark.parametrize(
    "command",
    ["train", "eval", "export", "deploy", "status", "system-info", "list", "workflow"],
)
def test_help(command):
    result = runner.invoke(app, ["workbench", "mjlab", command, "--help"])
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("command", ["train", "eval", "export"])
def test_dry_run_has_no_fabricated_metrics(command, monkeypatch):
    monkeypatch.setattr(
        runtime.StorageClient,
        "from_environment",
        lambda: pytest.fail("dry-run reached storage"),
    )
    args = [
        "workbench",
        "mjlab",
        command,
        "--output-path",
        "s3://fixture/out",
        "--dry-run",
    ]
    if command != "train":
        args += ["--checkpoint", "s3://fixture/model.pt"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    value = json.loads(result.stdout)
    assert value["status"] == "planned"
    assert value["executed"] is False
    assert "score" not in value and "passed" not in value


def test_eval_rejects_supplied_score():
    result = runner.invoke(
        app,
        [
            "workbench",
            "mjlab",
            "eval",
            "--checkpoint",
            "s3://fixture/model.pt",
            "--output-path",
            "s3://fixture/out",
            "--score",
            "0.9",
        ],
    )
    assert result.exit_code != 0
    assert "score" in result.output


def test_train_keeps_upstream_defaults(monkeypatch):
    captured = []
    monkeypatch.setattr(
        runtime,
        "train",
        lambda request, **kwargs: captured.append(request) or {"status": "completed"},
    )
    result = runner.invoke(
        app, ["workbench", "mjlab", "train", "--output-path", "s3://fixture/out"]
    )
    assert result.exit_code == 0, result.output
    assert captured[0].iterations is None
    assert captured[0].num_envs is None


def test_env_dry_run(monkeypatch):
    monkeypatch.setenv("NPA_DRY_RUN", "1")
    result = runner.invoke(
        app, ["workbench", "mjlab", "train", "--output-path", "s3://fixture/out"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["executed"] is False


def test_invalid_paths_are_errors(tmp_path):
    result = runner.invoke(
        app, ["workbench", "mjlab", "train", "--output-path", str(tmp_path / "model")]
    )
    assert result.exit_code == 1
    assert "s3://" in result.output


def test_status_does_not_claim_available(monkeypatch):
    monkeypatch.setattr(runtime, "system_info", lambda: {"installed": False})
    result = runner.invoke(app, ["workbench", "mjlab", "status", "--output", "json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout) == {"installed": False}
