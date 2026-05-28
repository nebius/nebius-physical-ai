from __future__ import annotations

import json

from typer.testing import CliRunner

from npa.cli.main import app


runner = CliRunner()


def test_mjlab_registered_under_workbench() -> None:
    result = runner.invoke(app, ["workbench", "--help"])

    assert result.exit_code == 0
    assert "mjlab" in result.output


def test_mjlab_command_help() -> None:
    for command in ("eval", "workflow", "status", "list"):
        result = runner.invoke(app, ["workbench", "mjlab", command, "--help"])

        assert result.exit_code == 0
        assert "Usage:" in result.output


def test_mjlab_eval_writes_local_json(tmp_path) -> None:
    output_dir = tmp_path / "mjlab"

    result = runner.invoke(
        app,
        [
            "workbench",
            "mjlab",
            "eval",
            "--input-path",
            "s3://bucket/sonic/retargeted/",
            "--checkpoint",
            "s3://bucket/sonic/training/checkpoint_smoke.json",
            "--output-path",
            str(output_dir),
            "--score",
            "0.9",
            "--success-threshold",
            "0.75",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["backend"] == "mjlab"
    assert payload["passed"] is True
    written = output_dir / "mjlab_eval.json"
    assert written.exists()
    assert json.loads(written.read_text(encoding="utf-8"))["score"] == 0.9


def test_mjlab_eval_respects_env_dry_run(monkeypatch, tmp_path) -> None:
    output_dir = tmp_path / "mjlab"
    monkeypatch.setenv("NPA_DRY_RUN", "1")

    result = runner.invoke(
        app,
        [
            "workbench",
            "mjlab",
            "eval",
            "--input-path",
            "s3://bucket/sonic/retargeted/",
            "--checkpoint",
            "s3://bucket/sonic/training/checkpoint_smoke.json",
            "--output-path",
            str(output_dir),
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0
    assert json.loads(result.output)["dry_run"] is True
    assert not output_dir.exists()


def test_mjlab_workflow_path() -> None:
    result = runner.invoke(app, ["workbench", "mjlab", "workflow", "--output", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["workflow"] == "npa/workflows/workbench/skypilot/mjlab-eval.yaml"
