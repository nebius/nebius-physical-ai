from __future__ import annotations

import json

from typer.testing import CliRunner

from npa.cli.main import app


runner = CliRunner()


def test_workbench_vlm_eval_command_help() -> None:
    result = runner.invoke(app, ["workbench", "vlm-eval", "--help"])

    assert result.exit_code == 0
    assert "Stub VLM evaluation" in result.output


def test_workbench_vlm_eval_run_writes_local_json(tmp_path) -> None:
    output_dir = tmp_path / "eval"

    result = runner.invoke(
        app,
        [
            "workbench",
            "vlm-eval",
            "run",
            "--input-path",
            "s3://bucket/cosmos/out/",
            "--output-path",
            str(output_dir),
            "--score",
            "0.9",
            "--success-threshold",
            "0.8",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["backend"] == "stub"
    assert payload["passed"] is True
    written = output_dir / "vlm_eval_stub.json"
    assert written.exists()
    assert json.loads(written.read_text(encoding="utf-8"))["score"] == 0.9


def test_workbench_vlm_eval_dry_run_does_not_write(tmp_path) -> None:
    output_dir = tmp_path / "eval"

    result = runner.invoke(
        app,
        [
            "workbench",
            "vlm-eval",
            "run",
            "--input-path",
            "s3://bucket/cosmos/out/",
            "--output-path",
            str(output_dir),
            "--dry-run",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["dry_run"] is True
    assert not output_dir.exists()


def test_workbench_vlm_eval_respects_env_dry_run(monkeypatch, tmp_path) -> None:
    output_dir = tmp_path / "eval"
    monkeypatch.setenv("NPA_DRY_RUN", "1")

    result = runner.invoke(
        app,
        [
            "workbench",
            "vlm-eval",
            "run",
            "--input-path",
            "s3://bucket/cosmos/out/",
            "--output-path",
            str(output_dir),
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0
    assert json.loads(result.output)["dry_run"] is True
    assert not output_dir.exists()
