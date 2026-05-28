from __future__ import annotations

import json

from typer.testing import CliRunner

from npa.cli.main import app


runner = CliRunner()


def test_retargeting_registered_under_workbench() -> None:
    result = runner.invoke(app, ["workbench", "--help"])

    assert result.exit_code == 0
    assert "retargeting" in result.output


def test_retargeting_command_help() -> None:
    for command in ("run", "workflow", "status", "list"):
        result = runner.invoke(app, ["workbench", "retargeting", command, "--help"])

        assert result.exit_code == 0
        assert "Usage:" in result.output


def test_retargeting_run_writes_local_manifest(tmp_path) -> None:
    output_dir = tmp_path / "retargeted"

    result = runner.invoke(
        app,
        [
            "workbench",
            "retargeting",
            "run",
            "--input-path",
            "s3://bucket/motions/source/",
            "--output-path",
            str(output_dir),
            "--source-format",
            "bvh",
            "--embodiment",
            "unitree-g1",
            "--frame-rate",
            "50",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "retargeted"
    assert payload["source_format"] == "bvh"
    written = output_dir / "retargeting_manifest.json"
    assert written.exists()
    assert json.loads(written.read_text(encoding="utf-8"))["embodiment"] == "unitree-g1"


def test_retargeting_respects_env_dry_run(monkeypatch, tmp_path) -> None:
    output_dir = tmp_path / "retargeted"
    monkeypatch.setenv("NPA_DRY_RUN", "1")

    result = runner.invoke(
        app,
        [
            "workbench",
            "retargeting",
            "run",
            "--input-path",
            "s3://bucket/motions/source/",
            "--output-path",
            str(output_dir),
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0
    assert json.loads(result.output)["dry_run"] is True
    assert not output_dir.exists()


def test_retargeting_rejects_negative_frame_limit() -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "retargeting",
            "run",
            "--input-path",
            "s3://bucket/motions/source/",
            "--output-path",
            "s3://bucket/out/",
            "--max-frames",
            "-1",
        ],
    )

    assert result.exit_code == 1
    assert "--max-frames must be non-negative" in result.output


def test_retargeting_workflow_path() -> None:
    result = runner.invoke(app, ["workbench", "retargeting", "workflow", "--output", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["workflow"] == "npa/workflows/workbench/skypilot/retargeting.yaml"
