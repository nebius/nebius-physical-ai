"""CLI coverage for read-only Cosmos Evaluator report inspection."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from npa.cli.main import app


runner = CliRunner()


def _report(*, passed: bool) -> dict:
    score = 0.9 if passed else 0.2
    return {
        "schema": "npa.cosmos_evaluator.report.v1",
        "status": "completed",
        "score": score,
        "passed": passed,
        "threshold": 0.8,
        "clip_count": 1,
        "passed_clips": int(passed),
        "augment_uri": "s3://example-bucket/source/",
        "clips": [
            {
                "clip_id": "variant-001",
                "status": "completed",
                "score": score,
                "passed": passed,
                "input_conditioned": False,
                "attribute_verification": {
                    "score": score,
                    "passed": passed,
                    "threshold": 0.8,
                    "total_checks": 1,
                    "passed_checks": int(passed),
                    "failed_checks": int(not passed),
                },
            }
        ],
    }


def _write_report(tmp_path: Path, *, passed: bool) -> Path:
    path = tmp_path / "cosmos_evaluator.json"
    path.write_text(json.dumps(_report(passed=passed)), encoding="utf-8")
    return path


def test_report_command_emits_a_bounded_json_projection(tmp_path: Path) -> None:
    path = _write_report(tmp_path, passed=True)

    result = runner.invoke(
        app,
        [
            "workbench",
            "cosmos-evaluator",
            "report",
            "--input-path",
            str(path),
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["reported_gate"]["passed"] is True
    assert payload["variants"][0]["diagnostics"]["attributes"]["status"] == "evaluated"
    assert "s3://example-bucket" not in result.output


def test_report_command_inspects_a_failed_quality_outcome(tmp_path: Path) -> None:
    path = _write_report(tmp_path, passed=False)

    result = runner.invoke(
        app,
        [
            "workbench",
            "cosmos-evaluator",
            "report",
            "--input-path",
            str(path),
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["reported_gate"]["passed"] is False
    assert payload["evidence_complete"] is True


def test_report_command_text_states_score_and_multiview_limits(tmp_path: Path) -> None:
    path = _write_report(tmp_path, passed=False)

    result = runner.invoke(
        app,
        ["workbench", "cosmos-evaluator", "report", "--input-path", str(path)],
    )

    assert result.exit_code == 0, result.output
    assert "score is not calibrated confidence" in result.output
    assert "multi-view assessment is not evaluated" in result.output


def test_report_command_text_labels_missing_enforcement_as_unverified(
    tmp_path: Path,
) -> None:
    report = _report(passed=True)
    report["clips"][0].pop("input_conditioned")
    path = tmp_path / "cosmos_evaluator.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    result = runner.invoke(
        app,
        ["workbench", "cosmos-evaluator", "report", "--input-path", str(path)],
    )

    assert result.exit_code == 0, result.output
    assert "unverified" in result.output
