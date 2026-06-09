from __future__ import annotations

import json

from typer.testing import CliRunner

from npa.cli.main import app

runner = CliRunner()


def test_doctor_registered_at_top_level() -> None:
    result = runner.invoke(app, ["doctor", "--help"])
    assert result.exit_code == 0
    assert "Preflight checks" in result.output


def test_doctor_sim2real_help_lists_checks() -> None:
    result = runner.invoke(app, ["doctor", "sim2real", "--help"])
    assert result.exit_code == 0
    assert "--checks" in result.output


def test_doctor_static_checks_pass_with_bucket() -> None:
    result = runner.invoke(
        app,
        ["doctor", "sim2real", "--checks", "config,coherence", "--s3-bucket", "real-bucket"],
    )
    assert result.exit_code == 0
    assert "three-tier-coherence" in result.output
    assert "PASS" in result.output


def test_doctor_fails_without_required_bucket(monkeypatch) -> None:
    for key in ("NPA_S3_BUCKET", "NPA_SIM2REAL_BUCKET", "S3_BUCKET"):
        monkeypatch.delenv(key, raising=False)
    result = runner.invoke(app, ["doctor", "sim2real", "--checks", "config"])
    assert result.exit_code == 1
    assert "FAIL" in result.output


def test_doctor_warn_only_suppresses_exit_code(monkeypatch) -> None:
    for key in ("NPA_S3_BUCKET", "NPA_SIM2REAL_BUCKET", "S3_BUCKET"):
        monkeypatch.delenv(key, raising=False)
    result = runner.invoke(
        app, ["doctor", "sim2real", "--checks", "config", "--warn-only"]
    )
    assert result.exit_code == 0


def test_doctor_json_output_is_valid() -> None:
    result = runner.invoke(
        app,
        [
            "doctor",
            "sim2real",
            "--checks",
            "config,coherence",
            "--s3-bucket",
            "real-bucket",
            "--json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert {c["name"] for c in payload["checks"]} == {"config", "three-tier-coherence"}


def test_doctor_rejects_unknown_check() -> None:
    result = runner.invoke(app, ["doctor", "sim2real", "--checks", "bogus"])
    assert result.exit_code != 0
    assert "unknown check" in result.output.lower()
