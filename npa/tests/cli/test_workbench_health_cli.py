from __future__ import annotations

import json

from typer.testing import CliRunner

from npa.cli.main import app

runner = CliRunner()


def test_health_registered_under_workbench() -> None:
    result = runner.invoke(app, ["workbench", "health", "--help"])
    assert result.exit_code == 0
    assert "Preflight health checks" in result.output


def test_health_not_registered_at_top_level() -> None:
    result = runner.invoke(app, ["doctor", "--help"])
    assert result.exit_code != 0


def test_health_sim2real_help_lists_checks() -> None:
    result = runner.invoke(app, ["workbench", "health", "sim2real", "--help"])
    assert result.exit_code == 0
    assert "--checks" in result.output


def test_health_static_checks_pass_with_bucket() -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "health",
            "sim2real",
            "--checks",
            "config,coherence",
            "--s3-bucket",
            "real-bucket",
        ],
    )
    assert result.exit_code == 0
    assert "three-tier-coherence" in result.output
    assert "PASS" in result.output


def test_health_checks_all_expands_to_full_set() -> None:
    # `--checks all` is the documented shorthand used by operator runbooks and the
    # 10-minute demo script; it must expand to the full check set, not error.
    result = runner.invoke(
        app,
        ["workbench", "health", "sim2real", "--checks", "all", "--s3-bucket", "real-bucket"],
    )
    assert "unknown check" not in result.output
    assert "config" in result.output
    assert "three-tier-coherence" in result.output


def test_health_fails_without_required_bucket(monkeypatch) -> None:
    for key in ("NPA_S3_BUCKET", "NPA_SIM2REAL_BUCKET", "S3_BUCKET"):
        monkeypatch.delenv(key, raising=False)
    result = runner.invoke(app, ["workbench", "health", "sim2real", "--checks", "config"])
    assert result.exit_code == 1
    assert "FAIL" in result.output


def test_health_warn_only_suppresses_exit_code(monkeypatch) -> None:
    for key in ("NPA_S3_BUCKET", "NPA_SIM2REAL_BUCKET", "S3_BUCKET"):
        monkeypatch.delenv(key, raising=False)
    result = runner.invoke(
        app, ["workbench", "health", "sim2real", "--checks", "config", "--warn-only"]
    )
    assert result.exit_code == 0


def test_health_json_output_is_valid() -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "health",
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


def test_health_rejects_unknown_check() -> None:
    result = runner.invoke(app, ["workbench", "health", "sim2real", "--checks", "bogus"])
    assert result.exit_code != 0
    assert "unknown check" in result.output.lower()
