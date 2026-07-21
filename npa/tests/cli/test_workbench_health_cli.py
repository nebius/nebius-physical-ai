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


def test_health_help_lists_preflight_not_deprecated_sim2real() -> None:
    result = runner.invoke(app, ["workbench", "health", "--help"])
    assert result.exit_code == 0
    # The generic credential preflight is the advertised command; the sim2real
    # one is hidden/deprecated in favor of `workbench workflow submit`. Assert on
    # the command *rows* (Typer renders each listed command as "│ <name> ...")
    # rather than a broad substring, so help copy mentioning "sim2real" elsewhere
    # can't silently break this.
    assert "preflight" in result.output
    command_rows = [
        line for line in result.output.splitlines() if line.strip().startswith("│ ")
    ]
    listed_commands = {line.split()[1] for line in command_rows if len(line.split()) > 1}
    assert "preflight" in listed_commands
    assert "sim2real" not in listed_commands


class _EmptyCreds:
    hf_token = ""
    ngc_api_key = ""
    token_factory_api_key = ""
    s3_access_key_id = ""
    s3_secret_access_key = ""
    s3_endpoint = ""
    s3_bucket = ""


def test_preflight_offline_all_warn_exit_zero(monkeypatch) -> None:
    from npa.cli.workbench import health as health_module

    monkeypatch.setattr(health_module, "load_credentials", lambda *a, **k: _EmptyCreds())
    result = runner.invoke(app, ["workbench", "health", "preflight", "--offline"])
    assert result.exit_code == 0
    for name in ("hf", "ngc", "s3", "token_factory"):
        assert name in result.output
    assert "0 fail" in result.output


def test_preflight_json_offline(monkeypatch) -> None:
    from npa.cli.workbench import health as health_module

    monkeypatch.setattr(health_module, "load_credentials", lambda *a, **k: _EmptyCreds())
    result = runner.invoke(
        app, ["workbench", "health", "preflight", "--offline", "--json"]
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert {c["name"] for c in payload["checks"]} == {"hf", "ngc", "s3", "token_factory"}


def test_preflight_selected_check(monkeypatch) -> None:
    from npa.cli.workbench import health as health_module

    monkeypatch.setattr(health_module, "load_credentials", lambda *a, **k: _EmptyCreds())
    result = runner.invoke(
        app, ["workbench", "health", "preflight", "--offline", "--checks", "hf"]
    )
    assert result.exit_code == 0
    assert "hf:" in result.output
    assert "token_factory:" not in result.output


def test_preflight_rejects_unknown_check(monkeypatch) -> None:
    from npa.cli.workbench import health as health_module

    monkeypatch.setattr(health_module, "load_credentials", lambda *a, **k: _EmptyCreds())
    result = runner.invoke(
        app, ["workbench", "health", "preflight", "--checks", "bogus"]
    )
    assert result.exit_code != 0
    assert "unknown check" in result.output.lower()


def test_preflight_fails_on_bad_s3(monkeypatch) -> None:
    from npa.cli.workbench import health as health_module

    class _Creds(_EmptyCreds):
        s3_access_key_id = "AK"
        s3_secret_access_key = "SK"
        s3_endpoint = "https://storage.eu-north1.nebius.cloud"
        s3_bucket = "s3://bkt/"

    class _Client:
        def list_checkpoints(self, uri):
            raise RuntimeError("403 Forbidden")

    captured_kwargs: dict = {}

    def _from_environment(cls, **kwargs):
        captured_kwargs.update(kwargs)
        return _Client()

    monkeypatch.setattr(health_module, "load_credentials", lambda *a, **k: _Creds())
    monkeypatch.setattr(
        health_module.StorageClient, "from_environment", classmethod(_from_environment)
    )
    result = runner.invoke(
        app, ["workbench", "health", "preflight", "--checks", "s3"]
    )
    assert result.exit_code == 1
    assert "FAIL" in result.output
    # The probe must be built from the resolved credentials (endpoint/keys often
    # live in ~/.npa, not the process env), not env-only defaults.
    assert captured_kwargs["endpoint_url"] == "https://storage.eu-north1.nebius.cloud"
    assert captured_kwargs["aws_access_key_id"] == "AK"
    assert captured_kwargs["aws_secret_access_key"] == "SK"


def test_preflight_warn_only_suppresses_exit(monkeypatch) -> None:
    from npa.cli.workbench import health as health_module

    class _Creds(_EmptyCreds):
        s3_access_key_id = "AK"
        s3_secret_access_key = "SK"
        s3_endpoint = "https://storage.eu-north1.nebius.cloud"
        s3_bucket = "s3://bkt/"

    class _Client:
        def list_checkpoints(self, uri):
            raise RuntimeError("403 Forbidden")

    monkeypatch.setattr(health_module, "load_credentials", lambda *a, **k: _Creds())
    monkeypatch.setattr(
        health_module.StorageClient, "from_environment", classmethod(lambda cls, **k: _Client())
    )
    result = runner.invoke(
        app, ["workbench", "health", "preflight", "--checks", "s3", "--warn-only"]
    )
    assert result.exit_code == 0
