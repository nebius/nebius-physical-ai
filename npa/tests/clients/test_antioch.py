"""Verify Antioch credentials reach only the intended local client process."""

import json
import os
import subprocess
import sys
import traceback
from types import SimpleNamespace

import pytest
import yaml

from npa.clients import credentials
from npa.clients.antioch import antioch_environment
from npa.workflows.xr1_antioch import operator, operator_data, transport


@pytest.fixture
def credential_file(tmp_path, monkeypatch):
    path = tmp_path / "credentials.yaml"
    monkeypatch.setattr(credentials, "CREDENTIALS_PATH", path)
    monkeypatch.delenv("ANTIOCH_TOKEN", raising=False)
    credentials.write_credentials_file({"tokens": {
        "ANTIOCH_TOKEN": "antioch-unit-secret", "HF_TOKEN": "hf-unit-secret",
    }}, path=path)
    return path


def test_saved_token_is_private_and_excluded_from_shared_exports(credential_file):
    resolved = credentials.load_credentials(environ={})
    assert resolved.antioch_token == "antioch-unit-secret"
    assert "ANTIOCH_TOKEN" not in resolved.tokens
    assert "ANTIOCH_TOKEN" not in credentials.shared_credential_env(resolved)
    assert "antioch-unit-secret" not in repr(resolved)
    assert credential_file.stat().st_mode & 0o777 == 0o600


def test_global_export_does_not_export_saved_antioch_token(credential_file, monkeypatch):
    # load_credentials intentionally exports other supported shared tokens.
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf-original")
    credentials.load_credentials(export_to_environment=True)
    assert "ANTIOCH_TOKEN" not in os.environ


@pytest.mark.parametrize("override", [None, "antioch-env-secret"])
def test_child_environment_resolves_precedence_without_mutating_parent(credential_file, monkeypatch, override):
    if override:
        monkeypatch.setenv("ANTIOCH_TOKEN", override)
    before = dict(os.environ)
    environment = antioch_environment()
    assert environment["ANTIOCH_TOKEN"] == (override or "antioch-unit-secret")
    assert environment.get("HF_TOKEN") == before.get("HF_TOKEN")
    assert dict(os.environ) == before


def test_missing_token_preserves_native_login_and_env_only_is_supported(credential_file, monkeypatch):
    credential_file.unlink()
    assert "ANTIOCH_TOKEN" not in antioch_environment()
    monkeypatch.setenv("ANTIOCH_TOKEN", "antioch-env-secret")
    assert antioch_environment()["ANTIOCH_TOKEN"] == "antioch-env-secret"


def test_malformed_store_does_not_fall_back_to_another_identity(credential_file):
    credential_file.write_text("tokens: [\nantioch-unit-secret")
    with pytest.raises(credentials.CredentialStoreError) as error:
        antioch_environment()
    assert "antioch-unit-secret" not in str(error.value)
    assert "antioch-unit-secret" not in "".join(traceback.format_exception(error.value))


def test_environment_import_persists_token_without_exposing_value(tmp_path):
    path = tmp_path / "credentials.yaml"
    report = credentials.persist_supported_env_credentials(
        path=path, environ={"ANTIOCH_TOKEN": "antioch-unit-secret"},
    )
    assert report["persisted"] == ["ANTIOCH_TOKEN"]
    assert "antioch-unit-secret" not in json.dumps(report)
    assert yaml.safe_load(path.read_text())["tokens"]["ANTIOCH_TOKEN"] == "antioch-unit-secret"
    assert path.stat().st_mode & 0o777 == 0o600


def test_configure_imports_antioch_pat_without_printing_it(credential_file, monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from npa.cli.main import app
    from npa.clients import config

    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.yaml")
    monkeypatch.setenv("ANTIOCH_TOKEN", "antioch-import-secret")
    result = CliRunner().invoke(app, ["configure", "--save-env-credentials", "--no-interactive"])
    assert result.exit_code == 0, result.output
    assert "antioch-import-secret" not in result.output
    assert yaml.safe_load(credential_file.read_text())["tokens"]["ANTIOCH_TOKEN"] == "antioch-import-secret"
    assert credential_file.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("mode", ["antioch", "exec"])
def test_cli_and_attached_sdk_receive_token_only_in_environment(credential_file, monkeypatch, tmp_path, mode):
    calls = []
    monkeypatch.setattr(operator.subprocess, "run", lambda argv, **kwargs:
                        calls.append((argv, kwargs)) or SimpleNamespace(returncode=7))
    options = [mode, "--antioch-project", str(tmp_path)]
    if mode == "exec":
        options += ["--antioch-python", sys.executable]
    args = operator._parser().parse_args(options + ["--", "python", "literal $(not-a-shell)"])
    assert operator._run_antioch(args) == 7
    argv, kwargs = calls[0]
    assert kwargs["env"]["ANTIOCH_TOKEN"] == "antioch-unit-secret"
    assert kwargs["cwd"] == tmp_path
    assert "antioch-unit-secret" not in repr(argv)
    assert argv[-2:] == ["python", "literal $(not-a-shell)"]
    assert not kwargs.get("shell")
    assert "ANTIOCH_TOKEN" not in os.environ


def test_actual_operator_subprocess_uses_saved_token_without_echo(tmp_path):
    config = tmp_path / "config"
    credentials.write_credentials_file({"tokens": {"ANTIOCH_TOKEN": "antioch-unit-secret"}},
                                       path=config / "credentials.yaml")
    executable = tmp_path / "antioch"
    executable.write_text(f"#!{sys.executable}\nimport os, sys\n"
                          "assert os.environ['ANTIOCH_TOKEN'] == 'antioch-unit-secret'\n"
                          "assert sys.argv[1:] == ['auth', 'whoami', '--json']\n"
                          "print('credential received')\n")
    executable.chmod(0o700)
    environment = {**os.environ, "NPA_CONFIG_DIR": str(config),
                   "PATH": str(tmp_path) + os.pathsep + os.environ.get("PATH", "")}
    environment.pop("ANTIOCH_TOKEN", None)
    result = subprocess.run([sys.executable, "-m", "npa.workflows.xr1_antioch.operator",
                             "antioch", "--antioch-project", str(tmp_path), "--",
                             "auth", "whoami", "--json"], env=environment,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "credential received"
    assert "antioch-unit-secret" not in result.stdout + result.stderr


def test_transfer_workers_receive_scoped_token(credential_file, monkeypatch, tmp_path):
    calls = []
    receipt = {"files": {}, "download_verified": True}
    monkeypatch.setattr(transport.subprocess, "run", lambda argv, **kwargs:
                        calls.append((argv, kwargs)) or SimpleNamespace(
                            returncode=0, stdout=json.dumps(receipt)))
    assert transport._worker(tmp_path, {"operation": "manifest", "root": "/outputs"}) == receipt
    storage = SimpleNamespace(s3=SimpleNamespace())
    assert transport.fetch_inputs(tmp_path, "/inputs", "s3://example-bucket/run", {}, storage) == receipt
    assert len(calls) == 2
    for argv, kwargs in calls:
        assert kwargs["env"]["ANTIOCH_TOKEN"] == "antioch-unit-secret"
        assert "antioch-unit-secret" not in repr(argv) + kwargs["input"]


def test_collection_probe_and_run_receive_scoped_token(credential_file, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(operator_data.subprocess, "check_output", lambda argv, **kwargs:
                        calls.append((argv, kwargs)) or "False")
    monkeypatch.setattr(operator_data.subprocess, "run", lambda argv, **kwargs:
                        calls.append((argv, kwargs)) or SimpleNamespace(returncode=0))
    monkeypatch.setattr(operator_data, "publish_artifacts", lambda *args: {})
    monkeypatch.setattr(operator_data, "_receipt", lambda *args: {"verified": True})
    args = SimpleNamespace(remote_root="/outputs", output_path=tmp_path,
                           antioch_project=tmp_path, source_root="/source")
    result = operator_data._collect_one(args, {"episode_id": "train-1", "seed": 1},
                                        "train", object(), "example-bucket", "run")
    assert result == {"verified": True}
    assert len(calls) == 2
    for argv, kwargs in calls:
        assert kwargs["env"]["ANTIOCH_TOKEN"] == "antioch-unit-secret"
        assert "antioch-unit-secret" not in repr(argv)


def test_attached_operator_rejects_missing_argv_and_preserves_interrupt(credential_file, monkeypatch, tmp_path):
    args = operator._parser().parse_args(["exec", "--antioch-project", str(tmp_path),
                                         "--antioch-python", sys.executable, "--"])
    with pytest.raises(ValueError, match="arguments"):
        operator._run_antioch(args)
    args.argv = ["--", "python", "-V"]
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(operator.subprocess, "run", interrupt)
    assert operator._run_antioch(args) == 130
