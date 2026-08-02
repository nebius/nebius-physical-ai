"""`npa cleanup` — local residue report + wipe after teardown."""

from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from npa.cli.main import app

runner = CliRunner()


def _seed_residue() -> tuple[Path, Path, Path, Path]:
    """Create the caches a teardown leaves behind under the isolated HOME."""
    home = Path.home()
    npa = home / ".npa"
    sky_venv = npa / "skypilot-venv" / "bin"
    sky_venv.mkdir(parents=True)
    (sky_venv / "sky").write_text("#!/bin/sh\n")
    tf_cache = npa / "terraform-plugin-cache" / "registry.terraform.io"
    tf_cache.mkdir(parents=True)
    (tf_cache / "provider").write_text("x" * 1024)
    sky_home = home / ".sky" / "state"
    sky_home.mkdir(parents=True)
    (sky_home / "db").write_text("y" * 2048)
    empty_alias = npa / "agents" / "test-rtx"
    empty_alias.mkdir(parents=True)
    return npa / "skypilot-venv", npa / "terraform-plugin-cache", home / ".sky", empty_alias


def test_cleanup_reports_residue_without_removing(monkeypatch) -> None:
    from npa.clients import config as config_module

    sky_venv, tf_cache, sky_home, empty_alias = _seed_residue()
    # A persisted sky_bin should be reported but not touched without --yes.
    config_module.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    config_module.CONFIG_PATH.write_text(yaml.safe_dump({"skypilot": {"sky_bin": "/x/sky"}}))

    result = runner.invoke(app, ["cleanup"])

    assert result.exit_code == 0, result.output
    assert "SkyPilot venv" in result.output
    assert "Terraform provider cache" in result.output
    assert "~/.sky" in result.output
    assert "Re-run with --yes" in result.output
    # Nothing removed in report mode.
    assert sky_venv.exists() and tf_cache.exists() and sky_home.exists() and empty_alias.exists()


def test_cleanup_yes_removes_local_caches_but_keeps_tokens(monkeypatch) -> None:
    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module

    sky_venv, tf_cache, sky_home, empty_alias = _seed_residue()
    config_module.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    config_module.CONFIG_PATH.write_text(yaml.safe_dump({"skypilot": {"sky_bin": "/x/sky"}}))
    credentials_module.CREDENTIALS_PATH.write_text(
        yaml.safe_dump({"tokens": {"HF_TOKEN": "hf_keep"}})
    )

    result = runner.invoke(app, ["cleanup", "--yes"])

    assert result.exit_code == 0, result.output
    assert not sky_venv.exists()
    assert not tf_cache.exists()
    assert not sky_home.exists()
    assert not empty_alias.exists()
    # sky_bin cleared from config; tokens untouched.
    saved_config = yaml.safe_load(config_module.CONFIG_PATH.read_text()) or {}
    assert "sky_bin" not in saved_config.get("skypilot", {})
    assert yaml.safe_load(credentials_module.CREDENTIALS_PATH.read_text())["tokens"]["HF_TOKEN"] == "hf_keep"


def test_cleanup_keep_sky_leaves_dot_sky(monkeypatch) -> None:
    sky_venv, _tf, sky_home, _empty = _seed_residue()

    result = runner.invoke(app, ["cleanup", "--yes", "--keep-sky"])

    assert result.exit_code == 0, result.output
    assert not sky_venv.exists()
    assert sky_home.exists()  # ~/.sky preserved


def test_cleanup_iam_note_names_the_storage_service_account(monkeypatch) -> None:
    from npa.clients import credentials as credentials_module

    credentials_module.CREDENTIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    credentials_module.CREDENTIALS_PATH.write_text(
        yaml.safe_dump({"nebius": {"service_account_id": "serviceaccount-lerobot-training"}})
    )

    result = runner.invoke(app, ["cleanup"])

    assert result.exit_code == 0, result.output
    assert "serviceaccount-lerobot-training" in result.output
    assert "nebius iam service-account delete" in result.output


def test_cleanup_is_quiet_when_nothing_is_left() -> None:
    result = runner.invoke(app, ["cleanup"])

    assert result.exit_code == 0, result.output
    assert "No local NPA/SkyPilot residue" in result.output
