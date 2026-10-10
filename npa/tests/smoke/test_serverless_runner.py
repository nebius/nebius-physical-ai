"""Unit tests for npa.smoke.serverless_runner project-id resolution (item 9).

These are import-safe and touch no real infrastructure: they only exercise the
``_project_id`` resolution precedence (explicit > env > config.yaml >
credentials.yaml) that ``npa configure`` relies on.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from npa.clients import config as client_config
from npa.smoke import serverless_runner


@pytest.fixture()
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cfg = tmp_path / ".npa" / "config.yaml"
    monkeypatch.setattr(client_config, "CONFIG_PATH", cfg)
    # ``_project_id`` reads ~/.npa/credentials.yaml via expanduser().
    monkeypatch.setenv("HOME", str(tmp_path))
    for env_var in ("NEBIUS_PROJECT_ID", "NPA_PROJECT_ID"):
        monkeypatch.delenv(env_var, raising=False)
    return tmp_path


def _write_config_project(cfg: Path, project_id: str) -> None:
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        yaml.safe_dump(
            {
                "default_project": "eu",
                "projects": {"eu": {"project_id": project_id}},
            }
        )
    )


def test_project_id_explicit_wins(isolated_home: Path) -> None:
    _write_config_project(isolated_home / ".npa" / "config.yaml", "project-from-config")
    assert serverless_runner._project_id("project-explicit") == "project-explicit"


def test_project_id_env_wins_over_config(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_config_project(isolated_home / ".npa" / "config.yaml", "project-from-config")
    monkeypatch.setenv("NEBIUS_PROJECT_ID", "project-from-env")
    assert serverless_runner._project_id(None) == "project-from-env"


def test_project_id_reads_config_yaml(isolated_home: Path) -> None:
    # The regression this fixes: no env vars, id lives only in config.yaml
    # (where `npa configure` writes it).
    _write_config_project(isolated_home / ".npa" / "config.yaml", "project-from-config")
    assert serverless_runner._project_id(None) == "project-from-config"


def test_project_id_reads_credentials_yaml(isolated_home: Path) -> None:
    creds = isolated_home / ".npa" / "credentials.yaml"
    creds.parent.mkdir(parents=True, exist_ok=True)
    creds.write_text(yaml.safe_dump({"nebius": {"project_id": "project-from-creds"}}))
    assert serverless_runner._project_id(None) == "project-from-creds"


def test_project_id_error_names_npa_configure(isolated_home: Path) -> None:
    # Nothing configured anywhere -> actionable error that names `npa configure`.
    with pytest.raises(RuntimeError, match="npa configure"):
        serverless_runner._project_id(None)
