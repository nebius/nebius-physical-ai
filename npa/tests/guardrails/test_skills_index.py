from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml
from click.utils import strip_ansi
from typer.testing import CliRunner

from npa.cli.main import app
from npa.clients import config, credentials


REPO_ROOT = Path(__file__).resolve().parents[3]
SKILLS_ROOT = REPO_ROOT / "skills"
INDEX_PATH = SKILLS_ROOT / "index.yaml"


@pytest.fixture(scope="module")
def skills_index() -> dict:
    return yaml.safe_load(INDEX_PATH.read_text())


def test_legacy_skill_paths_are_root_symlinks() -> None:
    for legacy in (REPO_ROOT / ".agents" / "skills", REPO_ROOT / ".claude" / "skills"):
        assert legacy.is_symlink()
        assert os.readlink(legacy) == "../skills"
        assert legacy.resolve() == SKILLS_ROOT


def test_skills_index_covers_every_skill(skills_index: dict) -> None:
    entries = skills_index["skills"]
    names = [entry["name"] for entry in entries]
    paths = [entry["path"] for entry in entries]

    assert len(names) == len(set(names))
    assert set(entry["category"] for entry in entries) == {"atomic", "tools", "workflows"}
    assert len(entries) >= 25

    indexed_paths = {REPO_ROOT / path for path in paths}
    actual_paths = set(SKILLS_ROOT.glob("*/*/SKILL.md"))
    assert indexed_paths == actual_paths

    for entry in entries:
        skill_path = REPO_ROOT / entry["path"]
        assert skill_path.exists(), entry["path"]
        assert entry["when_to_use"].strip()
        assert entry.get("smoke"), entry["name"]
        frontmatter = _frontmatter(skill_path)
        assert frontmatter["name"] == entry["name"]
        assert frontmatter["description"].strip()

    for license_entry in skills_index["licenses"]:
        assert (REPO_ROOT / license_entry["path"]).exists()
        for name in license_entry["applies_to"]:
            assert name in names


def test_skill_smoke_examples_run(skills_index: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    for entry in skills_index["skills"]:
        for smoke in entry["smoke"]:
            smoke_type = smoke["type"]
            if smoke_type == "cli_help":
                _assert_cli_help(runner, smoke["args"], smoke["contains"])
            elif smoke_type == "workflow_yaml":
                for relative_path in smoke["paths"]:
                    payloads = [
                        payload
                        for payload in yaml.safe_load_all((REPO_ROOT / relative_path).read_text())
                        if payload is not None
                    ]
                    assert payloads, relative_path
                    for payload in payloads:
                        assert isinstance(payload, dict), relative_path
                        assert payload.get("name") or payload.get("resources"), relative_path
            elif smoke_type == "file_exists":
                assert (REPO_ROOT / smoke["path"]).exists(), smoke["path"]
            elif smoke_type == "configure_provision_dry_run":
                _assert_configure_provision_dry_run(runner, tmp_path, monkeypatch)
            else:
                raise AssertionError(f"unknown smoke type: {smoke_type}")


def _frontmatter(path: Path) -> dict[str, str]:
    text = path.read_text()
    assert text.startswith("---\n"), path
    _, raw, _ = text.split("---\n", 2)
    data = yaml.safe_load(raw)
    assert isinstance(data, dict), path
    return data


def _assert_cli_help(runner: CliRunner, args: list[str], expected: str) -> None:
    result = runner.invoke(app, args)
    output = strip_ansi(result.output)
    assert result.exit_code == 0, output
    assert expected in output


def _assert_configure_provision_dry_run(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    npa_home = tmp_path / ".npa"
    monkeypatch.setattr(config, "CONFIG_PATH", npa_home / "config.yaml")
    monkeypatch.setattr(credentials, "CREDENTIALS_PATH", npa_home / "credentials.yaml")
    for env_var in config.ENV_MAP.values():
        monkeypatch.delenv(env_var, raising=False)

    configure = runner.invoke(app, ["configure", "--show"])
    assert configure.exit_code == 0, configure.output
    assert "storage:" in configure.output

    npa_home.mkdir(parents=True, exist_ok=True)
    (npa_home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "default_project": "ci",
                "projects": {
                    "ci": {
                        "project_id": "project-ci",
                        "tenant_id": "tenant-ci",
                        "region": "eu-north1",
                        "registry_id": "registry-ci",
                        "storage": {
                            "checkpoint_bucket": "s3://ci-bucket/checkpoints/",
                            "endpoint_url": "https://storage.eu-north1.nebius.cloud",
                        },
                    }
                },
            }
        )
    )

    provision = runner.invoke(
        app,
        [
            "provision-if-absent",
            "--project",
            "ci",
            "--dry-run",
            "--skip-validate",
            "--output-format",
            "json",
            "--kubeconfig",
            str(tmp_path / "missing-kubeconfig"),
        ],
    )
    assert provision.exit_code == 0, provision.output
    payload = json.loads(provision.output)
    assert payload["status"] == "ok"
    assert "s3:dry-run ensure bucket ci-bucket" in payload["actions"]
    assert "k8s:dry-run terraform apply deploy/cluster" in payload["actions"]
