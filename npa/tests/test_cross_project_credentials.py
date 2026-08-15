from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner
import yaml

from npa.cli.demo import stage_artifacts
from npa.cli.main import app
from npa.clients import config
from npa.clients import credentials as credentials_mod
from npa.clients.project_credentials import resolve_credentials, storage_env_for_project
from npa.clients.project_credential_store import write_project_credentials
from npa.errors import ScopedCredentialError
from fakes import _access_denied, _fake_s3_factory, _manifest


runner = CliRunner()


@pytest.fixture()
def cross_project_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cfg = tmp_path / ".npa" / "config.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(
        yaml.safe_dump(
            {
                "default_project": "project-source",
                "projects": {
                    "project-source": {
                        "storage": {
                            "endpoint_url": "https://source-storage.example",
                            "aws_access_key_id": "src-key",
                            "aws_secret_access_key": "src-secret",
                            "bucket": "s3://source/default/",
                        }
                    },
                    "project-target": {
                        "storage": {
                            "endpoint_url": "https://target-storage.example",
                            "aws_access_key_id": "tgt-key",
                            "aws_secret_access_key": "tgt-secret",
                            "bucket": "s3://target/default/",
                        }
                    },
                },
            },
            sort_keys=False,
        )
    )
    monkeypatch.setattr(config, "CONFIG_PATH", cfg)
    return cfg


def test_resolve_credentials_default_project(cross_project_config: Path) -> None:
    credentials = resolve_credentials(project=None)

    assert credentials.project is None
    assert credentials.aws_access_key_id == "src-key"
    assert credentials.aws_secret_access_key == "src-secret"


def test_resolve_credentials_other_project(cross_project_config: Path) -> None:
    credentials = resolve_credentials(project="project-target")

    assert credentials.project == "project-target"
    assert credentials.endpoint_url == "https://target-storage.example"
    assert credentials.aws_access_key_id == "tgt-key"


def test_storage_endpoint_override_keeps_project_credentials(
    cross_project_config: Path,
) -> None:
    env = storage_env_for_project(
        "project-target", endpoint_url="https://target-storage.correct-region"
    )

    assert env["AWS_ENDPOINT_URL"] == "https://target-storage.correct-region"
    assert env["NEBIUS_S3_ENDPOINT"] == "https://target-storage.correct-region"
    assert env["AWS_ACCESS_KEY_ID"] == "tgt-key"
    assert env["AWS_SECRET_ACCESS_KEY"] == "tgt-secret"


def test_validated_exact_project_storage_overrides_stale_inline_config(
    cross_project_config: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credential_path = tmp_path / "credentials.yaml"
    monkeypatch.setattr(credentials_mod, "CREDENTIALS_PATH", credential_path)
    project_id = "project-target-id"
    config_data = yaml.safe_load(cross_project_config.read_text())
    config_data["projects"]["project-target"]["project_id"] = project_id
    cross_project_config.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    write_project_credentials(
        project_id,
        {
            "storage": {
                "bucket": "s3://validated-bucket/",
                "endpoint_url": "https://storage.us-central1.example",
                "aws_access_key_id": "validated-key",
                "aws_secret_access_key": "validated-secret",
            }
        },
        alias="project-target",
    )

    resolved = config.resolve_project_storage("project-target")

    assert resolved.checkpoint_bucket == "s3://validated-bucket/"
    assert resolved.endpoint_url == "https://storage.us-central1.example"
    assert resolved.aws_access_key_id == "validated-key"
    assert resolved.aws_secret_access_key == "validated-secret"


def test_partial_exact_project_storage_record_is_ignored_atomically(
    cross_project_config: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial generation may not borrow the config stanza's missing field."""

    credential_path = tmp_path / "credentials.yaml"
    monkeypatch.setattr(credentials_mod, "CREDENTIALS_PATH", credential_path)
    project_id = "project-target-id"
    config_data = yaml.safe_load(cross_project_config.read_text())
    config_data["projects"]["project-target"]["project_id"] = project_id
    cross_project_config.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    write_project_credentials(
        project_id,
        {
            "storage": {
                # Deliberately lacks a bucket. Mixing this key generation with
                # the config bucket would create an unverified storage identity.
                "endpoint_url": "https://partial-storage.example",
                "aws_access_key_id": "partial-key",
                "aws_secret_access_key": "partial-secret",
            }
        },
        alias="project-target",
    )

    resolved = config.resolve_project_storage("project-target")

    # The config stanza still supplies the field absent from the credential
    # record, but atomic fallback means it supplies the whole compatible group.
    assert resolved.checkpoint_bucket == "s3://target/default/"
    assert resolved.endpoint_url == "https://target-storage.example"
    assert resolved.aws_access_key_id == "tgt-key"
    assert resolved.aws_secret_access_key == "tgt-secret"


def test_resolve_credentials_nonexistent_project_raises(
    cross_project_config: Path,
) -> None:
    with pytest.raises(ScopedCredentialError, match="missing-project"):
        resolve_credentials(project="missing-project")


@pytest.fixture()
def user_credentials_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Empty machine config plus a user ~/.npa/credentials.yaml with S3 keys."""
    cfg = tmp_path / ".npa" / "config.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("projects:\n  proj-x: {}\n")
    monkeypatch.setattr(config, "CONFIG_PATH", cfg)

    for var in (
        "AWS_ENDPOINT_URL",
        "NEBIUS_S3_ENDPOINT",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    ):
        monkeypatch.delenv(var, raising=False)

    creds_file = tmp_path / ".npa" / "credentials.yaml"
    creds_file.write_text(
        yaml.safe_dump(
            {
                "storage": {
                    "endpoint_url": "https://host-storage.example",
                    "aws_access_key_id": "host-key",
                    "aws_secret_access_key": "host-secret",
                    "bucket": "s3://host-bucket/",
                }
            }
        )
    )
    monkeypatch.setattr(credentials_mod, "CREDENTIALS_PATH", creds_file)
    return creds_file


def test_resolve_credentials_default_falls_back_to_user_credentials_file(
    user_credentials_file: Path,
) -> None:
    resolved = resolve_credentials(project=None)

    assert resolved.endpoint_url == "https://host-storage.example"
    assert resolved.aws_access_key_id == "host-key"
    assert resolved.aws_secret_access_key == "host-secret"
    assert resolved.uses_host_credentials is True


def test_resolve_credentials_named_project_requires_allow_host_creds(
    user_credentials_file: Path,
) -> None:
    with pytest.raises(ScopedCredentialError, match="--allow-host-creds"):
        resolve_credentials(project="proj-x")

    resolved = resolve_credentials(project="proj-x", allow_host_creds=True)
    assert resolved.aws_access_key_id == "host-key"
    assert resolved.uses_host_credentials is True


def test_demo_stage_cross_project_uses_source_and_target_credentials(
    tmp_path: Path,
    cross_project_config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients = _fake_s3_factory(monkeypatch)

    result = stage_artifacts(
        target_bucket="target",
        manifest_path=_manifest(tmp_path / "manifest.yaml"),
        source_project="project-source",
        target_project="project-target",
    )

    assert result == [{"name": "file-one", "action": "upload"}]
    assert clients["src-key"].get_calls == [("source", "path/file.bin")]
    assert clients["tgt-key"].put_calls == [("target", "staged/file.bin")]


def test_demo_stage_cross_project_failure_names_target_project(
    tmp_path: Path,
    cross_project_config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients = _fake_s3_factory(monkeypatch)
    clients["tgt-key"].fail_put = _access_denied()

    with pytest.raises(ScopedCredentialError, match="project-target") as exc_info:
        stage_artifacts(
            target_bucket="target",
            manifest_path=_manifest(tmp_path / "manifest.yaml"),
            source_project="project-source",
            target_project="project-target",
        )
    assert exc_info.value.source_project == "project-source"
    assert exc_info.value.target_project == "project-target"
    assert exc_info.value.failed_project == "project-target"
    assert "--allow-host-creds" in str(exc_info.value)


def test_demo_stage_cross_project_failure_names_source_project(
    tmp_path: Path,
    cross_project_config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients = _fake_s3_factory(monkeypatch)
    clients["src-key"].fail_get = _access_denied()

    with pytest.raises(ScopedCredentialError, match="project-source") as exc_info:
        stage_artifacts(
            target_bucket="target",
            manifest_path=_manifest(tmp_path / "manifest.yaml"),
            source_project="project-source",
            target_project="project-target",
        )
    assert exc_info.value.source_project == "project-source"
    assert exc_info.value.target_project == "project-target"
    assert exc_info.value.failed_project == "project-source"
    assert "--allow-host-creds" in str(exc_info.value)


def test_demo_stage_cross_project_host_fallback_uses_target_host_credentials(
    tmp_path: Path,
    cross_project_config: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog,
) -> None:
    clients = _fake_s3_factory(monkeypatch)
    clients["tgt-key"].fail_put = _access_denied()

    with caplog.at_level("WARNING"):
        result = stage_artifacts(
            target_bucket="target",
            manifest_path=_manifest(tmp_path / "manifest.yaml"),
            source_project="project-source",
            target_project="project-target",
            allow_host_creds=True,
        )

    assert result == [{"name": "file-one", "action": "upload"}]
    assert clients["host:https://target-storage.example"].put_calls == [
        ("target", "staged/file.bin")
    ]
    assert "project-target" in caplog.text
    assert "falling back to host credentials" in caplog.text


def test_cross_project_flags_are_visible_in_help() -> None:
    commands = [
        ["demo", "stage"],
        ["rerun", "host"],
        ["rerun", "share"],
        ["workbench", "cosmos", "infer"],
        ["workbench", "groot", "infer"],
        ["workbench", "isaac-lab", "export-lerobot"],
    ]
    for command in commands:
        result = runner.invoke(app, [*command, "--help"])
        assert result.exit_code == 0
        assert "target-project" in result.output
        if command[-1] != "export-lerobot":
            assert "source-project" in result.output

    for command in [["demo", "verify"], ["rerun", "list-shares"], ["rerun", "revoke"]]:
        result = runner.invoke(app, [*command, "--help"])
        assert result.exit_code == 0
        assert "target-project" in result.output
