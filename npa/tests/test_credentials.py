from __future__ import annotations

from pathlib import Path
import stat
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

from npa.cli.main import app
from npa.clients.config import SSHConfig
from npa.clients.credentials import (
    CredentialStoreError,
    CredentialsConfig,
    load_credentials,
    persist_supported_env_credentials,
    preflight_private_yaml_store,
    set_token_factory_api_key,
    shared_credential_env,
    warn_if_hf_token_missing,
    write_credentials_file,
)
from npa.clients.ssh import SSHClient


def test_load_credentials_from_yaml(tmp_path: Path) -> None:
    credentials_path = tmp_path / "credentials.yaml"
    credentials_path.write_text(
        yaml.safe_dump(
            {
                "tokens": {
                    "HF_TOKEN": "hf-file",
                    "NGC_API_KEY": "ngc-file",
                }
            }
        )
    )

    resolved = load_credentials(path=credentials_path, environ={})

    assert resolved.tokens == {
        "HF_TOKEN": "hf-file",
        "NGC_API_KEY": "ngc-file",
    }


def test_load_credentials_reads_top_level_shared_tokens(tmp_path: Path) -> None:
    credentials_path = tmp_path / "credentials.yaml"
    credentials_path.write_text(
        yaml.safe_dump(
            {
                "HF_TOKEN": "hf-top",
                "NGC_API_KEY": "ngc-top",
            }
        )
    )

    resolved = load_credentials(path=credentials_path, environ={})

    assert resolved.tokens == {
        "HF_TOKEN": "hf-top",
        "NGC_API_KEY": "ngc-top",
    }


def test_load_credentials_reads_ngc_section_and_env_overrides(tmp_path: Path) -> None:
    credentials_path = tmp_path / "credentials.yaml"
    credentials_path.write_text(
        yaml.safe_dump(
            {
                "tokens": {"HF_TOKEN": "hf-file"},
                "ngc": {
                    "api_key": "ngc-file",
                    "org": "org-file",
                    "team": "team-file",
                },
            }
        )
    )

    resolved = load_credentials(
        path=credentials_path,
        environ={
            "NGC_API_KEY": "ngc-env",
            "NGC_TEAM": "team-env",
        },
    )

    assert resolved.hf_token == "hf-file"
    assert resolved.ngc_api_key == "ngc-env"
    assert resolved.ngc_org == "org-file"
    assert resolved.ngc_team == "team-env"
    assert shared_credential_env(resolved)["NGC_API_KEY"] == "ngc-env"
    assert shared_credential_env(resolved)["NGC_ORG"] == "org-file"
    assert shared_credential_env(resolved)["NGC_TEAM"] == "team-env"


def test_load_credentials_reads_nebius_token_factory_key(tmp_path: Path) -> None:
    credentials_path = tmp_path / "credentials.yaml"
    credentials_path.write_text(
        yaml.safe_dump({"tokens": {"NEBIUS_TOKEN_FACTORY_KEY": "tf-file"}})
    )

    resolved = load_credentials(path=credentials_path, environ={})

    assert resolved.token_factory_api_key == "tf-file"
    assert resolved.tokens["NEBIUS_TOKEN_FACTORY_KEY"] == "tf-file"
    assert shared_credential_env(resolved)["NEBIUS_TOKEN_FACTORY_KEY"] == "tf-file"


def test_load_credentials_ignores_legacy_token_factory_alias(tmp_path: Path) -> None:
    credentials_path = tmp_path / "credentials.yaml"
    credentials_path.write_text(
        yaml.safe_dump({"tokens": {"NEBIUS_TOKEN_FACTORY_API_KEY": "tf-legacy-file"}})
    )

    resolved = load_credentials(path=credentials_path, environ={})

    assert resolved.token_factory_api_key == ""
    assert "NEBIUS_TOKEN_FACTORY_KEY" not in shared_credential_env(resolved)


def test_load_credentials_reads_service_shaped_sections(tmp_path: Path) -> None:
    """`token_factory:` / `huggingface:` blocks are accepted, not silently dropped.

    The Physical AI Data Factory deploy guide documented this shape, so
    hand-written files used it and every hosted call 401'd against a
    credentials.yaml that visibly contained the keys.
    """
    credentials_path = tmp_path / "credentials.yaml"
    credentials_path.write_text(
        yaml.safe_dump(
            {
                "token_factory": {"api_key": "tf-section"},
                "huggingface": {"token": "hf-section"},
            }
        )
    )

    resolved = load_credentials(path=credentials_path, environ={})

    assert resolved.token_factory_api_key == "tf-section"
    assert resolved.hf_token == "hf-section"
    env = shared_credential_env(resolved)
    assert env["NEBIUS_TOKEN_FACTORY_KEY"] == "tf-section"
    assert env["HF_TOKEN"] == "hf-section"


def test_canonical_tokens_win_over_service_sections(tmp_path: Path) -> None:
    credentials_path = tmp_path / "credentials.yaml"
    credentials_path.write_text(
        yaml.safe_dump(
            {
                "tokens": {
                    "HF_TOKEN": "hf-canonical",
                    "NEBIUS_TOKEN_FACTORY_KEY": "tf-canonical",
                },
                "token_factory": {"api_key": "tf-section"},
                "huggingface": {"token": "hf-section"},
            }
        )
    )

    resolved = load_credentials(path=credentials_path, environ={})

    assert resolved.token_factory_api_key == "tf-canonical"
    assert resolved.hf_token == "hf-canonical"


def test_empty_service_sections_are_ignored(tmp_path: Path) -> None:
    credentials_path = tmp_path / "credentials.yaml"
    credentials_path.write_text(
        yaml.safe_dump(
            {
                "token_factory": {"api_key": ""},
                "huggingface": None,
            }
        )
    )

    resolved = load_credentials(path=credentials_path, environ={})

    assert resolved.token_factory_api_key == ""
    assert resolved.hf_token == ""


def test_token_factory_key_env_overrides_file(tmp_path: Path) -> None:
    credentials_path = tmp_path / "credentials.yaml"
    credentials_path.write_text(
        yaml.safe_dump({"tokens": {"NEBIUS_TOKEN_FACTORY_KEY": "tf-file"}})
    )

    resolved = load_credentials(
        path=credentials_path,
        environ={"NEBIUS_TOKEN_FACTORY_KEY": "tf-env"},
    )

    assert resolved.token_factory_api_key == "tf-env"


def test_set_token_factory_api_key_merges_into_existing_credentials(
    tmp_path: Path,
) -> None:
    credentials_path = tmp_path / "credentials.yaml"
    credentials_path.write_text(
        yaml.safe_dump(
            {
                "tokens": {"HF_TOKEN": "hf-existing"},
                "storage": {"aws_access_key_id": "AKIAEXISTING"},
            }
        )
    )

    set_token_factory_api_key("tf-new-key", path=credentials_path)

    resolved = load_credentials(path=credentials_path, environ={})
    stored = yaml.safe_load(credentials_path.read_text())
    assert resolved.token_factory_api_key == "tf-new-key"
    assert resolved.hf_token == "hf-existing"
    assert stored["tokens"]["NEBIUS_TOKEN_FACTORY_KEY"] == "tf-new-key"
    assert stored["storage"]["aws_access_key_id"] == "AKIAEXISTING"


def test_persist_env_credentials_is_atomic_private_redacted_and_preserves_unrelated(
    tmp_path: Path,
) -> None:
    credentials_path = tmp_path / "credentials.yaml"
    credentials_path.write_text(
        yaml.safe_dump(
            {
                "tokens": {"UNRELATED_TOKEN": "keep-me"},
                "ssh": {"host": "example.invalid"},
            }
        ),
        encoding="utf-8",
    )
    environment = {
        "HF_TOKEN": "hf_super_secret_value",
        "NEBIUS_TOKEN_FACTORY_KEY": "tf_super_secret_value",
        "NGC_API_KEY": "nvapi-super-secret",
        "AWS_ACCESS_KEY_ID": "synthetic-access",
        "AWS_SECRET_ACCESS_KEY": "synthetic-secret",
        "NPA_STORAGE_ENDPOINT": "https://storage.example.invalid",
        "NPA_CHECKPOINT_BUCKET": "s3://synthetic-bucket/checkpoints/",
    }

    report = persist_supported_env_credentials(
        path=credentials_path, environ=environment
    )
    # Repeating the same update is safe and does not duplicate or discard fields.
    second = persist_supported_env_credentials(
        path=credentials_path, environ=environment
    )

    stored = yaml.safe_load(credentials_path.read_text(encoding="utf-8"))
    assert stored["tokens"]["UNRELATED_TOKEN"] == "keep-me"
    assert stored["tokens"]["HF_TOKEN"] == environment["HF_TOKEN"]
    assert (
        stored["tokens"]["NEBIUS_TOKEN_FACTORY_KEY"]
        == environment["NEBIUS_TOKEN_FACTORY_KEY"]
    )
    assert stored["ngc"]["api_key"] == environment["NGC_API_KEY"]
    assert stored["ssh"]["host"] == "example.invalid"
    assert stat.S_IMODE(credentials_path.stat().st_mode) == 0o600
    assert report == second
    serialized_report = repr(report)
    assert all(secret not in serialized_report for secret in environment.values())
    assert set(report["persisted"]) == set(environment)


def test_persist_partial_s3_credentials_reports_names_not_values(
    tmp_path: Path,
) -> None:
    secret = "synthetic-partial-secret"

    report = persist_supported_env_credentials(
        path=tmp_path / "credentials.yaml",
        environ={"AWS_SECRET_ACCESS_KEY": secret},
    )

    assert report["detected"] == ["AWS_SECRET_ACCESS_KEY"]
    assert report["persisted"] == ["AWS_SECRET_ACCESS_KEY"]
    assert "incomplete S3 credential pair" in report["warnings"][0]
    assert secret not in repr(report)


def test_saved_environment_credentials_feed_workflow_and_agent_after_env_clear(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Health/submit/deploy share the exact persisted credential resolver."""

    from npa.cli.agent import _resolve_deploy_llm_credentials
    from npa.cli.agent_preflight import _agent_token_factory_result
    from npa.clients import credentials as credentials_module
    from npa.orchestration.npa_workflow.submit_credentials import (
        resolve_submit_credentials,
    )

    path = tmp_path / ".npa" / "credentials.yaml"
    saved = {
        "HF_TOKEN": "hf_saved_test_value",
        "NGC_API_KEY": "nvapi-saved-test-value",
        "NEBIUS_TOKEN_FACTORY_KEY": "tf_saved_test_value",
        "AWS_ACCESS_KEY_ID": "saved-access",
        "AWS_SECRET_ACCESS_KEY": "saved-secret",
        "NPA_STORAGE_ENDPOINT": "https://storage.example.invalid",
        "NPA_CHECKPOINT_BUCKET": "s3://saved-bucket/checkpoints/",
    }
    persist_supported_env_credentials(path=path, environ=saved)
    for key in saved:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", path)
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.submit_credentials.resolve_project_storage",
        lambda project=None: SimpleNamespace(
            checkpoint_bucket="",
            endpoint_url="",
            aws_access_key_id="",
            aws_secret_access_key="",
        ),
    )

    workflow = resolve_submit_credentials(
        project="isolated",
        requested=(
            "HF_TOKEN",
            "NGC_API_KEY",
            "NEBIUS_TOKEN_FACTORY_KEY",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
        ),
        environ={},
    )
    agent_key, _model = _resolve_deploy_llm_credentials()

    assert workflow.missing == ()
    assert workflow.secret_values == {
        key: saved[key]
        for key in (
            "HF_TOKEN",
            "NGC_API_KEY",
            "NEBIUS_TOKEN_FACTORY_KEY",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
        )
    }
    assert agent_key == saved["NEBIUS_TOKEN_FACTORY_KEY"]
    assert _agent_token_factory_result(agent_key).status == "PASS"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_persist_env_credentials_refuses_symlink_destination(tmp_path: Path) -> None:
    destination = tmp_path / "credentials.yaml"
    destination.symlink_to(tmp_path / "elsewhere.yaml")

    with pytest.raises(OSError, match="symlink"):
        persist_supported_env_credentials(
            path=destination, environ={"HF_TOKEN": "hf_not_written"}
        )

    assert not (tmp_path / "elsewhere.yaml").exists()


def test_private_store_preflight_does_not_create_an_absent_store(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "protected" / "credentials.yaml"

    assert preflight_private_yaml_store(destination) == destination

    assert not destination.exists()
    assert stat.S_IMODE(destination.parent.stat().st_mode) == 0o700


def test_private_store_preflight_refuses_a_symlink(tmp_path: Path) -> None:
    destination = tmp_path / "credentials.yaml"
    destination.symlink_to(tmp_path / "elsewhere.yaml")

    with pytest.raises(OSError, match="symlink"):
        preflight_private_yaml_store(destination)


def test_atomic_credential_replace_failure_preserves_previous_file(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.clients import credentials as credentials_module

    destination = tmp_path / "credentials.yaml"
    destination.write_text("tokens:\n  HF_TOKEN: hf_old\n", encoding="utf-8")
    monkeypatch.setattr(
        credentials_module.os,
        "replace",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError("synthetic replace failure")
        ),
    )

    with pytest.raises(OSError, match="synthetic replace failure"):
        persist_supported_env_credentials(
            path=destination, environ={"HF_TOKEN": "hf_new"}
        )

    assert destination.read_text(encoding="utf-8") == "tokens:\n  HF_TOKEN: hf_old\n"
    assert list(tmp_path.glob(".credentials.yaml.*")) == []


def test_write_credentials_file_does_not_normalize_legacy_token_factory_key(
    tmp_path: Path,
) -> None:
    credentials_path = tmp_path / "credentials.yaml"
    credentials_path.write_text(
        yaml.safe_dump({"tokens": {"NEBIUS_TOKEN_FACTORY_API_KEY": "tf-legacy-file"}})
    )

    write_credentials_file(
        {"storage": {"bucket": "s3://bucket/checkpoints/"}},
        path=credentials_path,
    )

    stored = yaml.safe_load(credentials_path.read_text())
    assert stored["tokens"]["NEBIUS_TOKEN_FACTORY_API_KEY"] == "tf-legacy-file"
    assert "NEBIUS_TOKEN_FACTORY_KEY" not in stored["tokens"]
    assert stored["storage"]["bucket"] == "s3://bucket/checkpoints/"


def test_generic_service_account_update_migrates_legacy_storage_ownership(
    tmp_path: Path,
) -> None:
    credentials_path = tmp_path / "credentials.yaml"
    credentials_path.write_text(
        yaml.safe_dump(
            {
                "nebius": {
                    "service_account_id": "serviceaccount-storage",
                    "service_account_name": "lerobot-training",
                    "service_account_project_id": "project-a",
                    "service_account_managed_by": "npa",
                }
            }
        )
    )

    write_credentials_file(
        {"nebius": {"service_account_id": "serviceaccount-agent"}},
        path=credentials_path,
    )

    stored = yaml.safe_load(credentials_path.read_text())
    assert stored["nebius"] == {"service_account_id": "serviceaccount-agent"}
    assert stored["storage_iam"] == {
        "service_account_id": "serviceaccount-storage",
        "service_account_name": "lerobot-training",
        "service_account_project_id": "project-a",
        "service_account_managed_by": "npa",
    }


def test_shared_credential_env_never_emits_ai_cloud_key(tmp_path: Path) -> None:
    """The removed NEBIUS_AI_CLOUD_KEY must not leak back into the shared env."""
    credentials_path = tmp_path / "credentials.yaml"
    credentials_path.write_text(
        yaml.safe_dump({"tokens": {"NEBIUS_AI_CLOUD_KEY": "stale-ai-cloud-key"}})
    )

    resolved = load_credentials(path=credentials_path, environ={})

    assert not hasattr(resolved, "ai_cloud_api_key")
    assert not hasattr(resolved, "nebius_api_key")
    assert "NEBIUS_AI_CLOUD_KEY" not in shared_credential_env(resolved)


def test_load_credentials_reads_byovm_ssh_config(tmp_path: Path) -> None:
    credentials_path = tmp_path / "credentials.yaml"
    credentials_path.write_text(
        yaml.safe_dump(
            {
                "ssh": {
                    "host": "203.0.113.10",
                    "user": "robot",
                    "key_path": "~/.ssh/byovm",
                }
            }
        )
    )

    resolved = load_credentials(path=credentials_path, environ={})

    assert resolved.ssh_host == "203.0.113.10"
    assert resolved.ssh_user == "robot"
    assert resolved.ssh_key_path == "~/.ssh/byovm"


def test_load_credentials_reads_shared_s3_storage(tmp_path: Path) -> None:
    credentials_path = tmp_path / "credentials.yaml"
    credentials_path.write_text(
        yaml.safe_dump(
            {
                "tokens": {"HF_TOKEN": "hf-file"},
                "storage": {
                    "aws_access_key_id": "access",
                    "aws_secret_access_key": "secret",
                    "endpoint_url": "https://storage.example",
                    "bucket": "s3://bucket/checkpoints/",
                },
            }
        )
    )

    resolved = load_credentials(path=credentials_path, environ={})

    assert shared_credential_env(resolved) == {
        "HF_TOKEN": "hf-file",
        "HUGGING_FACE_HUB_TOKEN": "hf-file",
        "AWS_ACCESS_KEY_ID": "access",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "AWS_ENDPOINT_URL": "https://storage.example",
        "NEBIUS_S3_ENDPOINT": "https://storage.example",
        "NEBIUS_S3_BUCKET": "s3://bucket/checkpoints/",
    }


def test_load_credentials_byovm_env_overrides_ssh_config(tmp_path: Path) -> None:
    credentials_path = tmp_path / "credentials.yaml"
    credentials_path.write_text(
        yaml.safe_dump(
            {
                "ssh": {
                    "host": "file-host",
                    "user": "file-user",
                    "key_path": "/file/key",
                }
            }
        )
    )

    resolved = load_credentials(
        path=credentials_path,
        environ={
            "NPA_BYOVM_HOST": "env-host",
            "NPA_BYOVM_SSH_USER": "env-user",
            "NPA_BYOVM_SSH_KEY": "/env/key",
        },
    )

    assert resolved.ssh_host == "env-host"
    assert resolved.ssh_user == "env-user"
    assert resolved.ssh_key_path == "/env/key"


def test_load_credentials_env_overrides_file(tmp_path: Path) -> None:
    credentials_path = tmp_path / "credentials.yaml"
    credentials_path.write_text(yaml.safe_dump({"tokens": {"HF_TOKEN": "hf-file"}}))

    resolved = load_credentials(
        path=credentials_path,
        environ={"HF_TOKEN": "hf-env"},
    )

    assert resolved.tokens["HF_TOKEN"] == "hf-env"


def test_load_credentials_missing_file_returns_empty(tmp_path: Path) -> None:
    resolved = load_credentials(path=tmp_path / "missing.yaml", environ={})

    assert resolved.tokens == {}
    assert resolved.warnings == []


def test_load_credentials_distinguishes_malformed_store(tmp_path: Path) -> None:
    credentials_path = tmp_path / "credentials.yaml"
    credentials_path.write_text("tokens: [unterminated\n", encoding="utf-8")

    with pytest.raises(CredentialStoreError) as raised:
        load_credentials(path=credentials_path, environ={})

    assert raised.value.kind == "malformed"
    assert "malformed" in str(raised.value)


def test_load_credentials_distinguishes_unreadable_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    credentials_path = tmp_path / "credentials.yaml"
    credentials_path.write_text("{}\n", encoding="utf-8")
    original_open = Path.open

    def deny_target(path: Path, *args, **kwargs):
        if path == credentials_path:
            raise PermissionError("permission denied")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", deny_target)

    with pytest.raises(CredentialStoreError) as raised:
        load_credentials(path=credentials_path, environ={})

    assert raised.value.kind == "unreadable"
    assert "permission denied" in str(raised.value)


def test_warn_if_hf_token_missing_uses_standard_message() -> None:
    warnings: list[str] = []

    missing = warn_if_hf_token_missing(CredentialsConfig(), warn=warnings.append)

    assert missing is True
    assert warnings == [
        "Warning: HF_TOKEN not found in ~/.npa/credentials.yaml. "
        "Gated model downloads will fail."
    ]


def test_load_credentials_warns_when_readable_by_other_users(tmp_path: Path) -> None:
    credentials_path = tmp_path / "credentials.yaml"
    credentials_path.write_text(yaml.safe_dump({"tokens": {"HF_TOKEN": "hf-file"}}))
    credentials_path.chmod(0o644)
    warnings: list[str] = []

    resolved = load_credentials(
        path=credentials_path,
        environ={},
        warn=warnings.append,
    )

    assert resolved.tokens["HF_TOKEN"] == "hf-file"
    assert warnings == [
        "credentials.yaml is readable by other users. Run chmod 600 ~/.npa/credentials.yaml."
    ]


def test_cosmos_deploy_dry_run_probes_upstream_when_hf_token_missing(
    tmp_path: Path, mocker
) -> None:
    mocker.patch("npa.cli.workbench.load_credentials", return_value=CredentialsConfig())
    mocker.patch("npa.cli.cosmos.resolve_credentials", return_value=CredentialsConfig())
    access = mocker.patch(
        "npa.cli.cosmos.validate_hf_access",
        return_value=SimpleNamespace(
            ok=False,
            error="HF_TOKEN does not have upstream access to the selected model",
        ),
    )
    apply = mocker.patch("npa.cli.cosmos.provisioner.apply")

    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "cosmos",
            "deploy",
            "--project-id",
            "project",
            "--tenant-id",
            "tenant",
            "--region",
            "eu-north1",
            "--tf-dir",
            str(tmp_path),
            "--gpu-type",
            "gpu-h100-sxm",
            "--gpu-preset",
            "1gpu-16vcpu-200gb",
            "--dry-run",
        ],
    )

    assert result.exit_code == 1
    assert "HF_TOKEN does not have upstream access" in result.output
    access.assert_called_once_with(
        "", "nvidia/Cosmos-1.0-Diffusion-7B-Text2World"
    )
    apply.assert_not_called()


def test_cosmos_deploy_dry_run_prints_redacted_shared_credentials(
    tmp_path: Path, mocker
) -> None:
    credentials = CredentialsConfig(
        tokens={"HF_TOKEN": "hf_123456789"},
        s3_access_key_id="AKIA123456",
        s3_secret_access_key="secret$!`'\"\\value",
        s3_endpoint="https://storage.example",
        s3_bucket="s3://bucket/checkpoints/",
    )
    mocker.patch("npa.cli.cosmos.resolve_credentials", return_value=credentials)
    mocker.patch(
        "npa.cli.cosmos.validate_hf_access",
        return_value=SimpleNamespace(ok=True, error=""),
    )
    apply = mocker.patch("npa.cli.cosmos.provisioner.apply")

    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "cosmos",
            "deploy",
            "--project-id",
            "project",
            "--tenant-id",
            "tenant",
            "--region",
            "eu-north1",
            "--tf-dir",
            str(tmp_path),
            "--gpu-type",
            "gpu-h100-sxm",
            "--gpu-preset",
            "1gpu-16vcpu-200gb",
            "--runtime",
            "container",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "HF access ok: nvidia/Cosmos-1.0-Diffusion-7B-Text2World" in result.output
    assert "HF_TOKEN='hf_1****'" in result.output
    assert "AWS_ACCESS_KEY_ID='AKIA****'" in result.output
    assert "AWS_SECRET_ACCESS_KEY='secr****'" in result.output
    assert "secret$!" not in result.output
    apply.assert_not_called()


def test_ssh_forwards_tokens_into_remote_environment(mocker) -> None:
    channel = mocker.MagicMock()
    channel.recv.side_effect = [b"ok\n", b""]
    channel.recv_stderr.side_effect = [b""]
    channel.recv_exit_status.return_value = 0
    transport = mocker.MagicMock()
    transport.open_session.return_value = channel
    remote_file = mocker.MagicMock()
    sftp = mocker.MagicMock()
    sftp.open.return_value.__enter__.return_value = remote_file
    paramiko_client = mocker.MagicMock()
    paramiko_client.get_transport.return_value = transport
    paramiko_client.open_sftp.return_value = sftp
    mocker.patch("paramiko.SSHClient", return_value=paramiko_client)
    mocker.patch(
        "npa.clients.ssh.uuid.uuid4", return_value=SimpleNamespace(hex="abc123")
    )

    client = SSHClient(
        SSHConfig(
            host="host",
            user="ubuntu",
            key_path="key",
            tokens={"HF_TOKEN": "hf-file", "NGC_API_KEY": "ngc-file"},
        )
    )

    result = client.run("echo hello")

    assert result == (0, "ok\n", "")
    remote_command = channel.exec_command.call_args.args[0]
    assert "hf-file" not in remote_command
    assert "ngc-file" not in remote_command
    assert ". /tmp/.npa-env-abc123" in remote_command
    assert "rm -f /tmp/.npa-env-abc123" in remote_command
    assert "echo hello" in remote_command
    sftp.open.assert_called_once_with("/tmp/.npa-env-abc123", "w")
    sftp.chmod.assert_called_once_with("/tmp/.npa-env-abc123", 0o600)
    remote_env = remote_file.write.call_args.args[0]
    assert "export HF_TOKEN='hf-file'" in remote_env
    assert "export NGC_API_KEY='ngc-file'" in remote_env
