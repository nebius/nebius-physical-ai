from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml
from typer.testing import CliRunner

from npa.cli.main import app
from npa.clients.config import SSHConfig
from npa.clients.credentials import (
    CredentialsConfig,
    load_credentials,
    shared_credential_env,
    warn_if_hf_token_missing,
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


def test_load_credentials_reads_project_scoped_storage(tmp_path: Path) -> None:
    credentials_path = tmp_path / "credentials.yaml"
    credentials_path.write_text(
        yaml.safe_dump(
            {
                "projects": {
                    "proj": {
                        "storage": {
                            "aws_access_key_id": "project-access",
                            "aws_secret_access_key": "project-secret",
                            "endpoint_url": "https://storage.project",
                            "bucket": "s3://project-bucket/checkpoints/",
                        }
                    }
                }
            }
        )
    )

    resolved = load_credentials(path=credentials_path, environ={})

    assert resolved.project_storage["proj"].aws_access_key_id == "project-access"
    assert resolved.project_storage["proj"].aws_secret_access_key == "project-secret"
    assert resolved.project_storage["proj"].endpoint_url == "https://storage.project"
    assert resolved.project_storage["proj"].bucket == "s3://project-bucket/checkpoints/"


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


def test_cosmos_deploy_dry_run_fails_when_hf_token_missing(tmp_path: Path, mocker) -> None:
    mocker.patch("npa.cli.workbench.load_credentials", return_value=CredentialsConfig())
    mocker.patch("npa.cli.cosmos.resolve_credentials", return_value=CredentialsConfig())
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
    assert "Warning: HF_TOKEN not found in ~/.npa/credentials.yaml" in result.output
    assert "~/.npa/credentials.yaml" in result.output
    apply.assert_not_called()


def test_cosmos_deploy_dry_run_prints_redacted_shared_credentials(tmp_path: Path, mocker) -> None:
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
    mocker.patch("npa.clients.ssh.uuid.uuid4", return_value=SimpleNamespace(hex="abc123"))

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
