from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

from npa.cli.main import app
from npa.clients.config import SSHConfig
from npa.clients.credentials import CredentialsConfig, load_credentials
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


def test_cosmos_deploy_requires_hf_token(tmp_path: Path, mocker) -> None:
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
        ],
    )

    assert result.exit_code == 1
    assert "HF_TOKEN required for Cosmos deployment" in result.output
    assert "~/.npa/credentials.yaml" in result.output
    assert "under tokens:" in result.output
    assert "set HF_TOKEN as an environment" in result.output
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
    assert "export HF_TOKEN=hf-file" in remote_env
    assert "export NGC_API_KEY=ngc-file" in remote_env
