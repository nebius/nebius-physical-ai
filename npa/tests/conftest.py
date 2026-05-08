import os
import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def tmp_workspace(tmp_path):
    """A clean temp directory simulating a workspace."""
    return tmp_path


@pytest.fixture
def sample_config(tmp_path):
    """Write a minimal valid config YAML and return its path."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "tenant: test-tenant\n"
        "project: test-project\n"
        "region: eu-north1\n"
        "bucket: test-bucket\n"
    )
    return cfg


@pytest.fixture
def mock_ssh(mocker):
    """Patch paramiko.SSHClient universally."""
    mock_client = mocker.MagicMock()
    mock_client.exec_command.return_value = (
        mocker.MagicMock(),  # stdin
        mocker.MagicMock(read=lambda: b"ok\n"),  # stdout
        mocker.MagicMock(read=lambda: b""),  # stderr
    )
    mocker.patch("paramiko.SSHClient", return_value=mock_client)
    return mock_client


@pytest.fixture
def mock_s3(mocker):
    """Patch boto3 S3 client."""
    mock_client = mocker.MagicMock()
    mocker.patch("boto3.client", return_value=mock_client)
    return mock_client
