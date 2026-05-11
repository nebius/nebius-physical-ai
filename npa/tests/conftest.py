import os
import pytest

from npa.clients.project_credentials import CredentialPair
from npa.errors import ScopedCredentialError

os.environ.setdefault("NPA_PROJECT_ID", "project-test-00000000")
os.environ.setdefault("NPA_S3_BUCKET", "test-bucket-00000000")


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


@pytest.fixture
def mock_cross_project_creds(monkeypatch):
    """Mock credential resolution to simulate distinct credentials per project."""
    creds_by_project = {
        "project-source": CredentialPair(
            project="project-source",
            endpoint_url="https://source-storage.example",
            aws_access_key_id="src-key",
            aws_secret_access_key="src-secret",
        ),
        "project-target": CredentialPair(
            project="project-target",
            endpoint_url="https://target-storage.example",
            aws_access_key_id="tgt-key",
            aws_secret_access_key="tgt-secret",
        ),
        None: CredentialPair(
            project=None,
            endpoint_url="https://source-storage.example",
            aws_access_key_id="src-key",
            aws_secret_access_key="src-secret",
        ),
    }

    def fake_resolve(project, allow_host_creds=False):
        if project in creds_by_project:
            return creds_by_project[project]
        if allow_host_creds:
            return CredentialPair(
                project=project,
                endpoint_url="https://host-storage.example",
                aws_access_key_id="",
                aws_secret_access_key="",
                uses_host_credentials=True,
            )
        raise ScopedCredentialError(
            project or "default",
            f"resolve storage credentials for project '{project or 'default'}'",
            failed_project=project or "default",
        )

    monkeypatch.setattr(
        "npa.clients.project_credentials.resolve_credentials", fake_resolve
    )
    monkeypatch.setattr("npa.cli.demo.resolve_credentials", fake_resolve)
    return creds_by_project
