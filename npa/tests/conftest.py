import os
from urllib.parse import urlparse

import httpx
import pytest

from npa.clients.project_credentials import CredentialPair
from npa.errors import ScopedCredentialError
from npa.guardrails.pytest_collection import assert_nonzero_collection

os.environ.setdefault("NPA_PROJECT_ID", "project-test-00000000")
os.environ.setdefault("NPA_S3_BUCKET", "test-bucket-00000000")

# Live markers whose tests intentionally use real ambient credentials.
_LIVE_MARKERS = frozenset(
    {
        "byovm_live",
        "e2e",
        "e2e_pipeline",
        "e2e_serverless",
        "e2e_skypilot",
        "gpu",
        "multi_gpu",
        "ngc_e2e",
    }
)

# Credential/storage env vars that override file config. A contributor who has
# followed the quickstart and exported real Nebius creds must still get a
# hermetic unit suite, so these are scrubbed for non-live tests. Without this,
# e.g. an exported AWS_ENDPOINT_URL leaks into deploy-config assertions.
_AMBIENT_CREDENTIAL_ENV_VARS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_ENDPOINT_URL",
    "AWS_PROFILE",
    "AWS_DEFAULT_REGION",
    "AWS_REGION",
    "NEBIUS_S3_ENDPOINT",
    "NEBIUS_S3_BUCKET",
    "NPA_STORAGE_ENDPOINT",
    "NPA_CHECKPOINT_BUCKET",
    "NEBIUS_PROJECT_ID",
    "NEBIUS_TENANT_ID",
    "NPA_REGISTRY",
    "NPA_REGISTRY_ID",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "NGC_API_KEY",
    "NGC_ORG",
    "NGC_TEAM",
    "NPA_BYOVM_HOST",
    "NPA_SSH_HOST",
    "NPA_BYOVM_SSH_USER",
    "NPA_SSH_USER",
    "NPA_BYOVM_SSH_KEY",
    "NPA_SSH_KEY",
)


def pytest_collection_finish(session: pytest.Session) -> None:
    assert_nonzero_collection(len(session.items))


@pytest.fixture(autouse=True)
def scrub_ambient_credential_env(monkeypatch, request):
    """Isolate non-live tests from real credentials exported in the shell."""
    if any(request.node.get_closest_marker(marker) for marker in _LIVE_MARKERS):
        return
    for env_var in _AMBIENT_CREDENTIAL_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)


def _is_huggingface_url(url: object) -> bool:
    host = urlparse(str(url)).hostname or ""
    return host == "huggingface.co" or host.endswith(".huggingface.co")


@pytest.fixture(autouse=True)
def block_live_huggingface_http(monkeypatch, request):
    """Keep unit tests from depending on live Hugging Face availability."""
    live_markers = {
        "byovm_live",
        "e2e",
        "e2e_pipeline",
        "e2e_serverless",
        "gpu",
        "multi_gpu",
        "ngc_e2e",
    }
    if any(request.node.get_closest_marker(marker) for marker in live_markers):
        return

    def blocked(method: str, url: object) -> None:
        if _is_huggingface_url(url):
            raise AssertionError(
                f"Live Hugging Face HTTP is blocked in unit tests: {method} {url}. "
                "Mock the access check instead."
            )

    original_head = httpx.head
    original_get = httpx.get
    original_post = httpx.post
    original_request = httpx.request
    original_client_request = httpx.Client.request
    original_async_client_request = httpx.AsyncClient.request

    def guarded_head(url, *args, **kwargs):
        blocked("HEAD", url)
        return original_head(url, *args, **kwargs)

    def guarded_get(url, *args, **kwargs):
        blocked("GET", url)
        return original_get(url, *args, **kwargs)

    def guarded_post(url, *args, **kwargs):
        blocked("POST", url)
        return original_post(url, *args, **kwargs)

    def guarded_request(method, url, *args, **kwargs):
        blocked(str(method).upper(), url)
        return original_request(method, url, *args, **kwargs)

    def guarded_client_request(self, method, url, *args, **kwargs):
        blocked(str(method).upper(), url)
        return original_client_request(self, method, url, *args, **kwargs)

    async def guarded_async_client_request(self, method, url, *args, **kwargs):
        blocked(str(method).upper(), url)
        return await original_async_client_request(self, method, url, *args, **kwargs)

    monkeypatch.setattr(httpx, "head", guarded_head)
    monkeypatch.setattr(httpx, "get", guarded_get)
    monkeypatch.setattr(httpx, "post", guarded_post)
    monkeypatch.setattr(httpx, "request", guarded_request)
    monkeypatch.setattr(httpx.Client, "request", guarded_client_request)
    monkeypatch.setattr(httpx.AsyncClient, "request", guarded_async_client_request)


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
    return creds_by_project
