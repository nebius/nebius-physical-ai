"""Unit tests for `npa.clients.project_credentials`."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from npa.clients import project_credentials
from npa.clients.project_credentials import (
    CredentialPair,
    resolve_credentials,
    s3_client_for_project,
    storage_client_for_project,
    storage_env_for_project,
)
from npa.errors import ScopedCredentialError


@pytest.fixture()
def _mock_project_storage(monkeypatch: pytest.MonkeyPatch):
    """Provide a controllable resolve_project_storage + list_projects pair."""

    state = {
        "projects": {"proj-known"},
        "storage": SimpleNamespace(
            endpoint_url="https://s3.example",
            aws_access_key_id="ak",
            aws_secret_access_key="sk",
            checkpoint_bucket="bucket",
        ),
    }

    monkeypatch.setattr(
        project_credentials,
        "list_projects",
        lambda: {p: {} for p in state["projects"]},
    )
    monkeypatch.setattr(
        project_credentials,
        "resolve_project_storage",
        lambda project=None: state["storage"],
    )
    # Default-clean env so resolve_credentials does not pick host values up.
    for var in (
        "AWS_ENDPOINT_URL",
        "NEBIUS_S3_ENDPOINT",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    return state


# ── resolve_credentials ───────────────────────────────────────────────────


def test_resolve_credentials_returns_storage_when_complete(_mock_project_storage) -> None:
    creds = resolve_credentials("proj-known")
    assert creds == CredentialPair(
        project="proj-known",
        endpoint_url="https://s3.example",
        aws_access_key_id="ak",
        aws_secret_access_key="sk",
    )
    assert creds.uses_host_credentials is False


def test_resolve_credentials_uses_storage_dataclass_property(_mock_project_storage) -> None:
    creds = resolve_credentials("proj-known")
    storage_cfg = creds.storage
    assert storage_cfg.endpoint_url == "https://s3.example"
    assert storage_cfg.aws_access_key_id == "ak"
    assert storage_cfg.checkpoint_bucket == ""


def test_resolve_credentials_unknown_project_raises(_mock_project_storage) -> None:
    with pytest.raises(ScopedCredentialError):
        resolve_credentials("proj-missing")


def test_resolve_credentials_none_project_uses_default(_mock_project_storage) -> None:
    # `None` skips the membership check and uses the default project.
    creds = resolve_credentials(None)
    assert creds.project is None
    assert creds.aws_access_key_id == "ak"


def test_resolve_credentials_falls_back_to_env(
    monkeypatch: pytest.MonkeyPatch, _mock_project_storage
) -> None:
    _mock_project_storage["storage"] = SimpleNamespace(
        endpoint_url="",
        aws_access_key_id="",
        aws_secret_access_key="",
        checkpoint_bucket="",
    )
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://env.example")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "env-ak")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "env-sk")

    creds = resolve_credentials("proj-known")
    assert creds.endpoint_url == "https://env.example"
    assert creds.aws_access_key_id == "env-ak"


def test_resolve_credentials_host_creds_allowed(
    monkeypatch: pytest.MonkeyPatch, _mock_project_storage
) -> None:
    _mock_project_storage["storage"] = SimpleNamespace(
        endpoint_url="",
        aws_access_key_id="",
        aws_secret_access_key="",
        checkpoint_bucket="",
    )
    monkeypatch.setenv("NEBIUS_S3_ENDPOINT", "https://host.example")

    creds = resolve_credentials("proj-known", allow_host_creds=True)
    assert creds.uses_host_credentials is True
    assert creds.endpoint_url == "https://host.example"
    assert creds.aws_access_key_id == ""


def test_resolve_credentials_raises_when_nothing_configured(_mock_project_storage) -> None:
    _mock_project_storage["storage"] = SimpleNamespace(
        endpoint_url="",
        aws_access_key_id="",
        aws_secret_access_key="",
        checkpoint_bucket="",
    )
    with pytest.raises(ScopedCredentialError):
        resolve_credentials("proj-known")


# ── Convenience constructors ──────────────────────────────────────────────


def test_s3_client_for_project_passes_credentials(
    _mock_project_storage, mocker
) -> None:
    boto3_client = mocker.patch("npa.clients.project_credentials.boto3.client")

    s3_client_for_project("proj-known")

    kwargs = boto3_client.call_args.kwargs
    assert boto3_client.call_args.args == ("s3",)
    assert kwargs["endpoint_url"] == "https://s3.example"
    assert kwargs["aws_access_key_id"] == "ak"
    assert kwargs["aws_secret_access_key"] == "sk"
    assert kwargs["config"].signature_version == "s3v4"


def test_storage_client_for_project_uses_resolved_credentials(
    _mock_project_storage, mocker
) -> None:
    factory = mocker.patch(
        "npa.clients.project_credentials.StorageClient.from_environment",
        return_value="storage-client",
    )

    result = storage_client_for_project("proj-known")
    assert result == "storage-client"
    factory.assert_called_once_with(
        endpoint_url="https://s3.example",
        aws_access_key_id="ak",
        aws_secret_access_key="sk",
    )


def test_storage_env_for_project_returns_full_mapping(_mock_project_storage) -> None:
    env = storage_env_for_project("proj-known")
    assert env == {
        "AWS_ENDPOINT_URL": "https://s3.example",
        "NEBIUS_S3_ENDPOINT": "https://s3.example",
        "AWS_ACCESS_KEY_ID": "ak",
        "AWS_SECRET_ACCESS_KEY": "sk",
    }


def test_storage_env_for_project_omits_missing_keys(
    _mock_project_storage, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_project_storage["storage"] = SimpleNamespace(
        endpoint_url="https://only-endpoint.example",
        aws_access_key_id="",
        aws_secret_access_key="",
        checkpoint_bucket="",
    )
    env = storage_env_for_project("proj-known", allow_host_creds=True)
    assert env == {
        "AWS_ENDPOINT_URL": "https://only-endpoint.example",
        "NEBIUS_S3_ENDPOINT": "https://only-endpoint.example",
    }
