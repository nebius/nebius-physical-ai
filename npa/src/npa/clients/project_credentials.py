"""Project-aware S3 credential resolution helpers."""

from __future__ import annotations

from dataclasses import dataclass
import os

import boto3
from botocore.config import Config as BotoConfig

from npa.clients.config import StorageConfig, list_projects, resolve_project_storage
from npa.clients.credentials import load_credentials
from npa.clients.storage import StorageClient
from npa.errors import ScopedCredentialError


@dataclass(frozen=True)
class CredentialPair:
    project: str | None
    endpoint_url: str
    aws_access_key_id: str
    aws_secret_access_key: str
    uses_host_credentials: bool = False

    @property
    def storage(self) -> StorageConfig:
        return StorageConfig(
            checkpoint_bucket="",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
        )


def resolve_credentials(
    project: str | None,
    allow_host_creds: bool = False,
) -> CredentialPair:
    """Resolve S3 credentials for a project alias.

    `project=None` preserves the existing default-project behavior.
    """
    normalized = project or None
    if normalized is not None and normalized not in list_projects():
        raise ScopedCredentialError(
            normalized,
            f"resolve storage credentials for project '{normalized}'",
            remediation=(
                f"Configure project '{normalized}' in ~/.npa/config.yaml, "
                "or pass an existing project alias."
            ),
            failed_project=normalized,
        )

    storage = resolve_project_storage(normalized)
    endpoint_url = (
        storage.endpoint_url
        or os.environ.get("AWS_ENDPOINT_URL", "")
        or os.environ.get("NEBIUS_S3_ENDPOINT", "")
    )
    access_key = storage.aws_access_key_id or os.environ.get("AWS_ACCESS_KEY_ID", "")
    secret_key = storage.aws_secret_access_key or os.environ.get(
        "AWS_SECRET_ACCESS_KEY", ""
    )
    if endpoint_url and access_key and secret_key:
        return CredentialPair(
            project=normalized,
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

    # Fall back to the user credential file (~/.npa/credentials.yaml) that the
    # rest of npa uses for S3 access. These are host-level credentials, so use
    # them for the default project or when the caller opts in with
    # allow_host_creds.
    if normalized is None or allow_host_creds:
        user_creds = load_credentials()
        host_endpoint = endpoint_url or user_creds.s3_endpoint
        host_access = access_key or user_creds.s3_access_key_id
        host_secret = secret_key or user_creds.s3_secret_access_key
        if host_endpoint and host_access and host_secret:
            return CredentialPair(
                project=normalized,
                endpoint_url=host_endpoint,
                aws_access_key_id=host_access,
                aws_secret_access_key=host_secret,
                uses_host_credentials=True,
            )

    if allow_host_creds and endpoint_url:
        return CredentialPair(
            project=normalized,
            endpoint_url=endpoint_url,
            aws_access_key_id="",
            aws_secret_access_key="",
            uses_host_credentials=True,
        )

    label = normalized or "default"
    raise ScopedCredentialError(
        storage.checkpoint_bucket or label,
        f"resolve storage credentials for project '{label}'",
        remediation=(
            "Configure object-storage credentials for this project, "
            "or pass --allow-host-creds to use host credentials."
        ),
        failed_project=label,
    )


def s3_client_for_project(project: str | None, *, allow_host_creds: bool = False):
    credentials = resolve_credentials(project, allow_host_creds=allow_host_creds)
    return boto3.client(
        "s3",
        endpoint_url=credentials.endpoint_url or None,
        aws_access_key_id=credentials.aws_access_key_id or None,
        aws_secret_access_key=credentials.aws_secret_access_key or None,
        config=BotoConfig(signature_version="s3v4"),
    )


def storage_client_for_project(
    project: str | None,
    *,
    allow_host_creds: bool = False,
) -> StorageClient:
    credentials = resolve_credentials(project, allow_host_creds=allow_host_creds)
    return StorageClient.from_environment(
        endpoint_url=credentials.endpoint_url,
        aws_access_key_id=credentials.aws_access_key_id,
        aws_secret_access_key=credentials.aws_secret_access_key,
    )


def storage_env_for_project(
    project: str | None,
    *,
    allow_host_creds: bool = False,
) -> dict[str, str]:
    credentials = resolve_credentials(project, allow_host_creds=allow_host_creds)
    env = {
        "AWS_ENDPOINT_URL": credentials.endpoint_url,
        "NEBIUS_S3_ENDPOINT": credentials.endpoint_url,
    }
    if credentials.aws_access_key_id:
        env["AWS_ACCESS_KEY_ID"] = credentials.aws_access_key_id
    if credentials.aws_secret_access_key:
        env["AWS_SECRET_ACCESS_KEY"] = credentials.aws_secret_access_key
    return env
