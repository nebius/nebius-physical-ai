"""Self-contained S3 resolver for the minimal Antioch adapter image."""

from __future__ import annotations

import os
from typing import Callable

from npa.clients.storage import StorageClient


DEFAULT_NEBIUS_STORAGE_ENDPOINT = "https://storage.eu-north1.nebius.cloud"


def _host_project_storage() -> tuple[str, str, str] | None:
    """Use the full host resolver when installed, without requiring it in the image."""

    try:
        from npa.clients.config import resolve_project_storage
    except ModuleNotFoundError as exc:
        if exc.name in {
            "npa.clients.config",
            "npa.clients.credentials",
            "npa.config_schema",
            "npa.deploy",
        }:
            return None
        raise
    configured = resolve_project_storage()
    if not configured.endpoint_url:
        return None
    return (
        configured.endpoint_url,
        configured.aws_access_key_id,
        configured.aws_secret_access_key,
    )


def resolve_storage_client(
    *,
    host_resolver: Callable[[], tuple[str, str, str] | None] = _host_project_storage,
) -> StorageClient:
    """Resolve host config or fall back to boto's workload-identity credential chain."""

    environment_endpoint = (
        os.environ.get("AWS_ENDPOINT_URL", "").strip()
        or os.environ.get("NEBIUS_S3_ENDPOINT", "").strip()
        or os.environ.get("NPA_STORAGE_ENDPOINT", "").strip()
    )
    if environment_endpoint:
        return StorageClient.from_environment(endpoint_url=environment_endpoint)
    host_storage = host_resolver()
    if host_storage is not None:
        endpoint, access_key, secret_key = host_storage
        return StorageClient.from_environment(
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
    return StorageClient.from_environment(endpoint_url=DEFAULT_NEBIUS_STORAGE_ENDPOINT)
