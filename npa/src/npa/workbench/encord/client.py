"""Lazy Encord SDK seam and exact identifier resolution."""

from __future__ import annotations

import base64
import os
import re
from typing import Any

from npa.workbench.encord.schemas import EncordAuthError, EncordToolError

ENCORD_SSH_KEY_ENV = "ENCORD_SSH_KEY"
ENCORD_SSH_KEY_B64_ENV = "ENCORD_SSH_KEY_B64"
ENCORD_SSH_KEY_FILE_ENV = "ENCORD_SSH_KEY_FILE"
ENCORD_DOMAIN_ENV = "ENCORD_DOMAIN"
DEFAULT_ENCORD_DOMAIN = "https://api.encord.com"

AUTH_REMEDY = (
    "Set ENCORD_SSH_KEY, ENCORD_SSH_KEY_B64, or ENCORD_SSH_KEY_FILE in the "
    "environment or NPA credentials."
)
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def looks_like_id(value: str) -> bool:
    return bool(_UUID_RE.fullmatch(value.strip()))


def resolve_domain(environ: dict[str, str] | None = None) -> str:
    env = environ if environ is not None else os.environ
    return env.get(ENCORD_DOMAIN_ENV, "").strip() or DEFAULT_ENCORD_DOMAIN


def resolve_public_endpoint(environ: dict[str, str] | None = None) -> str:
    env = environ if environ is not None else os.environ
    endpoint = (
        env.get("AWS_ENDPOINT_URL", "").strip()
        or env.get("NEBIUS_S3_ENDPOINT", "").strip()
    )
    if not endpoint and environ is None:
        from npa.clients.credentials import load_credentials

        endpoint = load_credentials().s3_endpoint.strip()
    if not endpoint:
        raise EncordToolError("no S3 endpoint is configured for object URL construction")
    return endpoint.rstrip("/")


def _resolve_auth_env(environ: dict[str, str] | None = None) -> dict[str, str]:
    if environ is not None:
        return dict(environ)
    from npa.clients.credentials import load_credentials

    tokens = load_credentials().tokens
    return {
        name: tokens.get(name, "")
        for name in (ENCORD_SSH_KEY_ENV, ENCORD_SSH_KEY_B64_ENV, ENCORD_SSH_KEY_FILE_ENV)
    }


def _default_user_client(environ: dict[str, str] | None = None) -> Any:
    env = _resolve_auth_env(environ)
    ssh_key = env.get(ENCORD_SSH_KEY_ENV, "").strip()
    ssh_key_b64 = env.get(ENCORD_SSH_KEY_B64_ENV, "").strip()
    ssh_key_file = env.get(ENCORD_SSH_KEY_FILE_ENV, "").strip()
    if not (ssh_key or ssh_key_b64 or ssh_key_file):
        raise EncordAuthError(f"No Encord credential found. {AUTH_REMEDY}")
    if not ssh_key and ssh_key_b64:
        try:
            ssh_key = base64.b64decode(ssh_key_b64, validate=True).decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise EncordAuthError("ENCORD_SSH_KEY_B64 is not valid base64 UTF-8") from exc
    try:
        from encord.user_client import EncordUserClient
    except ModuleNotFoundError as exc:
        raise EncordToolError("Install the optional Encord SDK with `npa[encord]`.") from exc
    kwargs = {"domain": resolve_domain(environ)}
    if ssh_key:
        kwargs["ssh_private_key"] = ssh_key
    else:
        kwargs["ssh_private_key_path"] = ssh_key_file
    try:
        return EncordUserClient.create_with_ssh_private_key(**kwargs)
    except Exception as exc:  # noqa: BLE001 - SDK exceptions are not stable
        raise EncordAuthError(f"Encord authentication failed. {AUTH_REMEDY}") from exc


def resolve_integration(user_client: Any, value: str) -> tuple[str, str]:
    value = value.strip()
    if not value:
        raise EncordToolError("integration must not be empty")
    rows = list(user_client.get_cloud_integrations())
    if looks_like_id(value):
        matches = [row for row in rows if str(row.id).lower() == value.lower()]
    else:
        matches = [row for row in rows if str(row.title) == value]
    if len(matches) != 1:
        raise EncordToolError(f"integration {value!r} did not resolve uniquely")
    return str(matches[0].id), str(matches[0].title)


def find_folder(user_client: Any, value: str) -> Any | None:
    value = value.strip()
    if not value:
        raise EncordToolError("folder must not be empty")
    if looks_like_id(value):
        return user_client.get_storage_folder(value)
    matches = [
        row
        for row in user_client.list_storage_folders(search=value, page_size=1000)
        if str(row.name) == value
    ]
    if len(matches) > 1:
        raise EncordToolError(f"folder {value!r} did not resolve uniquely")
    return matches[0] if matches else None


def create_folder(user_client: Any, name: str) -> Any:
    return user_client.create_storage_folder(
        name.strip(), description="Created by npa workbench encord push"
    )


def find_dataset(user_client: Any, value: str) -> tuple[Any, str, str] | None:
    value = value.strip()
    if not value:
        raise EncordToolError("dataset reference must not be empty")
    if looks_like_id(value):
        try:
            dataset = user_client.get_dataset(value, include_data_rows=False)
        except TypeError:
            dataset = user_client.get_dataset(value)
        return dataset, value, str(getattr(dataset, "title", ""))
    rows = list(user_client.get_datasets(title_eq=value))
    if len(rows) > 1:
        raise EncordToolError(f"dataset {value!r} did not resolve uniquely")
    if not rows:
        return None
    info = rows[0]["dataset"]
    dataset_hash = str(info.dataset_hash)
    try:
        dataset = user_client.get_dataset(dataset_hash, include_data_rows=False)
    except TypeError:
        dataset = user_client.get_dataset(dataset_hash)
    return dataset, dataset_hash, value


def create_dataset(user_client: Any, title: str) -> tuple[str, str]:
    """Create a dataset and return its durable identity without hydrating it."""

    try:
        from encord.orm.dataset import StorageLocation

        storage_location: Any = StorageLocation.CORD_STORAGE
    except ModuleNotFoundError:
        storage_location = "CORD_STORAGE"
    response = user_client.create_dataset(
        title.strip(),
        storage_location,
        dataset_description="Created by npa workbench encord push",
        create_backing_folder=False,
    )
    dataset_hash = str(
        getattr(response, "dataset_hash", "")
        or (response.get("dataset_hash") if isinstance(response, dict) else "")
    )
    if not dataset_hash:
        raise EncordToolError("dataset creation returned no dataset hash")
    return dataset_hash, title.strip()


def resolve_project(user_client: Any, value: str) -> tuple[Any, str, str]:
    value = value.strip()
    if looks_like_id(value):
        project = user_client.get_project(value)
        return project, value, str(getattr(project, "title", ""))
    rows = list(user_client.get_projects(title_eq=value))
    if len(rows) != 1:
        raise EncordToolError(f"project {value!r} did not resolve uniquely")
    info = rows[0]["project"]
    project_hash = str(info.project_hash)
    return user_client.get_project(project_hash), project_hash, value


def resolve_collection(user_client: Any, value: str) -> tuple[Any, str, str]:
    value = value.strip()
    if looks_like_id(value):
        collection = user_client.get_collection(value)
        return collection, value, str(getattr(collection, "name", ""))
    matches = [row for row in user_client.list_collections() if str(row.name) == value]
    if len(matches) != 1:
        raise EncordToolError(f"collection {value!r} did not resolve uniquely")
    return matches[0], str(matches[0].uuid), value
