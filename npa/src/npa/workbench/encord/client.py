"""Encord SaaS seam: auth, domain, public endpoint, and title-or-id resolution.

Everything that talks to the Encord SDK funnels through this module so tests can
monkeypatch ``default_user_client`` (or inject ``user_client=``) and the
``encord`` package stays a lazy, optional import.
"""

from __future__ import annotations

import base64
import os
import re
from dataclasses import dataclass
from typing import Any, Sequence

from npa.clients.credentials import ENCORD_TOKEN_KEYS
from npa.workbench.encord.schemas import (
    EncordAuthError,
    EncordSdkMissingError,
    EncordToolError,
)

# Exactly two credential transports (a raw multi-line PEM pasted into YAML or
# an env var is truncation-prone and was an observed live failure):
# - ENCORD_SSH_KEY_B64: base64 of the PEM — survives env/secret transport (pods)
# - ENCORD_SSH_KEY_FILE: path to the downloaded key file (laptops)
# The names themselves are declared once, with the other tokens, in
# npa.clients.credentials (ENCORD_TOKEN_KEYS).
ENCORD_SSH_KEY_B64_ENV, ENCORD_SSH_KEY_FILE_ENV = ENCORD_TOKEN_KEYS
ENCORD_CREDENTIAL_TRANSPORTS = ENCORD_TOKEN_KEYS
ENCORD_DOMAIN_ENV = "ENCORD_DOMAIN"
DEFAULT_ENCORD_DOMAIN = "https://api.encord.com"
# encord.orm.dataset.StorageLocation.CORD_STORAGE, the dataset type for
# link_items-driven datasets: items live in our own storage folder and are
# linked explicitly, so the dataset needs no backing folder. Pinned as the
# enum's integer value so resolving a dataset never imports the SDK — the
# injected-client seam must work without it. test_encord checks this against
# the real enum whenever the SDK is installed.
CORD_STORAGE_LOCATION = 0

AUTH_REMEDY = (
    "Set ENCORD_SSH_KEY_B64 (base64 of the PEM: `base64 < key.pem | tr -d '\\n'`) "
    "in the environment or under tokens: in ~/.npa/credentials.yaml, or point "
    "ENCORD_SSH_KEY_FILE at the key file. Generate the key pair in the Encord app "
    "under public keys, and pass the secret to workflow submits with "
    "--secret-env ENCORD_SSH_KEY_B64."
)

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def looks_like_id(value: str) -> bool:
    """Whether ``value`` is UUID/hash-shaped (Encord hashes are UUIDs)."""

    return bool(_UUID_RE.match(value.strip()))


def resolve_domain(environ: dict[str, str] | None = None) -> str:
    env = environ if environ is not None else os.environ
    return env.get(ENCORD_DOMAIN_ENV, "").strip() or DEFAULT_ENCORD_DOMAIN


def resolve_public_endpoint(environ: dict[str, str] | None = None) -> str:
    """Endpoint host used to build the public objectUrls Encord registers."""

    env = environ if environ is not None else os.environ
    endpoint = (
        env.get("AWS_ENDPOINT_URL", "").strip()
        or env.get("NEBIUS_S3_ENDPOINT", "").strip()
    )
    if not endpoint and environ is None:
        from npa.clients.credentials import load_credentials

        endpoint = load_credentials().s3_endpoint.strip()
    if not endpoint:
        raise EncordToolError(
            "No S3 endpoint configured for objectUrl construction. Set "
            "AWS_ENDPOINT_URL / NEBIUS_S3_ENDPOINT or storage.endpoint_url in "
            "~/.npa/credentials.yaml."
        )
    return endpoint.rstrip("/")


def resolve_auth_env(environ: dict[str, str] | None = None) -> dict[str, str]:
    """The Encord auth names, from an injected env or the resolved credentials.

    ``load_credentials`` merges the process environment over the ``tokens:``
    section for these names, so the default path needs no extra merging.
    """

    if environ is not None:
        return dict(environ)
    from npa.clients.credentials import load_credentials

    tokens = load_credentials().tokens
    return {name: tokens.get(name, "") for name in ENCORD_CREDENTIAL_TRANSPORTS}


def configured_credential_transports(environ: dict[str, str] | None = None) -> list[str]:
    """Names (never values) of the Encord credential transports that are set."""

    env = resolve_auth_env(environ)
    return [name for name in ENCORD_CREDENTIAL_TRANSPORTS if env.get(name, "").strip()]


def default_user_client(environ: dict[str, str] | None = None) -> Any:
    """Build an authenticated EncordUserClient from env/NPA credentials."""

    env = resolve_auth_env(environ)
    ssh_key = ""
    ssh_key_b64 = env.get(ENCORD_SSH_KEY_B64_ENV, "").strip()
    ssh_key_file = env.get(ENCORD_SSH_KEY_FILE_ENV, "").strip()
    if not (ssh_key_b64 or ssh_key_file):
        raise EncordAuthError(f"No Encord credential found. {AUTH_REMEDY}")
    if ssh_key_b64:
        try:
            ssh_key = base64.b64decode(ssh_key_b64, validate=True).decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise EncordAuthError(
                f"ENCORD_SSH_KEY_B64 is not valid base64-encoded UTF-8: {exc}"
            ) from exc

    try:
        from encord.user_client import EncordUserClient
    except ModuleNotFoundError as exc:
        raise EncordSdkMissingError(
            "The encord SDK is not installed. Install it with "
            "`pip install 'npa[encord]'` or `pip install encord`."
        ) from exc

    domain = resolve_domain(environ)
    try:
        if ssh_key:
            return EncordUserClient.create_with_ssh_private_key(
                ssh_private_key=ssh_key, domain=domain
            )
        return EncordUserClient.create_with_ssh_private_key(
            ssh_private_key_path=ssh_key_file, domain=domain
        )
    except Exception as exc:  # noqa: BLE001 - SDK raises assorted exception types
        raise EncordAuthError(
            f"Encord authentication failed: {exc}. {AUTH_REMEDY}"
        ) from exc


@dataclass(frozen=True)
class ResolvedRef:
    """One Encord object resolved by uuid/hash or by exact, unique title."""

    obj: Any
    id: str
    title: str
    created: bool = False


def _require_reference(value: str, what: str) -> str:
    value = value.strip()
    if not value:
        raise EncordToolError(f"{what} must not be empty.")
    return value


def _unique_title_match(
    kind: str, value: str, matches: Sequence[Any], *, id_hint: str
) -> Any | None:
    """The single exact-title match, or None when absent; ambiguity fails closed.

    Every resolver shares this 0/1/many contract: a unique title resolves, a
    missing title is the caller's decision (create it, or fail with the
    caller's remedy), and several same-titled objects are never disambiguated
    by guess — the caller must pass the id.
    """

    if len(matches) > 1:
        raise EncordToolError(
            f"Multiple Encord {kind}s titled {value!r}; pass the {id_hint} instead."
        )
    return matches[0] if matches else None


def resolve_integration(user_client: Any, value: str) -> ResolvedRef:
    """Resolve an Encord cloud integration by uuid or exact title.

    Integrations are never created here: they hold cloud credentials and must be
    created once in the Encord app (S3-compatible/MinIO pattern for Nebius).
    """

    value = _require_reference(value, "--integration")
    integrations = list(user_client.get_cloud_integrations())
    if looks_like_id(value):
        matches = [i for i in integrations if str(i.id).lower() == value.lower()]
        if not matches:
            raise EncordToolError(
                f"No Encord cloud integration with id {value!r}. Available titles: "
                f"{sorted(str(i.title) for i in integrations)}"
            )
    else:
        matches = [i for i in integrations if str(i.title) == value]
    found = _unique_title_match(
        "cloud integration", value, matches, id_hint="integration id"
    )
    if found is None:
        raise EncordToolError(
            f"No Encord cloud integration titled {value!r}. Available: "
            f"{sorted(str(i.title) for i in integrations)}. Create an "
            "S3-compatible integration in the Encord app first."
        )
    return ResolvedRef(found, str(found.id), str(found.title))


def resolve_folder(user_client: Any, value: str, *, create: bool = True) -> ResolvedRef:
    """Resolve a storage folder by uuid or exact title; create the title if asked."""

    value = _require_reference(value, "--folder")
    if looks_like_id(value):
        folder = user_client.get_storage_folder(value)
        return ResolvedRef(folder, str(folder.uuid), str(folder.name))
    matches = [
        folder
        for folder in user_client.list_storage_folders(search=value, page_size=1000)
        if str(folder.name) == value
    ]
    found = _unique_title_match("storage folder", value, matches, id_hint="folder uuid")
    if found is not None:
        return ResolvedRef(found, str(found.uuid), str(found.name))
    if not create:
        raise EncordToolError(f"No Encord storage folder named {value!r}.")
    folder = user_client.create_storage_folder(
        value, description="Created by npa workbench encord push"
    )
    return ResolvedRef(folder, str(folder.uuid), str(folder.name), created=True)


def resolve_dataset(user_client: Any, value: str, *, create: bool = True) -> ResolvedRef:
    """Resolve a dataset by hash or exact title; create the title if asked."""

    value = _require_reference(value, "Dataset reference")
    if looks_like_id(value):
        dataset = user_client.get_dataset(value)
        return ResolvedRef(dataset, value, str(getattr(dataset, "title", "")))
    rows = list(user_client.get_datasets(title_eq=value))
    found = _unique_title_match("dataset", value, rows, id_hint="dataset hash")
    if found is not None:
        dataset_hash = str(found["dataset"].dataset_hash)
        return ResolvedRef(user_client.get_dataset(dataset_hash), dataset_hash, value)
    if not create:
        raise EncordToolError(f"No Encord dataset titled {value!r}.")
    response = user_client.create_dataset(
        value,
        CORD_STORAGE_LOCATION,
        dataset_description="Created by npa workbench encord push",
        create_backing_folder=False,
    )
    dataset_hash = str(response["dataset_hash"])
    return ResolvedRef(
        user_client.get_dataset(dataset_hash), dataset_hash, value, created=True
    )


def resolve_project(user_client: Any, value: str) -> ResolvedRef:
    """Resolve a project by hash or exact title (never created)."""

    value = _require_reference(value, "Project reference")
    if looks_like_id(value):
        project = user_client.get_project(value)
        return ResolvedRef(project, value, str(getattr(project, "title", "")))
    rows = list(user_client.get_projects(title_eq=value))
    found = _unique_title_match("project", value, rows, id_hint="project hash")
    if found is None:
        raise EncordToolError(f"No Encord project titled {value!r}.")
    project_hash = str(found["project"].project_hash)
    return ResolvedRef(user_client.get_project(project_hash), project_hash, value)


def resolve_collection(
    user_client: Any,
    value: str,
    *,
    create_in_folder_uuid: str = "",
) -> ResolvedRef:
    """Resolve a collection by uuid or exact name.

    Index collections are scoped to a top-level storage folder, so a non-empty
    ``create_in_folder_uuid`` (curate's collection-by-title path) both scopes
    the title search to that folder server-side and is where a missing title is
    created; without it a missing title is an error (pull's path).
    """

    value = _require_reference(value, "Collection reference")
    if looks_like_id(value):
        collection = user_client.get_collection(value)
        return ResolvedRef(collection, value, str(getattr(collection, "name", "")))
    scope = (
        {"top_level_folder_uuid": create_in_folder_uuid}
        if create_in_folder_uuid
        else {}
    )
    matches = [
        collection
        for collection in user_client.list_collections(**scope)
        if str(collection.name) == value
    ]
    found = _unique_title_match("collection", value, matches, id_hint="collection uuid")
    if found is not None:
        return ResolvedRef(found, str(found.uuid), value)
    if not create_in_folder_uuid:
        raise EncordToolError(f"No Encord collection named {value!r}.")
    collection = user_client.create_collection(
        top_level_folder_uuid=create_in_folder_uuid,
        name=value,
        description="Created by npa workbench encord curate",
    )
    return ResolvedRef(collection, str(collection.uuid), value, created=True)
