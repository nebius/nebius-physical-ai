"""Versioned exact-project credential and identity store.

Top-level ``storage``/``storage_iam``/``nebius`` keys are compatibility views
only. They are regenerated from the explicitly selected project and are never
used as authoritative multi-project state.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "npa.project-credentials.v2"


class ProjectCredentialStoreError(RuntimeError):
    pass


class AmbiguousLegacyCredentialError(ProjectCredentialStoreError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _path(path: Path | None) -> Path:
    if path is not None:
        return path
    from npa.clients.credentials import CREDENTIALS_PATH

    return CREDENTIALS_PATH


def _legacy_owner(document: Mapping[str, Any]) -> str:
    iam = document.get("storage_iam")
    if isinstance(iam, Mapping):
        owner = str(iam.get("service_account_project_id") or iam.get("project_id") or "").strip()
        if owner:
            return owner
    nebius = document.get("nebius")
    if isinstance(nebius, Mapping) and str(
        nebius.get("service_account_managed_by") or ""
    ) in {"npa", "npa-recovery-attested"}:
        owner = str(
            nebius.get("service_account_project_id") or nebius.get("project_id") or ""
        ).strip()
        if owner:
            return owner
    storage = document.get("storage")
    storage = storage if isinstance(storage, Mapping) else {}
    bucket = str(storage.get("bucket") or storage.get("s3_bucket") or "").removeprefix("s3://").strip("/")
    setup = document.get("storage_setup")
    projects = setup.get("projects") if isinstance(setup, Mapping) else None
    matches: list[str] = []
    if bucket and isinstance(projects, Mapping):
        for project_id, record in projects.items():
            if not isinstance(record, Mapping) or record.get("status") != "complete":
                continue
            record_bucket = str(record.get("bucket_name") or "").strip()
            resources = record.get("resources")
            if not record_bucket and isinstance(resources, Mapping):
                item = resources.get("bucket")
                if isinstance(item, Mapping):
                    record_bucket = str(item.get("name") or "").strip()
            if record_bucket == bucket:
                matches.append(str(project_id))
    return matches[0] if len(matches) == 1 else ""


def _root(document: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    value = document.get("project_credentials")
    if value is None:
        return {"schema_version": SCHEMA_VERSION, "projects": {}}, {}
    if not isinstance(value, Mapping):
        raise ProjectCredentialStoreError("project credential store is not a mapping")
    schema = value.get("schema_version")
    if schema != SCHEMA_VERSION:
        raise ProjectCredentialStoreError(f"unsupported project credential schema {schema!r}")
    projects = value.get("projects")
    if not isinstance(projects, Mapping):
        raise ProjectCredentialStoreError("project credential store projects must be a mapping")
    return deepcopy(dict(value)), deepcopy(dict(projects))


def _migrate_legacy(
    document: dict[str, Any], root: dict[str, Any], projects: dict[str, Any], project_id: str
) -> None:
    legacy_fields = {
        key: deepcopy(document[key])
        for key in ("storage", "storage_iam", "nebius")
        if isinstance(document.get(key), Mapping) and document.get(key)
    }
    if not legacy_fields:
        return
    owner = _legacy_owner(document)
    if not owner:
        raise AmbiguousLegacyCredentialError(
            "legacy global storage credentials have no unique exact-project ownership; "
            "they were preserved unchanged and must be reconciled explicitly"
        )
    if owner != project_id:
        raise AmbiguousLegacyCredentialError(
            f"legacy global storage credentials belong to exact project {owner}, not {project_id}; "
            "no credentials were copied"
        )
    saved_record = projects.get(project_id)
    record: dict[str, Any] = (
        deepcopy(dict(saved_record)) if isinstance(saved_record, Mapping) else {}
    )
    for key, value in legacy_fields.items():
        record.setdefault(key, value)
    record.setdefault("migrated_from", "legacy-global")
    record["updated_at"] = _now()
    projects[project_id] = record
    root["projects"] = projects


def _compatibility_views(document: dict[str, Any], root: Mapping[str, Any]) -> None:
    for key in ("storage", "storage_iam", "nebius"):
        document.pop(key, None)
    current = str(root.get("current_project_id") or "").strip()
    projects = root.get("projects")
    selected = projects.get(current) if current and isinstance(projects, Mapping) else None
    if isinstance(selected, Mapping):
        for key in ("storage", "storage_iam", "nebius"):
            if key in {"storage", "storage_iam"} and selected.get(
                "storage_selected"
            ) is False:
                continue
            value = selected.get(key)
            if isinstance(value, Mapping) and value:
                compatible = deepcopy(dict(value))
                if key == "storage_iam":
                    compatible = {
                        field: compatible[field]
                        for field in (
                            "service_account_id",
                            "service_account_name",
                            "service_account_project_id",
                            "service_account_managed_by",
                        )
                        if compatible.get(field)
                    }
                if compatible:
                    document[key] = compatible


def project_credential_record(
    project_id: str,
    *,
    alias: str = "",
    path: Path | None = None,
    migrate_legacy: bool = True,
) -> dict[str, Any]:
    """Resolve one exact project and safely migrate provable legacy state."""

    from npa.clients.credentials import update_private_yaml

    exact = str(project_id or "").strip()
    if not exact:
        raise ProjectCredentialStoreError("exact project ID is required")
    result: dict[str, Any] = {}

    def update(document: dict[str, Any]) -> dict[str, Any]:
        nonlocal result
        root, projects = _root(document)
        if exact not in projects and not projects and migrate_legacy:
            _migrate_legacy(document, root, projects, exact)
        record = projects.get(exact)
        result = deepcopy(dict(record)) if isinstance(record, Mapping) else {}
        if result and alias:
            aliases = sorted({*(str(item) for item in result.get("aliases", []) if item), alias})
            result["aliases"] = aliases
            result["project_id"] = exact
            result["updated_at"] = _now()
            projects[exact] = deepcopy(result)
            root["projects"] = projects
        document["project_credentials"] = root
        _compatibility_views(document, root)
        return document

    target = _path(path)
    if not migrate_legacy and not alias:
        if not target.exists():
            return {}
        from npa.clients.credentials import _read_credentials_document

        document = _read_credentials_document(target)
        _saved_root, saved_projects = _root(document)
        saved = saved_projects.get(exact)
        return deepcopy(dict(saved)) if isinstance(saved, Mapping) else {}
    if target.exists():
        update_private_yaml(target, update)
    return result


def write_project_credentials(
    project_id: str,
    payload: Mapping[str, Any],
    *,
    alias: str = "",
    path: Path | None = None,
    select: bool = True,
) -> Path:
    """Atomically merge credentials into one exact-project record."""

    from npa.clients.credentials import _prune_empty, update_private_yaml

    exact = str(project_id or "").strip()
    if not exact:
        raise ProjectCredentialStoreError("exact project ID is required")
    clean = _prune_empty(dict(payload))
    target = _path(path)

    def update(document: dict[str, Any]) -> dict[str, Any]:
        return merge_project_credentials_document(
            document, exact, clean, alias=alias, select=select
        )

    update_private_yaml(target, update)
    return target


def select_project_credentials(
    project_id: str,
    *,
    alias: str = "",
    path: Path | None = None,
    select_storage: bool,
) -> Path:
    """Select one exact project without adopting unrelated legacy storage.

    A project-only configure preserves ambiguous legacy fields under a
    non-authoritative quarantine record, then clears the top-level compatibility
    view. This keeps recovery evidence without making an old bucket look selected
    for a genuinely fresh configuration.
    """

    from npa.clients.credentials import update_private_yaml

    exact = str(project_id or "").strip()
    if not exact:
        raise ProjectCredentialStoreError("exact project ID is required")
    clean_alias = str(alias or "").strip()
    target = _path(path)

    def update(document: dict[str, Any]) -> dict[str, Any]:
        root, projects = _root(document)
        legacy_fields = {
            key: deepcopy(document[key])
            for key in ("storage", "storage_iam", "nebius")
            if isinstance(document.get(key), Mapping) and document.get(key)
        }
        if legacy_fields and not projects:
            owner = _legacy_owner(document)
            if owner:
                _migrate_legacy(document, root, projects, owner)
            else:
                root["legacy_unscoped"] = {
                    **legacy_fields,
                    "quarantined_at": _now(),
                    "reason": "ambiguous legacy project ownership",
                }

        saved = projects.get(exact)
        record = deepcopy(dict(saved)) if isinstance(saved, Mapping) else {}
        record["project_id"] = exact
        aliases = {str(item) for item in record.get("aliases", []) if item}
        if clean_alias:
            aliases.add(clean_alias)
        if aliases:
            record["aliases"] = sorted(aliases)
        record["storage_selected"] = bool(select_storage)
        record["updated_at"] = _now()
        projects[exact] = record
        root["projects"] = projects
        root["schema_version"] = SCHEMA_VERSION
        root["current_project_id"] = exact
        document["project_credentials"] = root
        _compatibility_views(document, root)
        return document

    update_private_yaml(target, update)
    return target


def persist_agent_service_account_id(project_id: str, service_account_id: str) -> None:
    """Persist one agent principal under its exact project identity."""

    from npa.clients.nebius import _saved_service_account_id

    exact = str(project_id or "").strip()
    account = str(service_account_id or "").strip()
    if not exact or not account or _saved_service_account_id(exact) == account:
        return
    write_project_credentials(
        exact,
        {
            "nebius": {
                "service_account_id": account,
                "service_account_project_id": exact,
            }
        },
    )


def persist_agent_terraform_credentials(
    project_id: str,
    *,
    alias: str,
    bucket: str,
    endpoint: str,
    access_key: str,
    secret_key: str,
) -> None:
    """Persist secret Terraform backend fields outside public project config."""

    write_project_credentials(
        project_id,
        {
            "terraform_state": {
                "bucket": bucket,
                "endpoint": endpoint,
                "access_key": access_key,
                "secret_key": secret_key,
            }
        },
        alias=alias,
    )


def merge_project_credentials_document(
    document: dict[str, Any],
    project_id: str,
    payload: Mapping[str, Any],
    *,
    alias: str = "",
    select: bool = True,
) -> dict[str, Any]:
    """Pure locked-updater form used by larger atomic lifecycle commits."""

    from npa.clients.credentials import _deep_merge, _prune_empty

    exact = str(project_id or "").strip()
    root, projects = _root(document)
    if exact not in projects and not projects:
        _migrate_legacy(document, root, projects, exact)
    saved_existing = projects.get(exact)
    existing: dict[str, Any] = (
        deepcopy(dict(saved_existing)) if isinstance(saved_existing, Mapping) else {}
    )
    incoming = _prune_empty(dict(payload))
    existing_iam = existing.get("storage_iam")
    incoming_iam = incoming.get("storage_iam")
    if isinstance(existing_iam, Mapping) and isinstance(incoming_iam, dict):
        generations: list[dict[str, Any]] = []
        seen: set[tuple[str, tuple[str, ...]]] = set()
        for source in (existing_iam.get("generations"), incoming_iam.get("generations")):
            if not isinstance(source, list):
                continue
            for item in source:
                if not isinstance(item, Mapping):
                    continue
                marker = (
                    str(item.get("service_account_id") or ""),
                    tuple(sorted(str(value) for value in item.get("access_key_ids", []) if value)),
                )
                if marker not in seen:
                    generations.append(deepcopy(dict(item)))
                    seen.add(marker)
        incoming_iam["generations"] = generations
    merged = _deep_merge(existing, incoming)
    if select and isinstance(incoming.get("storage"), Mapping):
        merged["storage_selected"] = True
    merged["project_id"] = exact
    aliases = {str(item) for item in merged.get("aliases", []) if item}
    if alias:
        aliases.add(alias)
    if aliases:
        merged["aliases"] = sorted(aliases)
    merged["updated_at"] = _now()
    projects[exact] = merged
    root["projects"] = projects
    root["schema_version"] = SCHEMA_VERSION
    if select:
        root["current_project_id"] = exact
    document["project_credentials"] = root
    _compatibility_views(document, root)
    return document


def forget_project_credentials(project_id: str, *, path: Path | None = None) -> bool:
    """Prune only one exact project's operational credentials and secret views."""

    from npa.clients.credentials import update_private_yaml

    exact = str(project_id or "").strip()
    removed = False

    def update(document: dict[str, Any]) -> dict[str, Any]:
        nonlocal removed
        root, projects = _root(document)
        removed = exact in projects
        projects.pop(exact, None)
        root["projects"] = projects
        if root.get("current_project_id") == exact:
            root.pop("current_project_id", None)
        document["project_credentials"] = root
        _compatibility_views(document, root)
        return document

    update_private_yaml(_path(path), update)
    return removed


def project_credential_residue(
    project_id: str, *, path: Path | None = None
) -> list[dict[str, str]]:
    """Return secret-free field paths/classes for actionable local residue."""

    target = _path(path)
    if not target.exists():
        return []
    from npa.clients.credentials import _read_credentials_document

    document = _read_credentials_document(target)
    root, projects = _root(document)
    del root
    record = projects.get(str(project_id or "").strip())
    if not isinstance(record, Mapping):
        return []
    residue: list[dict[str, str]] = []
    for section, fields in (
        ("storage", ("aws_access_key_id", "aws_secret_access_key")),
        ("terraform_state", ("access_key", "secret_key", "session_token")),
    ):
        value = record.get(section)
        if not isinstance(value, Mapping):
            continue
        for field in fields:
            if value.get(field) not in (None, ""):
                residue.append(
                    {
                        "path": f"project_credentials.projects.{project_id}.{section}.{field}",
                        "class": "live_project_credential",
                    }
                )
    if record.get("storage_iam"):
        residue.append(
            {
                "path": f"project_credentials.projects.{project_id}.storage_iam",
                "class": "operational_identity",
            }
        )
    return residue
