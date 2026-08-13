"""`npa storage` — object-storage teardown for resources npa created.

`npa configure` provisions a bucket (and sometimes a service account + access
key), so teardown needs an ownership-aware inverse. Deleting a versioned bucket
immediately is its own trap: non-current versions can remain and the API answers
``BucketNotEmpty``. Scheduling the purge (``--ttl``) is what actually works.
Storage IAM uses exact provider identity/scope checks, durable non-secret residue
markers, and explicit operator-attested recovery for legacy NPA-created accounts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NoReturn, TypedDict

import typer

from npa.lifecycle_intent import OperationIntent, intent_boundary, json_stdout_contract

app = typer.Typer(
    name="storage",
    help="Inspect and tear down npa-managed object storage.",
    no_args_is_help=True,
)

bucket_app = typer.Typer(
    name="bucket", help="Object-storage buckets.", no_args_is_help=True
)
app.add_typer(bucket_app, name="bucket")

service_account_app = typer.Typer(
    name="service-account",
    help="NPA-owned object-storage service accounts.",
    no_args_is_help=True,
)
app.add_typer(service_account_app, name="service-account")

DEFAULT_PURGE_TTL = "1m"


@dataclass(frozen=True)
class _OwnedStorageServiceAccount:
    account_id: str
    name: str
    project_id: str
    source: str
    access_key_ids: tuple[str, ...] = ()
    group_id: str = ""
    access_permit_id: str = ""
    binding_managed_by_npa: bool = False


class _BucketRow(TypedDict):
    name: str
    id: str
    configured: bool


@dataclass(frozen=True)
class _StorageIamContext:
    alias: str
    project_id: str
    tenant_id: str
    profile: str
    account_id: str = ""
    identity_source: str = "live_configuration"
    receipt_id: str = ""


@dataclass(frozen=True)
class _StorageIamObservation:
    outcome: str
    context: _StorageIamContext
    account_id: str = ""
    account_name: str = ""
    ownership: str = "unverified"
    ownership_note: str = ""
    detail: str = ""

    @property
    def present(self) -> bool:
        return self.outcome == "present"

    @property
    def verified_absent(self) -> bool:
        return self.outcome == "verified_absent"


@bucket_app.command("list")
@intent_boundary(OperationIntent.OBSERVE)
def list_buckets_cmd(
    project: str = typer.Option(
        "", "--project", help="NPA project alias to list buckets for."
    ),
    project_id: str = typer.Option(
        "", "--project-id", help="Nebius project id to list buckets for."
    ),
    output_json: bool = typer.Option(
        False, "--json", help="Emit JSON instead of a table."
    ),
) -> None:
    """List the object-storage buckets in a project, marking the configured one."""
    import json

    from npa.clients.config import resolve_environment
    from npa.clients.credentials import load_credentials
    from npa.clients.nebius import NebiusError, _list_project_buckets

    resolved_project = project_id.strip()
    if not resolved_project:
        saved = resolve_environment(project or None)
        resolved_project = str(getattr(saved, "project_id", "") or "")
    if not resolved_project:
        raise typer.BadParameter(
            "Cannot tell which Nebius project to list. Pass --project-id <id> or "
            "--project <alias> (after `npa configure`)."
        )
    try:
        items = _list_project_buckets(resolved_project)
    except NebiusError as exc:
        raise typer.BadParameter(
            f"Could not list buckets in {resolved_project}: {exc}"
        ) from exc

    configured = ""
    try:
        configured = _bucket_name_from_uri(
            str(load_credentials(environ={}).s3_bucket or "")
        )
    except Exception:  # noqa: BLE001 - listing works without readable credentials
        configured = ""

    rows: list[_BucketRow] = [
        {
            "name": str((item.get("metadata") or {}).get("name", "") or ""),
            "id": str((item.get("metadata") or {}).get("id", "") or ""),
            "configured": str((item.get("metadata") or {}).get("name", "") or "")
            == configured,
        }
        for item in items
    ]
    if output_json:
        typer.echo(json.dumps(rows, indent=2, sort_keys=True))
        return
    if not rows:
        typer.echo(f"No buckets in project {resolved_project}.")
        return
    width = max(len("NAME"), *(len(row["name"]) for row in rows))
    typer.echo(f"{'NAME'.ljust(width)}  ID")
    typer.echo(f"{'-' * width}  {'-' * 24}")
    for row in rows:
        marker = "  <- configured in ~/.npa" if row["configured"] else ""
        typer.echo(f"{row['name'].ljust(width)}  {row['id']}{marker}")
    typer.echo("")
    typer.echo("Delete one with `npa storage bucket delete --name <bucket>`.")


def _storage_service_account_record(
    *, account_id: str = "", project_id: str = "", receipt_id: str = ""
) -> tuple[_OwnedStorageServiceAccount | None, str]:
    """Read the persisted ownership proof for NPA-created storage IAM.

    A legacy ``nebius.service_account_id`` is not enough: configure has always
    accepted user-managed accounts, and deleting one based on its familiar name
    would silently destroy an unrelated identity. Only the provenance written
    at the successful create call is authoritative.
    """

    import yaml

    from npa.clients.credentials import CREDENTIALS_PATH
    from npa.clients.nebius import DEFAULT_SERVICE_ACCOUNT_NAME

    try:
        data = (
            yaml.safe_load(CREDENTIALS_PATH.read_text(encoding="utf-8")) or {}
            if CREDENTIALS_PATH.exists()
            else {}
        )
    except (OSError, yaml.YAMLError):
        data = {}
    if not isinstance(data, dict):
        data = {}

    candidates: list[_OwnedStorageServiceAccount] = []

    def _candidate(record_data: object, source: str) -> None:
        if not isinstance(record_data, dict):
            return
        account_id = str(record_data.get("service_account_id", "") or "").strip()
        managed_by = str(
            record_data.get("service_account_managed_by", "") or ""
        ).strip()
        name = str(record_data.get("service_account_name", "") or "").strip()
        project_id = str(
            record_data.get("service_account_project_id", "") or ""
        ).strip()
        recovery = record_data.get("recovery")
        recovery_matches = bool(
            managed_by == "npa-recovery-attested"
            and isinstance(recovery, dict)
            and recovery.get("schema_version") == "npa.storage-iam-recovery.v1"
            and str(recovery.get("service_account_id", "") or "").strip() == account_id
            and str(recovery.get("project_id", "") or "").strip() == project_id
            and str(recovery.get("reason", "") or "").strip()
            and str(recovery.get("attested_at", "") or "").strip()
            and str(recovery.get("attested_by", "") or "").strip()
            and recovery.get("provider_verified") is True
        )
        explicitly_created = managed_by == "npa" and bool(name)
        recovery_default = recovery_matches and name == DEFAULT_SERVICE_ACCOUNT_NAME
        if (
            managed_by in {"npa", "npa-recovery-attested"}
            and (explicitly_created or recovery_default)
            and account_id
            and project_id
        ):
            binding = record_data.get("binding")
            binding = binding if isinstance(binding, dict) else {}
            candidates.append(
                _OwnedStorageServiceAccount(
                    account_id,
                    name,
                    project_id,
                    source,
                    tuple(
                        sorted(
                            str(item).strip()
                            for item in record_data.get("iam_key_ids", [])
                            if str(item).strip()
                        )
                    )
                    if isinstance(record_data.get("iam_key_ids"), list)
                    else (),
                    str(binding.get("group_id") or "").strip(),
                    str(binding.get("access_permit_id") or "").strip(),
                    bool(
                        binding.get("group_managed_by") == "npa"
                        and binding.get("access_permit_managed_by") == "npa"
                    ),
                )
            )

    # A complete dedicated or legacy proof remains readable across upgrades.
    project_store = data.get("project_credentials")
    stored_projects = (
        project_store.get("projects") if isinstance(project_store, dict) else None
    )
    if isinstance(stored_projects, dict):
        selected_records = (
            {project_id: stored_projects.get(project_id)}
            if project_id
            else stored_projects
        )
        for exact_project, project_record in selected_records.items():
            if not isinstance(project_record, dict):
                continue
            iam_record = project_record.get("storage_iam")
            if not isinstance(iam_record, dict):
                continue
            generations = iam_record.get("generations")
            generation_records = (
                generations if isinstance(generations, list) else [iam_record]
            )
            for generation in generation_records:
                if not isinstance(generation, dict):
                    continue
                normalized = dict(generation)
                normalized.setdefault("service_account_project_id", exact_project)
                if (
                    not normalized.get("service_account_managed_by")
                    and normalized.get("ownership") == "npa"
                ):
                    normalized["service_account_managed_by"] = "npa"
                _candidate(normalized, f"project credential store for {exact_project}")

    storage_iam = data.get("storage_iam")
    _candidate(storage_iam, "storage_iam")
    nebius = data.get("nebius")
    _candidate(nebius, "legacy nebius record")

    # Failed/interrupted setup journals the successful create response before
    # the next provider step. This is equally strong ownership proof and remains
    # usable even though final credentials/storage_iam were never committed.
    setup = data.get("storage_setup")
    projects = setup.get("projects") if isinstance(setup, dict) else None
    if isinstance(projects, dict):
        for journal_project, project_record in projects.items():
            resources = (
                project_record.get("resources")
                if isinstance(project_record, dict)
                else None
            )
            account = (
                resources.get("service_account")
                if isinstance(resources, dict)
                else None
            )
            if not isinstance(account, dict):
                continue
            if (
                str(account.get("project_id", "") or "").strip() != str(journal_project)
                or not str(account.get("attempt_id", "") or "").strip()
            ):
                continue
            _candidate(
                {
                    "service_account_id": account.get("id"),
                    "service_account_name": account.get("name"),
                    "service_account_project_id": account.get("project_id"),
                    "service_account_managed_by": account.get("created_by"),
                },
                f"storage setup journal for {journal_project}",
            )

    if receipt_id:
        try:
            from npa.teardown_receipts import load_teardown_receipt

            receipt_payload = load_teardown_receipt(receipt_id)
            receipt_identity = receipt_payload.get("identity")
            receipt_identity = (
                receipt_identity if isinstance(receipt_identity, dict) else {}
            )
            receipt_storage = receipt_identity.get("storage_iam")
            if isinstance(receipt_storage, dict):
                generations = receipt_storage.get("generations")
                records = (
                    generations if isinstance(generations, list) else [receipt_storage]
                )
                for generation in records:
                    if not isinstance(generation, dict):
                        continue
                    _candidate(
                        {
                            "service_account_id": generation.get("service_account_id"),
                            "service_account_name": generation.get(
                                "service_account_name"
                            )
                            or DEFAULT_SERVICE_ACCOUNT_NAME,
                            "service_account_project_id": generation.get("project_id")
                            or receipt_payload.get("project_id"),
                            "service_account_managed_by": (
                                "npa" if generation.get("ownership") == "npa" else ""
                            ),
                            "iam_key_ids": generation.get("iam_key_ids") or [],
                        },
                        f"teardown receipt {receipt_id}",
                    )
        except (OSError, RuntimeError, ValueError):
            # The identity resolver reports malformed/missing selected receipts.
            # A secondary ownership source must never turn that into permission.
            pass

    selected_id = str(account_id or "").strip()
    selected_project = str(project_id or "").strip()
    if selected_id:
        exact_id_candidates = [
            item for item in candidates if item.account_id == selected_id
        ]
        wrong_projects = sorted(
            {
                item.project_id
                for item in exact_id_candidates
                if item.project_id != selected_project
            }
        )
        if selected_project and wrong_projects:
            return None, (
                f"NPA ownership evidence for {selected_id} names project(s) "
                f"{', '.join(wrong_projects)}, not selected project {selected_project}."
            )
        candidates = [
            item
            for item in exact_id_candidates
            if not selected_project or item.project_id == selected_project
        ]
        if not candidates:
            return None, (
                f"No trustworthy NPA storage IAM ownership record proves exact "
                f"service account {selected_id} in project {selected_project or '<unknown>'}."
            )

    unique = {(item.account_id, item.project_id) for item in candidates}
    if len(unique) == 1:
        preferred = next(
            (item for item in candidates if item.binding_managed_by_npa),
            next(
                (
                    item
                    for item in candidates
                    if item.source.startswith("project credential store")
                ),
                next(
                    (item for item in candidates if item.source == "storage_iam"),
                    candidates[0],
                ),
            ),
        )
        return preferred, ""
    if len(unique) > 1:
        return None, (
            "Conflicting NPA storage IAM ownership records were found; refusing "
            "to delete any IAM identity until the local provenance is reconciled."
        )

    legacy_id = ""
    if isinstance(nebius, dict):
        legacy_id = str(nebius.get("service_account_id", "") or "").strip()
    suffix = (
        f" The saved ID {legacy_id} is evidence, but is not proof of ownership "
        "and was left untouched."
        if legacy_id
        else ""
    )
    return None, f"No trustworthy NPA storage IAM ownership record is present.{suffix}"


def _remove_storage_service_account_record(account_id: str) -> bool:
    """Atomically prune only one deleted IAM generation from local truth."""

    from npa.clients.credentials import CREDENTIALS_PATH, update_private_yaml
    from npa.clients.project_credential_store import (
        _compatibility_views,
        _legacy_owner,
        _migrate_legacy,
        _root,
    )

    if not CREDENTIALS_PATH.exists():
        return True
    removed = False

    def update(data: dict[str, Any]) -> dict[str, Any] | None:
        nonlocal removed
        root, projects = _root(data)
        if not projects and _legacy_owner(data):
            legacy_project = _legacy_owner(data)
            _migrate_legacy(data, root, projects, legacy_project)
            root["current_project_id"] = legacy_project
        for exact_project, project_record in list(projects.items()):
            if not isinstance(project_record, dict):
                continue
            project_iam = project_record.get("storage_iam")
            if not isinstance(project_iam, dict):
                legacy_nebius = project_record.get("nebius")
                if (
                    isinstance(legacy_nebius, dict)
                    and str(legacy_nebius.get("service_account_id") or "").strip()
                    == account_id
                    and str(legacy_nebius.get("service_account_managed_by") or "")
                    in {"npa", "npa-recovery-attested"}
                ):
                    project_record.pop("nebius", None)
                    removed = True
                    if any(
                        project_record.get(section)
                        for section in (
                            "storage",
                            "storage_iam",
                            "terraform_state",
                            "nebius",
                        )
                    ):
                        projects[exact_project] = project_record
                    else:
                        projects.pop(exact_project, None)
                continue
            generations = project_iam.get("generations")
            records = generations if isinstance(generations, list) else [project_iam]
            retained = [
                item
                for item in records
                if not isinstance(item, dict)
                or str(item.get("service_account_id") or "").strip() != account_id
            ]
            if len(retained) == len(records):
                continue
            removed = True
            if retained:
                project_iam["generations"] = retained
                if (
                    str(project_iam.get("active_service_account_id") or "").strip()
                    == account_id
                ):
                    project_iam["active_service_account_id"] = str(
                        retained[-1].get("service_account_id") or ""
                    )
                project_record["storage_iam"] = project_iam
            else:
                project_record.pop("storage_iam", None)
            nebius = project_record.get("nebius")
            if (
                isinstance(nebius, dict)
                and str(nebius.get("service_account_id") or "").strip() == account_id
            ):
                project_record.pop("nebius", None)
            if any(
                project_record.get(section)
                for section in ("storage", "storage_iam", "terraform_state", "nebius")
            ):
                projects[exact_project] = project_record
            else:
                projects.pop(exact_project, None)
        root["projects"] = projects
        data["project_credentials"] = root

        setup = data.get("storage_setup")
        setup_projects = setup.get("projects") if isinstance(setup, dict) else None
        if isinstance(setup, dict) and isinstance(setup_projects, dict):
            for exact_project, project_record in list(setup_projects.items()):
                resources = (
                    project_record.get("resources")
                    if isinstance(project_record, dict)
                    else None
                )
                account = (
                    resources.get("service_account")
                    if isinstance(resources, dict)
                    else None
                )
                if not (
                    isinstance(account, dict)
                    and str(account.get("id") or "").strip() == account_id
                    and account.get("created_by") == "npa"
                    and str(account.get("project_id") or "").strip() == exact_project
                ):
                    continue
                assert isinstance(resources, dict)
                updated_resources = dict(resources)
                updated_resources.pop("service_account", None)
                updated_resources.pop("access_keys", None)
                if updated_resources:
                    project_record["resources"] = updated_resources
                else:
                    setup_projects.pop(exact_project, None)
                removed = True
            if setup_projects:
                setup["projects"] = setup_projects
                data["storage_setup"] = setup
            else:
                data.pop("storage_setup", None)
        _compatibility_views(data, root)
        if (
            not root.get("projects")
            and not data.get("storage_setup")
            and set(data) <= {"project_credentials"}
        ):
            return None
        return data

    try:
        update_private_yaml(CREDENTIALS_PATH, update)
    except (OSError, RuntimeError, ValueError):
        return False
    return removed


def _untrusted_storage_account_ids() -> set[str]:
    """Return exact saved IDs that are evidence, but not ownership proof."""

    import yaml

    from npa.clients.credentials import CREDENTIALS_PATH

    try:
        data = yaml.safe_load(CREDENTIALS_PATH.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return set()
    if not isinstance(data, dict):
        return set()
    ids: set[str] = set()
    for section_name in ("storage_iam", "nebius"):
        section = data.get(section_name)
        if isinstance(section, dict):
            account_id = str(section.get("service_account_id", "") or "").strip()
            if account_id:
                ids.add(account_id)
    return ids


def _resolve_storage_iam_context(
    project: str = "",
    project_id: str = "",
    *,
    tenant_id: str = "",
    profile: str = "",
    receipt: str = "",
    service_account_id: str = "",
) -> _StorageIamContext:
    """Resolve one stable alias/project/tenant/profile context for IAM calls."""

    from npa.cleanup_identity import resolve_cleanup_identity
    from npa.clients.config import ConfigError, resolve_environment
    from npa.clients.nebius_auth import nebius_profile

    alias = str(project or "").strip()
    explicit_project = str(project_id or "").strip()
    environment = None
    if alias:
        environment = resolve_environment(alias)
    elif not (explicit_project or receipt):
        environment = resolve_environment(None)
        from npa.clients.config import default_project_name

        alias = str(default_project_name() or "").strip()
    identity = resolve_cleanup_identity(
        explicit={
            "project_alias": alias,
            "project_id": explicit_project,
            "tenant_id": tenant_id,
            "profile": profile,
            "service_account_id": service_account_id,
        },
        receipt_id=receipt,
        live={
            "project_alias": alias,
            "project_id": str(getattr(environment, "project_id", "") or ""),
            "tenant_id": str(getattr(environment, "tenant_id", "") or ""),
            # An ambient active profile is not project configuration and must
            # not contradict a durable receipt after the alias was removed.
            # It remains a fallback only when no receipt selects the identity.
            "profile": "" if receipt else nebius_profile(),
        },
        phase="storage_iam",
    )
    resolved_project = str(identity.get("project_id") or "").strip()
    if not resolved_project:
        raise ConfigError(
            "A configured project alias or exact --project-id is required for IAM verification."
        )
    alias = str(identity.get("project_alias") or alias).strip()
    return _StorageIamContext(
        alias=alias,
        project_id=resolved_project,
        tenant_id=str(identity.get("tenant_id") or "").strip(),
        profile=str(identity.get("profile") or nebius_profile()).strip(),
        account_id=str(identity.get("service_account_id") or "").strip(),
        identity_source=identity.source,
        receipt_id=identity.receipt_id,
    )


def _observe_storage_iam(
    context: _StorageIamContext,
) -> _StorageIamObservation:
    """Use one strict resolver for dry-run, real deletion, and full cleanup."""

    from npa.clients.config import storage_iam_residue
    from npa.clients.nebius import (
        DEFAULT_SERVICE_ACCOUNT_NAME,
        NebiusError,
        get_service_account_id_by_name,
        get_service_account_identity,
    )

    # Provider-verified deletion is durable teardown truth. After bucket/config
    # removal there may intentionally be no credentials left with which to repeat
    # the get. Reuse only the selected receipt's exact project/account evidence;
    # a stale live alias or a receipt for another key/account cannot satisfy it.
    if context.receipt_id:
        from npa.teardown_receipts import load_teardown_receipt

        receipt = load_teardown_receipt(context.receipt_id)
        identity = receipt.get("identity")
        identity = identity if isinstance(identity, dict) else {}
        saved = identity.get("storage_iam")
        saved = saved if isinstance(saved, dict) else {}
        saved_account = str(saved.get("service_account_id") or "").strip()
        exact_match = bool(
            str(receipt.get("project_id") or "") == context.project_id
            and saved_account
            and (not context.account_id or context.account_id == saved_account)
        )
        terminal = [
            event
            for event in receipt.get("events") or []
            if isinstance(event, dict)
            and event.get("phase") == "storage_iam"
            and str(event.get("terminal_state") or "").lower()
            in {"verified_absent", "verified_deleted", "deleted"}
        ]
        if exact_match and terminal:
            latest = max(terminal, key=lambda item: str(item.get("recorded_at") or ""))
            verification = latest.get("verification")
            verification = verification if isinstance(verification, dict) else {}
            if (
                verification.get("provider_outcome") == "verified_absent"
                or verification.get("exact_service_account_absent") is True
            ):
                return _StorageIamObservation(
                    outcome="verified_absent",
                    context=context,
                    account_id=saved_account,
                    account_name=str(saved.get("service_account_name") or ""),
                    ownership="npa"
                    if saved.get("ownership") == "npa"
                    else "unverified",
                    detail="Durable exact provider verification proves the service account is absent.",
                )

    record, ownership_note = _storage_service_account_record(
        account_id=context.account_id,
        project_id=context.project_id,
        receipt_id=context.receipt_id,
    )
    expected_name = record.name if record is not None else DEFAULT_SERVICE_ACCOUNT_NAME
    if context.account_id and record is None:
        return _StorageIamObservation(
            outcome="verification_failed",
            context=context,
            account_id=context.account_id,
            ownership="unverified",
            ownership_note=ownership_note,
            detail=ownership_note,
        )
    if record is not None and record.project_id != context.project_id:
        return _StorageIamObservation(
            outcome="verification_failed",
            context=context,
            account_id=record.account_id,
            ownership="npa",
            ownership_note=ownership_note,
            detail=(
                f"Recorded ownership belongs to {record.project_id}, which does not "
                f"match project {context.project_id}"
            ),
        )
    candidates = (
        {context.account_id}
        if context.account_id
        else {record.account_id}
        if record is not None
        else set(_untrusted_storage_account_ids())
    )
    marker = storage_iam_residue(context.alias) if context.alias else {}
    if record is None:
        marker_id = str(marker.get("service_account_id", "") or "").strip()
        if marker_id:
            candidates.add(marker_id)
        marker_candidates = marker.get("candidate_service_account_ids")
        if isinstance(marker_candidates, list):
            candidates.update(
                str(item).strip() for item in marker_candidates if str(item).strip()
            )
    if len(candidates) > 1:
        return _StorageIamObservation(
            outcome="verification_failed",
            context=context,
            ownership_note=ownership_note,
            detail=(
                "Conflicting exact service-account IDs are present in local evidence: "
                + ", ".join(sorted(candidates))
            ),
        )

    candidate = next(iter(candidates), "")
    try:
        if not candidate:
            candidate = str(
                get_service_account_id_by_name(
                    context.project_id,
                    expected_name,
                    strict=True,
                    profile=context.profile,
                )
                or ""
            ).strip()
            if not candidate:
                return _StorageIamObservation(
                    outcome="verified_absent",
                    context=context,
                    ownership_note=ownership_note,
                    detail=(
                        "The provider authoritatively returned no exact service account "
                        f"named {expected_name!r} in {context.project_id}."
                    ),
                )
        identity = get_service_account_identity(
            candidate,
            project_id=context.project_id,
            tenant_id=context.tenant_id,
            expected_name=expected_name,
            profile=context.profile,
        )
    except NebiusError as exc:
        return _StorageIamObservation(
            outcome="verification_failed",
            context=context,
            account_id=candidate,
            ownership=(
                "npa"
                if record is not None and record.account_id == candidate
                else "unverified"
            ),
            ownership_note=ownership_note,
            detail=str(exc),
        )
    if identity is None:
        return _StorageIamObservation(
            outcome="verified_absent",
            context=context,
            account_id=candidate,
            ownership=(
                "npa"
                if record is not None and record.account_id == candidate
                else "unverified"
            ),
            ownership_note=ownership_note,
            detail="The provider authoritatively returned NotFound for the exact immutable ID.",
        )
    provider_mismatches: list[str] = []
    if identity.account_id != candidate:
        provider_mismatches.append("service-account ID")
    if identity.name != expected_name:
        provider_mismatches.append("service-account name")
    if identity.project_id != context.project_id:
        provider_mismatches.append("parent project")
    if context.tenant_id and identity.tenant_id != context.tenant_id:
        provider_mismatches.append("parent tenant")
    if provider_mismatches:
        return _StorageIamObservation(
            outcome="verification_failed",
            context=context,
            account_id=candidate,
            account_name=identity.name,
            ownership="npa" if record is not None else "unverified",
            ownership_note=ownership_note,
            detail=(
                "Provider identity did not match the selected cleanup target: "
                + ", ".join(provider_mismatches)
            ),
        )
    return _StorageIamObservation(
        outcome="present",
        context=context,
        account_id=identity.account_id,
        account_name=identity.name,
        ownership=(
            "npa"
            if record is not None and record.account_id == identity.account_id
            else "unverified"
        ),
        ownership_note=ownership_note,
        detail="Exact ID, name, project, tenant, and CLI profile context were verified.",
    )


def _persist_storage_iam_observation(observation: _StorageIamObservation) -> None:
    """Persist unresolved evidence, or clear it after authoritative absence."""

    from datetime import datetime, timezone

    from npa.clients.config import (
        ConfigError,
        clear_storage_iam_residue,
        mark_storage_iam_residue,
    )

    alias = observation.context.alias
    from npa.clients.config import resolve_environment

    alias_is_configured = bool(alias and resolve_environment(alias) is not None)
    if not alias_is_configured:
        from npa.teardown_receipts import record_teardown_event

        record_teardown_event(
            phase="storage_iam",
            resource=observation.account_id or "storage-iam",
            terminal_state=(
                "verified_absent"
                if observation.verified_absent
                else "verified_present"
                if observation.present
                else "verification_failed"
            ),
            project_alias=alias,
            project_id=observation.context.project_id,
            identity={
                "project_id": observation.context.project_id,
                "tenant_id": observation.context.tenant_id,
                "profile": observation.context.profile,
                "service_account_id": observation.account_id,
                "service_account_name": observation.account_name,
                "ownership": observation.ownership,
            },
            precheck={"identity_source": observation.context.identity_source},
            action={"kind": "exact_provider_check", "mutation": False},
            verification={"provider_outcome": observation.outcome},
            errors=[observation.detail]
            if observation.outcome == "verification_failed"
            else [],
        )
        return
    if observation.verified_absent:
        # An immutable NPA creation record is authoritative over an older
        # project residue marker. Collapse that historical marker even when it
        # names a stale candidate; unowned evidence retains the exact-ID guard.
        clear_storage_iam_residue(
            alias,
            account_id="" if observation.ownership == "npa" else observation.account_id,
        )
        return
    status = (
        "present_owned"
        if observation.present and observation.ownership == "npa"
        else "present_unverified_ownership"
        if observation.present
        else "verification_failed"
    )
    evidence = {
        "status": status,
        "project_id": observation.context.project_id,
        "tenant_id": observation.context.tenant_id,
        "profile": observation.context.profile or "active",
        "identity_source": observation.context.identity_source,
        "receipt_id": observation.context.receipt_id,
        "service_account_id": observation.account_id,
        "service_account_name": observation.account_name,
        "ownership": observation.ownership,
        "last_checked_at": datetime.now(timezone.utc).isoformat(),
        "detail": observation.detail,
    }
    try:
        mark_storage_iam_residue(alias, evidence)
    except ConfigError as exc:
        raise ConfigError(
            f"IAM is unresolved and its project residue marker could not be saved: {exc}"
        ) from exc


def _observation_dict(observation: _StorageIamObservation) -> dict[str, object]:
    return {
        "outcome": observation.outcome,
        "project": observation.context.alias,
        "project_id": observation.context.project_id,
        "tenant_id": observation.context.tenant_id,
        "profile": observation.context.profile or "active",
        "service_account_id": observation.account_id,
        "service_account_name": observation.account_name,
        "ownership": observation.ownership,
        "detail": observation.detail,
    }


def _record_storage_iam_teardown(
    observation: _StorageIamObservation,
    *,
    terminal_state: str,
    action: str,
    errors: tuple[str, ...] = (),
    access_key_ids: tuple[str, ...] = (),
) -> None:
    from npa.teardown_receipts import record_teardown_event

    record_teardown_event(
        phase="storage_iam",
        resource=observation.account_id or observation.account_name or "storage-iam",
        terminal_state=terminal_state,
        project_alias=observation.context.alias,
        project_id=observation.context.project_id,
        identity={
            "project_id": observation.context.project_id,
            "tenant_id": observation.context.tenant_id,
            "profile": observation.context.profile,
            "service_account_id": observation.account_id,
            "service_account_name": observation.account_name,
            "ownership": observation.ownership,
            "iam_key_ids": list(access_key_ids),
        },
        precheck={
            "provider_outcome": observation.outcome,
            "ownership": observation.ownership,
            "account_id": observation.account_id,
        },
        action={"kind": action},
        verification={"provider_outcome": observation.outcome},
        errors=errors,
    )


def _receipt_proves_bucket_cleanup(receipt_id: str, project_id: str) -> bool:
    """Require exact durable bucket completion before alias-free IAM deletion."""

    if not receipt_id:
        return False
    from npa.teardown_receipts import load_teardown_receipt

    receipt = load_teardown_receipt(receipt_id)
    if str(receipt.get("project_id") or "") != project_id:
        return False
    terminal = {
        "completed",
        "verified_absent",
        "verified_deleted",
        "deleted",
    }
    return any(
        isinstance(event, dict)
        and str(event.get("phase") or "") in {"bucket", "project_destroy_bucket"}
        and str(event.get("terminal_state") or "").lower() in terminal
        for event in receipt.get("events") or []
    )


def _partial_cleanup(
    message: str,
    *,
    output_json: bool = False,
    payload: dict[str, Any] | None = None,
) -> NoReturn:
    """Report an operator-actionable partial cleanup with stable exit semantics."""

    if output_json:
        import json

        document = dict(payload or {})
        document.update(
            {
                "result": str(document.get("result") or "partial_cleanup"),
                "message": message,
                "mutated": False,
                "verification_unresolved": True,
            }
        )
        typer.echo(json.dumps(document, indent=2, sort_keys=True))
    else:
        typer.echo(f"Partial cleanup: {message}", err=True)
    raise typer.Exit(code=2)


def _begin_bucket_iam_cleanup_tombstone(
    project: str, project_id: str, bucket_name: str
) -> tuple[str, dict[str, object]]:
    """Persist non-secret IAM ownership evidence before bucket credentials vanish."""

    from datetime import datetime, timezone

    import yaml

    from npa.clients.config import (
        ConfigError,
        mark_storage_iam_residue,
        project_alias_for_id,
        storage_iam_residue,
    )
    from npa.clients.credentials import CREDENTIALS_PATH

    alias = str(project or "").strip() or project_alias_for_id(project_id)
    existing = storage_iam_residue(alias) if alias else {}
    try:
        data = (
            yaml.safe_load(CREDENTIALS_PATH.read_text(encoding="utf-8")) or {}
            if CREDENTIALS_PATH.exists()
            else {}
        )
    except (OSError, yaml.YAMLError) as exc:
        raise typer.BadParameter(
            "Cannot preserve storage-IAM cleanup provenance because the local "
            f"credentials journal is unreadable: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise typer.BadParameter(
            "Cannot preserve storage-IAM cleanup provenance because the local "
            "credentials journal is not a YAML mapping."
        )

    account_ids: set[str] = set()
    account_name = ""
    access_key_ids: set[str] = set()
    creation_status = ""
    creation_phase = ""
    creation_attempt = ""
    for section_name in ("storage_iam", "nebius"):
        section = data.get(section_name)
        if not isinstance(section, dict):
            continue
        account_id = str(section.get("service_account_id", "") or "").strip()
        if account_id:
            account_ids.add(account_id)
        if section_name == "storage_iam":
            account_name = str(section.get("service_account_name", "") or "").strip()

    setup = data.get("storage_setup")
    setup_projects = setup.get("projects") if isinstance(setup, dict) else None
    setup_record = (
        setup_projects.get(project_id) if isinstance(setup_projects, dict) else None
    )
    if isinstance(setup_record, dict):
        creation_status = str(setup_record.get("status", "") or "").strip()
        creation_phase = str(setup_record.get("phase", "") or "").strip()
        creation_attempt = str(setup_record.get("attempt_id", "") or "").strip()
        resources = setup_record.get("resources")
        if isinstance(resources, dict):
            account = resources.get("service_account")
            if isinstance(account, dict):
                account_id = str(account.get("id", "") or "").strip()
                if account_id:
                    account_ids.add(account_id)
                account_name = (
                    account_name or str(account.get("name", "") or "").strip()
                )
            keys = resources.get("access_keys")
            if isinstance(keys, dict):
                access_key_ids.update(
                    str(key).strip() for key in keys if str(key).strip()
                )

    storage = data.get("storage")
    saved_bucket = ""
    if isinstance(storage, dict):
        saved_bucket = _bucket_name_from_uri(
            str(
                storage.get("bucket")
                or storage.get("s3_bucket")
                or storage.get("checkpoint_bucket")
                or ""
            )
        )
        for key in ("aws_access_key_id", "access_key_id", "nebius_api_key"):
            value = str(storage.get(key, "") or "").strip()
            if value:
                access_key_ids.add(value)

    setup_bucket = ""
    if isinstance(setup_record, dict):
        resources = setup_record.get("resources")
        bucket_record = resources.get("bucket") if isinstance(resources, dict) else None
        if isinstance(bucket_record, dict):
            setup_bucket = _bucket_name_from_uri(
                str(bucket_record.get("name", "") or "")
            )

    # A credentials file may describe a different live bucket. Its IAM identity
    # is not provenance for the explicitly selected deletion target, so leave
    # that entire lifecycle untouched.
    if (
        bucket_name
        and saved_bucket
        and saved_bucket != bucket_name
        and setup_bucket != bucket_name
    ):
        return "", {}

    owned_record, ownership_note = _storage_service_account_record()
    if owned_record is not None and owned_record.project_id == project_id:
        # A generic ``nebius.service_account_id`` may belong to the agent. Once
        # the dedicated storage journal proves one immutable identity, it is the
        # only IAM target this bucket lifecycle is allowed to carry forward.
        account_ids = {owned_record.account_id}
    owned = bool(
        owned_record is not None
        and owned_record.project_id == project_id
        and owned_record.account_id in account_ids
    )
    existing_id = str(existing.get("service_account_id", "") or "").strip()
    if existing_id:
        account_ids.add(existing_id)
    if len(account_ids) > 1:
        account_id = ""
        ownership_state = "unknown"
        detail = (
            "Conflicting immutable service-account IDs were preserved; reconciliation "
            "must resolve the exact identity before deletion: "
            + ", ".join(sorted(account_ids))
        )
    else:
        account_id = next(iter(account_ids), "")
        ownership_state = (
            "owned" if owned else ("pending-verification" if account_id else "unknown")
        )
        detail = (
            "NPA creation provenance is preserved for guarded deletion."
            if owned
            else ownership_note
            or "Historical provenance does not yet establish NPA ownership."
        )
    evidence: dict[str, object] = {
        "status": "present_owned" if owned else "present_unverified_ownership",
        "ownership_state": ownership_state,
        "ownership": "npa" if owned else "unverified",
        "project_id": project_id,
        "service_account_id": account_id,
        "service_account_name": account_name,
        "candidate_service_account_ids": sorted(account_ids),
        "access_key_ids": sorted(access_key_ids),
        "bucket_name": bucket_name,
        "bucket_cleanup_state": "pending",
        "creation_outcome": creation_status,
        "creation_phase": creation_phase,
        "creation_attempt_id": creation_attempt,
        "detail": detail,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if not alias:
        # Old credentials may predate project aliases. Bucket pruning removes
        # only the storage-secret section, so the non-secret ``nebius``,
        # ``storage_iam``, and setup-journal evidence remains available. Do not
        # make the bucket itself undeletable merely because that legacy evidence
        # cannot yet be copied into the per-project config tombstone.
        return "", evidence if account_ids or setup_record else {}
    try:
        return alias, mark_storage_iam_residue(alias, evidence)
    except ConfigError as exc:
        raise typer.BadParameter(
            "Bucket cleanup cannot safely proceed because its storage-IAM tombstone "
            f"could not be saved: {exc}. No bucket was changed."
        ) from exc


def _complete_bucket_iam_cleanup_tombstone(
    alias: str, marker: dict[str, object]
) -> None:
    """Mark bucket removal complete while leaving IAM cleanup visibly pending."""

    if not marker:
        return
    if not alias:
        typer.echo(
            "Bucket cleanup completed; storage-IAM cleanup remains pending in the "
            "non-secret credentials journal. Restore the project with `npa configure`, "
            "then continue with `npa storage service-account reconcile --project "
            "<alias> --id <exact-service-account-id> --dry-run`."
        )
        return
    from datetime import datetime, timezone

    from npa.clients.config import ConfigError, mark_storage_iam_residue

    try:
        finished = mark_storage_iam_residue(
            alias,
            {
                "bucket_cleanup_state": "complete",
                "bucket_cleanup_completed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    except (ConfigError, OSError) as exc:
        _partial_cleanup(
            "bucket cleanup completed, but its pending storage-IAM tombstone could "
            f"not be marked complete: {type(exc).__name__}: {exc}. The pending "
            "ownership evidence was preserved; fix local config permissions and retry."
        )
    account_id = str(finished.get("service_account_id", "") or "").strip()
    ownership_state = str(finished.get("ownership_state", "") or "unknown")
    if ownership_state == "owned":
        command = f"npa storage service-account delete --project {alias} --dry-run"
    else:
        exact_id = account_id or "<exact-service-account-id>"
        command = (
            "npa storage service-account reconcile "
            f"--project {alias} --id {exact_id} --dry-run"
        )
    typer.echo(
        "Bucket cleanup completed; storage-IAM cleanup remains pending "
        f"({ownership_state}). Continue with `{command}`."
    )


@service_account_app.command("delete")
@intent_boundary(OperationIntent.DESTROY)
@json_stdout_contract
def delete_service_account_cmd(
    project: str = typer.Option(
        "", "--project", help="NPA project alias owning the account."
    ),
    project_id: str = typer.Option(
        "", "--project-id", help="Nebius project id owning the account."
    ),
    tenant_id: str = typer.Option("", "--tenant-id", help="Exact Nebius tenant ID."),
    profile: str = typer.Option("", "--profile", help="Exact Nebius CLI profile."),
    receipt: str = typer.Option("", "--receipt", help="Opaque teardown receipt ID."),
    service_account_id: str = typer.Option(
        "", "--id", help="Exact immutable storage service-account ID."
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help=(
            "Verify and print exact NPA-owned IAM resources without deleting them; "
            "untrusted/failed verification exits 2."
        ),
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip the confirmation prompt."
    ),
    output_json: bool = typer.Option(
        False, "--json", help="Emit a machine-readable result."
    ),
) -> None:
    """Delete the storage service account only when NPA recorded creating it.

    Run this after deleting the configured bucket. Reused, legacy, incomplete,
    mismatched, and user-managed identities are always left untouched. Verified
    deletion/absence exits 0; missing trustworthy ownership or provider/auth
    verification failure is an operator-actionable partial cleanup (exit 2).
    """

    import json
    from npa.clients.config import ConfigError
    from npa.clients.credentials import load_credentials
    from npa.clients.nebius import (
        NebiusError,
        delete_access_key,
        delete_access_permit,
        delete_group,
        delete_service_account,
        is_not_found,
        list_access_keys_for_service_account,
    )

    def progress(message: str, *, err: bool = False) -> None:
        if not output_json:
            typer.echo(message, err=err)

    try:
        context = _resolve_storage_iam_context(
            project,
            project_id,
            tenant_id=tenant_id,
            profile=profile,
            receipt=receipt,
            service_account_id=service_account_id,
        )
        observation = _observe_storage_iam(context)
        # Preserve the established reconciliation behavior of --dry-run: it may
        # refresh/clear the non-secret residue journal, but never deletes cloud
        # IAM or the immutable ownership record. A mutating verified-absence
        # path defers even that journal collapse until confirmation.
        if dry_run or not observation.verified_absent:
            _persist_storage_iam_observation(observation)
    except ConfigError as exc:
        _partial_cleanup(str(exc), output_json=output_json)
    payload = _observation_dict(observation)
    if not output_json:
        typer.echo(f"identity_source: {context.identity_source}")
    if observation.outcome == "verification_failed":
        _record_storage_iam_teardown(
            observation,
            terminal_state="verification_failed",
            action="none",
            errors=(observation.detail,),
        )
        payload["result"] = "partial_verification_failure"
        if output_json:
            typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        _partial_cleanup(
            "provider/auth verification failed for storage IAM; nothing was "
            f"deleted and the project was preserved: {observation.detail}",
            output_json=output_json,
            payload=payload,
        )
    if observation.verified_absent:
        payload["result"] = "already_absent"
        record, _note = _storage_service_account_record(
            account_id=context.account_id,
            project_id=context.project_id,
            receipt_id=context.receipt_id,
        )
        from npa.clients.config import storage_iam_residue

        has_residue_marker = bool(context.alias and storage_iam_residue(context.alias))
        should_prune_record = bool(
            not dry_run
            and observation.account_id
            and record is not None
            and record.account_id == observation.account_id
        )
        should_mutate_local_state = bool(
            not dry_run and (should_prune_record or has_residue_marker)
        )
        if should_mutate_local_state:
            from npa.cli.destructive import require_destructive_confirmation

            require_destructive_confirmation(
                yes=yes,
                prompt=(
                    "Remove the stale local ownership record for provider-verified "
                    f"absent service account {observation.account_id}?"
                ),
                output_json=output_json,
                payload=payload,
            )
            try:
                _persist_storage_iam_observation(observation)
            except ConfigError as exc:
                _partial_cleanup(str(exc), output_json=output_json, payload=payload)
            if record is not None:
                if not _remove_storage_service_account_record(record.account_id):
                    _partial_cleanup(
                        "the account is verified absent, but its stale local ownership "
                        "record could not be removed; fix file permissions and retry.",
                        output_json=output_json,
                        payload=payload,
                    )
        if not dry_run:
            _record_storage_iam_teardown(
                observation,
                terminal_state="verified_absent",
                action="remove_verified_absent_local_evidence"
                if should_mutate_local_state
                else "none",
            )
        if output_json:
            typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        else:
            typer.echo(
                "Verified absence: the exact storage service account is already "
                f"absent in {context.project_id}. No IAM resource was deleted."
            )
        return

    record, note = _storage_service_account_record(
        account_id=context.account_id,
        project_id=context.project_id,
        receipt_id=context.receipt_id,
    )
    if (
        observation.ownership != "npa"
        or record is None
        or record.account_id != observation.account_id
    ):
        payload["result"] = "residual_unverified_ownership"
        if output_json:
            typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        _partial_cleanup(
            f"{note} Provider verification found exact service account "
            f"{observation.account_name} ({observation.account_id}) in "
            f"{context.project_id}, but identity/name is not ownership proof. It "
            "was left untouched and the project was preserved. Inspect the plan "
            "with `npa storage service-account reconcile --project "
            f"{context.alias or '<alias>'} --id {observation.account_id} --dry-run`.",
            output_json=output_json,
            payload=payload,
        )

    from npa.clients.config import resolve_environment

    alias_is_configured = bool(
        context.alias and resolve_environment(context.alias) is not None
    )
    if context.receipt_id and not alias_is_configured:
        if not _receipt_proves_bucket_cleanup(context.receipt_id, context.project_id):
            _partial_cleanup(
                "the alias is absent and the selected receipt does not prove exact "
                "bucket cleanup completed; storage IAM was preserved.",
                output_json=output_json,
                payload={
                    **payload,
                    "result": "bucket_interlock_unresolved",
                    "interlock": "delete_bucket_first",
                },
            )
    else:
        try:
            credentials = load_credentials(environ={})
        except Exception as exc:  # noqa: BLE001 - fail closed on credential uncertainty
            message = (
                "could not verify the delete-bucket-first interlock because the saved "
                f"credential/config store is unreadable or invalid: {exc}. Nothing was deleted."
            )
            _partial_cleanup(
                message,
                output_json=output_json,
                payload={
                    **payload,
                    "result": "bucket_interlock_unresolved",
                    "interlock": "delete_bucket_first",
                },
            )
        configured_bucket = str(getattr(credentials, "s3_bucket", "") or "").strip()
        if configured_bucket:
            raise typer.BadParameter(
                f"Object storage {configured_bucket} is still configured. Run `npa storage "
                "bucket delete --project <alias> --yes --wait` first so this account is "
                "not deleted while its bucket may still need it."
            )

    try:
        keys = list_access_keys_for_service_account(
            record.project_id,
            record.account_id,
            strict=True,
            profile=context.profile,
        )
    except NebiusError as exc:
        if is_not_found(str(exc)):
            # The access-key endpoint's NotFound is not itself authoritative
            # proof that the service account disappeared. Re-run the same exact
            # identity resolver before changing any ownership/residue state.
            recheck = _observe_storage_iam(context)
            if not recheck.verified_absent:
                _partial_cleanup(
                    "access-key inventory returned NotFound, but an immediate exact "
                    "service-account recheck did not prove absence; nothing was deleted.",
                    output_json=output_json,
                    payload=payload,
                )
            recheck_payload = _observation_dict(recheck)
            recheck_payload["result"] = "already_absent"
            if dry_run:
                try:
                    _persist_storage_iam_observation(recheck)
                except ConfigError as marker_exc:
                    _partial_cleanup(
                        str(marker_exc),
                        output_json=output_json,
                        payload=recheck_payload,
                    )
            else:
                from npa.cli.destructive import require_destructive_confirmation

                require_destructive_confirmation(
                    yes=yes,
                    prompt=(
                        "Remove stale local ownership evidence for provider-verified "
                        f"absent service account {record.account_id}?"
                    ),
                    output_json=output_json,
                    payload=recheck_payload,
                )
                try:
                    _persist_storage_iam_observation(recheck)
                except ConfigError as marker_exc:
                    _partial_cleanup(
                        str(marker_exc),
                        output_json=output_json,
                        payload=recheck_payload,
                    )
                if not _remove_storage_service_account_record(record.account_id):
                    _partial_cleanup(
                        "the account is verified absent, but its stale local ownership "
                        "record could not be removed; fix file permissions and retry.",
                        output_json=output_json,
                        payload=recheck_payload,
                    )
                _record_storage_iam_teardown(
                    recheck,
                    terminal_state="verified_absent",
                    action="remove_verified_absent_local_evidence",
                )
            if output_json:
                typer.echo(json.dumps(recheck_payload, indent=2, sort_keys=True))
            else:
                typer.echo(
                    f"Verified absence: NPA-owned service account {record.name} "
                    f"({record.account_id}) is already absent."
                )
            return
        _partial_cleanup(
            "provider/auth verification failed while inspecting access keys for "
            f"NPA-owned service account {record.account_id}; nothing was deleted: {exc}",
            output_json=output_json,
            payload=payload,
        )
    mismatched_keys = [
        str((key or {}).get("id", "") or "").strip() or "<missing-id>"
        for key in keys
        if str((key or {}).get("service_account_id", "") or "").strip()
        != record.account_id
    ]
    if mismatched_keys:
        _partial_cleanup(
            "access-key inventory did not prove the selected service-account "
            "relationship for key(s) "
            + ", ".join(mismatched_keys)
            + "; nothing was deleted.",
            output_json=output_json,
            payload=payload,
        )
    key_ids = [str((key or {}).get("id", "") or "").strip() for key in keys]
    key_ids = [key_id for key_id in key_ids if key_id]
    if dry_run:
        payload.update(
            {
                "result": "delete_planned",
                "access_key_ids": key_ids,
            }
        )
        if output_json:
            typer.echo(json.dumps(payload, indent=2, sort_keys=True))
            return
        typer.echo(
            f"Verified present: NPA creation provenance ({record.source}) owns "
            f"service account {record.account_id}."
        )
        for key_id in key_ids:
            typer.echo(f"Would delete access key {key_id}.")
        typer.echo(
            f"Would delete NPA-owned service account {record.name} "
            f"({record.account_id}) in {record.project_id}."
        )
        return

    from npa.cli.destructive import require_destructive_confirmation

    require_destructive_confirmation(
        yes=yes,
        prompt=(
            f"Delete NPA-owned service account {record.name} ({record.account_id}) "
            f"and {len(key_ids)} access key(s)?"
        ),
        output_json=output_json,
        payload=payload,
    )

    failed = False
    profile_kwargs = {"profile": context.profile} if context.profile else {}
    for key_id in key_ids:
        try:
            delete_access_key(key_id, **profile_kwargs)
        except NebiusError as exc:
            if is_not_found(str(exc)):
                progress(f"Access key {key_id} is already absent.")
            else:
                failed = True
                progress(
                    f"Warning: could not delete access key {key_id}: {exc}", err=True
                )
        else:
            progress(f"Deleted access key {key_id}.")
    if record.binding_managed_by_npa:
        permit_deleted = True
        try:
            delete_access_permit(record.access_permit_id, **profile_kwargs)
        except NebiusError as exc:
            if not is_not_found(str(exc)):
                permit_deleted = False
                failed = True
                progress(
                    f"Warning: could not delete exact NPA storage access permit "
                    f"{record.access_permit_id}: {exc}",
                    err=True,
                )
        if permit_deleted:
            try:
                delete_group(record.group_id, **profile_kwargs)
            except NebiusError as exc:
                if not is_not_found(str(exc)):
                    failed = True
                    progress(
                        f"Warning: could not delete exact NPA storage group "
                        f"{record.group_id}: {exc}",
                        err=True,
                    )
    try:
        delete_service_account(record.account_id, **profile_kwargs)
    except NebiusError as exc:
        if is_not_found(str(exc)):
            progress(
                f"Verified absence: NPA-owned service account {record.name} "
                f"({record.account_id}) is already absent."
            )
            if not _remove_storage_service_account_record(record.account_id):
                failed = True
                progress(
                    "Warning: the stale local service-account ownership record could "
                    "not be removed; retry after fixing its file permissions.",
                    err=True,
                )
            try:
                from npa.clients.config import clear_storage_iam_residue

                if context.alias:
                    clear_storage_iam_residue(context.alias, account_id="")
            except ConfigError as marker_exc:
                failed = True
                progress(
                    f"Warning: exact IAM absence was verified, but its residue marker "
                    f"could not be cleared: {marker_exc}",
                    err=True,
                )
        else:
            failed = True
            progress(
                f"Warning: could not delete NPA-owned service account {record.account_id}: {exc}",
                err=True,
            )
    else:
        progress(
            f"Verified deletion: NPA-owned service account {record.name} "
            f"({record.account_id}) was deleted."
        )
        deleted_observation = _StorageIamObservation(
            outcome="verified_absent",
            context=context,
            account_id=record.account_id,
            account_name=record.name,
            ownership="npa",
            detail="provider delete completed",
        )
        _record_storage_iam_teardown(
            deleted_observation,
            terminal_state="verified_deleted",
            action="delete_npa_owned_service_account",
            access_key_ids=tuple(key_ids),
        )
        if not _remove_storage_service_account_record(record.account_id):
            failed = True
            progress(
                "Warning: the service account is gone, but its local ownership record "
                "could not be removed; retry after fixing its file permissions.",
                err=True,
            )
        try:
            from npa.clients.config import clear_storage_iam_residue

            if context.alias:
                clear_storage_iam_residue(context.alias, account_id="")
        except ConfigError as exc:
            failed = True
            progress(
                f"Warning: the service account is gone, but IAM residue could not "
                f"be cleared: {exc}",
                err=True,
            )
    if failed:
        if output_json:
            payload["result"] = "partial"
            payload["mutated"] = True
            typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        raise typer.Exit(code=1)
    if output_json:
        payload["result"] = "deleted"
        payload["mutated"] = True
        payload["access_key_ids"] = key_ids
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@service_account_app.command("reconcile")
def reconcile_service_account_cmd(
    project: str = typer.Option("", "--project", help="Configured NPA project alias."),
    project_id: str = typer.Option("", "--project-id", help="Exact Nebius project ID."),
    tenant_id: str = typer.Option("", "--tenant-id", help="Exact Nebius tenant ID."),
    profile: str = typer.Option("", "--profile", help="Exact Nebius CLI profile."),
    receipt: str = typer.Option("", "--receipt", help="Opaque teardown receipt ID."),
    service_account_id: str = typer.Option(
        "",
        "--id",
        help="Exact immutable service-account ID. Defaults to saved residue evidence.",
    ),
    reason: str = typer.Option(
        "",
        "--reason",
        help="Non-secret reason legacy NPA ownership is being recovered.",
    ),
    attested_by: str = typer.Option(
        "",
        "--attested-by",
        help="Non-secret operator identity recorded in the recovery journal.",
    ),
    attest_npa_created: bool = typer.Option(
        False,
        "--attest-npa-created",
        help="Explicitly attest that the exact provider-verified identity was created by NPA.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Verify and show the recovery plan only."
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Confirm writing recovery provenance."
    ),
    output_json: bool = typer.Option(
        False, "--json", help="Emit a machine-readable result."
    ),
) -> None:
    """Reconcile a legacy NPA-created storage identity without weakening deletion guards."""

    import getpass
    import json
    from datetime import datetime, timezone

    import yaml

    from npa.clients.config import (
        ConfigError,
        mark_storage_iam_residue,
        storage_iam_residue,
    )
    from npa.clients.credentials import CREDENTIALS_PATH, write_private_yaml
    from npa.clients.nebius import (
        DEFAULT_SERVICE_ACCOUNT_NAME,
        NebiusError,
        get_service_account_identity,
    )

    try:
        context = _resolve_storage_iam_context(
            project,
            project_id,
            tenant_id=tenant_id,
            profile=profile,
            receipt=receipt,
            service_account_id=service_account_id,
        )
    except ConfigError as exc:
        _partial_cleanup(str(exc))
    if not context.tenant_id:
        _partial_cleanup(
            "Reconciliation requires the exact tenant_id from --tenant-id, a receipt, "
            "or configured project context before attesting ownership."
        )
    marker = storage_iam_residue(context.alias) if context.alias else {}
    saved_ids = _untrusted_storage_account_ids()
    marker_id = str(marker.get("service_account_id", "") or "").strip()
    if marker_id:
        saved_ids.add(marker_id)
    marker_candidates = marker.get("candidate_service_account_ids")
    if isinstance(marker_candidates, list):
        saved_ids.update(
            str(item).strip() for item in marker_candidates if str(item).strip()
        )
    exact_id = str(service_account_id or context.account_id or "").strip()
    if not exact_id:
        if len(saved_ids) != 1:
            _partial_cleanup(
                "Pass one exact immutable --id; local evidence did not select exactly one candidate."
            )
        exact_id = next(iter(saved_ids))
    elif saved_ids and exact_id not in saved_ids:
        _partial_cleanup(
            "The requested immutable ID conflicts with saved project evidence; no provenance was changed."
        )
    try:
        identity = get_service_account_identity(
            exact_id,
            project_id=context.project_id,
            tenant_id=context.tenant_id,
            expected_name=DEFAULT_SERVICE_ACCOUNT_NAME,
            profile=context.profile,
        )
    except NebiusError as exc:
        observation = _StorageIamObservation(
            outcome="verification_failed",
            context=context,
            account_id=exact_id,
            detail=str(exc),
        )
        _persist_storage_iam_observation(observation)
        if output_json:
            payload = _observation_dict(observation)
            payload["result"] = "partial_verification_failure"
            typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        _partial_cleanup(
            f"exact identity/project/tenant verification failed; no provenance was changed: {exc}"
        )
    if identity is None:
        observation = _StorageIamObservation(
            outcome="verified_absent",
            context=context,
            account_id=exact_id,
            detail="The provider authoritatively returned NotFound for the exact immutable ID.",
        )
        _persist_storage_iam_observation(observation)
        payload = _observation_dict(observation)
        payload["result"] = "already_absent"
        if output_json:
            typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        else:
            typer.echo(
                f"Verified absence: {exact_id} is already absent; no reconciliation is needed."
            )
        return

    observation = _StorageIamObservation(
        outcome="present",
        context=context,
        account_id=identity.account_id,
        account_name=identity.name,
        detail="Exact ID, name, project, tenant, and CLI profile context were verified.",
    )
    _persist_storage_iam_observation(observation)
    plan = _observation_dict(observation)
    plan.update(
        {
            "result": "reconciliation_planned" if dry_run else "reconciled",
            "requires_attestation": True,
            "will_delete": False,
        }
    )
    if dry_run:
        if output_json:
            typer.echo(json.dumps(plan, indent=2, sort_keys=True))
        else:
            typer.echo(
                f"Verified exact residual {identity.name} ({identity.account_id}) in "
                f"project {identity.project_id}, tenant {identity.tenant_id or '(provider-unset)'}."
            )
            typer.echo(
                "Would record operator-attested NPA recovery provenance; no IAM resource would be deleted."
            )
            typer.echo(
                "To attest, re-run with --reason <why> --attest-npa-created --yes; "
                "then use `npa storage service-account delete --project "
                f"{context.alias or '<alias>'} --dry-run`."
            )
        return
    if not (yes and attest_npa_created and reason.strip()):
        _partial_cleanup(
            "Writing ownership recovery requires --reason <why>, --attest-npa-created, "
            "and --yes after reviewing --dry-run. The identity remains unowned."
        )
    from npa.orchestration.skypilot.workflow_state import redact_text

    if redact_text(reason.strip()) != reason.strip():
        _partial_cleanup(
            "The recovery reason appears to contain credential material and was not stored. "
            "Use a non-secret evidence reference instead."
        )
    operator = str(attested_by or getpass.getuser() or "unknown").strip()
    if not operator:
        _partial_cleanup("A non-empty --attested-by identity is required.")
    try:
        data = yaml.safe_load(CREDENTIALS_PATH.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        data = {}
    except (OSError, yaml.YAMLError) as exc:
        _partial_cleanup(f"Could not read the existing credentials journal: {exc}")
    if not isinstance(data, dict):
        _partial_cleanup("The existing credentials journal is not a YAML mapping.")
    now = datetime.now(timezone.utc).isoformat()
    recovery = {
        "schema_version": "npa.storage-iam-recovery.v1",
        "service_account_id": identity.account_id,
        "service_account_name": identity.name,
        "project_alias": context.alias,
        "project_id": identity.project_id,
        "tenant_id": identity.tenant_id,
        "profile": identity.profile or "active",
        "provider_verified": True,
        "verified_at": now,
        "attested_by": operator,
        "attested_at": now,
        "reason": reason.strip(),
    }
    existing = data.get("storage_iam")
    if isinstance(existing, dict):
        existing_id = str(existing.get("service_account_id", "") or "").strip()
        if existing_id and existing_id != identity.account_id:
            _partial_cleanup(
                "A different storage IAM ownership record already exists; no provenance was changed."
            )
    data["storage_iam"] = {
        "service_account_id": identity.account_id,
        "service_account_name": identity.name,
        "service_account_project_id": identity.project_id,
        "service_account_managed_by": "npa-recovery-attested",
        "recovery": recovery,
    }
    write_private_yaml(CREDENTIALS_PATH, data)
    record, record_error = _storage_service_account_record(
        account_id=identity.account_id,
        project_id=identity.project_id,
    )
    if record is None or record.account_id != identity.account_id:
        _partial_cleanup(
            f"Recovery journal failed its ownership validation and deletion remains blocked: {record_error}"
        )
    marker_payload = {
        "status": "reconciled_pending_delete",
        "project_id": identity.project_id,
        "tenant_id": identity.tenant_id,
        "profile": identity.profile or "active",
        "service_account_id": identity.account_id,
        "service_account_name": identity.name,
        "ownership": "npa-recovery-attested",
        "reconciled_at": now,
        "recovery_reason": reason.strip(),
        "attested_by": operator,
    }
    if context.alias:
        mark_storage_iam_residue(context.alias, marker_payload)
    else:
        _record_storage_iam_teardown(
            _StorageIamObservation(
                outcome="present",
                context=context,
                account_id=identity.account_id,
                account_name=identity.name,
                ownership="npa",
                detail="Operator-attested recovery provenance persisted globally.",
            ),
            terminal_state="operator_action_remains",
            action="record_exact_ownership_attestation",
        )
    if output_json:
        typer.echo(json.dumps(plan, indent=2, sort_keys=True))
    else:
        typer.echo(
            f"Recorded recovery provenance for {identity.name} ({identity.account_id}); "
            "no IAM resource was deleted."
        )
        typer.echo(
            "Continue through the guarded path: `npa storage service-account delete "
            f"--project {context.alias or '<alias>'} --dry-run`, then repeat with --yes."
        )


@bucket_app.command("delete")
@intent_boundary(OperationIntent.DESTROY)
@json_stdout_contract
def delete_bucket_cmd(
    name: str = typer.Option(
        "", "--name", help="Bucket name. Defaults to the configured bucket."
    ),
    bucket_id: str = typer.Option(
        "", "--id", help="Bucket resource id (skips the name lookup)."
    ),
    project: str = typer.Option(
        "", "--project", help="NPA project alias holding the bucket."
    ),
    project_id: str = typer.Option(
        "", "--project-id", help="Nebius project id holding the bucket."
    ),
    ttl: str = typer.Option(
        DEFAULT_PURGE_TTL,
        "--ttl",
        help=(
            "Schedule the purge this far out (Nebius duration, e.g. 1m/2h). A bucket "
            "with objects or object versions cannot be deleted immediately; pass an "
            "empty value to attempt an immediate delete."
        ),
    ),
    prune_config: bool = typer.Option(
        True,
        "--prune-config/--keep-config",
        help=(
            "Also drop this bucket's saved S3 credentials from "
            "~/.npa/credentials.yaml and its Terraform remote-state keys from "
            "~/.npa/config.yaml."
        ),
    ),
    wait: bool = typer.Option(
        False,
        "--wait",
        help="Poll until the bucket is actually gone (a scheduled purge is async).",
    ),
    wait_timeout: int = typer.Option(
        300, "--wait-timeout", help="Max seconds to wait when --wait is set."
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip the confirmation prompt."
    ),
    output_json: bool = typer.Option(
        False, "--json", help="Emit a machine-readable confirmation refusal."
    ),
) -> None:
    """Delete an object-storage bucket npa provisioned, contents and versions included."""
    from npa.clients.config import resolve_environment
    from npa.clients.credentials import load_credentials
    from npa.clients.nebius import NebiusError, delete_bucket, get_bucket_by_name

    credentials = None
    try:
        credentials = load_credentials(environ={})
    except Exception:  # noqa: BLE001 - teardown must work without readable credentials
        credentials = None
    bucket_name = _bucket_name_from_uri(name.strip()) or _bucket_name_from_uri(
        str(getattr(credentials, "s3_bucket", "") or "")
    )
    resolved_project = project_id.strip()
    if not resolved_project:
        saved = resolve_environment(project or None)
        resolved_project = str(getattr(saved, "project_id", "") or "")

    resolved_id = bucket_id.strip()
    if not resolved_id:
        if not bucket_name:
            raise typer.BadParameter(
                "Pass --name <bucket> or --id <bucket-id> (no bucket is configured in "
                "~/.npa/credentials.yaml)."
            )
        if not resolved_project:
            raise typer.BadParameter(
                "Cannot tell which Nebius project holds the bucket. Pass --project-id "
                "<id> or --project <alias>, or pass --id <bucket-id> directly."
            )
        try:
            item = get_bucket_by_name(resolved_project, bucket_name)
        except NebiusError as exc:
            raise typer.BadParameter(
                f"Could not list buckets in {resolved_project}: {exc}"
            ) from exc
        if item is None:
            if prune_config:
                from npa.cli.destructive import require_destructive_confirmation

                require_destructive_confirmation(
                    yes=yes,
                    prompt=(
                        f"Remove saved local credentials/state for provider-verified "
                        f"absent bucket {bucket_name!r}?"
                    ),
                    output_json=output_json,
                    payload={"bucket_id": "", "bucket_name": bucket_name},
                )
            if not output_json:
                typer.echo(
                    f"Bucket {bucket_name!r} does not exist in project {resolved_project}."
                )
            iam_alias, iam_marker = _begin_bucket_iam_cleanup_tombstone(
                project, resolved_project, bucket_name
            )
            from npa.teardown_receipts import record_teardown_event

            record_teardown_event(
                phase="bucket",
                resource=bucket_name,
                terminal_state="verified_absent",
                project_alias=project,
                project_id=resolved_project,
                precheck={"provider_lookup": "absent"},
                action={"kind": "none"},
                verification={"bucket_absent": True},
            )
            if prune_config:
                _prune_local_state(bucket_name)
            _complete_bucket_iam_cleanup_tombstone(iam_alias, iam_marker)
            if output_json:
                import json

                typer.echo(
                    json.dumps(
                        {
                            "result": "already_absent",
                            "bucket_id": "",
                            "bucket_name": bucket_name,
                            "project_id": resolved_project,
                            "verified_absent": True,
                        },
                        sort_keys=True,
                    )
                )
            return
        resolved_id = str((item.get("metadata") or {}).get("id", "") or "")
        bucket_name = bucket_name or str(
            (item.get("metadata") or {}).get("name", "") or ""
        )

    target = f"{bucket_name or resolved_id} ({resolved_id})"
    from npa.cli.destructive import require_destructive_confirmation

    require_destructive_confirmation(
        yes=yes,
        prompt=f"Delete bucket {target} and everything in it?",
        output_json=output_json,
        payload={"bucket_id": resolved_id, "bucket_name": bucket_name},
    )

    iam_alias, iam_marker = _begin_bucket_iam_cleanup_tombstone(
        project, resolved_project, bucket_name
    )

    try:
        delete_bucket(resolved_id, ttl=ttl)
    except NebiusError as exc:
        message = str(exc)
        if (
            "notempty" in message.replace(" ", "").lower()
            and not str(ttl or "").strip()
        ):
            raise typer.BadParameter(
                f"Bucket {target} is not empty (objects or non-current versions remain). "
                f"Re-run with --ttl {DEFAULT_PURGE_TTL} to schedule the purge."
            ) from exc
        # Once a purge is scheduled the API is inconsistent: `delete` answers
        # NoSuchBucket while `list`/`get` still return the bucket as
        # SCHEDULED_FOR_DELETION. Re-deleting is then a no-op, not a failure.
        pending = _scheduled_deletion_state(resolved_project, bucket_name)
        if "nosuchbucket" in message.replace(" ", "").lower() and pending:
            if not output_json:
                typer.echo(f"Bucket {target} is already {pending}.")
            verified_gone = False
            if wait:
                verified_gone = _call_bucket_waiter(
                    resolved_project,
                    bucket_name,
                    target,
                    wait_timeout,
                    bucket_id=resolved_id,
                    quiet=output_json,
                )
            from npa.teardown_receipts import record_teardown_event

            record_teardown_event(
                phase="bucket",
                resource=bucket_name or resolved_id,
                terminal_state=("verified_deleted" if verified_gone else "in_progress"),
                project_alias=project,
                project_id=resolved_project,
                precheck={"provider_state": pending},
                action={"kind": "scheduled_bucket_purge"},
                verification={"bucket_absent": verified_gone},
            )
            if prune_config and bucket_name and (not wait or verified_gone):
                _prune_local_state(bucket_name)
            if not wait or verified_gone:
                _complete_bucket_iam_cleanup_tombstone(iam_alias, iam_marker)
            if wait and not verified_gone:
                if output_json:
                    import json

                    typer.echo(
                        json.dumps(
                            {
                                "result": "partial",
                                "bucket_id": resolved_id,
                                "bucket_name": bucket_name,
                                "project_id": resolved_project,
                                "provider_state": pending,
                                "verified_absent": False,
                            },
                            sort_keys=True,
                        )
                    )
                raise typer.Exit(code=2)
            if output_json:
                import json

                typer.echo(
                    json.dumps(
                        {
                            "result": "already_scheduled",
                            "bucket_id": resolved_id,
                            "bucket_name": bucket_name,
                            "project_id": resolved_project,
                            "provider_state": pending,
                            "verified_absent": verified_gone,
                        },
                        sort_keys=True,
                    )
                )
            return
        raise typer.BadParameter(f"Bucket delete failed: {message}") from exc

    if str(ttl or "").strip() and not output_json:
        typer.echo(f"Bucket {target} scheduled for purge in {ttl}.")
    elif not output_json:
        typer.echo(f"Bucket {target} deleted.")
    verified_gone = not str(ttl or "").strip()
    if wait:
        verified_gone = _call_bucket_waiter(
            resolved_project,
            bucket_name,
            target,
            wait_timeout,
            bucket_id=resolved_id,
            quiet=output_json,
        )
    from npa.teardown_receipts import record_teardown_event

    record_teardown_event(
        phase="bucket",
        resource=bucket_name or resolved_id,
        terminal_state="verified_deleted" if verified_gone else "in_progress",
        project_alias=project,
        project_id=resolved_project,
        precheck={"bucket_id": resolved_id},
        action={
            "kind": "bucket_delete",
            "scheduled": bool(str(ttl or "").strip()),
        },
        verification={"bucket_absent": verified_gone},
    )
    if prune_config and bucket_name and (not wait or verified_gone):
        _prune_local_state(bucket_name)
    if not wait or verified_gone:
        _complete_bucket_iam_cleanup_tombstone(iam_alias, iam_marker)
    if wait and not verified_gone:
        if output_json:
            import json

            typer.echo(
                json.dumps(
                    {
                        "result": "partial",
                        "bucket_id": resolved_id,
                        "bucket_name": bucket_name,
                        "project_id": resolved_project,
                        "verified_absent": False,
                    },
                    sort_keys=True,
                )
            )
        raise typer.Exit(code=2)
    if output_json:
        import json

        typer.echo(
            json.dumps(
                {
                    "result": "deleted" if verified_gone else "scheduled",
                    "bucket_id": resolved_id,
                    "bucket_name": bucket_name,
                    "project_id": resolved_project,
                    "verified_absent": verified_gone,
                },
                sort_keys=True,
            )
        )


def _call_bucket_waiter(
    project_id: str,
    bucket_name: str,
    target: str,
    timeout: int,
    *,
    bucket_id: str,
    quiet: bool = False,
) -> bool:
    """Call the identity-aware waiter while preserving old injected test hooks."""

    import inspect

    parameters = inspect.signature(_wait_for_bucket_gone).parameters
    if "bucket_id" not in parameters:
        result = _wait_for_bucket_gone(project_id, bucket_name, target, timeout)
        return True if result is None else bool(result)
    kwargs: dict[str, Any] = {"bucket_id": bucket_id}
    if "quiet" in parameters:
        kwargs["quiet"] = quiet
    return _wait_for_bucket_gone(project_id, bucket_name, target, timeout, **kwargs)


def _wait_for_bucket_gone(
    project_id: str,
    bucket_name: str,
    target: str,
    timeout: int,
    *,
    bucket_id: str = "",
    quiet: bool = False,
) -> bool:
    """Poll until *bucket_name* no longer exists (a scheduled purge is async).

    `storage bucket delete --ttl` returns while the bucket is still
    ``SCHEDULED_FOR_DELETION``; --wait blocks until Nebius has actually removed it
    so a caller can proceed knowing the name is free.
    """
    import time

    from npa.clients.nebius import NebiusError, get_bucket_by_name
    from npa.progress import WaitProgress

    if not project_id or not bucket_name:
        if not quiet:
            typer.echo("--wait skipped: no project/bucket name to poll.")
        return False
    deadline = time.monotonic() + max(1, int(timeout))
    if not quiet:
        typer.echo(f"Waiting up to {timeout}s for {bucket_name} to be purged...")
    progress = WaitProgress(
        "bucket deletion",
        monotonic=time.monotonic,
        emit=lambda message: typer.echo(message, err=True),
    )
    progress.start(f"id={bucket_id or 'unknown'} name={bucket_name}")
    while time.monotonic() < deadline:
        try:
            item = get_bucket_by_name(project_id, bucket_name)
        except NebiusError:
            typer.echo(
                f"Bucket {target} deletion could not be verified because provider "
                "inventory was forbidden or unavailable; it was not reported gone.",
                err=True,
            )
            progress.finish("verification_failed")
            return False
        if item is None:
            if not quiet:
                typer.echo(f"Bucket {target} is gone.")
            progress.finish("verified_absent")
            return True
        live_id = str((item.get("metadata") or {}).get("id") or "")
        if bucket_id and live_id and live_id != bucket_id:
            typer.echo(
                f"Bucket id {bucket_id} is gone; name {bucket_name!r} now belongs to "
                f"a different bucket id ({live_id}) and was left untouched; name reuse "
                "is not authoritative absence for this wait contract.",
                err=True,
            )
            progress.finish("verification_failed", "name_reused=true")
            return False
        progress.tick("provider_state=present")
        time.sleep(5)
    state = _scheduled_deletion_state(project_id, bucket_name) or "still present"
    overdue = _purge_is_overdue(project_id, bucket_name)
    if overdue:
        # Saying "it will be removed by Nebius" is not true once purge_at has
        # passed and the objects are still there; that is a platform-side stall.
        if not quiet:
            typer.echo(
                f"Bucket {bucket_name} is {state} and its purge_at has already passed, "
                f"so the purge has stalled rather than merely being slower than the {timeout}s "
                "wait. The name stays reserved until Nebius clears it; raise it with Nebius "
                "support if it does not resolve."
            )
        progress.finish("timed_out", "purge_overdue=true")
        return False
    if not quiet:
        typer.echo(
            f"Bucket {bucket_name} is {state} after {timeout}s "
            "(a scheduled purge can take longer than the wait); it will be removed by "
            "Nebius. Re-run with a larger --wait-timeout to keep watching."
        )
    progress.finish("timed_out", f"provider_state={state}")
    return False


def _bucket_item(project_id: str, bucket_name: str) -> dict | None:
    from npa.clients.nebius import NebiusError, get_bucket_by_name

    if not project_id or not bucket_name:
        return None
    try:
        return get_bucket_by_name(project_id, bucket_name)
    except NebiusError:
        return None


def _scheduled_deletion_state(project_id: str, bucket_name: str) -> str:
    """Return a human state (e.g. ``SCHEDULED_FOR_DELETION``) when still listed."""

    item = _bucket_item(project_id, bucket_name)
    if not isinstance(item, dict):
        return ""
    raw_status = item.get("status")
    status: dict = raw_status if isinstance(raw_status, dict) else {}
    return str(status.get("state") or status.get("status") or "").strip()


def _purge_is_overdue(project_id: str, bucket_name: str) -> bool:
    """Whether the bucket is still listed past its own ``purge_at``."""

    from datetime import datetime, timezone

    item = _bucket_item(project_id, bucket_name)
    if not isinstance(item, dict):
        return False
    raw_status = item.get("status")
    status: dict = raw_status if isinstance(raw_status, dict) else {}
    raw = str(status.get("purge_at") or status.get("purgeAt") or "").strip()
    if not raw:
        return False
    try:
        purge_at = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return False
    if purge_at.tzinfo is None:
        purge_at = purge_at.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) > purge_at


def _bucket_name_from_uri(value: str) -> str:
    return str(value or "").strip().removeprefix("s3://").strip("/").split("/", 1)[0]


def _prune_local_state(bucket_name: str) -> None:
    """Drop every on-disk secret tied to a now-deleted bucket.

    Two files hold them: ``credentials.yaml`` (the object-storage access key) and
    ``config.yaml`` (the Terraform remote-state backend key under
    ``projects.<alias>.terraform_state``). A bucket delete that cleaned only the
    former left live-looking HMAC keys for the deleted bucket in config.yaml.
    """
    try:
        _prune_storage_credentials(bucket_name)
    except (OSError, ValueError) as exc:
        _partial_cleanup(
            f"bucket deletion/absence was recorded, but the atomic credential rewrite "
            f"failed and the prior credential document was left intact: {exc}. Fix "
            "the credential path/permissions and retry local cleanup."
        )
    from npa.clients.config import (
        CONFIG_PATH,
        ConfigError,
        clear_terraform_state_for_bucket,
    )

    try:
        cleared = clear_terraform_state_for_bucket(bucket_name)
    except (ConfigError, OSError) as exc:
        _partial_cleanup(
            f"bucket deletion/absence was recorded and matching S3 credentials were "
            f"pruned, but Terraform backend state could not be rewritten: {exc}. "
            "Retry local cleanup after fixing config permissions."
        )
    if cleared:
        typer.echo(
            f"Removed the Terraform remote-state keys for {bucket_name} from "
            f"{CONFIG_PATH} (projects: {', '.join(cleared)})."
        )


# All key aliases `npa configure` / hand-written files use for the storage
# section. `configure` writes the aws_* / endpoint_url forms; the loader accepts
# the rest, so a robust prune must drop every alias, not just the canonical one.
_STORAGE_SECRET_KEYS = (
    "aws_access_key_id",
    "aws_secret_access_key",
    "access_key_id",
    "secret_access_key",
    "access_key",
    "secret_key",
    "nebius_api_key",
    "nebius_secret_key",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "endpoint_url",
    "endpoint",
    "s3_endpoint",
    "AWS_ENDPOINT_URL",
    "NEBIUS_S3_ENDPOINT",
    "bucket",
    "checkpoint_bucket",
    "s3_bucket",
    "NEBIUS_S3_BUCKET",
    "NPA_CHECKPOINT_BUCKET",
)
# The credentials.yaml section names the loader accepts for storage.
_STORAGE_SECTION_KEYS = ("storage", "s3", "object-storage", "object_storage")


def _prune_storage_credentials(bucket_name: str) -> None:
    """Drop saved S3 credentials that point at a bucket that no longer exists.

    Leaving them behind means the next `npa configure` / deploy reuses an access
    key for a deleted bucket — the stale-secret half of the teardown report.
    Removes the access key, secret key, endpoint and bucket from the storage
    section (under whatever key names the file uses). IAM identity state is a
    separate lifecycle: its ID and any NPA ownership proof remain until the
    explicit ownership-gated service-account teardown confirms it is gone.
    """
    from copy import deepcopy

    from npa.clients.credentials import CREDENTIALS_PATH, update_private_yaml

    removed: list[str] = []

    def prune(existing: dict[str, Any]) -> dict[str, Any]:
        data = deepcopy(existing)
        saved_bucket = ""
        for section_key in _STORAGE_SECTION_KEYS:
            section = data.get(section_key)
            if not isinstance(section, dict):
                continue
            for bucket_key in ("bucket", "checkpoint_bucket", "s3_bucket"):
                candidate = _bucket_name_from_uri(str(section.get(bucket_key) or ""))
                if candidate:
                    saved_bucket = candidate
                    break
            if saved_bucket:
                break
        if saved_bucket == bucket_name:
            for section_key in _STORAGE_SECTION_KEYS:
                section = data.get(section_key)
                if not isinstance(section, dict):
                    continue
                section = dict(section)
                for key in _STORAGE_SECRET_KEYS:
                    if section.get(key) not in (None, ""):
                        section.pop(key, None)
                        removed.append(key)
                if section:
                    data[section_key] = section
                else:
                    data.pop(section_key, None)

        # The setup journal is durable ownership evidence, but a provider-verified
        # bucket deletion must retire that bucket's exact entry.  Preserve the
        # service-account/access-key records until their separately guarded IAM
        # teardown; once those records are gone too, no operational setup state
        # should survive merely because the completed transaction had metadata.
        setup = data.get("storage_setup")
        projects = setup.get("projects") if isinstance(setup, dict) else None
        if isinstance(setup, dict) and isinstance(projects, dict):
            for project_id, project_record in list(projects.items()):
                resources = (
                    project_record.get("resources")
                    if isinstance(project_record, dict)
                    else None
                )
                bucket = (
                    resources.get("bucket") if isinstance(resources, dict) else None
                )
                if not (
                    isinstance(project_record, dict)
                    and isinstance(resources, dict)
                    and isinstance(bucket, dict)
                    and str(bucket.get("name", "") or "").strip() == bucket_name
                    and str(bucket.get("project_id", "") or "").strip()
                    == str(project_id)
                ):
                    continue
                resources = dict(resources)
                resources.pop("bucket", None)
                if resources:
                    project_record["resources"] = resources
                    projects[project_id] = project_record
                else:
                    projects.pop(project_id, None)
            if projects:
                setup["projects"] = projects
                data["storage_setup"] = setup
            else:
                data.pop("storage_setup", None)
        return data

    # Never prune the nebius section here. Even an unproven legacy ID is useful
    # evidence for an operator, while still being categorically insufficient for
    # NPA to delete it. Coupling IAM evidence to bucket cleanup was what made the
    # documented bucket -> service-account order unverifiable after step one.

    if not CREDENTIALS_PATH.exists():
        return
    update_private_yaml(CREDENTIALS_PATH, prune)
    if not removed:
        return
    unique_removed = list(dict.fromkeys(removed))
    typer.echo(
        f"Removed the saved S3 {', '.join(unique_removed)} for {bucket_name} from "
        f"{CREDENTIALS_PATH} (they pointed at a deleted bucket). Re-run `npa configure` "
        "to provision new storage."
    )
