"""IAM leftovers for ``npa agent destroy``.

Extracted from the ``npa.cli.agent`` monolith (kept under a size ratchet). Agent
deploy creates a long-lived ``npa-agent`` service account and an access key for
the VM (see ``npa.clients.nebius.bootstrap_agent_environment``); destroy only ever
removed the VM and its Terraform stack, so "destroyed: <project>/<agent>" left
credentials that outlive the thing they were made for.

The service account is shared by every agent in the project, so it is only
removable once the last agent record is gone — this module reports what remains
either way, and deletes it when the caller opts in.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

StatusFn = Callable[[str], None]


class AgentIAMCleanupError(RuntimeError):
    """Exact agent infrastructure is absent, but owned IAM did not converge."""


def _agent_iam_records() -> tuple[dict[str, Any], Any]:
    """Load the owner-only agent IAM journal and return it with its path."""

    from npa.clients.credentials import (
        CREDENTIALS_PATH,
        _read_credentials_document,
        _validate_private_destination,
    )

    _validate_private_destination(CREDENTIALS_PATH)
    if not CREDENTIALS_PATH.exists():
        return {}, CREDENTIALS_PATH
    return _read_credentials_document(CREDENTIALS_PATH), CREDENTIALS_PATH


def _locked_journal_update(function):
    """Use the shared credential-store lock for every read/modify/write."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        from npa.clients.credentials import CREDENTIALS_PATH, _private_store_lock

        with _private_store_lock(CREDENTIALS_PATH):
            return function(*args, **kwargs)

    return wrapped


def preflight_agent_iam_journal() -> None:
    """Prove durable, structurally valid creation evidence before provider changes."""
    from npa.clients.credentials import CREDENTIALS_PATH, preflight_private_yaml_store

    preflight_private_yaml_store(CREDENTIALS_PATH)
    data, _path = _agent_iam_records()
    root = data.get("agent_iam", {})
    if not isinstance(root, dict) or root.get("version", 1) != 1:
        raise AgentIAMCleanupError("agent IAM ownership journal is malformed")
    projects = root.get("projects", {})
    if not isinstance(projects, dict):
        raise AgentIAMCleanupError("agent IAM project journal is malformed")
    for project_id, record in projects.items():
        if (
            not isinstance(project_id, str)
            or not project_id
            or not isinstance(record, dict)
        ):
            raise AgentIAMCleanupError("agent IAM project journal is malformed")
        resources = record.get("resources", {})
        if not isinstance(resources, dict):
            raise AgentIAMCleanupError("agent IAM resource journal is malformed")
        for kind, entries in resources.items():
            if kind not in {
                "service_account",
                "access_keys",
                "agent_group",
                "agent_permit",
                "agent_membership",
            } or not isinstance(entries, dict):
                raise AgentIAMCleanupError("agent IAM resource journal is malformed")
            rows = (
                [(entries.get("id"), entries)]
                if kind == "service_account"
                else entries.items()
            )
            for identity, metadata in rows:
                if (
                    not isinstance(identity, str)
                    or not identity
                    or not isinstance(metadata, dict)
                    or metadata.get("id") != identity
                    or metadata.get("project_id") != project_id
                ):
                    raise AgentIAMCleanupError(
                        "agent IAM creation identity is malformed"
                    )


@_locked_journal_update
def record_agent_iam_resource(
    project_id: str, kind: str, metadata: dict[str, str], *, status: str = "in_progress"
) -> None:
    """Atomically record exact agent IAM creation metadata."""

    from datetime import datetime, timezone

    from npa.clients.credentials import write_private_yaml

    data, path = _agent_iam_records()

    def mapping(parent: dict, key: str) -> dict:
        value = parent.get(key, {})
        if not isinstance(value, dict):
            raise AgentIAMCleanupError("agent IAM ownership journal is malformed")
        return dict(value)

    root = mapping(data, "agent_iam")
    projects = mapping(root, "projects")
    record = mapping(projects, project_id)
    resources = mapping(record, "resources")
    clean = {key: str(value or "").strip() for key, value in metadata.items() if value}
    clean.update(
        {
            "created_by": "npa",
            "project_id": project_id,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
    )
    if kind == "access_key":
        keys = resources.get("access_keys")
        keys = dict(keys) if isinstance(keys, dict) else {}
        keys[clean["id"]] = clean
        resources["access_keys"] = keys
    elif kind == "service_account":
        resources[kind] = clean
    elif kind in {"agent_group", "agent_permit", "agent_membership"}:
        if (
            clean.get("ownership_source") != "provider-create-response"
            or not all(
                clean.get(key)
                for key in (
                    "id",
                    "tenant_id",
                    "service_account_id",
                    "group_name",
                    "role",
                )
            )
            or kind != "agent_group"
            and not clean.get("group_id")
        ):
            raise ValueError(
                "agent project binding requires exact provider creation evidence"
            )
        entries = mapping(resources, kind)
        if clean["id"] in entries and (
            not isinstance(entries[clean["id"]], dict)
            or entries[clean["id"]].get("project_id") != project_id
        ):
            raise ValueError("agent IAM creation conflicts with another project")
        entries[clean["id"]] = clean
        resources[kind] = entries
    else:
        raise ValueError(f"unsupported agent IAM resource kind: {kind}")
    record.update({"status": status, "resources": resources})
    projects[project_id] = record
    root.update({"version": 1, "projects": projects})
    data["agent_iam"] = root
    write_private_yaml(path, data)


@_locked_journal_update
def mark_agent_iam_status(project_id: str, status: str) -> None:
    from npa.clients.credentials import write_private_yaml

    data, path = _agent_iam_records()
    root = data.get("agent_iam")
    if not isinstance(root, dict):
        return
    projects = root.get("projects")
    if not isinstance(projects, dict) or not isinstance(projects.get(project_id), dict):
        return
    projects[project_id]["status"] = status
    write_private_yaml(path, data)


def agent_iam_owned(project_id: str, account_id: str) -> bool:
    data, _path = _agent_iam_records()
    root = data.get("agent_iam")
    projects = root.get("projects") if isinstance(root, dict) else None
    record = projects.get(project_id) if isinstance(projects, dict) else None
    resources = record.get("resources") if isinstance(record, dict) else None
    account = resources.get("service_account") if isinstance(resources, dict) else None
    return bool(
        isinstance(account, dict)
        and account.get("created_by") == "npa"
        and account.get("project_id") == project_id
        and account.get("name") == "npa-agent"
        and account.get("id") == account_id
    )


@_locked_journal_update
def clear_agent_iam_record(project_id: str, account_id: str) -> bool:
    """Remove the journal only when it names the exact deleted account."""

    from npa.clients.credentials import write_private_yaml

    data, path = _agent_iam_records()
    root = data.get("agent_iam")
    projects = root.get("projects") if isinstance(root, dict) else None
    record = projects.get(project_id) if isinstance(projects, dict) else None
    resources = record.get("resources") if isinstance(record, dict) else None
    account = resources.get("service_account") if isinstance(resources, dict) else None
    if (
        not isinstance(root, dict)
        or not isinstance(projects, dict)
        or not isinstance(account, dict)
        or account.get("id") != account_id
    ):
        return False
    # Account absence does not prove project-parented access keys or permission
    # objects are absent. Keep every unverified receipt for exact reconciliation.
    bindings = {
        key: value
        for key, value in resources.items()
        if key in {"access_keys", "agent_group", "agent_permit", "agent_membership"}
        and value
    }
    if bindings:
        if bindings.get("access_keys"):
            bindings["service_account"] = account
        record.update(resources=bindings, status="partial")
        projects[project_id] = record
    else:
        projects.pop(project_id, None)
    if projects:
        root["projects"] = projects
        data["agent_iam"] = root
    else:
        data.pop("agent_iam", None)
    write_private_yaml(path, data)
    return True


@_locked_journal_update
def remove_agent_iam_resource(project_id: str, kind: str, resource_id: str) -> bool:
    """Forget one conclusively removed creation; return whether resources remain."""

    from npa.clients.credentials import write_private_yaml

    data, path = _agent_iam_records()
    root = data.get("agent_iam")
    projects = root.get("projects") if isinstance(root, dict) else None
    record = projects.get(project_id) if isinstance(projects, dict) else None
    resources = record.get("resources") if isinstance(record, dict) else None
    if (
        not isinstance(root, dict)
        or not isinstance(projects, dict)
        or not isinstance(record, dict)
        or not isinstance(resources, dict)
    ):
        return False
    resources = dict(resources)
    if kind == "access_key":
        keys = resources.get("access_keys")
        keys = dict(keys) if isinstance(keys, dict) else {}
        saved = keys.get(resource_id)
        if isinstance(saved, dict) and saved.get("project_id") == project_id:
            keys.pop(resource_id, None)
        if keys:
            resources["access_keys"] = keys
        else:
            resources.pop("access_keys", None)
    elif kind == "service_account":
        saved = resources.get("service_account")
        if isinstance(saved, dict) and saved.get("id") == resource_id:
            resources.pop("service_account", None)
    elif kind in {"agent_group", "agent_permit", "agent_membership"}:
        entries = resources.get(kind)
        entries = dict(entries) if isinstance(entries, dict) else {}
        saved = entries.get(resource_id)
        if isinstance(saved, dict) and saved.get("project_id") == project_id:
            entries.pop(resource_id, None)
        if entries:
            resources[kind] = entries
        else:
            resources.pop(kind, None)
    else:
        raise ValueError(f"unsupported agent IAM resource kind: {kind}")
    if resources:
        record["resources"] = resources
        projects[project_id] = record
    else:
        projects.pop(project_id, None)
    if projects:
        root["projects"] = projects
        data["agent_iam"] = root
    else:
        data.pop("agent_iam", None)
    write_private_yaml(path, data)
    return bool(resources)


def agent_iam_binding_resources(
    project_id: str,
) -> dict[str, dict[str, dict[str, str]]]:
    """Return durable binding creation records without inferring ownership."""
    data, _path = _agent_iam_records()
    root = data.get("agent_iam", {})
    if not isinstance(root, dict):
        raise AgentIAMCleanupError("agent IAM journal is malformed")
    projects = root.get("projects", {})
    if not isinstance(projects, dict):
        raise AgentIAMCleanupError("agent IAM project journal is malformed")
    record = projects.get(project_id, {})
    if not isinstance(record, dict) or not isinstance(
        record.get("resources", {}), dict
    ):
        raise AgentIAMCleanupError("agent IAM resource journal is malformed")
    return {
        kind: entries
        for kind, entries in record.get("resources", {}).items()
        if kind in {"agent_group", "agent_permit", "agent_membership"}
    }


def _account_iam_leftovers(project_id: str) -> dict[str, Any]:
    """Return the ``npa-agent`` service account and its access keys, if any.

    Provider inventory failures are explicit and block IAM deletion. An unreadable
    inventory must never look like proof that no peer agent depends on the account.
    """
    try:
        from npa.clients.nebius import (
            AGENT_SERVICE_ACCOUNT_NAME,
            get_service_account_id_by_name,
            list_access_keys_for_service_account,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced as unresolved inventory
        return {
            "service_account_id": "",
            "service_account_name": "",
            "access_keys": [],
            "inventory_verified": False,
            "inventory_error": str(exc),
            "dependents": [],
        }

    if not project_id:
        return {
            "service_account_id": "",
            "service_account_name": AGENT_SERVICE_ACCOUNT_NAME,
            "access_keys": [],
            "inventory_verified": False,
            "inventory_error": "exact project ID is missing",
            "dependents": [],
        }
    try:
        sa_id = (
            get_service_account_id_by_name(
                project_id, AGENT_SERVICE_ACCOUNT_NAME, strict=True
            )
            or ""
        )
    except Exception as exc:  # noqa: BLE001 - fail closed and report exact blocker
        return {
            "project_id": project_id,
            "service_account_id": "",
            "service_account_name": AGENT_SERVICE_ACCOUNT_NAME,
            "access_keys": [],
            "inventory_verified": False,
            "inventory_error": str(exc),
            "dependents": [],
        }
    keys: list[dict[str, str]] = []
    if sa_id:
        try:
            keys = list_access_keys_for_service_account(project_id, sa_id, strict=True)
        except Exception as exc:  # noqa: BLE001 - fail closed and report exact blocker
            return {
                "project_id": project_id,
                "service_account_id": sa_id,
                "service_account_name": AGENT_SERVICE_ACCOUNT_NAME,
                "access_keys": [],
                "owned_by_npa": agent_iam_owned(project_id, sa_id),
                "inventory_verified": False,
                "inventory_error": str(exc),
                "dependents": [],
            }
        try:
            dependents = _provider_agent_dependents(project_id, sa_id)
        except Exception as exc:  # noqa: BLE001 - exact receipt fallback below
            proof, proof_error = _receipt_proves_agent_graphs_absent(project_id, sa_id)
            if not proof:
                return {
                    "project_id": project_id,
                    "service_account_id": sa_id,
                    "service_account_name": AGENT_SERVICE_ACCOUNT_NAME,
                    "access_keys": keys,
                    "owned_by_npa": agent_iam_owned(project_id, sa_id),
                    "inventory_verified": False,
                    "inventory_error": f"{exc}; exact receipt fallback: {proof_error}",
                    "dependents": [],
                }
            dependents = []
    else:
        dependents = []
    return {
        "project_id": project_id,
        "service_account_id": sa_id,
        "service_account_name": AGENT_SERVICE_ACCOUNT_NAME,
        "access_keys": keys,
        "owned_by_npa": agent_iam_owned(project_id, sa_id),
        "inventory_verified": True,
        "inventory_error": "",
        "dependents": dependents,
    }


def _verify_access_key_absent(key_id: str) -> None:
    """Inspect only the allowed ID scalar, never a secret-bearing key response."""
    import re

    from npa.clients.nebius import (
        NebiusError,
        _access_key_metadata_scalar,
        is_permission_denied,
    )

    try:
        _access_key_metadata_scalar(key_id, "id", optional=False, identifier=True)
    except NebiusError as exc:
        # Missing JSONPath fields can say "not found" too. Only the provider's
        # explicit status token proves that the immutable resource is absent.
        if re.search(r"\bNotFound\b", str(exc)) and not is_permission_denied(str(exc)):
            return
        raise
    raise NebiusError("access-key deletion is not verified absent")


def _reconcile_absent_access_keys(project_id: str) -> None:
    """Forget only exact owned keys independently proven absent; never delete here."""
    for key_id, record in _recorded_access_keys(project_id).items():
        if (
            not isinstance(record, dict)
            or record.get("id") != key_id
            or record.get("project_id") != project_id
            or record.get("created_by") != "npa"
        ):
            raise AgentIAMCleanupError("access-key creation ownership is unresolved")
        _verify_access_key_absent(key_id)
        remove_agent_iam_resource(project_id, "access_key", key_id)


def _recorded_access_keys(project_id: str) -> dict:
    data, _path = _agent_iam_records()
    root = data.get("agent_iam", {})
    projects = root.get("projects", {}) if isinstance(root, dict) else {}
    record = projects.get(project_id, {}) if isinstance(projects, dict) else {}
    resources = record.get("resources", {}) if isinstance(record, dict) else {}
    keys = resources.get("access_keys", {}) if isinstance(resources, dict) else {}
    if not isinstance(keys, dict):
        raise AgentIAMCleanupError("agent access-key creation journal is malformed")
    return keys


def _recorded_owned_account_id(project_id: str) -> str:
    data, _path = _agent_iam_records()
    root = data.get("agent_iam", {})
    projects = root.get("projects", {}) if isinstance(root, dict) else {}
    record = projects.get(project_id, {}) if isinstance(projects, dict) else {}
    resources = record.get("resources", {}) if isinstance(record, dict) else {}
    account = (
        resources.get("service_account", {}) if isinstance(resources, dict) else {}
    )
    identity = account.get("id", "") if isinstance(account, dict) else ""
    return (
        identity
        if isinstance(identity, str) and agent_iam_owned(project_id, identity)
        else ""
    )


def agent_iam_leftovers(project_id: str) -> dict[str, Any]:
    """Include owned project bindings, even after the account itself is gone."""
    try:
        leftovers = _account_iam_leftovers(project_id)
    except Exception:  # noqa: BLE001 - unreadable creation evidence must never permit cleanup
        return {
            "project_id": project_id,
            "inventory_verified": False,
            "inventory_error": "agent account ownership inventory is unresolved",
        }
    if not leftovers.get("inventory_verified"):
        return leftovers
    try:
        bindings = agent_iam_binding_resources(project_id)
        accounts = set()
        recorded_account = _recorded_owned_account_id(project_id)
        if not leftovers.get("service_account_id") and recorded_account:
            from npa.clients.agent_iam_binding import _get

            if _get("service-account", recorded_account) is not None:
                raise AgentIAMCleanupError(
                    "named account absence disagrees with exact identity"
                )
            leftovers["verified_absent_owned_account_id"] = recorded_account
            accounts.add(recorded_account)
        for entries in bindings.values():
            if not isinstance(entries, dict):
                raise AgentIAMCleanupError(
                    "agent binding ownership inventory is malformed"
                )
            for identity, record in entries.items():
                if (
                    not isinstance(record, dict)
                    or record.get("id") != identity
                    or record.get("project_id") != project_id
                    or not isinstance(record.get("service_account_id"), str)
                    or not record["service_account_id"]
                ):
                    raise AgentIAMCleanupError(
                        "agent binding account inventory is malformed"
                    )
                accounts.add(record["service_account_id"])
        dependents = list(leftovers.get("dependents") or [])
        for account in accounts - {leftovers.get("service_account_id")}:
            dependents.extend(_provider_agent_dependents(project_id, account))
        leftovers.update(binding_resources=bindings, dependents=sorted(set(dependents)))
    except Exception:  # noqa: BLE001 - no cleanup when journal or dependency inventory is unreadable
        leftovers.update(
            inventory_verified=False,
            inventory_error="agent binding ownership or dependency inventory is unresolved",
        )
    return leftovers


def _purge_agent_bindings(leftovers: dict[str, Any], on_status: StatusFn) -> list[str]:
    from npa.clients.agent_iam_binding import cleanup_agent_project_binding

    bindings = leftovers.get("binding_resources") or {}
    if not any(bindings.values()):
        return []
    project_id = str(leftovers.get("project_id") or "")
    try:
        if not leftovers.get("inventory_verified") or leftovers.get("dependents"):
            raise AgentIAMCleanupError("agent binding dependency absence is unverified")
        # Recheck every bound account immediately before deleting permissions.
        # A missing named account is not proof that no VM still references it.
        accounts = {
            record["service_account_id"]
            for entries in bindings.values()
            for record in entries.values()
        }
        for account in accounts:
            if _provider_agent_dependents(project_id, account):
                raise AgentIAMCleanupError("agent binding has dependent VMs")
        deleted = cleanup_agent_project_binding(
            project_id,
            bindings,
            on_removed=lambda kind, identity: remove_agent_iam_resource(
                project_id, kind, identity
            ),
        )
    except Exception as exc:  # noqa: BLE001 - preserve account and exact unremoved binding journal
        mark_agent_iam_status(project_id, "partial")
        raise AgentIAMCleanupError(
            "exact agent project binding cleanup remains partial"
        ) from exc
    for kind in deleted:
        on_status(f"Deleted owned agent IAM {kind.removeprefix('agent_')}.")
    return deleted


def _receipt_proves_agent_graphs_absent(
    project_id: str, account_id: str
) -> tuple[bool, str]:
    """Use exact terminal Terraform-graph receipts when broad inventory is invalid.

    This fallback never treats an empty/missing receipt set as absence.  Every
    receipt agent tied to the selected NPA-created service account must have an
    immutable instance ID and a terminal verified delete/absence event.  Those
    events are written only after Terraform destroys the VM, disk, network,
    subnet, security group and public-IP graph and an exact VM get returns
    NotFound.
    """

    from npa.teardown_receipts import list_teardown_receipts

    terminal = {"verified_absent", "verified_deleted", "deleted"}
    candidates: set[tuple[str, str]] = set()
    verified: set[tuple[str, str]] = set()
    for receipt in list_teardown_receipts(project_id=project_id, legacy="exclude"):
        identity = receipt.get("identity")
        identity = identity if isinstance(identity, dict) else {}
        for item in identity.get("agents") or []:
            if not isinstance(item, dict):
                continue
            if str(item.get("service_account_id") or "") != account_id:
                continue
            name = str(item.get("agent_name") or "").strip()
            instance_id = str(item.get("instance_id") or "").strip()
            if name and instance_id:
                candidates.add((name, instance_id))
        for event in receipt.get("events") or []:
            if not isinstance(event, dict) or event.get("phase") != "agent":
                continue
            event_identity = event.get("identity")
            event_identity = event_identity if isinstance(event_identity, dict) else {}
            if str(event_identity.get("service_account_id") or "") != account_id:
                continue
            name = str(
                event_identity.get("agent_name") or event.get("resource") or ""
            ).strip()
            instance_id = str(event_identity.get("instance_id") or "").strip()
            if (
                name
                and instance_id
                and str(event.get("terminal_state") or "").lower() in terminal
                and isinstance(event.get("verification"), dict)
                and event["verification"].get("exact_instance_absent") is True
                and event["verification"].get("terraform_destroy_completed") is True
                and {
                    "compute_instance",
                    "boot_disk",
                    "network",
                    "subnet",
                    "security_group",
                    "public_ip",
                }.issubset(
                    set(event["verification"].get("terraform_dependency_graph") or [])
                )
            ):
                verified.add((name, instance_id))
    if not candidates:
        return False, "no exact agent dependency graph is recorded"
    missing = sorted(candidates - verified)
    if missing:
        return False, "non-terminal exact agent graph(s): " + ", ".join(
            f"{name}/{instance_id}" for name, instance_id in missing
        )
    return True, "all exact receipt-recorded agent dependency graphs are absent"


def _provider_agent_dependents(project_id: str, account_id: str) -> list[str]:
    """List exact provider VMs attached to the shared agent service account."""

    from npa.clients.nebius import NebiusError, _run_json

    payload = _run_json(
        ["compute", "instance", "list", "--parent-id", project_id, "--all"]
    )
    # Successful ProtoJSON omits empty repeated fields. With --all, only an
    # absent/empty terminal token is complete; unknown fields and explicit
    # null/non-list items remain errors, never absence evidence.
    items = payload.get("items", []) if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or set(payload) - {"items", "next_page_token"}
        or not isinstance(items, list)
        or payload.get("next_page_token", "") != ""
    ):
        raise NebiusError(
            "Nebius returned an incomplete or schema-invalid compute inventory"
        )
    dependents: list[str] = []
    seen_ids: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            raise NebiusError("Nebius returned a non-object compute inventory item")
        metadata = item.get("metadata")
        spec = item.get("spec")
        if not isinstance(metadata, dict) or not isinstance(spec, dict):
            raise NebiusError("Nebius returned incomplete compute identity/spec data")
        identity, name = metadata.get("id"), metadata.get("name")
        if (
            metadata.get("parent_id") != project_id
            or not isinstance(identity, str)
            or not identity
            or identity != identity.strip()
            or identity.startswith("-")
            or any(char.isspace() for char in identity)
            or not isinstance(name, str)
            or not name
            or identity in seen_ids
        ):
            raise NebiusError(
                "Nebius returned compute inventory outside the exact project or without unique identity"
            )
        seen_ids.add(identity)
        account = spec.get("account", {})
        if not isinstance(account, dict):
            raise NebiusError("Nebius returned malformed compute account attachment")
        nested = account.get("service_account", {})
        if not isinstance(nested, dict):
            raise NebiusError(
                "Nebius returned malformed compute service-account attachment"
            )
        attached_ids = [
            nested.get("id", ""),
            account.get("service_account_id", ""),
            spec.get("service_account_id", ""),
        ]
        if any(not isinstance(value, str) for value in attached_ids):
            raise NebiusError("Nebius returned malformed compute account identity")
        attached_ids = [value for value in attached_ids if value]
        if any(
            value != value.strip()
            or value.startswith("-")
            or any(char.isspace() for char in value)
            for value in attached_ids
        ):
            raise NebiusError("Nebius returned a non-exact compute account identity")
        if len(set(attached_ids)) > 1:
            raise NebiusError("Nebius returned conflicting compute account identities")
        if account_id in attached_ids:
            dependents.append(f"{name} ({identity})")
    return sorted(dependents)


def purge_agent_iam(leftovers: dict[str, Any], *, on_status: StatusFn) -> list[str]:
    """Delete bindings and keys before a scoped, readback-verified account removal."""
    from npa.clients.nebius import (
        NebiusError,
        delete_access_key,
        is_not_found,
        is_permission_denied,
    )
    from npa.clients.agent_iam_binding import remove_created_agent_account

    project_id = str(leftovers.get("project_id", "") or "")
    sa_id = str(leftovers.get("service_account_id", "") or "")

    def verify_account_unused() -> None:
        if not sa_id:
            return
        try:
            dependents = _provider_agent_dependents(project_id, sa_id)
        except NebiusError:
            # Legacy accounts retain the existing exact terminal graph receipt
            # fallback; project bindings require complete live inventory.
            proof, _error = _receipt_proves_agent_graphs_absent(project_id, sa_id)
            if not proof or any((leftovers.get("binding_resources") or {}).values()):
                raise
            dependents = []
        if dependents:
            raise NebiusError("agent account has dependent VMs")

    # Storage keys are shared by every VM attached to this account. Recheck
    # dependencies before the first destructive action, not only account removal.
    try:
        verify_account_unused()
    except NebiusError as exc:
        mark_agent_iam_status(project_id, "partial")
        raise AgentIAMCleanupError(
            "exact agent IAM dependency inventory remains unresolved"
        ) from exc

    deleted = _purge_agent_bindings(leftovers, on_status)
    failures: list[str] = []
    for key in leftovers.get("access_keys") or []:
        key_id = str((key or {}).get("id", "") or "")
        if not key_id:
            continue
        try:
            delete_access_key(key_id)
            _verify_access_key_absent(key_id)
        except NebiusError as exc:
            if is_not_found(str(exc)) and not is_permission_denied(str(exc)):
                try:
                    _verify_access_key_absent(key_id)
                except NebiusError:
                    failures.append("access-key absence remains unverified")
                    continue
                deleted.append(f"access key {key_id}")
                remove_agent_iam_resource(project_id, "access_key", key_id)
                continue
            on_status(f"Warning: could not delete access key {key_id}: {exc}")
            failures.append(f"access key {key_id}: {exc}")
            continue
        deleted.append(f"access key {key_id}")
        remove_agent_iam_resource(project_id, "access_key", key_id)
    if sa_id and not failures:
        try:
            verify_account_unused()
            remove_created_agent_account(project_id, "", sa_id)
            clear_agent_iam_record(project_id, sa_id)
            if agent_iam_owned(project_id, sa_id) and _recorded_access_keys(project_id):
                raise NebiusError(
                    "exact access-key creation receipts still require absence verification"
                )
            deleted.append(
                f"service account {leftovers.get('service_account_name') or sa_id} ({sa_id})"
            )
        except (NebiusError, OSError) as exc:
            mark_agent_iam_status(project_id, "partial")
            on_status(
                "Warning: exact agent account or access-key cleanup remains unresolved."
            )
            failures.append(f"service account {sa_id}: {exc}")
    for item in deleted:
        on_status(f"Deleted {item}.")
    if failures:
        raise AgentIAMCleanupError(
            "exact agent IAM cleanup remains partial: " + "; ".join(failures)
        )
    return deleted


def report_destroyed_agent_iam(
    project: str, name: str, *, record: dict[str, Any] | None, purge: bool = True
) -> None:
    """Surface the npa-agent service account/keys that outlive the destroyed VM."""
    import typer

    from npa.cli.agent import resolve_project_agents
    from npa.clients.config import resolve_environment

    project_id = str((record or {}).get("project_id", "") or "")
    if not project_id:
        saved_env = resolve_environment(project)
        project_id = str(getattr(saved_env, "project_id", "") or "")
    remaining = len([key for key in resolve_project_agents(project) if key != name])
    report_agent_iam(
        project_id=project_id,
        remaining_agents=remaining,
        purge=purge,
        on_status=lambda message: typer.echo(f"  {message}", err=True),
        strict=purge,
    )


def report_agent_iam(
    *,
    project_id: str,
    remaining_agents: int,
    purge: bool,
    on_status: StatusFn,
    strict: bool = False,
) -> list[str]:
    """Report (and optionally delete) the IAM the agent VM left behind.

    Returns the deleted-resource descriptions, so a caller can tell whether the
    teardown was complete.
    """
    leftovers = agent_iam_leftovers(project_id)
    if not leftovers.get("inventory_verified"):
        on_status(
            "Keeping the npa-agent service account: exact provider dependency "
            "inventory is unresolved ("
            + str(leftovers.get("inventory_error") or "unknown provider error")
            + "). No IAM resources were deleted."
        )
        if strict and purge:
            raise AgentIAMCleanupError(
                "exact provider dependency inventory for agent IAM is unresolved: "
                + str(leftovers.get("inventory_error") or "unknown provider error")
            )
        return []
    has_bindings = any((leftovers.get("binding_resources") or {}).values())
    absent_account = leftovers.get("verified_absent_owned_account_id")
    if (
        not leftovers.get("service_account_id")
        and not has_bindings
        and not absent_account
    ):
        return []
    provider_dependents = list(leftovers.get("dependents") or [])
    last_agent = remaining_agents == 0 and not provider_dependents
    owned = bool(leftovers.get("owned_by_npa"))
    if purge and last_agent and owned:
        return purge_agent_iam(leftovers, on_status=on_status)
    deleted_bindings: list[str] = []
    if purge and last_agent and not owned:
        deleted_bindings = _purge_agent_bindings(leftovers, on_status)
        if not leftovers.get("service_account_id"):
            if absent_account:
                from npa.clients.agent_iam_binding import _get
                from npa.clients.nebius import NebiusError

                if (
                    not agent_iam_owned(project_id, absent_account)
                    or _provider_agent_dependents(project_id, absent_account)
                    or _get("service-account", absent_account) is not None
                ):
                    raise AgentIAMCleanupError(
                        "owned account absence remains unresolved"
                    )
                try:
                    _reconcile_absent_access_keys(project_id)
                except (NebiusError, RuntimeError, OSError) as exc:
                    mark_agent_iam_status(project_id, "partial")
                    raise AgentIAMCleanupError(
                        "owned account is absent but exact access-key receipts remain unverified"
                    ) from exc
                clear_agent_iam_record(project_id, absent_account)
                on_status(
                    "Reconciled the exact owned service account already absent at the provider."
                )
            return deleted_bindings
        on_status(
            "Keeping the npa-agent service account: its familiar name is not proof "
            "that NPA created it. The account and its access keys were preserved."
        )
        if strict:
            raise AgentIAMCleanupError(
                "agent IAM remains because exact NPA creation ownership is unproven"
            )
    if purge and not last_agent:
        if provider_dependents:
            on_status(
                "Keeping the npa-agent service account: exact provider inventory "
                "shows dependent VM(s): " + ", ".join(provider_dependents) + "."
            )
        else:
            on_status(
                "Keeping the npa-agent service account: "
                f"{remaining_agents} other local agent record(s) still use it."
            )
        if strict and provider_dependents:
            raise AgentIAMCleanupError(
                "agent IAM remains because exact provider inventory reports dependent VM(s)"
            )
    for line in format_iam_leftovers(
        leftovers, project_id=project_id, last_agent=last_agent
    ):
        on_status(line)
    return deleted_bindings


def format_iam_leftovers(
    leftovers: dict[str, Any], *, project_id: str, last_agent: bool
) -> list[str]:
    """Return report lines naming what destroy did not delete, with NPA guidance."""
    sa_id = str(leftovers.get("service_account_id", "") or "")
    if not sa_id:
        return []
    name = leftovers.get("service_account_name") or "npa-agent"
    keys = [
        str((key or {}).get("id", "") or "")
        for key in leftovers.get("access_keys") or []
    ]
    keys = [key for key in keys if key]
    lines = [
        f"Left in place: service account {name} ({sa_id})"
        + (f" and {len(keys)} access key(s)" if keys else "")
        + ".",
    ]
    if last_agent and leftovers.get("owned_by_npa"):
        lines.append(
            "  This project has no agents left, so nothing needs it. Re-run the "
            "same NPA teardown with explicit IAM cleanup: `npa agent destroy "
            "--project <alias> --name <name> --purge-iam --yes`."
        )
    elif last_agent:
        lines.append(
            "  NPA has no creation provenance for this account. Verify ownership "
            "and dependencies without deleting it; do not bypass NPA's ownership guard."
        )
    else:
        lines.append(
            "  Other agents in this project still use it, so it was kept. "
            f"`npa agent status --project <alias>` lists them (project {project_id})."
        )
    return lines
