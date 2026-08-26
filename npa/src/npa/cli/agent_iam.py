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

from typing import Any, Callable

StatusFn = Callable[[str], None]


class AgentIAMCleanupError(RuntimeError):
    """Exact agent infrastructure is absent, but owned IAM did not converge."""


def _agent_iam_records() -> tuple[dict[str, Any], Any]:
    """Load the owner-only agent IAM journal and return it with its path."""

    import yaml

    from npa.clients.credentials import CREDENTIALS_PATH

    if not CREDENTIALS_PATH.exists():
        return {}, CREDENTIALS_PATH
    try:
        data = yaml.safe_load(CREDENTIALS_PATH.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        data = {}
    return (data if isinstance(data, dict) else {}), CREDENTIALS_PATH


def record_agent_iam_resource(
    project_id: str, kind: str, metadata: dict[str, str], *, status: str = "in_progress"
) -> None:
    """Atomically record exact agent IAM creation metadata."""

    from datetime import datetime, timezone

    from npa.clients.credentials import write_private_yaml

    data, path = _agent_iam_records()
    root = data.get("agent_iam")
    root = dict(root) if isinstance(root, dict) else {"version": 1}
    projects = root.get("projects")
    projects = dict(projects) if isinstance(projects, dict) else {}
    record = projects.get(project_id)
    record = dict(record) if isinstance(record, dict) else {}
    resources = record.get("resources")
    resources = dict(resources) if isinstance(resources, dict) else {}
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
    else:
        raise ValueError(f"unsupported agent IAM resource kind: {kind}")
    record.update({"status": status, "resources": resources})
    projects[project_id] = record
    root.update({"version": 1, "projects": projects})
    data["agent_iam"] = root
    write_private_yaml(path, data)


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
    projects.pop(project_id, None)
    if projects:
        root["projects"] = projects
        data["agent_iam"] = root
    else:
        data.pop("agent_iam", None)
    write_private_yaml(path, data)
    return True


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


def agent_iam_leftovers(project_id: str) -> dict[str, Any]:
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
    items = payload.get("items")
    if not isinstance(items, list):
        raise NebiusError("Nebius returned a schema-invalid compute inventory")
    dependents: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            raise NebiusError("Nebius returned a non-object compute inventory item")
        metadata = item.get("metadata")
        spec = item.get("spec")
        if not isinstance(metadata, dict) or not isinstance(spec, dict):
            raise NebiusError("Nebius returned incomplete compute identity/spec data")
        account = spec.get("account")
        account = account if isinstance(account, dict) else {}
        nested = account.get("service_account")
        nested = nested if isinstance(nested, dict) else {}
        attached = str(
            nested.get("id")
            or account.get("service_account_id")
            or spec.get("service_account_id")
            or ""
        ).strip()
        if attached != account_id:
            continue
        identity = str(metadata.get("id") or "").strip()
        name = str(metadata.get("name") or "").strip()
        if not identity or not name:
            raise NebiusError("Nebius returned an attached VM without exact identity")
        dependents.append(f"{name} ({identity})")
    return sorted(dependents)


def purge_agent_iam(leftovers: dict[str, Any], *, on_status: StatusFn) -> list[str]:
    """Delete the access keys then the service account. Returns what was deleted."""
    from npa.clients.nebius import (
        NebiusError,
        delete_access_key,
        delete_service_account,
        is_not_found,
    )

    deleted: list[str] = []
    failures: list[str] = []
    for key in leftovers.get("access_keys") or []:
        key_id = str((key or {}).get("id", "") or "")
        if not key_id:
            continue
        try:
            delete_access_key(key_id)
        except NebiusError as exc:
            if is_not_found(str(exc)):
                deleted.append(f"access key {key_id}")
                remove_agent_iam_resource(
                    str(leftovers.get("project_id", "") or ""), "access_key", key_id
                )
                continue
            on_status(f"Warning: could not delete access key {key_id}: {exc}")
            failures.append(f"access key {key_id}: {exc}")
            continue
        deleted.append(f"access key {key_id}")
    sa_id = str(leftovers.get("service_account_id", "") or "")
    if sa_id:
        try:
            delete_service_account(sa_id)
        except NebiusError as exc:
            if is_not_found(str(exc)):
                deleted.append(
                    f"service account {leftovers.get('service_account_name') or sa_id} ({sa_id})"
                )
                clear_agent_iam_record(
                    str(leftovers.get("project_id", "") or ""), sa_id
                )
            else:
                on_status(f"Warning: could not delete service account {sa_id}: {exc}")
                failures.append(f"service account {sa_id}: {exc}")
        else:
            deleted.append(
                f"service account {leftovers.get('service_account_name') or sa_id} ({sa_id})"
            )
            clear_agent_iam_record(str(leftovers.get("project_id", "") or ""), sa_id)
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
    if not leftovers.get("service_account_id"):
        return []
    provider_dependents = list(leftovers.get("dependents") or [])
    last_agent = remaining_agents == 0 and not provider_dependents
    owned = bool(leftovers.get("owned_by_npa"))
    if purge and last_agent and owned:
        return purge_agent_iam(leftovers, on_status=on_status)
    if purge and last_agent and not owned:
        on_status(
            "Keeping the npa-agent service account: its familiar name is not proof "
            "that NPA created it. No IAM resources were deleted."
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
    return []


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
