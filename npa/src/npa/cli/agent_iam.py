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

from collections.abc import Mapping
from typing import Any, Callable

StatusFn = Callable[[str], None]


class AgentIAMCleanupError(RuntimeError):
    """Exact agent infrastructure is absent, but owned IAM did not converge."""


class AgentIAMRecordError(RuntimeError):
    """The owner-only IAM journal is present but invalid or unreadable."""


def _safe_error(exc: BaseException) -> str:
    from npa.clients.nebius import redact_nebius_output

    return redact_nebius_output(str(exc))[:1000]


def _agent_iam_records() -> tuple[dict[str, Any], Any]:
    """Load the owner-only agent IAM journal and return it with its path."""

    import yaml

    from npa.clients.credentials import CREDENTIALS_PATH

    if not CREDENTIALS_PATH.exists():
        return {}, CREDENTIALS_PATH
    try:
        loaded = yaml.safe_load(CREDENTIALS_PATH.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise AgentIAMRecordError("agent IAM journal is unreadable") from exc
    data = {} if loaded is None else loaded
    if not isinstance(data, dict):
        raise AgentIAMRecordError("credentials root is not an object")
    if "agent_iam" not in data:
        return data, CREDENTIALS_PATH
    root = data["agent_iam"]
    if not isinstance(root, dict) or type(root.get("version")) is not int:
        raise AgentIAMRecordError("agent IAM journal root/schema is invalid")
    if (
        root["version"] != 1
        or "projects" not in root
        or not isinstance(root["projects"], dict)
    ):
        raise AgentIAMRecordError("agent IAM journal version/projects are invalid")
    for project_id, record in root["projects"].items():
        if not isinstance(project_id, str) or not project_id.strip():
            raise AgentIAMRecordError("agent IAM journal has an invalid project key")
        if (
            not isinstance(record, dict)
            or "resources" not in record
            or not isinstance(record["resources"], dict)
        ):
            raise AgentIAMRecordError(
                f"agent IAM journal for project {project_id!r} is incomplete"
            )
        resources = record.get("resources", {})
        account = resources.get("service_account")
        if account is not None and (
            not isinstance(account, dict)
            or account.get("created_by") != "npa"
            or account.get("project_id") != project_id
            or not isinstance(account.get("id"), str)
            or not account.get("id")
            or not isinstance(account.get("name"), str)
            or not account.get("name")
        ):
            raise AgentIAMRecordError(
                f"agent IAM service-account record for {project_id!r} is invalid"
            )
        keys = resources.get("access_keys", {})
        if not isinstance(keys, dict):
            raise AgentIAMRecordError(
                f"agent IAM access-key records for {project_id!r} are invalid"
            )
        for key_id, key in keys.items():
            if (
                not isinstance(key_id, str)
                or not key_id
                or not isinstance(key, dict)
                or key.get("id") != key_id
                or key.get("project_id") != project_id
                or key.get("created_by") != "npa"
            ):
                raise AgentIAMRecordError(
                    f"agent IAM access-key record for {project_id!r} is invalid"
                )
    return data, CREDENTIALS_PATH


def _owned_agent_account(project_id: str) -> dict[str, str] | None:
    data, _path = _agent_iam_records()
    root = data.get("agent_iam")
    projects = root.get("projects") if isinstance(root, dict) else None
    record = projects.get(project_id) if isinstance(projects, dict) else None
    resources = record.get("resources") if isinstance(record, dict) else None
    account = resources.get("service_account") if isinstance(resources, dict) else None
    return dict(account) if isinstance(account, dict) else None


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
    account = _owned_agent_account(project_id)
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
            NebiusError,
            get_service_account_identity,
            get_service_account_id_by_name,
            list_access_keys_for_service_account,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced as unresolved inventory
        return {
            "service_account_id": "",
            "service_account_name": "",
            "access_keys": [],
            "inventory_verified": False,
            "inventory_error": _safe_error(exc),
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
        owned_account = _owned_agent_account(project_id)
    except AgentIAMRecordError as exc:
        return {
            "project_id": project_id,
            "service_account_id": "",
            "service_account_name": AGENT_SERVICE_ACCOUNT_NAME,
            "access_keys": [],
            "inventory_verified": False,
            "inventory_error": _safe_error(exc),
            "dependents": [],
        }
    owned_id = str((owned_account or {}).get("id") or "")
    try:
        named_id = (
            get_service_account_id_by_name(
                project_id, AGENT_SERVICE_ACCOUNT_NAME, strict=True
            )
            or ""
        )
        exact_identity = (
            get_service_account_identity(
                owned_id,
                project_id=project_id,
            )
            if owned_id
            else None
        )
        if not isinstance(named_id, str):
            raise NebiusError(
                "Nebius returned an invalid named service-account identity"
            )
        named_id = named_id.strip()
        if exact_identity is not None and (
            not isinstance(getattr(exact_identity, "account_id", None), str)
            or exact_identity.account_id != owned_id
            or not isinstance(getattr(exact_identity, "name", None), str)
            or not exact_identity.name.strip()
            or not isinstance(getattr(exact_identity, "project_id", None), str)
            or exact_identity.project_id != project_id
        ):
            raise NebiusError(
                "Nebius returned an invalid exact service-account identity"
            )
    except Exception as exc:  # noqa: BLE001 - fail closed and report exact blocker
        return {
            "project_id": project_id,
            "service_account_id": owned_id,
            "service_account_name": AGENT_SERVICE_ACCOUNT_NAME,
            "access_keys": [],
            "owned_by_npa": bool(owned_id),
            "inventory_verified": False,
            "inventory_error": _safe_error(exc),
            "dependents": [],
        }
    if exact_identity is not None:
        if exact_identity.name != AGENT_SERVICE_ACCOUNT_NAME:
            return {
                "project_id": project_id,
                "service_account_id": owned_id,
                "service_account_name": exact_identity.name,
                "access_keys": [],
                "owned_by_npa": True,
                "inventory_verified": False,
                "inventory_error": (
                    "owned exact service account is present under a different name"
                ),
                "dependents": [],
            }
        if named_id != owned_id:
            return {
                "project_id": project_id,
                "service_account_id": owned_id,
                "service_account_name": AGENT_SERVICE_ACCOUNT_NAME,
                "access_keys": [],
                "owned_by_npa": True,
                "inventory_verified": False,
                "inventory_error": (
                    "named and owned service-account identities conflict"
                ),
                "dependents": [],
            }
        sa_id = owned_id
    else:
        # Exact owned absence does not make a same-name replacement NPA-owned.
        sa_id = named_id
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
                "owned_by_npa": bool(owned_id and sa_id == owned_id),
                "inventory_verified": False,
                "inventory_error": _safe_error(exc),
                "dependents": [],
            }
        if not isinstance(keys, list):
            return {
                "project_id": project_id,
                "service_account_id": sa_id,
                "service_account_name": AGENT_SERVICE_ACCOUNT_NAME,
                "access_keys": [],
                "owned_by_npa": bool(owned_id and sa_id == owned_id),
                "inventory_verified": False,
                "inventory_error": "access-key inventory is schema-invalid",
                "dependents": [],
            }
        key_ids: list[str] = []
        for key in keys:
            key_id = key.get("id") if isinstance(key, Mapping) else None
            if not isinstance(key_id, str) or not key_id.strip():
                return {
                    "project_id": project_id,
                    "service_account_id": sa_id,
                    "service_account_name": AGENT_SERVICE_ACCOUNT_NAME,
                    "access_keys": [],
                    "owned_by_npa": bool(owned_id and sa_id == owned_id),
                    "inventory_verified": False,
                    "inventory_error": "access-key inventory is schema-invalid",
                    "dependents": [],
                }
            key_ids.append(key_id.strip())
        if len(key_ids) != len(set(key_ids)):
            return {
                "project_id": project_id,
                "service_account_id": sa_id,
                "service_account_name": AGENT_SERVICE_ACCOUNT_NAME,
                "access_keys": [],
                "owned_by_npa": bool(owned_id and sa_id == owned_id),
                "inventory_verified": False,
                "inventory_error": "access-key inventory has duplicate identities",
                "dependents": [],
            }
        try:
            dependents = _provider_agent_dependents(project_id, sa_id)
        except Exception as exc:  # noqa: BLE001 - current inventory is mandatory
            return {
                "project_id": project_id,
                "service_account_id": sa_id,
                "service_account_name": AGENT_SERVICE_ACCOUNT_NAME,
                "access_keys": keys,
                "owned_by_npa": bool(owned_id and sa_id == owned_id),
                "inventory_verified": False,
                "inventory_error": _safe_error(exc),
                "dependents": [],
            }
    else:
        dependents = []
    return {
        "project_id": project_id,
        "service_account_id": sa_id,
        "verified_absent_owned_id": owned_id if owned_id and not sa_id else "",
        "service_account_name": AGENT_SERVICE_ACCOUNT_NAME,
        "access_keys": keys,
        "owned_by_npa": bool(owned_id and sa_id == owned_id),
        "inventory_verified": True,
        "inventory_error": "",
        "dependents": dependents,
    }


def _provider_agent_dependents(
    project_id: str, account_id: str
) -> list[dict[str, str]]:
    """List exact provider VMs attached to the shared agent service account."""

    from npa.clients.nebius import NebiusError, _run_json

    payload = _run_json(
        ["compute", "instance", "list", "--parent-id", project_id, "--all"]
    )
    items = payload.get("items")
    if not isinstance(items, list):
        raise NebiusError("Nebius returned a schema-invalid compute inventory")
    dependents: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            raise NebiusError("Nebius returned a non-object compute inventory item")
        metadata = item.get("metadata")
        spec = item.get("spec")
        if not isinstance(metadata, dict) or not isinstance(spec, dict):
            raise NebiusError("Nebius returned incomplete compute identity/spec data")
        raw_identity = metadata.get("id")
        raw_name = metadata.get("name")
        if (
            not isinstance(raw_identity, str)
            or not raw_identity.strip()
            or not isinstance(raw_name, str)
            or not raw_name.strip()
        ):
            raise NebiusError("Nebius returned a VM without exact identity")
        identity = raw_identity.strip()
        name = raw_name.strip()
        raw_account = spec.get("account")
        if raw_account is not None and not isinstance(raw_account, dict):
            raise NebiusError("Nebius returned invalid compute account data")
        account = raw_account if isinstance(raw_account, dict) else {}
        raw_nested = account.get("service_account")
        if raw_nested is not None and not isinstance(raw_nested, dict):
            raise NebiusError(
                "Nebius returned invalid nested compute service-account data"
            )
        nested = raw_nested if isinstance(raw_nested, dict) else {}
        candidates = (
            nested.get("id"),
            account.get("service_account_id"),
            spec.get("service_account_id"),
        )
        if any(value is not None and not isinstance(value, str) for value in candidates):
            raise NebiusError(
                "Nebius returned a compute service-account identity with invalid type"
            )
        attached = next(
            (str(value).strip() for value in candidates if str(value or "").strip()),
            "",
        )
        if attached != account_id:
            continue
        dependents.append({"name": name, "instance_id": identity})
    instance_ids = [item["instance_id"] for item in dependents]
    if len(instance_ids) != len(set(instance_ids)):
        raise NebiusError("Nebius returned duplicate compute instance identities")
    return sorted(dependents, key=lambda item: (item["name"], item["instance_id"]))


def purge_agent_iam(leftovers: dict[str, Any], *, on_status: StatusFn) -> list[str]:
    """Delete exact owned IAM and prove absence before retiring its journal."""
    from npa.clients.nebius import (
        NebiusError,
        delete_access_key,
        delete_service_account,
        get_service_account_identity,
        is_not_found,
        list_access_keys_for_service_account,
    )

    project_id = leftovers.get("project_id")
    sa_id = leftovers.get("service_account_id")
    if not isinstance(project_id, str) or not project_id:
        raise AgentIAMCleanupError("exact project ID is required for IAM cleanup")
    if not isinstance(sa_id, str) or not sa_id:
        raise AgentIAMCleanupError(
            "exact service-account ID is required for IAM cleanup"
        )
    raw_keys = leftovers.get("access_keys")
    if not isinstance(raw_keys, list):
        raise AgentIAMCleanupError("access-key inventory is schema-invalid")
    key_ids: list[str] = []
    for key in raw_keys:
        key_id = key.get("id") if isinstance(key, Mapping) else None
        if not isinstance(key_id, str) or not key_id or key_id in key_ids:
            raise AgentIAMCleanupError("access-key inventory has invalid identities")
        key_ids.append(key_id)

    failures: list[str] = []
    for key_id in key_ids:
        try:
            delete_access_key(key_id)
        except NebiusError as exc:
            if not is_not_found(exc):
                failures.append(f"access key {key_id}: {_safe_error(exc)}")
    if failures:
        raise AgentIAMCleanupError(
            "exact agent IAM cleanup remains partial: " + "; ".join(failures)
        )
    try:
        remaining_keys = list_access_keys_for_service_account(
            project_id, sa_id, strict=True
        )
    except Exception as exc:  # noqa: BLE001 - postcheck must fail closed
        raise AgentIAMCleanupError(
            "access-key post-delete verification is unresolved: " + _safe_error(exc)
        ) from exc
    if not isinstance(remaining_keys, list):
        raise AgentIAMCleanupError(
            "access-key post-delete verification returned invalid schema"
        )
    postcheck_ids: list[str] = []
    for item in remaining_keys:
        key_id = item.get("id") if isinstance(item, Mapping) else None
        if not isinstance(key_id, str) or not key_id.strip():
            raise AgentIAMCleanupError(
                "access-key post-delete verification returned invalid schema"
            )
        postcheck_ids.append(key_id.strip())
    if len(postcheck_ids) != len(set(postcheck_ids)):
        raise AgentIAMCleanupError(
            "access-key post-delete verification returned duplicate identities"
        )
    remaining_ids = set(postcheck_ids)
    surviving = sorted(set(key_ids) & remaining_ids)
    if surviving:
        raise AgentIAMCleanupError(
            "provider still reports deleted access key(s): " + ", ".join(surviving)
        )
    try:
        for key_id in key_ids:
            remove_agent_iam_resource(project_id, "access_key", key_id)
    except (AgentIAMRecordError, OSError, ValueError) as exc:
        raise AgentIAMCleanupError(
            "provider access keys are absent, but exact local key ownership "
            "evidence could not be retired: " + _safe_error(exc)
        ) from exc

    try:
        delete_service_account(sa_id)
    except NebiusError as exc:
        if not is_not_found(exc):
            raise AgentIAMCleanupError(
                f"exact service-account deletion failed for {sa_id}: "
                + _safe_error(exc)
            ) from exc
    try:
        remaining_account = get_service_account_identity(
            sa_id,
            project_id=project_id,
        )
    except Exception as exc:  # noqa: BLE001 - postcheck must fail closed
        raise AgentIAMCleanupError(
            "service-account post-delete verification is unresolved: "
            + _safe_error(exc)
        ) from exc
    if remaining_account is not None:
        raise AgentIAMCleanupError(
            f"provider still reports service account {sa_id} after deletion"
        )
    try:
        local_record_cleared = clear_agent_iam_record(project_id, sa_id)
    except (AgentIAMRecordError, OSError, ValueError) as exc:
        raise AgentIAMCleanupError(
            "provider IAM is absent but its exact local ownership record could "
            "not be retired: " + _safe_error(exc)
        ) from exc
    if not local_record_cleared:
        raise AgentIAMCleanupError(
            "provider IAM is absent but its exact local ownership record remains"
        )
    deleted = [*(f"access key {key_id}" for key_id in key_ids)]
    deleted.append(
        f"service account {leftovers.get('service_account_name') or sa_id} ({sa_id})"
    )
    for item in deleted:
        on_status(f"Deleted {item}.")
    return deleted


def report_destroyed_agent_iam(
    project: str, name: str, *, record: dict[str, Any] | None, purge: bool = True
) -> str:
    """Surface the npa-agent service account/keys that outlive the destroyed VM."""
    import typer

    from npa.cli.agent import resolve_project_agents
    from npa.clients.config import resolve_environment

    project_id = str((record or {}).get("project_id", "") or "")
    if not project_id:
        saved_env = resolve_environment(project)
        project_id = str(getattr(saved_env, "project_id", "") or "")
    remaining_records = {
        key: value
        for key, value in resolve_project_agents(project).items()
        if key != name
    }
    remaining = len(remaining_records)
    local_dependent_instance_ids: set[str] | None = None
    if purge:
        from npa.cli.agent_records import AgentRecordState, decode_agent_record

        local_dependent_instance_ids = set()
        for peer_name in remaining_records:
            try:
                decoded = decode_agent_record(project, peer_name)
            except (OSError, RuntimeError, ValueError) as exc:
                raise AgentIAMCleanupError(
                    f"local agent dependency record {peer_name!r} is unreadable"
                ) from exc
            if decoded.state is not AgentRecordState.COMPLETE:
                description = (
                    "empty or absent"
                    if decoded.state in {
                        AgentRecordState.ABSENT,
                        AgentRecordState.INCOMPLETE,
                    }
                    else decoded.state.value
                )
                raise AgentIAMCleanupError(
                    f"local agent dependency record {peer_name!r} is {description}: "
                    f"{decoded.detail}"
                )
            peer_record = decoded.record
            peer_project = peer_record.get("project_id")
            if not isinstance(peer_project, str) or peer_project.strip() != project_id:
                raise AgentIAMCleanupError(
                    f"local agent dependency record {peer_name!r} has no matching "
                    "immutable project ID"
                )
            peer_instance = peer_record.get("instance_id")
            if not isinstance(peer_instance, str) or not peer_instance.strip():
                raise AgentIAMCleanupError(
                    f"local agent dependency record {peer_name!r} has no immutable "
                    "instance ID"
                )
            peer_instance = peer_instance.strip()
            if peer_instance in local_dependent_instance_ids:
                raise AgentIAMCleanupError(
                    "local agent dependency records reuse immutable instance ID "
                    f"{peer_instance!r}"
                )
            local_dependent_instance_ids.add(peer_instance)
    dispositions: list[str] = []
    report_agent_iam(
        project_id=project_id,
        remaining_agents=remaining,
        local_dependent_instance_ids=local_dependent_instance_ids,
        purge=purge,
        on_status=lambda message: typer.echo(f"  {message}", err=True),
        on_disposition=dispositions.append,
        strict=purge,
    )
    return dispositions[-1] if dispositions else "absent"


def report_agent_iam(
    *,
    project_id: str,
    remaining_agents: int,
    local_dependent_instance_ids: set[str] | None = None,
    purge: bool,
    on_status: StatusFn,
    on_disposition: StatusFn | None = None,
    strict: bool = False,
) -> list[str]:
    """Report (and optionally delete) the IAM the agent VM left behind.

    Returns the deleted-resource descriptions, so a caller can tell whether the
    teardown was complete.
    """
    leftovers = agent_iam_leftovers(project_id)
    if not isinstance(leftovers, Mapping):
        inventory_error = "provider inventory returned an invalid result"
    else:
        inventory_error = str(
            leftovers.get("inventory_error") or "unknown provider error"
        )
    if (
        not isinstance(leftovers, Mapping)
        or leftovers.get("inventory_verified") is not True
    ):
        if on_disposition is not None:
            on_disposition("verification_unresolved")
        on_status(
            "Keeping the npa-agent service account: exact provider dependency "
            "inventory is unresolved ("
            + inventory_error
            + "). No IAM resources were deleted."
        )
        if strict and purge:
            raise AgentIAMCleanupError(
                "exact provider dependency inventory for agent IAM is unresolved: "
                + inventory_error
            )
        return []
    service_account_id = leftovers.get("service_account_id")
    if not isinstance(service_account_id, str):
        if on_disposition is not None:
            on_disposition("verification_unresolved")
        if strict and purge:
            raise AgentIAMCleanupError(
                "exact provider agent IAM inventory has an invalid service-account ID"
            )
        return []
    if not service_account_id:
        absent_owned_id = leftovers.get("verified_absent_owned_id", "")
        if purge and absent_owned_id:
            try:
                retired_absent_record = bool(
                    isinstance(absent_owned_id, str)
                    and clear_agent_iam_record(project_id, absent_owned_id)
                )
            except (AgentIAMRecordError, OSError, ValueError):
                retired_absent_record = False
            if not retired_absent_record:
                if on_disposition is not None:
                    on_disposition("verification_unresolved")
                if strict:
                    raise AgentIAMCleanupError(
                        "provider verified exact agent IAM absent, but its local "
                        "ownership record could not be retired"
                    )
                return []
        if on_disposition is not None:
            on_disposition("absent")
        return []
    provider_dependents = leftovers.get("dependents")
    if not isinstance(provider_dependents, list) or not all(
        isinstance(item, Mapping)
        and isinstance(item.get("name"), str)
        and bool(item.get("name"))
        and isinstance(item.get("instance_id"), str)
        and bool(item.get("instance_id"))
        for item in provider_dependents
    ):
        if on_disposition is not None:
            on_disposition("verification_unresolved")
        if strict and purge:
            raise AgentIAMCleanupError(
                "exact provider agent dependency inventory is schema-invalid"
            )
        return []
    provider_dependency_count = len(provider_dependents)
    local_dependency_count = max(remaining_agents, 0)
    provider_dependent_instance_ids: set[str] = set()
    for dependent in provider_dependents:
        instance_id = str(dependent["instance_id"])
        if not instance_id or instance_id in provider_dependent_instance_ids:
            if on_disposition is not None:
                on_disposition("verification_unresolved")
            if strict and purge:
                raise AgentIAMCleanupError(
                    "exact provider agent dependency inventory contains duplicate IDs"
                )
            return []
        provider_dependent_instance_ids.add(instance_id)
    dependency_inventory_agrees = bool(
        provider_dependency_count == local_dependency_count
        and local_dependent_instance_ids is not None
        and (
            len(provider_dependent_instance_ids) == provider_dependency_count
            and local_dependent_instance_ids == provider_dependent_instance_ids
        )
    )
    last_agent = local_dependency_count == 0 and provider_dependency_count == 0
    owned = leftovers.get("owned_by_npa") is True
    if purge and last_agent and owned:
        deleted = purge_agent_iam(leftovers, on_status=on_status)
        if on_disposition is not None:
            on_disposition("deleted")
        return deleted
    if purge and last_agent and not owned:
        if on_disposition is not None:
            on_disposition("retained_unowned")
        on_status(
            "Keeping the npa-agent service account: its familiar name is not proof "
            "that NPA created it. No IAM resources were deleted."
        )
        if strict:
            raise AgentIAMCleanupError(
                "agent IAM remains because exact NPA creation ownership is unproven"
            )
    if purge and not last_agent:
        if provider_dependents and dependency_inventory_agrees:
            if on_disposition is not None:
                on_disposition("retained_shared")
            on_status(
                "Keeping the npa-agent service account: exact provider inventory "
                "and local lifecycle records agree on exact dependent VM(s): "
                + ", ".join(
                    f"{item['name']} ({item['instance_id']})"
                    for item in provider_dependents
                )
                + "."
            )
        elif not provider_dependents:
            if on_disposition is not None:
                on_disposition("retained_local_dependents")
            on_status(
                "Keeping the npa-agent service account: "
                f"{local_dependency_count} other local agent record(s) still use it, "
                "but exact provider inventory reports no dependent VM."
            )
            if strict:
                raise AgentIAMCleanupError(
                    "agent IAM remains on local-only dependency evidence; exact "
                    "provider inventory reports no dependent VM"
                )
        else:
            if on_disposition is not None:
                on_disposition("retained_dependency_disagreement")
            on_status(
                "Keeping the npa-agent service account: provider/local dependency "
                "inventories disagree ("
                f"provider={provider_dependency_count}, local={local_dependency_count}); "
                "provider-dependent VM(s): "
                + ", ".join(
                    f"{item['name']} ({item['instance_id']})"
                    for item in provider_dependents
                )
                + "."
            )
            if strict:
                raise AgentIAMCleanupError(
                    "agent IAM remains because provider/local dependency inventories "
                    "disagree"
                )
        # Only exact provider dependents corroborated by local lifecycle records
        # make intentional IAM retention a successful strict purge disposition.
    elif not purge and on_disposition is not None:
        on_disposition("retained_by_request")
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
