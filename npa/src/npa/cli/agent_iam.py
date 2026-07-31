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


def agent_iam_leftovers(project_id: str) -> dict[str, Any]:
    """Return the ``npa-agent`` service account and its access keys, if any.

    Best-effort: any lookup failure reports "nothing found" rather than blocking a
    teardown that has already removed the VM.
    """
    try:
        from npa.clients.nebius import (
            AGENT_SERVICE_ACCOUNT_NAME,
            get_service_account_id_by_name,
            list_access_keys_for_service_account,
        )
    except Exception:  # noqa: BLE001 - import-time failure means "cannot check"
        return {"service_account_id": "", "service_account_name": "", "access_keys": []}

    if not project_id:
        return {"service_account_id": "", "service_account_name": AGENT_SERVICE_ACCOUNT_NAME, "access_keys": []}
    try:
        sa_id = get_service_account_id_by_name(project_id, AGENT_SERVICE_ACCOUNT_NAME) or ""
    except Exception:  # noqa: BLE001 - unreadable IAM is not a destroy failure
        sa_id = ""
    keys: list[dict[str, str]] = []
    if sa_id:
        try:
            keys = list_access_keys_for_service_account(project_id, sa_id)
        except Exception:  # noqa: BLE001
            keys = []
    return {
        "service_account_id": sa_id,
        "service_account_name": AGENT_SERVICE_ACCOUNT_NAME,
        "access_keys": keys,
    }


def purge_agent_iam(leftovers: dict[str, Any], *, on_status: StatusFn) -> list[str]:
    """Delete the access keys then the service account. Returns what was deleted."""
    from npa.clients.nebius import NebiusError, delete_access_key, delete_service_account

    deleted: list[str] = []
    for key in leftovers.get("access_keys") or []:
        key_id = str((key or {}).get("id", "") or "")
        if not key_id:
            continue
        try:
            delete_access_key(key_id)
        except NebiusError as exc:
            on_status(f"Warning: could not delete access key {key_id}: {exc}")
            continue
        deleted.append(f"access key {key_id}")
    sa_id = str(leftovers.get("service_account_id", "") or "")
    if sa_id:
        try:
            delete_service_account(sa_id)
        except NebiusError as exc:
            on_status(f"Warning: could not delete service account {sa_id}: {exc}")
        else:
            deleted.append(f"service account {leftovers.get('service_account_name') or sa_id} ({sa_id})")
    for item in deleted:
        on_status(f"Deleted {item}.")
    return deleted


def report_destroyed_agent_iam(
    project: str, name: str, *, record: dict[str, Any] | None, purge: bool
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
    )


def report_agent_iam(
    *,
    project_id: str,
    remaining_agents: int,
    purge: bool,
    on_status: StatusFn,
) -> list[str]:
    """Report (and optionally delete) the IAM the agent VM left behind.

    Returns the deleted-resource descriptions, so a caller can tell whether the
    teardown was complete.
    """
    leftovers = agent_iam_leftovers(project_id)
    if not leftovers.get("service_account_id"):
        return []
    last_agent = remaining_agents == 0
    if purge and last_agent:
        return purge_agent_iam(leftovers, on_status=on_status)
    if purge and not last_agent:
        on_status(
            "Keeping the npa-agent service account: "
            f"{remaining_agents} other agent(s) in this project still use it."
        )
    for line in format_iam_leftovers(leftovers, project_id=project_id, last_agent=last_agent):
        on_status(line)
    return []


def format_iam_leftovers(leftovers: dict[str, Any], *, project_id: str, last_agent: bool) -> list[str]:
    """Return report lines naming what destroy did not delete, with commands."""
    sa_id = str(leftovers.get("service_account_id", "") or "")
    if not sa_id:
        return []
    name = leftovers.get("service_account_name") or "npa-agent"
    keys = [str((key or {}).get("id", "") or "") for key in leftovers.get("access_keys") or []]
    keys = [key for key in keys if key]
    lines = [
        f"Left in place: service account {name} ({sa_id})"
        + (f" and {len(keys)} access key(s)" if keys else "")
        + ".",
    ]
    if last_agent:
        lines.append(
            "  This project has no agents left, so nothing needs it. Remove it with "
            "`npa agent destroy --purge-iam` next time, or now:"
        )
        for key_id in keys:
            lines.append(f"    nebius iam v2 access-key delete --id {key_id}")
        lines.append(f"    nebius iam service-account delete --id {sa_id}")
    else:
        lines.append(
            "  Other agents in this project still use it, so it was kept. "
            f"`npa agent status --project <alias>` lists them (project {project_id})."
        )
    return lines
