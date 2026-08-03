"""`npa storage` — object-storage teardown for resources npa created.

`npa configure` provisions a bucket (and a service account + access key) with no
inverse, so cleaning up meant dropping to raw `nebius` commands. Deleting the
bucket by hand is its own trap: a versioned bucket that `aws s3 rb --force`
appears to empty still holds non-current versions, and the API answers
``BucketNotEmpty``. Scheduling the purge (``--ttl``) is what actually works.
"""

from __future__ import annotations

from dataclasses import dataclass

import typer

app = typer.Typer(
    name="storage",
    help="Inspect and tear down npa-managed object storage.",
    no_args_is_help=True,
)

bucket_app = typer.Typer(name="bucket", help="Object-storage buckets.", no_args_is_help=True)
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


@bucket_app.command("list")
def list_buckets_cmd(
    project: str = typer.Option("", "--project", help="NPA project alias to list buckets for."),
    project_id: str = typer.Option("", "--project-id", help="Nebius project id to list buckets for."),
    output_json: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
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
        raise typer.BadParameter(f"Could not list buckets in {resolved_project}: {exc}") from exc

    configured = ""
    try:
        configured = _bucket_name_from_uri(str(load_credentials(environ={}).s3_bucket or ""))
    except Exception:  # noqa: BLE001 - listing works without readable credentials
        configured = ""

    rows = [
        {
            "name": str((item.get("metadata") or {}).get("name", "") or ""),
            "id": str((item.get("metadata") or {}).get("id", "") or ""),
            "configured": str((item.get("metadata") or {}).get("name", "") or "") == configured,
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


def _storage_service_account_record() -> tuple[_OwnedStorageServiceAccount | None, str]:
    """Read the persisted ownership proof for NPA-created storage IAM.

    A legacy ``nebius.service_account_id`` is not enough: configure has always
    accepted user-managed accounts, and deleting one based on its familiar name
    would silently destroy an unrelated identity. Only the provenance written
    at the successful create call is authoritative.
    """

    import yaml

    from npa.clients.credentials import CREDENTIALS_PATH
    from npa.clients.nebius import DEFAULT_SERVICE_ACCOUNT_NAME

    if not CREDENTIALS_PATH.exists():
        return None, "No NPA-owned storage service account is recorded."
    try:
        data = yaml.safe_load(CREDENTIALS_PATH.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None, "No readable NPA-owned storage service-account record was found."
    nebius = data.get("nebius") if isinstance(data, dict) else None
    if not isinstance(nebius, dict):
        return None, "No NPA-owned storage service account is recorded."

    account_id = str(nebius.get("service_account_id", "") or "").strip()
    managed_by = str(nebius.get("service_account_managed_by", "") or "").strip()
    name = str(nebius.get("service_account_name", "") or "").strip()
    project_id = str(nebius.get("service_account_project_id", "") or "").strip()
    if managed_by != "npa":
        suffix = (
            f" The saved ID {account_id} is not proof of ownership and was left untouched."
            if account_id
            else ""
        )
        return None, f"No NPA-owned storage service account is recorded.{suffix}"
    if not account_id or name != DEFAULT_SERVICE_ACCOUNT_NAME or not project_id:
        return None, (
            "The storage service-account ownership record is incomplete; refusing to "
            "delete any IAM identity."
        )
    return _OwnedStorageServiceAccount(account_id, name, project_id), ""


def _remove_storage_service_account_record(account_id: str) -> bool:
    """Remove a deleted account's saved ownership record; report success."""

    import yaml

    from npa.clients.credentials import CREDENTIALS_PATH

    if not CREDENTIALS_PATH.exists():
        return True
    try:
        data = yaml.safe_load(CREDENTIALS_PATH.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return False
    if not isinstance(data, dict):
        return False
    nebius = data.get("nebius")
    if not isinstance(nebius, dict):
        return False
    if str(nebius.get("service_account_id", "") or "").strip() != account_id:
        return False
    for key in (
        "service_account_id",
        "service_account_name",
        "service_account_project_id",
        "service_account_managed_by",
    ):
        nebius.pop(key, None)
    if nebius:
        data["nebius"] = nebius
    else:
        data.pop("nebius", None)
    if not data:
        try:
            CREDENTIALS_PATH.unlink()
        except OSError:
            return False
        return True
    try:
        CREDENTIALS_PATH.write_text(
            yaml.safe_dump(data, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )
        CREDENTIALS_PATH.chmod(0o600)
    except OSError:
        return False
    return True


@service_account_app.command("delete")
def delete_service_account_cmd(
    project: str = typer.Option("", "--project", help="NPA project alias owning the account."),
    project_id: str = typer.Option(
        "", "--project-id", help="Nebius project id owning the account."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the exact NPA-owned IAM resources without deleting them."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Delete the storage service account only when NPA recorded creating it.

    Run this after deleting the configured bucket. Reused, legacy, incomplete,
    mismatched, and user-managed identities are always left untouched.
    """

    import sys

    from npa.clients.config import ConfigError, resolve_environment
    from npa.clients.credentials import load_credentials
    from npa.clients.nebius import (
        NebiusError,
        delete_access_key,
        delete_service_account,
        is_not_found,
        list_access_keys_for_service_account,
    )

    record, note = _storage_service_account_record()
    if record is None:
        typer.echo(note)
        return

    resolved_project = project_id.strip()
    if not resolved_project:
        try:
            saved = resolve_environment(project or None)
        except ConfigError as exc:
            raise typer.BadParameter(str(exc)) from exc
        resolved_project = str(getattr(saved, "project_id", "") or "").strip()
    if not resolved_project:
        resolved_project = record.project_id
    if resolved_project != record.project_id:
        typer.echo(
            f"Recorded NPA-owned service account {record.account_id} belongs to "
            f"{record.project_id}, which does not match project {resolved_project}; "
            "nothing was deleted."
        )
        return

    try:
        credentials = load_credentials(environ={})
    except Exception:  # noqa: BLE001 - IAM teardown can still use the ownership record
        credentials = None
    configured_bucket = str(getattr(credentials, "s3_bucket", "") or "").strip()
    if configured_bucket:
        raise typer.BadParameter(
            f"Object storage {configured_bucket} is still configured. Run `npa storage "
            "bucket delete --project <alias> --yes --wait` first so this account is "
            "not deleted while its bucket may still need it."
        )

    try:
        keys = list_access_keys_for_service_account(
            record.project_id, record.account_id, strict=True
        )
    except NebiusError as exc:
        raise typer.BadParameter(
            f"Could not inspect access keys for NPA-owned service account "
            f"{record.account_id}; nothing was deleted: {exc}"
        ) from exc
    key_ids = [str((key or {}).get("id", "") or "").strip() for key in keys]
    key_ids = [key_id for key_id in key_ids if key_id]
    if dry_run:
        for key_id in key_ids:
            typer.echo(f"Would delete access key {key_id}.")
        typer.echo(
            f"Would delete NPA-owned service account {record.name} "
            f"({record.account_id}) in {record.project_id}."
        )
        return

    if not yes:
        prompt = (
            f"Delete NPA-owned service account {record.name} ({record.account_id}) "
            f"and {len(key_ids)} access key(s)?"
        )
        if not sys.stdin.isatty():
            typer.echo(f"{prompt} Re-run with --yes, or use --dry-run to inspect the plan.")
            raise typer.Exit(code=1)
        if not typer.confirm(prompt, default=False):
            typer.echo("Aborted.")
            raise typer.Exit(code=1)

    failed = False
    for key_id in key_ids:
        try:
            delete_access_key(key_id)
        except NebiusError as exc:
            failed = True
            typer.echo(f"Warning: could not delete access key {key_id}: {exc}", err=True)
        else:
            typer.echo(f"Deleted access key {key_id}.")
    try:
        delete_service_account(record.account_id)
    except NebiusError as exc:
        if is_not_found(str(exc)):
            typer.echo(
                f"NPA-owned service account {record.name} ({record.account_id}) "
                "is already absent."
            )
            if not _remove_storage_service_account_record(record.account_id):
                failed = True
                typer.echo(
                    "Warning: the stale local service-account ownership record could "
                    "not be removed; retry after fixing its file permissions.",
                    err=True,
                )
        else:
            failed = True
            typer.echo(
                f"Warning: could not delete NPA-owned service account {record.account_id}: {exc}",
                err=True,
            )
    else:
        typer.echo(
            f"Deleted NPA-owned service account {record.name} ({record.account_id})."
        )
        if not _remove_storage_service_account_record(record.account_id):
            failed = True
            typer.echo(
                "Warning: the service account is gone, but its local ownership record "
                "could not be removed; retry after fixing its file permissions.",
                err=True,
            )
    if failed:
        raise typer.Exit(code=1)


@bucket_app.command("delete")
def delete_bucket_cmd(
    name: str = typer.Option("", "--name", help="Bucket name. Defaults to the configured bucket."),
    bucket_id: str = typer.Option("", "--id", help="Bucket resource id (skips the name lookup)."),
    project: str = typer.Option("", "--project", help="NPA project alias holding the bucket."),
    project_id: str = typer.Option("", "--project-id", help="Nebius project id holding the bucket."),
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
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Delete an object-storage bucket npa provisioned, contents and versions included."""
    import sys

    from npa.clients.config import resolve_environment
    from npa.clients.credentials import load_credentials
    from npa.clients.nebius import NebiusError, delete_bucket, get_bucket_by_name

    credentials = None
    try:
        credentials = load_credentials(environ={})
    except Exception:  # noqa: BLE001 - teardown must work without readable credentials
        credentials = None
    bucket_name = name.strip() or _bucket_name_from_uri(
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
            raise typer.BadParameter(f"Could not list buckets in {resolved_project}: {exc}") from exc
        if item is None:
            typer.echo(f"Bucket {bucket_name!r} does not exist in project {resolved_project}.")
            if prune_config:
                _prune_local_state(bucket_name)
            return
        resolved_id = str((item.get("metadata") or {}).get("id", "") or "")
        bucket_name = bucket_name or str((item.get("metadata") or {}).get("name", "") or "")

    target = f"{bucket_name or resolved_id} ({resolved_id})"
    if not yes and sys.stdin.isatty():
        if not typer.confirm(f"Delete bucket {target} and everything in it?", default=False):
            typer.echo("Aborted.")
            raise typer.Exit(code=1)

    try:
        delete_bucket(resolved_id, ttl=ttl)
    except NebiusError as exc:
        message = str(exc)
        if "notempty" in message.replace(" ", "").lower() and not str(ttl or "").strip():
            raise typer.BadParameter(
                f"Bucket {target} is not empty (objects or non-current versions remain). "
                f"Re-run with --ttl {DEFAULT_PURGE_TTL} to schedule the purge."
            ) from exc
        # Once a purge is scheduled the API is inconsistent: `delete` answers
        # NoSuchBucket while `list`/`get` still return the bucket as
        # SCHEDULED_FOR_DELETION. Re-deleting is then a no-op, not a failure.
        pending = _scheduled_deletion_state(resolved_project, bucket_name)
        if "nosuchbucket" in message.replace(" ", "").lower() and pending:
            typer.echo(f"Bucket {target} is already {pending}.")
            if wait:
                _wait_for_bucket_gone(resolved_project, bucket_name, target, wait_timeout)
            if prune_config and bucket_name:
                _prune_local_state(bucket_name)
            return
        raise typer.BadParameter(f"Bucket delete failed: {message}") from exc

    if str(ttl or "").strip():
        typer.echo(f"Bucket {target} scheduled for purge in {ttl}.")
    else:
        typer.echo(f"Bucket {target} deleted.")
    if wait:
        _wait_for_bucket_gone(resolved_project, bucket_name, target, wait_timeout)
    if prune_config and bucket_name:
        _prune_local_state(bucket_name)


def _wait_for_bucket_gone(project_id: str, bucket_name: str, target: str, timeout: int) -> None:
    """Poll until *bucket_name* no longer exists (a scheduled purge is async).

    `storage bucket delete --ttl` returns while the bucket is still
    ``SCHEDULED_FOR_DELETION``; --wait blocks until Nebius has actually removed it
    so a caller can proceed knowing the name is free.
    """
    import time

    from npa.clients.nebius import NebiusError, get_bucket_by_name

    if not project_id or not bucket_name:
        typer.echo("--wait skipped: no project/bucket name to poll.")
        return
    deadline = time.monotonic() + max(1, int(timeout))
    typer.echo(f"Waiting up to {timeout}s for {bucket_name} to be purged...")
    while time.monotonic() < deadline:
        try:
            item = get_bucket_by_name(project_id, bucket_name)
        except NebiusError:
            # Transient list failure: keep waiting rather than declaring done.
            item = {"metadata": {"name": bucket_name}}
        if item is None:
            typer.echo(f"Bucket {target} is gone.")
            return
        time.sleep(5)
    state = _scheduled_deletion_state(project_id, bucket_name) or "still present"
    overdue = _purge_is_overdue(project_id, bucket_name)
    if overdue:
        # Saying "it will be removed by Nebius" is not true once purge_at has
        # passed and the objects are still there; that is a platform-side stall.
        typer.echo(
            f"Bucket {bucket_name} is {state} and its purge_at has already passed, "
            f"so the purge has stalled rather than merely being slower than the {timeout}s "
            "wait. The name stays reserved until Nebius clears it; raise it with Nebius "
            "support if it does not resolve."
        )
        return
    typer.echo(
        f"Bucket {bucket_name} is {state} after {timeout}s "
        "(a scheduled purge can take longer than the wait); it will be removed by "
        "Nebius. Re-run with a larger --wait-timeout to keep watching."
    )


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
    _prune_storage_credentials(bucket_name)
    from npa.clients.config import CONFIG_PATH, clear_terraform_state_for_bucket

    cleared = clear_terraform_state_for_bucket(bucket_name)
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
    section (under whatever key names the file uses) and the
    ``nebius.service_account_id`` of the deleted bucket's storage principal.
    """
    from npa.clients.credentials import CREDENTIALS_PATH, load_credentials

    try:
        credentials = load_credentials(environ={})
    except Exception:  # noqa: BLE001
        return
    saved_bucket = _bucket_name_from_uri(str(getattr(credentials, "s3_bucket", "") or ""))
    if not saved_bucket or saved_bucket != bucket_name:
        return
    if not CREDENTIALS_PATH.exists():
        return
    import yaml

    try:
        data = yaml.safe_load(CREDENTIALS_PATH.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return
    if not isinstance(data, dict):
        return

    removed: list[str] = []
    for section_key in _STORAGE_SECTION_KEYS:
        section = data.get(section_key)
        if not isinstance(section, dict):
            continue
        for key in _STORAGE_SECRET_KEYS:
            if section.get(key) not in (None, ""):
                section.pop(key, None)
                removed.append(key)
        if section:
            data[section_key] = section
        else:
            data.pop(section_key, None)

    # Keep a verifiable NPA ownership record until the explicit IAM teardown
    # runs. A legacy/user-managed ID has no such provenance and retains the old
    # behavior of being detached from the deleted bucket's credentials.
    nebius = data.get("nebius")
    if isinstance(nebius, dict) and nebius.get("service_account_id") not in (None, ""):
        owned = str(nebius.get("service_account_managed_by", "") or "") == "npa"
        if not owned:
            nebius.pop("service_account_id", None)
            removed.append("service_account_id")
        if nebius:
            data["nebius"] = nebius
        else:
            data.pop("nebius", None)

    if not removed:
        return
    # Rewrite the file verbatim: write_credentials_file deep-merges and cannot
    # drop keys.
    CREDENTIALS_PATH.write_text(
        yaml.safe_dump(data, default_flow_style=False, sort_keys=False), encoding="utf-8"
    )
    CREDENTIALS_PATH.chmod(0o600)
    unique_removed = list(dict.fromkeys(removed))
    typer.echo(
        f"Removed the saved S3 {', '.join(unique_removed)} for {bucket_name} from "
        f"{CREDENTIALS_PATH} (they pointed at a deleted bucket). Re-run `npa configure` "
        "to provision new storage."
    )
