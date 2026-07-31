"""`npa storage` — object-storage teardown for resources npa created.

`npa configure` provisions a bucket (and a service account + access key) with no
inverse, so cleaning up meant dropping to raw `nebius` commands. Deleting the
bucket by hand is its own trap: a versioned bucket that `aws s3 rb --force`
appears to empty still holds non-current versions, and the API answers
``BucketNotEmpty``. Scheduling the purge (``--ttl``) is what actually works.
"""

from __future__ import annotations

import typer

app = typer.Typer(
    name="storage",
    help="Inspect and tear down npa-managed object storage.",
    no_args_is_help=True,
)

bucket_app = typer.Typer(name="bucket", help="Object-storage buckets.", no_args_is_help=True)
app.add_typer(bucket_app, name="bucket")

DEFAULT_PURGE_TTL = "1m"


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
        help="Also drop the saved S3 credentials for this bucket from ~/.npa/credentials.yaml.",
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
                _prune_storage_credentials(bucket_name)
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
        raise typer.BadParameter(f"Bucket delete failed: {message}") from exc

    if str(ttl or "").strip():
        typer.echo(f"Bucket {target} scheduled for purge in {ttl}.")
    else:
        typer.echo(f"Bucket {target} deleted.")
    if prune_config and bucket_name:
        _prune_storage_credentials(bucket_name)


def _bucket_name_from_uri(value: str) -> str:
    return str(value or "").strip().removeprefix("s3://").strip("/").split("/", 1)[0]


def _prune_storage_credentials(bucket_name: str) -> None:
    """Drop saved S3 credentials that point at a bucket that no longer exists.

    Leaving them behind means the next `npa configure` / deploy reuses an access
    key for a deleted bucket — the stale-secret half of the teardown report.
    """
    from npa.clients.credentials import CREDENTIALS_PATH, load_credentials, write_credentials_file

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
    storage = data.get("storage")
    if not isinstance(storage, dict):
        return
    removed = [key for key in ("bucket", "access_key_id", "secret_access_key") if storage.get(key)]
    if not removed:
        return
    for key in removed:
        storage.pop(key, None)
    data["storage"] = storage
    # write_credentials_file deep-merges, so rewrite the file to drop keys.
    CREDENTIALS_PATH.write_text(yaml.safe_dump(data, default_flow_style=False), encoding="utf-8")
    CREDENTIALS_PATH.chmod(0o600)
    write_credentials_file({})
    typer.echo(
        f"Removed the saved S3 {', '.join(removed)} for {bucket_name} from {CREDENTIALS_PATH} "
        "(they pointed at a deleted bucket). Re-run `npa configure` to provision new storage."
    )
