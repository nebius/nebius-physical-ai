"""Guarded teardown for registries in NPA-created disposable projects."""

from __future__ import annotations

import json

import typer

app = typer.Typer(
    name="registry",
    help="Inspect and tear down exact registries in NPA-created projects.",
    no_args_is_help=True,
)


@app.command("delete")
def delete_registry_cmd(
    project: str = typer.Option(..., "--project", help="Exact NPA project alias."),
    project_id: str = typer.Option(..., "--project-id", help="Exact project ID."),
    tenant_id: str = typer.Option(..., "--tenant-id", help="Exact tenant ID."),
    registry_id: str = typer.Option(..., "--id", help="Exact immutable registry ID."),
    name: str = typer.Option(..., "--name", help="Exact provider registry name."),
    profile: str = typer.Option("", "--profile", help="Exact Nebius CLI profile."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Confirm exact deletion."),
    output_json: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Delete one exact registry only with durable NPA project-creation proof."""

    from npa.clients.nebius import (
        NebiusError,
        delete_all_registry_images,
        delete_registry,
        get_project_identity,
        get_registry_identity,
        list_registry_image_ids,
    )
    from npa.project_destroy import _project_ownership_operation
    from npa.teardown_receipts import record_teardown_event

    expected_name = name.strip()
    try:
        project_identity = get_project_identity(
            project_id, tenant_id=tenant_id, profile=profile or None
        )
        if project_identity is None:
            raise RuntimeError("exact project is absent")
        ownership = _project_ownership_operation(project, project_id, tenant_id)
        if ownership is None:
            raise RuntimeError(
                "registry deletion requires unique durable NPA project-creation proof"
            )
        identity = get_registry_identity(registry_id, profile=profile or None)
        if identity is None:
            payload = {
                "outcome": "already_absent",
                "registry_id": registry_id,
                "project_id": project_id,
                "verified": True,
            }
        else:
            if identity.project_id != project_id or identity.name != expected_name:
                raise RuntimeError(
                    "provider registry identity does not match the exact project/name selectors"
                )
            payload = {
                "outcome": "planned",
                "registry_id": identity.registry_id,
                "registry_name": identity.name,
                "project_id": identity.project_id,
                "verified": True,
            }
            if yes:
                image_ids = list_registry_image_ids(
                    identity.registry_id, profile=profile or None
                )
                record_teardown_event(
                    phase="registry",
                    resource=identity.registry_id,
                    terminal_state="deletion_approved",
                    project_alias=project,
                    project_id=project_id,
                    identity={
                        "registry_id": identity.registry_id,
                        "registry_name": identity.name,
                        "project_id": identity.project_id,
                        "ownership": "npa_disposable_project",
                        "project_operation_id": ownership.operation_id,
                        "registry_image_ids": list(image_ids),
                    },
                    action={"kind": "delete_exact_registry"},
                )
                removed_images = delete_all_registry_images(
                    identity.registry_id, profile=profile or None
                )
                remaining_images = list_registry_image_ids(
                    identity.registry_id, profile=profile or None
                )
                if remaining_images:
                    raise RuntimeError(
                        "registry image cleanup did not converge to verified absence"
                    )
                delete_registry(identity.registry_id, profile=profile or None)
                if get_registry_identity(identity.registry_id, profile=profile or None):
                    raise RuntimeError(
                        "provider accepted deletion but registry remains present"
                    )
                record_teardown_event(
                    phase="registry",
                    resource=identity.registry_id,
                    terminal_state="verified_deleted",
                    project_alias=project,
                    project_id=project_id,
                    identity={
                        "registry_id": identity.registry_id,
                        "registry_name": identity.name,
                        "project_id": identity.project_id,
                    },
                    verification={"exact_registry_absent": True},
                )
                payload["outcome"] = "verified_deleted"
                payload["images_removed"] = list(removed_images)
    except (NebiusError, RuntimeError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    if output_json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(f"registry {payload['registry_id']}: {payload['outcome']}")
