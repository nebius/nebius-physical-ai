"""Read-only proof for explicitly configured external Terraform storage.

Selecting a remote backend does not adopt its bucket as compute-project
infrastructure. The default backend must still exist in the compute project;
an external backend requires exact saved ownership and same-tenant proof.
"""

from __future__ import annotations

from npa.clients.config import resolve_environment, resolve_terraform_state


def verify_external_backend(
    *, project_alias: str, project_id: str, bucket_name: str, endpoint: str
) -> dict[str, str] | None:
    """Return verified external identity, or None for the ordinary local path."""
    from npa.clients.nebius import NebiusError, _run_json
    from npa.lifecycle_intent import OperationIntent, operation_intent

    if not project_alias:
        return None
    with operation_intent(OperationIntent.OBSERVE):
        saved = resolve_terraform_state(project_alias)
    owner = str(getattr(saved, "owner_project_id", "") or "").strip()
    bucket_id = str(getattr(saved, "bucket_id", "") or "").strip()
    if not owner and not bucket_id:
        return None
    if (
        not owner
        or not bucket_id
        or saved.bucket != bucket_name
        or not endpoint
        or saved.endpoint != endpoint
    ):
        raise NebiusError(
            "Explicit Terraform bucket binding is incomplete or mismatched"
        )
    selected = resolve_environment(project_alias)
    if selected.project_id != project_id or not selected.tenant_id:
        raise NebiusError(
            "Terraform bucket binding requires the exact selected project and tenant"
        )
    for identity in dict.fromkeys((project_id, owner)):
        result = _run_json(["iam", "project", "get", "--id", identity])
        metadata = result.get("metadata")
        if (
            not isinstance(metadata, dict)
            or metadata.get("id") != identity
            or metadata.get("parent_id") != selected.tenant_id
        ):
            raise NebiusError(
                "Terraform bucket owner is not proven in the selected tenant"
            )
    result = _run_json(["storage", "bucket", "get", "--id", bucket_id])
    metadata = result.get("metadata")
    if (
        not isinstance(metadata, dict)
        or metadata.get("id") != bucket_id
        or metadata.get("parent_id") != owner
        or metadata.get("name") != bucket_name
    ):
        raise NebiusError(
            "Terraform bucket identity or actual owner does not match its binding"
        )
    return {"owner_project_id": owner, "bucket_id": bucket_id}
