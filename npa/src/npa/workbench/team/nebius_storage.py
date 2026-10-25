"""Provision a fresh personal Nebius storage principal without project-wide fallback."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from npa.clients import nebius
from npa.clients.config import resolve_environment

from .errors import BackendError
from .models import StorageGrant


def create_storage(project: str, endpoint: str, output_dir: Path):
    """Create a dedicated bucket, account, capability group, and private credential.

    Args:
        project: Saved Nebius project alias belonging to the administering operator.
        endpoint: Explicit HTTPS object-storage endpoint for this project.
        output_dir: New private receipt directory; partial resources are retained.
    Returns:
        Private configuration-fragment and receipt paths, never key contents.
    Raises:
        BackendError, OSError: Provisioning failed; receipt identifies partial resources.
    """
    environment = resolve_environment(project=project)
    name = "npa-team-" + uuid.uuid4().hex
    credential_path = output_dir / "storage-credentials.json"
    StorageGrant(
        endpoint=endpoint,
        bucket=name,
        prefix="personal",
        credentials_file=credential_path,
        principal="pending",
    )
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    receipt = _Receipt(output_dir)
    receipt.record(
        "provision-intent", {"name": name, "project": environment.project_id}
    )
    try:
        _provision_storage(environment, name, endpoint, output_dir, receipt)
    except (nebius.NebiusError, ValueError, KeyError) as exc:
        raise BackendError(
            "personal storage provisioning failed; retain the private receipt for recovery"
        ) from exc
    return {
        "storage_grant": str(output_dir / "storage-grant.json"),
        "receipt": str(output_dir / "receipt.json"),
    }


def _provision_storage(environment, name, endpoint, output_dir, receipt):
    account, bucket = _resources(environment, name, receipt)
    evidence = _binding(environment, name, account, bucket, receipt)
    access, secret = nebius.ensure_access_key(
        environment.project_id,
        account,
        key_name=name,
        on_created=lambda key, key_name: receipt.record(
            "access-key", {"id": key, "name": key_name, "service_account": account}
        ),
    )
    credential_path = output_dir / "storage-credentials.json"
    _write(
        credential_path, {"aws_access_key_id": access, "aws_secret_access_key": secret}
    )
    grant = StorageGrant(
        endpoint=endpoint,
        bucket=name,
        prefix="personal",
        credentials_file=credential_path.resolve(),
        principal=account,
    )
    _write(output_dir / "storage-grant.json", grant.model_dump(mode="json"))
    receipt.record("complete", {"scope": evidence.scope_id})


def _resources(environment, name, receipt):
    account = nebius.ensure_service_account(
        environment.project_id,
        name=name,
        allow_saved_fallback=False,
        description="Personal Workbench artifact access",
        on_created=lambda identifier: receipt.record(
            "service-account", {"id": identifier}
        ),
    )
    if not receipt.has("service-account", account):
        raise BackendError("refusing to adopt an existing storage service account")
    nebius.ensure_bucket(
        environment.project_id,
        name,
        allow_existing=False,
        on_created=lambda bucket: receipt.record("bucket", {"name": bucket}),
    )
    bucket = nebius.get_bucket_by_name(environment.project_id, name)
    if not bucket or not bucket.get("metadata", {}).get("id"):
        raise BackendError("created storage bucket identity cannot be verified")
    return account, bucket["metadata"]["id"]


def _binding(environment, name, account, bucket, receipt):
    evidence = nebius.ensure_storage_capability_binding(
        project_id=environment.project_id,
        tenant_id=environment.tenant_id,
        bucket_id=bucket,
        service_account_id=account,
        allow_editors_fallback=False,
        binding_group_name=name,
        on_resource_created=receipt.record,
    )
    if evidence.compatibility_fallback or evidence.scope_id != bucket:
        raise BackendError(
            "personal storage did not receive an exact bucket capability"
        )
    if not receipt.has("iam_group", evidence.group_id):
        raise BackendError(
            "refusing to reuse a capability group with unknown members or permits"
        )
    return evidence


class _Receipt:
    def __init__(self, root):
        self.path = root / "receipt.json"
        self.events = []
        _write(self.path, self.events)

    def record(self, kind, resource):
        self.events.append({"kind": kind, **resource})
        _write(self.path, self.events)

    def has(self, kind, identifier):
        return any(
            item["kind"] == kind and item.get("id") == identifier
            for item in self.events
        )


def _write(path, payload):
    path.touch(mode=0o600, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
