"""Terraform working-dir and variable helpers for ``npa agent``.

Extracted from the ``npa.cli.agent`` monolith (kept under a size ratchet) so the
Terraform plumbing for destroy/reclaim lives in one small module. These helpers are
re-exported from ``npa.cli.agent`` for the existing call sites and tests.
"""

from __future__ import annotations

import shutil
from typing import Any, Mapping

import typer

from npa.clients.config import (
    ConfigError,
    resolve_environment,
    resolve_terraform_state,
)
from npa.deploy import provisioner
from npa.deploy.provisioner import ProvisionerError
from npa.provisioning_journal import current_operation, list_operations


def _ensure_terraform_state_bucket(
    *,
    project_id: str,
    bucket_name: str,
    endpoint: str = "",
    access_key: str = "",
    secret_key: str = "",
    region: str = "",
    project_alias: str = "",
    agent_name: str = "default",
) -> None:
    """Verify the configured backend immediately before Terraform mutation."""

    project = str(project_id or "").strip()
    bucket = str(bucket_name or "").strip()
    if not project or not bucket:
        return
    from npa.clients.nebius import NebiusError, bucket_exists

    try:
        exists = bucket_exists(project, bucket)
    except NebiusError as exc:
        raise NebiusError(
            f"Terraform backend inventory check failed before apply: {exc}"
        ) from exc
    if not exists:
        raise NebiusError(
            f"Terraform backend bucket {bucket!r} is missing from project {project}. "
            "NPA preserved configuration and will not silently recreate or adopt it."
        )
    if all((endpoint, access_key, secret_key)):
        from npa.clients.storage_validation import (
            probe_terraform_backend,
            terraform_backend_fingerprint,
            terraform_state_key,
        )

        backend_key = terraform_state_key(project_alias or project, agent_name)
        probe = probe_terraform_backend(
            bucket=bucket,
            state_key=backend_key,
            endpoint_url=endpoint,
            access_key_id=access_key,
            secret_access_key=secret_key,
            region=region,
        )
        if not probe.ok:
            raise NebiusError(
                "Terraform backend capability probe failed immediately before "
                f"apply: {probe.summary}"
            )
        operation = current_operation()
        if operation is not None:
            operation.update_identity(
                backend={
                    "bucket": bucket,
                    "endpoint": endpoint,
                    "region": region,
                    "state_key": backend_key,
                    "addressing_style": "path",
                    "credential_source": "project_resolver",
                    "config_fingerprint": terraform_backend_fingerprint(
                        bucket=bucket,
                        state_key=backend_key,
                        endpoint_url=endpoint,
                        access_key_id=access_key,
                        secret_access_key=secret_key,
                        region=region,
                    ),
                }
            )


def _persist_agent_project_config(
    *,
    project: str,
    project_id: str,
    tenant_id: str,
    region: str,
    merged_vars: dict[str, str],
) -> None:
    from npa.cli import agent as agent_module

    operation = current_operation()
    if operation is not None:
        operation.record_config_mutation(
            store="config.yaml",
            fields=[
                f"projects.{project}.project_id",
                f"projects.{project}.tenant_id",
                f"projects.{project}.region",
                f"projects.{project}.terraform_state",
            ],
            secret_fields=[
                f"projects.{project}.terraform_state.access_key",
                f"projects.{project}.terraform_state.secret_key",
                f"projects.{project}.terraform_state.session_token",
            ],
        )
    terraform_state = {
        "bucket": merged_vars.get("s3_bucket", ""),
        "endpoint": merged_vars.get("s3_endpoint", ""),
        "access_key": merged_vars.get("nebius_api_key", ""),
        "secret_key": merged_vars.get("nebius_secret_key", ""),
        "session_token": merged_vars.get("s3_session_token", ""),
        "region": region,
        "addressing_style": "path",
    }
    from npa.clients.project_credential_store import write_project_credentials

    write_project_credentials(
        project_id,
        {"terraform_state": terraform_state},
        alias=project,
    )
    agent_module.write_config(
        {
            "projects": {
                project: {
                    "project_id": project_id,
                    "tenant_id": tenant_id,
                    "region": region,
                    "terraform_state": {
                        "bucket": terraform_state["bucket"],
                        "endpoint": terraform_state["endpoint"],
                        "region": region,
                        "addressing_style": "path",
                        "credential_source": "project_credentials_v2",
                    },
                }
            }
        }
    )


def _preserve_available_state(operation, tf_dir) -> None:
    for candidate in (tf_dir / "errored.tfstate", tf_dir / "terraform.tfstate"):
        if candidate.is_file():
            operation.preserve_state_file(candidate, name=candidate.stem)


def _apply_agent_terraform(
    *,
    project: str,
    name: str,
    merged_vars: dict[str, str],
    env_region: str,
    backend_credentials: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Apply agent Terraform and durably verify its remote state."""

    from npa.lifecycle_intent import forbid_destructive_provisioning

    forbid_destructive_provisioning("apply_agent_terraform")

    from npa.cli.agent import (
        _AGENT_TERRAFORM_RUNTIME_ONLY_VARS,
        _looks_like_compute_permission_denied,
    )

    backend = dict(backend_credentials or {})
    backend_access_key = str(
        backend.get("access_key") or merged_vars.get("nebius_api_key", "")
    )
    backend_secret_key = str(
        backend.get("secret_key") or merged_vars.get("nebius_secret_key", "")
    )
    backend_session_token = str(
        backend.get("session_token") or merged_vars.get("s3_session_token", "")
    )

    def revalidate_backend() -> None:
        _ensure_terraform_state_bucket(
            project_id=str(merged_vars.get("nebius_project_id", "")),
            bucket_name=str(merged_vars.get("s3_bucket", "")),
            endpoint=str(merged_vars.get("s3_endpoint", "")),
            access_key=backend_access_key,
            secret_key=backend_secret_key,
            region=env_region,
            project_alias=project,
            agent_name=name,
        )

    revalidate_backend()
    tf_dir = provisioner.prepare_working_dir(
        project,
        name,
        bucket=merged_vars.get("s3_bucket", ""),
        region=env_region,
        endpoint=merged_vars.get("s3_endpoint", ""),
    )
    provisioner.init(
        tf_dir=tf_dir,
        backend_config={
            "access_key": backend_access_key,
            "secret_key": backend_secret_key,
            "session_token": backend_session_token,
            "region": env_region,
            "endpoint": merged_vars.get("s3_endpoint", ""),
            "addressing_style": "path",
        },
    )
    tf_vars = {
        key: value
        for key, value in merged_vars.items()
        if key not in _AGENT_TERRAFORM_RUNTIME_ONLY_VARS
    }
    operation = current_operation()
    if operation is not None:
        preserved = operation.state_copies()
        if preserved and not provisioner.state_list(tf_dir):
            revalidate_backend()
            provisioner.state_push(preserved[0], tf_dir)
    try:
        try:
            revalidate_backend()
            result = provisioner.apply(tf_dir=tf_dir, tf_vars=tf_vars)
        except ProvisionerError as exc:
            sa_id = str(merged_vars.get("service_account_id", "")).strip()
            if not (sa_id and _looks_like_compute_permission_denied(str(exc))):
                raise
            typer.echo(
                "  WARNING: compute create was denied WITH the VM service-account "
                "attachment, so it is being retried WITHOUT it.",
                err=True,
            )
            typer.echo(
                "  WARNING: without an attached service account the agent VM "
                "CANNOT self-mint Nebius IAM tokens (no metadata/token-file "
                "source), so IAM-dependent actions (provisioning clusters, "
                "buckets, access keys, registries) will fail until you provide an "
                "alternative token source. Grant the deploying identity "
                "'compute.admin' (or equivalent) on the project so the SA can be "
                "attached, or inject a token on the VM via NEBIUS_IAM_TOKEN / a "
                "token file. Re-run `npa agent setup` once the permission is "
                "granted.",
                err=True,
            )
            retry_vars = dict(tf_vars)
            retry_vars["service_account_id"] = ""
            revalidate_backend()
            result = provisioner.apply(tf_dir=tf_dir, tf_vars=retry_vars)
    except ProvisionerError as exc:
        if operation is not None:
            _preserve_available_state(operation, tf_dir)
            operation.transition("recovery-required", error=str(exc))
        raise
    if operation is not None:
        operation.transition("resource-created")
        instance_id = str(result.get("instance_id", "") or "")
        instance_name = str(merged_vars.get("instance_name", name))
        operation.record_resource(
            resource_type="compute_instance",
            requested_name=instance_name,
            provider_id=instance_id,
            ownership="created_by_this_operation",
            ownership_source="terraform-output-and-operation-label",
            project_id=str(merged_vars.get("nebius_project_id", "")),
            labels={"npa-operation-id": operation.operation_id},
        )
        for resource_type, requested_name in (
            ("boot_disk", f"{instance_name}-boot"),
            ("network", f"{instance_name}-network"),
            ("subnet", f"{instance_name}-subnet"),
            ("security_group", f"{instance_name}-sg"),
            ("public_ip", f"{instance_name}:eth0"),
        ):
            operation.record_resource(
                resource_type=resource_type,
                requested_name=requested_name,
                ownership="created_by_this_operation",
                ownership_source="terraform-state-and-operation-instance-label",
                project_id=str(merged_vars.get("nebius_project_id", "")),
                labels={"npa-operation-id": operation.operation_id},
            )
        try:
            revalidate_backend()
            state = provisioner.state_pull(tf_dir)
            operation.preserve_state_bytes(state, name="verified-remote")
        except ProvisionerError as exc:
            _preserve_available_state(operation, tf_dir)
            operation.transition("recovery-required", error=str(exc))
            raise
        operation.transition("state-durable")
    return result


def _agent_terraform_state_exists(project: str, name: str) -> bool:
    tf_dir = provisioner.working_dir_path(project, name)
    if any(
        candidate.exists()
        for candidate in (
            tf_dir / ".terraform",
            tf_dir / "terraform.tfstate",
            tf_dir / "errored.tfstate",
        )
    ):
        return True
    from npa.provisioning_journal import list_operations

    return any(
        operation.state_copies()
        for operation in list_operations(
            project_alias=project,
            resource_type="agent",
            requested_name=name,
        )
    )


def _record_agent_destroy_event(
    project: str,
    name: str,
    *,
    terminal_state: str,
    record_present: bool | None = None,
    terraform_state_present: bool | None = None,
    purge_iam: bool | None = None,
    error: str = "",
    identity: dict[str, Any] | None = None,
    project_id: str = "",
    identity_source: str = "live_configuration",
    terraform_graph_absent: bool = False,
) -> None:
    """Persist agent destroy evidence outside the removable project record."""

    from npa.teardown_receipts import record_teardown_event

    environment = resolve_environment(project) if project else None
    exact_project_id = str(project_id or getattr(environment, "project_id", "") or "")
    precheck: dict[str, object] = {
        "identity_resolved": bool(exact_project_id),
        "identity_source": identity_source,
    }
    if record_present is not None:
        precheck["local_record_present"] = record_present
    if terraform_state_present is not None:
        precheck["terraform_state_present"] = terraform_state_present
    action: dict[str, object] = {"kind": "terraform_agent_destroy"}
    if purge_iam is not None:
        action["purge_iam"] = purge_iam
    verification: dict[str, object] = {
        "remote_destroy": {
            "in_progress": "pending",
            "failed": "failed",
            "verified_deleted": "completed",
            "verified_absent": "already_absent",
        }.get(terminal_state, terminal_state)
    }
    if terminal_state in {"verified_deleted", "verified_absent"}:
        verification["exact_instance_absent"] = True
        if terraform_graph_absent:
            verification.update(
                {
                    "terraform_destroy_completed": True,
                    "terraform_dependency_graph": [
                        "compute_instance",
                        "boot_disk",
                        "network",
                        "subnet",
                        "security_group",
                        "public_ip",
                    ],
                }
            )
    record_teardown_event(
        phase="agent",
        resource=name,
        terminal_state=terminal_state,
        project_alias=project,
        project_id=exact_project_id,
        identity=identity,
        precheck=precheck,
        action=action,
        verification=verification,
        errors=[error] if error else [],
    )


def _resolve_destroy_tf_vars(
    project: str,
    name: str,
    record: dict[str, Any] | None,
    *,
    backend_override: dict[str, str] | None = None,
) -> dict[str, str]:
    # Imported lazily: npa.cli.agent imports this module.
    from npa.cli.agent import (
        DEFAULT_AGENT_IMAGE_FAMILY,
        DEFAULT_AGENT_PORT,
        _resolve_agent_service_account_id,
    )
    from npa.clients.nebius import get_iam_token

    try:
        state = resolve_terraform_state(project)
    except ConfigError:  # exact journal-backed recovery may lack config
        state = None
    saved_env = resolve_environment(project)
    region = str(
        (record or {}).get("region", "")
        or (saved_env.region if saved_env else "")
        or "eu-north1"
    )
    project_id = str(
        (record or {}).get("project_id", "")
        or (saved_env.project_id if saved_env else "")
    )
    service_account_id = str((record or {}).get("service_account_id", "")).strip()
    if not service_account_id:
        creds = (record or {}).get("credentials", {})
        if isinstance(creds, dict):
            service_account_id = str(creds.get("service_account_id", "")).strip()
    if not service_account_id:
        service_account_id = _resolve_agent_service_account_id(project, record or {})
    ssh_key_path = str(
        (record or {}).get("ssh_public_key_path")
        or (record or {}).get("ssh_key_path")
        or "~/.ssh/id_ed25519"
    )
    ssh_public_key_path = (
        ssh_key_path if ssh_key_path.endswith(".pub") else f"{ssh_key_path}.pub"
    )

    iam_token = get_iam_token()
    return {
        "nebius_project_id": project_id,
        "nebius_region": region,
        "service_account_id": service_account_id,
        "iam_token": iam_token,
        "instance_name": f"agent-{project}-{name}",
        "server_port": str(DEFAULT_AGENT_PORT),
        "workbench_type": "agent",
        "gpu_platform": "cpu-d3",
        "gpu_preset": "8vcpu-32gb",
        "image_family": DEFAULT_AGENT_IMAGE_FAMILY,
        "enable_preemptible": "false",
        "ssh_public_key_path": ssh_public_key_path,
        "nebius_api_key": str(
            (backend_override or {}).get("access_key")
            or getattr(state, "access_key", "")
            or ""
        ),
        "nebius_secret_key": str(
            (backend_override or {}).get("secret_key")
            or getattr(state, "secret_key", "")
            or ""
        ),
        "s3_session_token": str(
            (backend_override or {}).get("session_token")
            or getattr(state, "session_token", "")
            or ""
        ),
        "s3_bucket": str(
            (backend_override or {}).get("bucket") or getattr(state, "bucket", "") or ""
        ),
        "s3_endpoint": str(
            (backend_override or {}).get("endpoint")
            or getattr(state, "endpoint", "")
            or ""
        ),
        "extra_ingress_ports": "[]",
    }


def _cleanup_orphan_agent_instances(
    project_id: str,
    instance_name: str,
    *,
    operation_id: str = "",
) -> None:
    """Delete only an exact-name orphan carrying this operation's ownership label."""

    project_id = str(project_id or "").strip()
    instance_name = str(instance_name or "").strip()
    if not project_id or not instance_name or not operation_id:
        return
    from npa.clients.nebius import NebiusError, _run, _run_json

    try:
        payload = _run_json(
            ["compute", "instance", "list", "--parent-id", project_id, "--all"]
        )
    except NebiusError as exc:
        raise ProvisionerError(
            f"Exact orphan inventory is unresolved; no by-name deletion ran: {exc}"
        ) from exc
    items = payload.get("items", [])
    if not isinstance(items, list):
        raise ProvisionerError(
            "Provider returned a schema-invalid exact orphan instance inventory"
        )
    for item in items:
        if not isinstance(item, dict):
            raise ProvisionerError(
                "Provider returned a non-object item in exact orphan inventory"
            )
        meta = item.get("metadata", {})
        if not isinstance(meta, dict):
            raise ProvisionerError(
                "Provider returned an orphan inventory item without metadata"
            )
        if str(meta.get("name", "")).strip() != instance_name:
            continue
        labels = meta.get("labels", {})
        if (
            not isinstance(labels, dict)
            or str(labels.get("npa-operation-id", "")).strip() != operation_id
        ):
            typer.echo(
                f"  Preserved same-name instance {instance_name!r}: it lacks exact "
                f"ownership label npa-operation-id={operation_id}.",
                err=True,
            )
            continue
        instance_id = str(meta.get("id", "")).strip()
        if not instance_id:
            raise ProvisionerError(
                f"Provider returned exact owned orphan {instance_name!r} without an "
                "immutable instance ID; no deletion ran"
            )
        try:
            _run(["compute", "instance", "delete", instance_id], check=True)
            after = _run_json(
                ["compute", "instance", "list", "--parent-id", project_id, "--all"]
            )
        except NebiusError as exc:
            raise ProvisionerError(
                f"Provider rejected or could not verify orphan deletion for exact "
                f"instance {instance_id}: {exc}"
            ) from exc
        remaining = after.get("items", [])
        if not isinstance(remaining, list):
            raise ProvisionerError(
                "Provider returned a schema-invalid instance inventory after deletion"
            )
        if any(
            isinstance(candidate, dict)
            and str((candidate.get("metadata") or {}).get("id") or "").strip()
            == instance_id
            for candidate in remaining
        ):
            raise ProvisionerError(
                f"Provider accepted deletion for orphan instance {instance_id}, but "
                "post-delete inventory still reports it present"
            )
        typer.echo(f"  Verified deleted orphan agent instance {instance_id}")


def _destroy_agent_terraform(
    project: str,
    name: str,
    *,
    record: dict[str, Any] | None = None,
    rollback_operation: bool = False,
    operation_id: str = "",
    project_id: str = "",
) -> None:
    """Destroy one exact state/journal-owned agent dependency graph."""

    from npa.cli import agent as agent_module
    from npa.cli.agent_network import destroy_with_default_security_group_recovery

    if operation_id:
        from npa.provisioning_journal import load_operation

        operations = [load_operation(operation_id)]
    else:
        operations = list_operations(
            project_alias=project,
            project_id=project_id,
            resource_type="agent",
            requested_name=name,
        )
    nonterminal_operations = [
        candidate
        for candidate in operations
        if candidate.read().get("phase")
        not in {"committed", "destroyed", "rolled-back"}
    ]
    candidate_operations = nonterminal_operations or operations
    candidate_project_ids = {
        str(candidate.read().get("project_id") or "")
        for candidate in candidate_operations
        if str(candidate.read().get("project_id") or "")
    }
    if len(candidate_project_ids) > 1:
        raise ProvisionerError(
            "Agent recovery is ambiguous across operation journals for different "
            "Nebius projects. Pass the exact recorded project alias; no resources "
            "were changed."
        )
    operation = candidate_operations[0] if candidate_operations else None
    operation_payload = operation.read() if operation is not None else {}
    if operation is not None and (
        str(operation_payload.get("resource_type") or "") != "agent"
        or str(operation_payload.get("requested_name") or "") != name
    ):
        raise ProvisionerError(
            "The selected operation journal does not own this exact agent name; "
            "no resources were changed."
        )
    # A prior invocation may have completed an exact Terraform no-op/destroy and
    # then stopped while trying to reconcile IAM (notably when deployment failed
    # before an instance ID was ever persisted).  The operation journal survives
    # bucket/config removal and is newer evidence than stale backend settings.
    # Requiring credentials for an already-destroyed backend would deadlock the
    # only remaining, independent IAM cleanup.
    if (
        operation is not None
        and str(operation_payload.get("phase") or "") == "destroyed"
    ):
        return
    journal_alias = str(operation_payload.get("project_alias") or "")
    if project and journal_alias and project != journal_alias:
        raise ProvisionerError(
            "Agent recovery identity mismatch: the selected operation journal "
            "belongs to a different project alias. No resources were changed."
        )
    backend = operation_payload.get("backend")
    backend = dict(backend) if isinstance(backend, dict) else {}
    journal_project_id = str(operation_payload.get("project_id", "") or "")
    record_project_id = str((record or {}).get("project_id", "") or "")
    if project_id and journal_project_id and project_id != journal_project_id:
        raise ProvisionerError(
            "Agent recovery identity mismatch: --project-id and the exact operation "
            "journal name different Nebius projects. No resources were changed."
        )
    if (
        journal_project_id
        and record_project_id
        and journal_project_id != record_project_id
    ):
        raise ProvisionerError(
            "Agent recovery identity mismatch: the local record and operation journal "
            "name different Nebius projects. No resources were changed."
        )
    if operation is not None:
        from npa.clients.credentials import load_credentials

        credentials = load_credentials(environ={})
        journal_bucket = str(backend.get("bucket", "") or "")
        configured_bucket = credentials.s3_bucket.removeprefix("s3://").strip("/")
        project_credentials_match = (
            bool(journal_project_id)
            and credentials.s3_project_id == journal_project_id
            and bool(configured_bucket)
        )
        if project_credentials_match and (
            not journal_bucket
            or configured_bucket == journal_bucket.removeprefix("s3://").strip("/")
        ):
            backend.update(
                {
                    "bucket": journal_bucket or configured_bucket,
                    "access_key": credentials.s3_access_key_id,
                    "secret_key": credentials.s3_secret_access_key,
                    "endpoint": backend.get("endpoint") or credentials.s3_endpoint,
                    "region": backend.get("region")
                    or str(operation_payload.get("region") or ""),
                }
            )
    recovery_record = dict(record or {})
    commands = operation_payload.get("recovery_commands")
    resume_argv = (
        list(commands.get("resume_argv") or []) if isinstance(commands, dict) else []
    )
    if "--ssh-public-key-path" in resume_argv:
        index = resume_argv.index("--ssh-public-key-path") + 1
        if index < len(resume_argv):
            recovery_record.setdefault("ssh_public_key_path", str(resume_argv[index]))
    if journal_project_id:
        recovery_record.setdefault("project_id", journal_project_id)
    recovery_record.setdefault(
        "tenant_id", str(operation_payload.get("tenant_id", "") or "")
    )
    recovery_record.setdefault("region", str(operation_payload.get("region", "") or ""))
    if backend:
        tf_vars = agent_module._resolve_destroy_tf_vars(
            project,
            name,
            recovery_record,
            backend_override={key: str(value or "") for key, value in backend.items()},
        )
    else:
        tf_vars = agent_module._resolve_destroy_tf_vars(project, name, recovery_record)
    tf_vars = {
        key: value
        for key, value in tf_vars.items()
        if key not in agent_module._AGENT_TERRAFORM_RUNTIME_ONLY_VARS
    }
    region = tf_vars["nebius_region"]
    instance_id = str((record or {}).get("instance_id", "")).strip()
    instance_name = tf_vars["instance_name"]
    project_id = tf_vars["nebius_project_id"]
    agent_module._cleanup_agent_ingress(instance_id)

    try:
        state = agent_module.resolve_terraform_state(project)
    except ConfigError:
        state = None
    have_local_state = agent_module._agent_terraform_state_exists(project, name)
    backend_bucket = str(backend.get("bucket") or getattr(state, "bucket", "") or "")
    backend_endpoint = str(
        backend.get("endpoint") or getattr(state, "endpoint", "") or ""
    )
    backend_access = str(
        backend.get("access_key") or getattr(state, "access_key", "") or ""
    )
    backend_secret = str(
        backend.get("secret_key") or getattr(state, "secret_key", "") or ""
    )
    backend_session = str(
        backend.get("session_token") or getattr(state, "session_token", "") or ""
    )
    have_remote_backend = bool(backend_bucket and backend_access and backend_secret)
    if not have_local_state and not have_remote_backend:
        if operation is None:
            raise ProvisionerError(
                "No Terraform state or exact operation ownership record is available; "
                "refusing an unguarded name-based orphan deletion."
            )
        raise ProvisionerError(
            "The exact operation journal is present, but no Terraform state copy "
            "or authenticated remote backend is available to recover the complete "
            "dependency graph. NPA preserved all resources and refused a VM-only "
            "name sweep. Restore project-matched backend credentials and retry: "
            + str(operation_payload.get("recovery_commands", {}).get("destroy", ""))
        )
    tf_dir = provisioner.prepare_working_dir(
        project,
        name,
        bucket=backend_bucket,
        region=region,
        endpoint=backend_endpoint,
    )
    copies = operation.state_copies() if operation is not None else []
    if have_remote_backend:
        provisioner.init(
            tf_dir=tf_dir,
            backend_config={
                "access_key": backend_access,
                "secret_key": backend_secret,
                "session_token": backend_session,
                "endpoint": backend_endpoint,
                "region": region,
                "addressing_style": "path",
            },
        )
        if copies and not provisioner.state_list(tf_dir):
            provisioner.state_push(copies[0], tf_dir)
    elif copies:
        (tf_dir / "backend.tf").unlink(missing_ok=True)
        shutil.rmtree(tf_dir / ".terraform", ignore_errors=True)
        shutil.copy2(copies[0], tf_dir / "terraform.tfstate")
        (tf_dir / "terraform.tfstate").chmod(0o600)
        provisioner.init(tf_dir=tf_dir, disable_backend=True)
    else:
        raise ProvisionerError(
            f"Terraform backend credentials are absent for project {project!r} and "
            "the exact operation journal has no preserved state. No name-based "
            "deletion was attempted. Restore project-matched credentials or use "
            + str(
                operation_payload.get("recovery_commands", {}).get(
                    "resume", "the resume command"
                )
            )
        )

    def _run_destroy() -> None:
        provisioner.destroy(tf_dir=tf_dir, tf_vars=tf_vars)

    destroy_with_default_security_group_recovery(
        run_destroy=_run_destroy,
        cleanup_ingress=lambda: agent_module._cleanup_agent_ingress(instance_id),
        tf_dir=tf_dir,
        tf_vars=tf_vars,
        cleanup_action=f"`npa agent destroy --project {project} --name {name} --yes`",
        on_status=lambda message: typer.echo(f"  {message}", err=True),
    )
    agent_module._cleanup_orphan_agent_instances(
        project_id,
        instance_name,
        operation_id=operation.operation_id if operation is not None else "",
    )
    active_operation = current_operation()
    if operation is not None and operation.read().get("phase") not in {
        "committed",
        "destroyed",
        "rolled-back",
    }:
        if (
            rollback_operation
            and active_operation is not None
            and active_operation.operation_id == operation.operation_id
        ):
            operation.transition("rolled-back")
        elif (
            active_operation is None
            or active_operation.operation_id != operation.operation_id
        ):
            operation.transition("destroyed")
