"""Project-scoped teardown planning and orchestration through existing NPA guards."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import re
import subprocess
import time
from typing import Any, Callable, Mapping

from npa.clients.json_output import parse_single_json_document
from npa.lifecycle_intent import OperationIntent, intent_boundary


Runner = Callable[..., subprocess.CompletedProcess[str]]

PROJECT_DELETE_VERIFY_TIMEOUT_SECONDS = 180.0
PROJECT_DELETE_VERIFY_INTERVAL_SECONDS = 2.0
PROJECT_STABLE_ABSENCE_OBSERVATIONS = 2

_FATAL_INVENTORY_DIAGNOSTIC = re.compile(
    r"(?i)\b(?:unauthenticated|unauthorized|permission denied|access denied|forbidden|"
    r"timeout|timed out|unreachable|connection (?:refused|reset)|transport error|traceback)\b"
)
_BENIGN_INVENTORY_DIAGNOSTIC = re.compile(
    r"(?i)^(?:warning:\s*)?(?:checking for updates|(?:skypilot\s+)?update check|telemetry|"
    r"skypilot usage collection|deprecated output formatting).*$"
)


def _project_bucket_name(project_id: str, state_bucket: str) -> str:
    """Resolve storage only from state or exact project-scoped credentials."""

    from npa.clients.storage_validation import bucket_name

    saved = str(state_bucket or "").strip()
    if saved:
        return bucket_name(saved)
    from npa.clients.project_credential_store import project_credential_record

    record = project_credential_record(project_id, migrate_legacy=False)
    storage = record.get("storage")
    if isinstance(storage, Mapping):
        return bucket_name(storage.get("bucket") or storage.get("s3_bucket") or "")
    # Compatibility adapter for callers/tests that supply a proven legacy view
    # without a writable store. Exact ownership remains mandatory.
    from npa.clients.credentials import load_credentials

    legacy = load_credentials(environ={})
    if legacy.s3_project_id == project_id:
        return bucket_name(legacy.s3_bucket)
    return ""


def _project_storage_iam_generation_ids(
    project: str, project_id: str
) -> tuple[str, ...]:
    """Return every exact NPA-owned storage principal generation for a project."""

    from npa.clients.project_credential_store import project_credential_record
    from npa.teardown_receipts import list_teardown_receipts

    ids: set[str] = set()
    record = project_credential_record(project_id, migrate_legacy=False)
    storage_iam = record.get("storage_iam")
    if isinstance(storage_iam, Mapping):
        generations = storage_iam.get("generations")
        rows = generations if isinstance(generations, list) else [storage_iam]
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            ownership = str(
                row.get("ownership") or row.get("service_account_managed_by") or ""
            ).strip()
            account_id = str(row.get("service_account_id") or "").strip()
            if ownership in {"npa", "npa-recovery-attested"} and account_id:
                ids.add(account_id)
    for receipt in list_teardown_receipts(
        project_alias=project, project_id=project_id, legacy="exclude"
    ):
        identity = receipt.get("identity")
        identity = identity if isinstance(identity, Mapping) else {}
        receipt_iam = identity.get("storage_iam")
        if not isinstance(receipt_iam, Mapping):
            continue
        generations = receipt_iam.get("generations")
        rows = generations if isinstance(generations, list) else [receipt_iam]
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            if str(row.get("project_id") or project_id).strip() != project_id:
                continue
            account_id = str(row.get("service_account_id") or "").strip()
            if row.get("ownership") in {"npa", "npa-recovery-attested"} and account_id:
                ids.add(account_id)
    return tuple(sorted(ids))


def _project_storage_iam_logical_names(project_id: str) -> tuple[str, ...]:
    from npa.clients.project_credential_store import project_credential_record
    from npa.teardown_receipts import list_teardown_receipts

    names: set[str] = set()
    record = project_credential_record(project_id, migrate_legacy=False)
    storage_iam = record.get("storage_iam")
    if isinstance(storage_iam, Mapping):
        generations = storage_iam.get("generations")
        rows = generations if isinstance(generations, list) else [storage_iam]
        for row in rows:
            if isinstance(row, Mapping):
                name = str(row.get("service_account_name") or "").strip()
                if name:
                    names.add(name)
    for receipt in list_teardown_receipts(project_id=project_id, legacy="exclude"):
        identity = receipt.get("identity")
        receipt_iam = (
            identity.get("storage_iam") if isinstance(identity, Mapping) else None
        )
        if not isinstance(receipt_iam, Mapping):
            continue
        generations = receipt_iam.get("generations")
        rows = generations if isinstance(generations, list) else [receipt_iam]
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            if str(row.get("project_id") or project_id).strip() != project_id:
                continue
            name = str(row.get("service_account_name") or "").strip()
            if name:
                names.add(name)
    return tuple(sorted(names))


def _verify_storage_iam_stable_absence(
    *, project_id: str, account_ids: tuple[str, ...], names: tuple[str, ...]
) -> dict[str, Any]:
    """Reject exact-ID residue or a same-name replacement across stable observations."""

    from npa.clients.nebius import (
        get_service_account_id_by_name,
        get_service_account_identity,
    )

    observations: list[dict[str, str]] = []
    for index in range(PROJECT_STABLE_ABSENCE_OBSERVATIONS):
        present: dict[str, str] = {}
        for account_id in account_ids:
            if (
                get_service_account_identity(account_id, project_id=project_id)
                is not None
            ):
                present[f"id:{account_id}"] = account_id
        for name in names:
            replacement = get_service_account_id_by_name(project_id, name, strict=True)
            if replacement:
                present[f"name:{name}"] = replacement
        observations.append(present)
        if present:
            raise RuntimeError(
                "storage IAM stable-absence verification found owned generation or "
                "same-name replacement: " + ", ".join(sorted(present))
            )
        if index + 1 < PROJECT_STABLE_ABSENCE_OBSERVATIONS:
            time.sleep(PROJECT_DELETE_VERIFY_INTERVAL_SECONDS)
    return {
        "stable_absence_observations": len(observations),
        "logical_names": list(names),
    }


def _verify_bucket_stable_absence(
    *, project_id: str, bucket_name: str
) -> dict[str, Any]:
    """Reject a stale completion receipt when the logical bucket is present."""

    from npa.clients.nebius import get_bucket_by_name, get_project_identity

    observations = 0
    for index in range(PROJECT_STABLE_ABSENCE_OBSERVATIONS):
        observations += 1
        project = get_project_identity(project_id)
        if project is not None:
            item = get_bucket_by_name(project_id, bucket_name)
            if item is not None:
                bucket_id = str((item.get("metadata") or {}).get("id") or "")
                raise RuntimeError(
                    "bucket stable-absence verification found a present logical "
                    f"replacement {bucket_name!r} ({bucket_id or 'unknown-id'})"
                )
        if index + 1 < PROJECT_STABLE_ABSENCE_OBSERVATIONS:
            time.sleep(PROJECT_DELETE_VERIFY_INTERVAL_SECONDS)
    return {
        "stable_absence_observations": observations,
        "logical_name": bucket_name,
    }


@dataclass(frozen=True)
class DestroyPhase:
    name: str
    commands: tuple[tuple[str, ...], ...]
    detail: str
    requires: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "phase": self.name,
            "commands": [list(command) for command in self.commands],
            "detail": self.detail,
            "requires": list(self.requires),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ResumeVerification:
    """Decision for a durably completed phase seen during teardown resume.

    A receipt is evidence that an earlier attempt converged, never current-state
    evidence.  Only an affirmative phase verifier may skip replay; every other
    phase re-enters its ordinary identity-scoped command path.
    """

    skip: bool
    evidence: dict[str, Any] = field(default_factory=dict)


def _verify_completed_phase_for_resume(
    phase: DestroyPhase, *, project: str, project_id: str
) -> ResumeVerification:
    """Re-verify stable absence for phases that can safely avoid replay."""

    prior = {"durable_prior_completion": True, "resume_contract": "reverify_or_replay"}
    if phase.name == "bucket":
        bucket_name = str(phase.metadata.get("logical_name") or "").strip()
        if not bucket_name:
            from npa.clients.storage_validation import bucket_name as normalize_bucket

            for command in phase.commands:
                if "--name" in command:
                    position = command.index("--name") + 1
                    if position < len(command):
                        bucket_name = normalize_bucket(command[position])
                        break
        if bucket_name:
            try:
                evidence = _verify_bucket_stable_absence(
                    project_id=project_id, bucket_name=bucket_name
                )
            except (OSError, RuntimeError, ValueError):
                return ResumeVerification(False, prior)
            return ResumeVerification(True, {**prior, **evidence})
    if phase.name == "storage_iam":
        raw_ids = phase.metadata.get("generation_ids", [])
        raw_names = phase.metadata.get("logical_names", [])
        ids = tuple(
            str(value)
            for value in (raw_ids if isinstance(raw_ids, (list, tuple)) else ())
            if str(value).strip()
        )
        names = tuple(
            str(value)
            for value in (raw_names if isinstance(raw_names, (list, tuple)) else ())
            if str(value).strip()
        )
        if ids or names:
            try:
                evidence = _verify_storage_iam_stable_absence(
                    project_id=project_id, account_ids=ids, names=names
                )
            except (OSError, RuntimeError, ValueError):
                return ResumeVerification(False, prior)
            return ResumeVerification(True, {**prior, **evidence})
    if phase.name == "delete_project":
        from npa.clients.nebius import NebiusError, get_project_identity

        observations = 0
        try:
            for index in range(PROJECT_STABLE_ABSENCE_OBSERVATIONS):
                if get_project_identity(project_id) is not None:
                    return ResumeVerification(False, prior)
                observations += 1
                if index + 1 < PROJECT_STABLE_ABSENCE_OBSERVATIONS:
                    time.sleep(PROJECT_DELETE_VERIFY_INTERVAL_SECONDS)
        except (NebiusError, OSError, RuntimeError, ValueError):
            return ResumeVerification(False, prior)
        return ResumeVerification(
            True,
            {
                **prior,
                "stable_absence_observations": observations,
                "project_id": project_id,
            },
        )
    if phase.name == "forget_alias":
        from npa.clients.config import resolve_environment

        try:
            current = resolve_environment(project)
        except (OSError, RuntimeError, ValueError):
            return ResumeVerification(False, prior)
        if current is None:
            return ResumeVerification(
                True, {**prior, "current_alias_state": "verified_absent"}
            )
    return ResumeVerification(False, prior)


def build_project_destroy_plan(
    project: str, *, delete_project: bool = False
) -> list[DestroyPhase]:
    """Build a read-only, exact-identity plan from project-local NPA state."""

    from npa.cli.agent import resolve_project_agents
    from npa.clients.config import resolve_environment, resolve_terraform_state
    from npa.cluster.state import list_local_clusters
    from npa.controller_ownership import controller_owner
    from npa.provisioning_journal import list_operations

    environment = resolve_environment(project)
    if environment is None or not environment.project_id:
        raise RuntimeError(
            f"Project {project!r} has no immutable project identity; refusing teardown."
        )
    project_id = str(environment.project_id)
    agents = resolve_project_agents(project)
    agent_names = set(agents)
    for operation in list_operations(
        project_alias=project, project_id=project_id, resource_type="agent"
    ):
        requested_name = str(operation.read().get("requested_name") or "").strip()
        if requested_name:
            agent_names.add(requested_name)
    agent_commands = tuple(
        (
            "npa",
            "agent",
            "destroy",
            "--project",
            project,
            "--name",
            str(name),
            "--yes",
            "--json",
        )
        for name in sorted(agent_names)
    )
    owner = controller_owner(project)
    controller_commands: tuple[tuple[str, ...], ...] = ()
    if owner is not None:
        controller_commands = (
            (
                "npa",
                "skypilot",
                "cleanup-controller",
                "--project",
                project,
                "--project-id",
                owner.project_id,
                "--context",
                owner.context,
                "--cluster-id",
                owner.cluster_id,
                "--cluster-name",
                owner.cluster_name,
                "--yes",
                "--json",
            ),
        )
    cluster_targets: dict[tuple[str, str], str] = {
        (cluster.name, cluster.cluster_id): ""
        for cluster in list_local_clusters()
        if cluster.project_id == project_id and cluster.cluster_id
    }
    # Retries intentionally produce a new operation and may receive a different
    # immutable cluster ID for the same context.  Inventory every attempt; a
    # destroyed historical ID is harmless audit evidence, while collapsing by
    # context would either forget it or deadlock it against the newer ID.
    for operation in list_operations(
        project_alias=project, project_id=project_id, resource_type="cluster"
    ):
        payload = operation.read()
        rollback = payload.get("rollback")
        rollback = rollback if isinstance(rollback, dict) else {}
        if str(payload.get("phase") or "") in {"destroyed", "rolled-back"} and (
            str(payload.get("phase") or "") == "destroyed"
            or rollback.get("completed") is True
        ):
            # Terminal attempt history is audit evidence, not live inventory.
            # Re-adding its exact ID produces a false unresolved destroy action
            # after provider/local inventory has already converged empty.
            continue
        context = str(payload.get("requested_name") or "").strip()
        for resource in payload.get("resources") or []:
            if not isinstance(resource, dict):
                continue
            cluster_id = str(resource.get("provider_id") or "").strip()
            resource_project = str(resource.get("project_id") or project_id).strip()
            if (
                resource.get("resource_type") == "managed_kubernetes_cluster"
                and context
                and cluster_id
                and resource_project == project_id
            ):
                cluster_targets.setdefault(
                    (context, cluster_id), operation.operation_id
                )
    cluster_commands = tuple(
        (
            "npa",
            "cluster",
            "down",
            "--project",
            project,
            "--project-id",
            project_id,
            "--cluster-id",
            cluster_id,
            "--context",
            context,
            *(("--operation-id", operation_id) if operation_id else ()),
            "--force",
            "--json",
        )
        for (context, cluster_id), operation_id in sorted(cluster_targets.items())
    )
    state = resolve_terraform_state(project)
    bucket_name = _project_bucket_name(project_id, str(state.bucket or ""))
    bucket_commands: tuple[tuple[str, ...], ...] = ()
    if bucket_name:
        bucket_commands = (
            (
                "npa",
                "storage",
                "bucket",
                "delete",
                "--project",
                project,
                "--project-id",
                project_id,
                "--name",
                bucket_name,
                "--yes",
                "--wait",
                "--json",
            ),
        )
    storage_iam_ids = _project_storage_iam_generation_ids(project, project_id)
    storage_iam_names = _project_storage_iam_logical_names(project_id)
    storage_iam_commands: tuple[tuple[str, ...], ...] = tuple(
        (
            "npa",
            "storage",
            "service-account",
            "delete",
            "--project",
            project,
            "--project-id",
            project_id,
            "--id",
            account_id,
            "--yes",
            "--json",
        )
        for account_id in storage_iam_ids
    )
    phases = [
        DestroyPhase(
            "workflows",
            (
                (
                    "npa",
                    "workbench",
                    "workflow",
                    "list",
                    "--project",
                    project,
                    "--json",
                ),
            ),
            "Inventory durable runs, then cancel each exact run before controller teardown.",
        ),
        DestroyPhase(
            "agents", agent_commands, "Destroy every configured project agent."
        ),
        DestroyPhase(
            "controller",
            controller_commands,
            "Remove the exact bound shared controller.",
            ("workflows",),
        ),
        DestroyPhase(
            "clusters",
            cluster_commands,
            "Destroy exact project-matched cluster identities.",
            ("workflows", "controller"),
        ),
        DestroyPhase(
            "bucket",
            bucket_commands,
            "Delete and verify the exact state bucket.",
            ("workflows", "agents", "controller", "clusters"),
            {"project_id": project_id, "logical_name": bucket_name},
        ),
        DestroyPhase(
            "storage_iam",
            storage_iam_commands,
            "Delete only project-scoped NPA-owned storage IAM.",
            ("workflows", "agents", "controller", "clusters", "bucket"),
            {
                "project_id": project_id,
                "generation_ids": list(storage_iam_ids),
                "logical_names": list(storage_iam_names),
            },
        ),
    ]
    if delete_project:
        ownership = _project_ownership_operation(
            project, project_id, str(getattr(environment, "tenant_id", "") or "")
        )
        phases.append(
            DestroyPhase(
                "network",
                (
                    (
                        "npa",
                        "network",
                        "delete-project-default",
                        "--project",
                        project,
                        "--project-id",
                        project_id,
                        "--tenant-id",
                        str(getattr(environment, "tenant_id", "") or ""),
                        "--yes",
                        "--json",
                    ),
                ),
                "Delete only the exact default topology of the NPA-created project.",
                ("agents", "clusters"),
            )
        )
        phases.append(
            DestroyPhase(
                "delete_project",
                (),
                "Delete the exact provider project only after ownership and empty-child inventory are proven.",
                (
                    "workflows",
                    "agents",
                    "controller",
                    "clusters",
                    "bucket",
                    "storage_iam",
                    "network",
                ),
                {
                    "deletion_requested": True,
                    "project_id": project_id,
                    "ownership_proven": ownership is not None,
                    "ownership_operation_id": ownership.operation_id
                    if ownership is not None
                    else "",
                    "provider_inventory": "required_before_mutation",
                },
            )
        )
    phases.extend(
        [
            DestroyPhase(
                "local_cleanup",
                (
                    (
                        "npa",
                        "cleanup",
                        "--project",
                        project,
                        "--yes",
                        "--keep-sky",
                        "--json",
                    ),
                ),
                "Remove only project-scoped local residue; preserve shared runtime state.",
                (("delete_project",) if delete_project else ()),
            ),
            DestroyPhase(
                "forget_alias",
                (("npa", "configure", "--forget-project", project),),
                "Forget the alias only after cloud and IAM phases converge.",
                (
                    "workflows",
                    "agents",
                    "controller",
                    "clusters",
                    "bucket",
                    "storage_iam",
                    *(("network",) if delete_project else ()),
                    *(("delete_project",) if delete_project else ()),
                    "local_cleanup",
                ),
            ),
            DestroyPhase(
                "final_audit",
                (
                    (
                        "npa",
                        "cleanup",
                        "--project",
                        project_id,
                        "--full",
                        "--yes",
                        "--include-sky",
                        "--skip-jobs",
                        "--attest-no-active-jobs",
                        "--json",
                    ),
                ),
                "Remove remaining exact-project secrets after owned jobs/controllers are "
                "stopped; shared machine SkyPilot state remains preserved.",
                ("forget_alias",),
            ),
        ]
    )
    return phases


def build_receipt_project_delete_plan(
    *,
    project: str,
    project_id: str,
    tenant_id: str,
    receipt_id: str,
) -> list[DestroyPhase]:
    """Build the narrow post-forget project-deletion recovery plan."""

    ownership = _project_ownership_operation(project, project_id, tenant_id)
    return [
        DestroyPhase(
            "network",
            (
                (
                    "npa",
                    "network",
                    "delete-project-default",
                    "--project",
                    project,
                    "--project-id",
                    project_id,
                    "--tenant-id",
                    tenant_id,
                    "--yes",
                    "--json",
                ),
            ),
            "Delete only the exact default topology of the NPA-created project.",
        ),
        DestroyPhase(
            "delete_project",
            (),
            "Delete the exact provider project after receipt identity, ownership, and empty-child inventory are proven.",
            ("network",),
            metadata={
                "deletion_requested": True,
                "project_id": project_id,
                "receipt_id": receipt_id,
                "identity_source": "durable_teardown_receipt",
                "ownership_proven": ownership is not None,
                "ownership_operation_id": ownership.operation_id
                if ownership is not None
                else "",
                "provider_inventory": "required_before_mutation",
            },
        ),
        DestroyPhase(
            "final_audit",
            (
                (
                    "npa",
                    "cleanup",
                    "--project",
                    project_id,
                    "--full",
                    "--yes",
                    "--include-sky",
                    "--skip-jobs",
                    "--attest-no-active-jobs",
                    "--json",
                ),
            ),
            "Finish the exact-project local secret and operational-residue audit after alias removal.",
            ("delete_project",),
        ),
    ]


def _project_ownership_operation(project: str, project_id: str, tenant_id: str):
    """Return exact durable NPA project-creation proof, or ``None``."""

    from npa.provisioning_journal import list_operations

    matches = []
    for operation in list_operations(project_id=project_id):
        payload = operation.read()
        if str(payload.get("tenant_id") or "") != tenant_id:
            continue
        for resource in payload.get("resources") or []:
            if not isinstance(resource, dict):
                continue
            labels = resource.get("labels")
            if (
                resource.get("resource_type") == "nebius_project"
                and resource.get("provider_id") == project_id
                and resource.get("project_id") == project_id
                and resource.get("ownership") == "created_by_this_operation"
                and resource.get("ownership_source") == "provider-create-response"
                and isinstance(labels, dict)
                and labels.get("tenant_id") == tenant_id
            ):
                matches.append(operation)
    return matches[0] if len(matches) == 1 else None


def _delete_owned_empty_project(
    *,
    project: str,
    project_id: str,
    tenant_id: str,
    region: str,
    profile: str = "",
) -> dict[str, Any]:
    """Guard, journal, delete, and verify one exact NPA-owned empty project."""

    from npa.clients.nebius import (
        NebiusError,
        delete_project,
        get_project_identity,
        list_project_dependencies,
    )
    from npa.teardown_receipts import record_teardown_event

    ownership = _project_ownership_operation(project, project_id, tenant_id)
    if ownership is None:
        raise RuntimeError(
            "exact project is external, shared, or lacks unique durable NPA creation proof"
        )
    identity = get_project_identity(
        project_id, tenant_id=tenant_id, profile=profile or None
    )
    if identity is None:
        record_teardown_event(
            phase="project",
            resource=project_id,
            terminal_state="verified_absent",
            project_alias=project,
            project_id=project_id,
            identity={
                "project_id": project_id,
                "tenant_id": tenant_id,
                "region": region,
            },
            verification={"exact_project_absent": True},
        )
        return {"outcome": "already_absent", "project_id": project_id}
    if identity.region != region:
        raise RuntimeError(
            f"provider project region {identity.region} does not match durable region {region}"
        )
    dependency_observations: list[dict[str, list[str]]] = []
    for observation_index in range(PROJECT_STABLE_ABSENCE_OBSERVATIONS):
        dependencies = list_project_dependencies(project_id, profile=profile or None)
        remaining = {kind: list(ids) for kind, ids in dependencies.items() if ids}
        dependency_observations.append(remaining)
        if remaining:
            raise RuntimeError(
                "provider dependency inventory is nonempty: "
                + ", ".join(
                    f"{kind}={len(ids)}" for kind, ids in sorted(remaining.items())
                )
            )
        if observation_index + 1 < PROJECT_STABLE_ABSENCE_OBSERVATIONS:
            time.sleep(PROJECT_DELETE_VERIFY_INTERVAL_SECONDS)
    # This durable intent receipt is a precondition to mutation. If it cannot be
    # written, deletion does not run. It carries only exact, non-secret identity.
    record_teardown_event(
        phase="project",
        resource=project_id,
        terminal_state="deletion_approved",
        project_alias=project,
        project_id=project_id,
        identity={
            "project_id": project_id,
            "tenant_id": tenant_id,
            "region": region,
            "project_operation_id": ownership.operation_id,
            "ownership": "npa_disposable_project",
        },
        precheck={
            "provider_identity_verified": True,
            "child_inventory": "verified_empty",
            "stable_empty_observations": len(dependency_observations),
        },
        action={"kind": "delete_exact_project", "project_id": project_id},
    )
    try:
        delete_project(project_id, profile=profile or None)
        deadline = time.monotonic() + PROJECT_DELETE_VERIFY_TIMEOUT_SECONDS
        absent_observations = 0
        while True:
            after = get_project_identity(
                project_id, tenant_id=tenant_id, profile=profile or None
            )
            if after is None:
                absent_observations += 1
            else:
                absent_observations = 0
            if (
                absent_observations >= PROJECT_STABLE_ABSENCE_OBSERVATIONS
                or time.monotonic() >= deadline
            ):
                break
            time.sleep(PROJECT_DELETE_VERIFY_INTERVAL_SECONDS)
    except NebiusError as exc:
        raise RuntimeError(f"exact provider project deletion failed: {exc}") from exc
    if after is not None:
        raise RuntimeError(
            "provider accepted deletion but exact project remains present; "
            f"last_observation=present stable_absence_observations={absent_observations}"
        )
    if absent_observations < PROJECT_STABLE_ABSENCE_OBSERVATIONS:
        raise RuntimeError(
            "provider accepted deletion and the last observation was absent, but stable "
            "absence could not be established before the verification deadline; "
            f"last_observation=absent stable_absence_observations={absent_observations} "
            f"required={PROJECT_STABLE_ABSENCE_OBSERVATIONS}"
        )
    record_teardown_event(
        phase="project",
        resource=project_id,
        terminal_state="verified_deleted",
        project_alias=project,
        project_id=project_id,
        identity={"project_id": project_id, "tenant_id": tenant_id, "region": region},
        verification={
            "exact_project_absent": True,
            "child_inventory": "verified_empty",
            "stable_absence_observations": absent_observations,
        },
    )
    return {"outcome": "verified_deleted", "project_id": project_id}


def _run(command: tuple[str, ...], runner: Runner) -> subprocess.CompletedProcess[str]:
    argv = (
        _internal_command_argv(command) if runner is subprocess.run else list(command)
    )
    try:
        kwargs: dict[str, Any] = {}
        if runner is subprocess.run:
            from npa.provisioning_journal import current_operation

            operation = current_operation()
            if operation is not None:
                child_env = os.environ.copy()
                child_env["NPA_PARENT_LIFECYCLE_OPERATION"] = operation.operation_id
                child_env["NPA_OPERATION_INTENT"] = "destroy"
                kwargs["env"] = child_env
            else:
                child_env = os.environ.copy()
                child_env["NPA_OPERATION_INTENT"] = "destroy"
                kwargs["env"] = child_env
        return runner(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            **kwargs,
        )
    except OSError as exc:
        return subprocess.CompletedProcess(
            argv,
            127,
            stdout="",
            stderr=f"{type(exc).__name__}: NPA command could not be started",
        )


def _internal_command_argv(command: tuple[str, ...]) -> list[str]:
    """Invoke NPA with this process's interpreter, independent of ``PATH``.

    Plans and recovery receipts intentionally retain the operator-facing
    ``npa ...`` spelling.  Only in-process orchestration replaces that first
    token, without a shell, so editable installs and console-entrypoint installs
    execute the same imported package under the active environment.
    """

    if not command or command[0] != "npa":
        return list(command)
    from npa.cli.invocation import internal_cli_argv

    return internal_cli_argv(command[1:])


def _parse_workflow_inventory(
    completed: subprocess.CompletedProcess[str],
) -> tuple[list[dict[str, Any]] | None, str]:
    """Classify the exact project workflow inventory without exit-code races."""

    payload = parse_single_json_document(completed.stdout or "")
    rows = payload.get("runs") if isinstance(payload, dict) else None
    if not isinstance(payload, dict) or not isinstance(rows, list):
        return None, "workflow inventory returned ambiguous JSON"
    if set(payload) != {"runs"}:
        return None, "workflow inventory returned ambiguous JSON"
    if any(
        not isinstance(row, dict) or not str(row.get("run_id") or "").strip()
        for row in rows
    ):
        return None, "workflow inventory returned schema-invalid run rows"
    diagnostic = str(completed.stderr or "").strip()
    if _FATAL_INVENTORY_DIAGNOSTIC.search(diagnostic):
        return (
            None,
            "workflow inventory failed with authentication, permission, or transport evidence",
        )
    if completed.returncode != 0:
        if rows:
            return None, "workflow inventory exited nonzero with nonempty evidence"
        unknown = [
            line.strip()
            for line in diagnostic.splitlines()
            if line.strip() and not _BENIGN_INVENTORY_DIAGNOSTIC.fullmatch(line.strip())
        ]
        if unknown:
            return None, "workflow inventory exited nonzero with unknown diagnostics"
    return [dict(row) for row in rows], ""


def _stream_kind(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "empty"
    return "json" if parse_single_json_document(text) is not None else "text"


def _completed_argv(
    completed: subprocess.CompletedProcess[str], logical: tuple[str, ...]
) -> list[str]:
    """Return the exact shell-free argv supplied to the subprocess runner."""

    args = completed.args
    if isinstance(args, (list, tuple)):
        return [str(value) for value in args]
    # Production calls are always argv lists. Retain a safe logical fallback for
    # injected legacy runners instead of splitting a shell-like string.
    return list(logical)


def _command_failure_detail(completed: subprocess.CompletedProcess[str]) -> str:
    """Preserve the primary subprocess error without exposing credential text."""

    from npa.orchestration.skypilot.workflow_state import redact_text

    stderr = str(completed.stderr or "").strip()
    stdout = str(completed.stdout or "").strip()
    stream = "stderr" if stderr else "stdout" if stdout else "none"
    detail = stderr or stdout or "no subprocess output"
    return (
        f"command failed (exit {completed.returncode}); primary_{stream}: "
        f"{redact_text(detail)[:500]}"
    )


def _redacted_stream_summary(value: object) -> str:
    """Keep bounded diagnostic evidence without persisting secret-bearing output."""

    from npa.orchestration.skypilot.workflow_state import redact_text

    return redact_text(str(value or "").strip())[:500]


def _command_evidence(
    completed: subprocess.CompletedProcess[str], command: tuple[str, ...]
) -> dict[str, Any]:
    return {
        "argv": _completed_argv(completed, command),
        "exit_code": completed.returncode,
        "stdout_kind": _stream_kind(completed.stdout),
        "stderr_kind": _stream_kind(completed.stderr),
        "stdout_summary": _redacted_stream_summary(completed.stdout),
        "stderr_summary": _redacted_stream_summary(completed.stderr),
    }


@intent_boundary(OperationIntent.DESTROY)
def execute_project_destroy(
    project: str,
    phases: list[DestroyPhase],
    *,
    runner: Runner = subprocess.run,
    on_phase: Callable[[str], None] | None = None,
    exact_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute every independent phase and report complete/partial convergence."""

    from npa.clients.config import resolve_environment

    environment = resolve_environment(project)
    supplied = dict(exact_identity or {})
    if supplied:
        project_id = str(supplied.get("project_id") or "").strip()
        tenant_id = str(supplied.get("tenant_id") or "").strip()
        region = str(supplied.get("region") or "").strip()
        profile = str(supplied.get("profile") or "").strip()
        if not project_id or not tenant_id or not region:
            raise RuntimeError(
                "durable project recovery identity requires exact project, tenant, and region IDs"
            )
        if environment is not None:
            configured = {
                "project_id": str(getattr(environment, "project_id", "") or ""),
                "tenant_id": str(getattr(environment, "tenant_id", "") or ""),
                "region": str(getattr(environment, "region", "") or ""),
            }
            conflicts = [
                key
                for key, value in configured.items()
                if value
                and value
                != {"project_id": project_id, "tenant_id": tenant_id, "region": region}[
                    key
                ]
            ]
            if conflicts:
                raise RuntimeError(
                    "live project configuration conflicts with durable receipt identity: "
                    + ", ".join(conflicts)
                )
    else:
        if environment is None or not environment.project_id:
            raise RuntimeError(
                f"Project {project!r} has no immutable project identity; refusing teardown."
            )
        project_id = str(environment.project_id)
        tenant_id = str(getattr(environment, "tenant_id", "") or "")
        region = str(getattr(environment, "region", "") or "")
        profile = ""
    results: list[dict[str, Any]] = []
    statuses: dict[str, str] = {}
    from npa.teardown_receipts import list_teardown_receipts

    completed_receipt_phases = {
        str(event.get("phase") or "")
        for receipt in list_teardown_receipts(
            project_alias=project, project_id=project_id, legacy="exclude"
        )
        for event in receipt.get("events", [])
        if isinstance(event, dict) and event.get("terminal_state") == "completed"
    }
    for phase in phases:
        resume_evidence: dict[str, Any] = {}
        if on_phase:
            on_phase(phase.name)
        if f"project_destroy_{phase.name}" in completed_receipt_phases:
            resume = _verify_completed_phase_for_resume(
                phase, project=project, project_id=project_id
            )
            resume_evidence = dict(resume.evidence)
            if resume.skip:
                statuses[phase.name] = "completed"
                results.append(
                    {
                        "phase": phase.name,
                        "status": "completed",
                        "commands": [],
                        "errors": [],
                        "warnings": [],
                        "recovery_commands": [],
                        "blocked_by": [],
                        "evidence": {**resume.evidence, "command_results": []},
                    }
                )
                continue
        commands = list(phase.commands)
        phase_errors: list[str] = []
        phase_warnings: list[str] = []
        phase_evidence: dict[str, Any] = dict(resume_evidence)
        command_results: list[dict[str, Any]] = []
        executed: list[list[str]] = []
        recovery_commands: list[list[str]] = []
        blocked_by = [
            dependency
            for dependency in phase.requires
            if statuses.get(dependency) not in {"completed", "degraded"}
        ]
        if blocked_by:
            phase_errors.append("dependency not converged: " + ", ".join(blocked_by))
            recovery_commands.extend([list(command) for command in commands])
        elif phase.name == "delete_project":
            from npa.clients.nebius import NebiusError

            try:
                phase_evidence = _delete_owned_empty_project(
                    project=project,
                    project_id=project_id,
                    tenant_id=tenant_id,
                    region=region,
                    profile=profile,
                )
                executed.append(["provider-adapter", "delete-project", project_id])
            except (NebiusError, OSError, RuntimeError, ValueError) as exc:
                phase_errors.append(str(exc))
        elif phase.name == "workflows" and commands:
            inventory = _run(commands[0], runner)
            executed.append(list(commands[0]))
            command_results.append(_command_evidence(inventory, commands[0]))
            rows, inventory_error = _parse_workflow_inventory(inventory)
            if rows is None:
                phase_errors.append(
                    f"{inventory_error}: {_command_failure_detail(inventory)}"
                )
                recovery_commands.append(list(commands[0]))
            else:
                for row in rows:
                    run_id = str(row.get("run_id") or "")
                    submission_state = str(
                        row.get("submission_state")
                        or row.get("status")
                        or row.get("submission_status")
                        or ""
                    ).upper()
                    if submission_state in {"NOT_SUBMITTED", "PLAN_ONLY"}:
                        continue
                    cancel_command = (
                        "npa",
                        "workbench",
                        "workflow",
                        "cancel",
                        run_id,
                        "--project",
                        project,
                        "--json",
                    )
                    completed = _run(cancel_command, runner)
                    executed.append(list(cancel_command))
                    command_results.append(_command_evidence(completed, cancel_command))
                    if completed.returncode != 0:
                        phase_errors.append(
                            f"workflow cancellation failed for {run_id}: "
                            + _command_failure_detail(completed)
                        )
                        recovery_commands.append(list(cancel_command))
        else:
            for command in commands:
                completed = _run(command, runner)
                executed.append(list(command))
                command_results.append(_command_evidence(completed, command))
                parsed = parse_single_json_document(completed.stdout or "")
                remote_only_converged = bool(
                    phase.name == "controller"
                    and isinstance(parsed, dict)
                    and parsed.get("outcome") == "degraded_local_metadata"
                    and parsed.get("remote_absence_verified") is True
                )
                infrastructure_only_converged = bool(
                    phase.name == "agents"
                    and isinstance(parsed, dict)
                    and parsed.get("infrastructure_absent") is True
                    and parsed.get("iam_cleanup_complete") is False
                )
                if remote_only_converged:
                    phase_warnings.append(
                        "exact remote controller absence verified; stale local "
                        "metadata remains for idempotent reconciliation"
                    )
                elif infrastructure_only_converged:
                    phase_warnings.append(
                        "exact agent infrastructure absence verified; agent IAM cleanup remains partial"
                    )
                elif completed.returncode != 0:
                    phase_errors.append(_command_failure_detail(completed))
                    recovery_commands.append(list(command))
            if phase.name == "storage_iam" and not phase_errors:
                raw_generation_ids = phase.metadata.get("generation_ids", [])
                raw_logical_names = phase.metadata.get("logical_names", [])
                generation_ids = tuple(
                    str(value)
                    for value in (
                        raw_generation_ids
                        if isinstance(raw_generation_ids, (list, tuple))
                        else []
                    )
                    if str(value).strip()
                )
                logical_names = tuple(
                    str(value)
                    for value in (
                        raw_logical_names
                        if isinstance(raw_logical_names, (list, tuple))
                        else []
                    )
                    if str(value).strip()
                )
                if generation_ids or logical_names:
                    try:
                        phase_evidence.update(
                            _verify_storage_iam_stable_absence(
                                project_id=project_id,
                                account_ids=generation_ids,
                                names=logical_names,
                            )
                        )
                    except (OSError, RuntimeError, ValueError) as exc:
                        phase_errors.append(str(exc))
                else:
                    phase_evidence.update(
                        {
                            "outcome": "verified_nothing_to_do",
                            "identity_source": "exact_project_records_and_receipts",
                            "generation_ids": [],
                        }
                    )
        phase_status = (
            "skipped_dependency"
            if blocked_by
            else "degraded"
            if phase_warnings and not phase_errors
            else "completed"
            if not phase_errors
            else "partial"
        )
        statuses[phase.name] = phase_status
        results.append(
            {
                "phase": phase.name,
                "status": phase_status,
                "commands": executed,
                "errors": phase_errors,
                "warnings": phase_warnings,
                "recovery_commands": recovery_commands,
                "blocked_by": blocked_by,
                "evidence": {**phase_evidence, "command_results": command_results},
            }
        )
        try:
            from npa.teardown_receipts import record_teardown_event

            record_teardown_event(
                phase=f"project_destroy_{phase.name}",
                resource=project,
                terminal_state=(
                    "degraded_local_metadata"
                    if phase_warnings and not phase_errors
                    else "completed"
                    if not phase_errors
                    else "partial"
                ),
                project_alias=project,
                project_id=project_id,
                precheck={"planned_command_count": len(commands)},
                action={"kind": "npa_guarded_phase", "executed_count": len(executed)},
                verification={
                    "converged": not phase_errors,
                    "remote_absence_only": bool(phase_warnings),
                },
                errors=phase_errors,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            phase_errors.append(f"receipt write failed: {type(exc).__name__}")
            statuses[phase.name] = "partial"
            results[-1]["status"] = "partial"
            results[-1]["errors"] = phase_errors
    complete = all(result["status"] == "completed" for result in results)
    return {
        "status": "success" if complete else "partial",
        "project": project,
        "phases": results,
    }
