"""Atomic project-scoped ownership for the shared SkyPilot jobs controller."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

from npa.clients.config import (
    CONFIG_PATH,
    default_project_name,
    resolve_environment,
    update_config_document,
)
from npa.cluster.state import load_cluster_state


class ClusterOwnerIdentityMismatchError(RuntimeError):
    """A controller operation targets a different immutable cluster owner."""


class ControllerIdentityUnavailableError(ClusterOwnerIdentityMismatchError):
    """Exact local/provider evidence was unavailable, so ownership is unchanged."""


_TERMINAL_CLUSTER_STATES = frozenset(
    {
        "ABSENT",
        "DELETED",
        "DELETING",
        "DESTROYED",
        "NOT_FOUND",
        "ROLLED_BACK",
        "ROLLED-BACK",
    }
)


@dataclass(frozen=True)
class ControllerOwner:
    project_alias: str
    project_id: str
    cluster_id: str
    cluster_name: str
    context: str
    context_fingerprint: str
    mode: str = "kubernetes"
    namespace: str = "sky-system"
    name: str = "jobs-controller"
    operation_id: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "schema_version": "npa.controller-owner.v1",
            "project_alias": self.project_alias,
            "project_id": self.project_id,
            "cluster_id": self.cluster_id,
            "cluster_name": self.cluster_name,
            "context": self.context,
            "context_fingerprint": self.context_fingerprint,
            "mode": self.mode,
            "namespace": self.namespace,
            "name": self.name,
            "operation_id": self.operation_id,
        }


def _owner_from_mapping(value: object) -> ControllerOwner | None:
    if not isinstance(value, dict):
        return None
    required = ("project_alias", "project_id", "cluster_id", "context")
    if any(not str(value.get(key) or "").strip() for key in required):
        return None
    return ControllerOwner(
        project_alias=str(value.get("project_alias") or ""),
        project_id=str(value.get("project_id") or ""),
        cluster_id=str(value.get("cluster_id") or ""),
        cluster_name=str(value.get("cluster_name") or ""),
        context=str(value.get("context") or ""),
        context_fingerprint=str(value.get("context_fingerprint") or ""),
        mode=str(value.get("mode") or "kubernetes"),
        namespace=str(value.get("namespace") or "sky-system"),
        name=str(value.get("name") or "jobs-controller"),
        operation_id=str(value.get("operation_id") or ""),
    )


def resolve_controller_candidate(project: str, context: str) -> ControllerOwner:
    alias = str(project or default_project_name() or "").strip()
    if not alias or not context:
        raise ClusterOwnerIdentityMismatchError(
            "Controller ownership requires an exact project alias and Kubernetes context."
        )
    environment = resolve_environment(alias)
    if environment is None or not environment.project_id:
        raise ClusterOwnerIdentityMismatchError(
            f"Project {alias!r} has no immutable project identity."
        )
    cluster = load_cluster_state(context)
    if cluster is None:
        raise ClusterOwnerIdentityMismatchError(
            f"No NPA cluster identity exists for context {context!r}; refusing controller adoption."
        )
    if cluster.project_id != environment.project_id:
        raise ClusterOwnerIdentityMismatchError(
            f"Context {context!r} belongs to project {cluster.project_id!r}, not "
            f"selected project {environment.project_id!r}."
        )
    cluster_state = str(cluster.last_seen_state or "").strip().upper()
    if cluster_state in _TERMINAL_CLUSTER_STATES:
        raise ClusterOwnerIdentityMismatchError(
            f"Context {context!r} is recorded as {cluster_state}; refusing controller adoption. "
            "Provision or adopt a live exact cluster before binding."
        )
    operation_id = ""
    try:
        from npa.provisioning_journal import current_operation, list_operations

        operation = current_operation()
        candidates = (
            [operation]
            if operation is not None
            else list_operations(
                project_alias=alias,
                project_id=environment.project_id,
                resource_type="cluster",
                requested_name=context,
            )
        )
        for operation in candidates:
            payload = operation.read()
            if any(
                isinstance(item, dict)
                and item.get("resource_type") == "managed_kubernetes_cluster"
                and str(item.get("provider_id") or "") == cluster.cluster_id
                for item in payload.get("resources", [])
            ):
                operation_id = operation.operation_id
                break
    except (OSError, RuntimeError, ValueError):
        operation_id = ""
    digest = hashlib.sha256()
    for value in (environment.project_id, cluster.cluster_id, context):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return ControllerOwner(
        project_alias=alias,
        project_id=environment.project_id,
        cluster_id=cluster.cluster_id,
        cluster_name=cluster.name,
        context=context,
        context_fingerprint=digest.hexdigest(),
        operation_id=operation_id,
    )


def _assert_owner_operation_is_live(candidate: ControllerOwner) -> None:
    if not candidate.operation_id:
        return
    try:
        from npa.provisioning_journal import load_operation

        payload = load_operation(candidate.operation_id).read()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ControllerIdentityUnavailableError(
            "The controller owner operation journal cannot be verified; ownership "
            f"was not changed: {type(exc).__name__}: {exc}"
        ) from exc
    phase = str(payload.get("phase") or "").strip().lower()
    if phase in {"destroyed", "rolled-back"}:
        raise ClusterOwnerIdentityMismatchError(
            f"Controller owner operation {candidate.operation_id} is {phase}; "
            "refusing to bind a cluster destroyed by that attempt."
        )
    matches = [
        item
        for item in payload.get("resources", [])
        if isinstance(item, dict)
        and item.get("resource_type") == "managed_kubernetes_cluster"
        and str(item.get("provider_id") or "") == candidate.cluster_id
        and str(item.get("project_id") or candidate.project_id) == candidate.project_id
    ]
    if not matches:
        raise ClusterOwnerIdentityMismatchError(
            "Controller owner operation does not prove the exact recorded cluster/project identity."
        )


def verify_live_controller_candidate(candidate: ControllerOwner) -> ControllerOwner:
    """Require exact local and provider evidence for one live controller owner."""

    local = resolve_controller_candidate(candidate.project_alias, candidate.context)
    if not _same_immutable_owner(local, candidate):
        raise ClusterOwnerIdentityMismatchError(
            "The proposed controller owner no longer matches exact local cluster state."
        )
    _assert_owner_operation_is_live(candidate)
    try:
        from npa.cluster.identity import (
            ClusterIdentityError,
            resolve_verified_cluster_identity,
        )

        verified = resolve_verified_cluster_identity(
            project=candidate.project_alias,
            context=candidate.context,
        )
    except ClusterIdentityError as exc:
        raise ControllerIdentityUnavailableError(str(exc)) from exc
    if verified.cluster_absent:
        raise ClusterOwnerIdentityMismatchError(
            f"Provider reports cluster {candidate.cluster_id} absent; refusing controller binding."
        )
    if (
        verified.project_id != candidate.project_id
        or verified.cluster_id != candidate.cluster_id
        or verified.context != candidate.context
    ):
        raise ClusterOwnerIdentityMismatchError(
            "Provider identity does not match the exact proposed controller owner; "
            "ownership was not changed."
        )
    return candidate


def ensure_controller_owner(project: str, context: str) -> ControllerOwner:
    """Verify and atomically bind an unowned exact cluster, or compare an owner."""

    from npa.lifecycle_intent import forbid_destructive_provisioning

    forbid_destructive_provisioning("ensure_controller_owner")

    candidate = verify_live_controller_candidate(
        resolve_controller_candidate(project, context)
    )
    existing = controller_owner()
    if existing is not None and not _same_immutable_owner(existing, candidate):
        raise ClusterOwnerIdentityMismatchError(
            "Shared controller owner mismatch: recorded "
            f"{existing.project_alias}/{existing.context}/{existing.cluster_id}, requested "
            f"{candidate.project_alias}/{candidate.context}/{candidate.cluster_id}. "
            "Finish or cancel the recorded owner's jobs, clean up its controller, then "
            "run provisioning again; Terraform was not changed."
        )
    return bind_controller_owner(candidate)


def controller_preflight(project: str, context: str) -> tuple[str, str]:
    """Classify ownership before paid mutation.

    Returns ``(ready|blocked|unknown, reason)``. A missing cluster with no saved
    owner is the only safely bindable-after-create case.
    """

    try:
        existing = controller_owner(strict=True)
    except (ClusterOwnerIdentityMismatchError, ControllerIdentityUnavailableError) as exc:
        return (
            "blocked",
            f"controller owner configuration is unsafe: {exc}; reconcile the exact "
            "recorded owner before provisioning",
        )
    except Exception as exc:  # noqa: BLE001 - total preflight must fail closed
        return (
            "unknown",
            f"controller owner resolution failed for project={project!r} "
            f"context={context!r}: {type(exc).__name__}: {exc}",
        )
    try:
        cluster = load_cluster_state(context)
    except Exception as exc:  # noqa: BLE001 - corrupt/unreadable state blocks mutation
        return (
            "blocked",
            f"cluster state for context {context!r} is unreadable: "
            f"{type(exc).__name__}: {exc}; repair or remove only that exact stale state",
        )
    if cluster is None:
        if existing is None:
            return "ready", "unowned new cluster will be bound after exact provider identity is durable"
        return (
            "blocked",
            "recorded controller owner references missing/stale cluster state; clean up "
            "that exact controller owner before provisioning a replacement",
        )
    try:
        candidate = verify_live_controller_candidate(
            resolve_controller_candidate(project, context)
        )
    except ControllerIdentityUnavailableError as exc:
        return "blocked", f"exact controller candidate cannot be verified: {exc}"
    except ClusterOwnerIdentityMismatchError as exc:
        return "blocked", str(exc)
    if existing is None:
        return "ready", f"exact live cluster {candidate.cluster_id} is safely bindable"
    try:
        verify_live_controller_candidate(existing)
    except ControllerIdentityUnavailableError as exc:
        return "blocked", f"recorded controller owner cannot be verified: {exc}"
    except ClusterOwnerIdentityMismatchError as exc:
        return "blocked", str(exc)
    if not _same_immutable_owner(existing, candidate):
        return (
            "blocked",
            "shared controller belongs to "
            f"{existing.project_alias}/{existing.context}/{existing.cluster_id}; "
            "cancel/finish its jobs and run `npa skypilot cleanup-controller --project "
            f"{existing.project_alias} --context {existing.context} --yes` before provisioning",
        )
    return "ready", f"compatible owner verified for exact cluster {candidate.cluster_id}"


def _same_immutable_owner(left: ControllerOwner, right: ControllerOwner) -> bool:
    return (
        left.project_id,
        left.cluster_id,
        left.context_fingerprint,
        left.mode,
        left.namespace,
        left.name,
    ) == (
        right.project_id,
        right.cluster_id,
        right.context_fingerprint,
        right.mode,
        right.namespace,
        right.name,
    )


def _configured_owner(payload: dict[str, Any]) -> ControllerOwner | None:
    skypilot = payload.get("skypilot")
    owner = _owner_from_mapping(
        skypilot.get("controller_owner") if isinstance(skypilot, dict) else None
    )
    if owner is not None:
        return owner
    projects = payload.get("projects")
    legacy = {
        candidate
        for candidate in (
            _owner_from_mapping(project.get("controller_owner"))
            for project in (projects.values() if isinstance(projects, dict) else [])
            if isinstance(project, dict)
        )
        if candidate is not None
    }
    if len(legacy) > 1:
        raise ClusterOwnerIdentityMismatchError(
            "Multiple legacy controller owners are recorded; explicit reconciliation is required."
        )
    return next(iter(legacy), None)


def controller_owner(project: str = "", *, strict: bool = False) -> ControllerOwner | None:
    import yaml

    try:
        payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        payload = {}
    except (OSError, yaml.YAMLError) as exc:
        if strict:
            raise ControllerIdentityUnavailableError(
                f"NPA configuration {CONFIG_PATH} could not be read: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return None
    if not isinstance(payload, dict):
        if strict:
            raise ClusterOwnerIdentityMismatchError(
                "NPA configuration root is not a mapping; controller ownership is ambiguous."
            )
        return None
    owner = _configured_owner(payload)
    if owner is None or not project:
        return owner
    if owner.project_alias == project:
        return owner
    try:
        environment = resolve_environment(project)
        return (
            owner
            if environment is not None and environment.project_id == owner.project_id
            else None
        )
    except Exception:  # noqa: BLE001 - lookup convenience, immutable ID still governs
        return None


def bind_controller_owner(
    candidate: ControllerOwner, *, allow_rebind: bool = False
) -> ControllerOwner:
    """Atomically create or compare-and-swap one controller owner record."""

    def mutate(payload: dict[str, Any]) -> dict[str, Any]:
        projects = payload.get("projects", {})
        if not isinstance(projects, dict):
            raise ClusterOwnerIdentityMismatchError(
                "NPA project configuration is invalid."
            )
        project = projects.get(candidate.project_alias)
        if not isinstance(project, dict):
            raise ClusterOwnerIdentityMismatchError(
                f"Project alias {candidate.project_alias!r} is not configured."
            )
        existing = _configured_owner(payload)
        if (
            existing is not None
            and not _same_immutable_owner(existing, candidate)
            and not allow_rebind
        ):
            raise ClusterOwnerIdentityMismatchError(
                "Shared controller owner mismatch: recorded "
                f"{existing.project_alias}/{existing.context}/{existing.cluster_id}, requested "
                f"{candidate.project_alias}/{candidate.context}/{candidate.cluster_id}. "
                "Use `npa skypilot bind-controller --rebind` only after managed jobs are terminal."
            )
        skypilot = payload.setdefault("skypilot", {})
        if not isinstance(skypilot, dict):
            raise ClusterOwnerIdentityMismatchError(
                "NPA SkyPilot configuration is invalid."
            )
        skypilot["controller_owner"] = candidate.to_dict()
        payload["skypilot"] = skypilot
        # Migrate the pre-release per-project representation atomically.
        for configured in projects.values():
            if isinstance(configured, dict):
                configured.pop("controller_owner", None)
        payload["projects"] = projects
        return payload

    update_config_document(mutate, path=CONFIG_PATH)
    return candidate


def verify_controller_owner(project: str, context: str) -> ControllerOwner:
    candidate = resolve_controller_candidate(project, context)
    existing = controller_owner()
    if existing is None:
        raise ClusterOwnerIdentityMismatchError(
            "No shared controller owner is bound. Run `npa skypilot bind-controller "
            f"--project {candidate.project_alias} --context {context}` before submit."
        )
    if not _same_immutable_owner(existing, candidate):
        raise ClusterOwnerIdentityMismatchError(
            "Shared controller owner does not match the selected immutable project/cluster context. "
            f"After verifying its jobs terminal, run `npa skypilot bind-controller --project "
            f"{candidate.project_alias} --context {context} --rebind`."
        )
    return existing


def verify_recorded_controller_owner() -> ControllerOwner | None:
    """Fail fast when a saved owner points at a replaced or missing local cluster."""

    existing = controller_owner()
    if existing is None:
        return None
    cluster = load_cluster_state(existing.context)
    if cluster is None:
        raise ClusterOwnerIdentityMismatchError(
            "The shared controller owner references a missing NPA cluster identity. "
            "Restore its exact receipt/state or clean up the old controller before rebinding."
        )
    digest = hashlib.sha256()
    for value in (existing.project_id, cluster.cluster_id, existing.context):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    if (
        cluster.project_id != existing.project_id
        or cluster.cluster_id != existing.cluster_id
        or digest.hexdigest() != existing.context_fingerprint
    ):
        raise ClusterOwnerIdentityMismatchError(
            "The shared controller owner does not match the current immutable cluster "
            "record. Verify old jobs terminal, then use `npa skypilot bind-controller "
            "--project <alias> --context <context> --rebind`."
        )
    return existing


def clear_controller_owner(
    project: str = "",
    *,
    project_id: str = "",
    cluster_id: str = "",
    context: str = "",
) -> bool:
    """Clear only a record matching the cluster just verified absent."""

    changed = False

    def mutate(payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal changed
        existing = _configured_owner(payload)
        if existing is None:
            return payload
        requested_project_id = str(project_id or "").strip()
        if project and not requested_project_id:
            try:
                environment = resolve_environment(project)
                requested_project_id = (
                    str(environment.project_id or "") if environment is not None else ""
                )
            except Exception:  # noqa: BLE001 - exact saved alias is accepted after stanza loss
                requested_project_id = ""
        if requested_project_id and existing.project_id != requested_project_id:
            raise ClusterOwnerIdentityMismatchError(
                "Refusing to clear controller ownership for a different immutable project id."
            )
        if not requested_project_id and existing.project_alias != project:
            raise ClusterOwnerIdentityMismatchError(
                "Refusing to clear controller ownership for a different project."
            )
        if cluster_id and existing.cluster_id != cluster_id:
            raise ClusterOwnerIdentityMismatchError(
                "Refusing to clear controller ownership for a different cluster id."
            )
        if context and existing.context != context:
            raise ClusterOwnerIdentityMismatchError(
                "Refusing to clear controller ownership for a different context."
            )
        skypilot = payload.get("skypilot")
        if isinstance(skypilot, dict):
            skypilot.pop("controller_owner", None)
        projects = payload.get("projects")
        if isinstance(projects, dict):
            for selected in projects.values():
                if isinstance(selected, dict):
                    selected.pop("controller_owner", None)
        changed = True
        return payload

    update_config_document(mutate, path=CONFIG_PATH)
    return changed
