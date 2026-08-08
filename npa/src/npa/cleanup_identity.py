"""Shared teardown identity resolution over receipts, journals, and live config."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

from npa.teardown_receipts import TERMINAL_STATES, load_teardown_receipt


class CleanupIdentityError(RuntimeError):
    """Cleanup identity is missing, conflicting, ambiguous, or unsafe."""


_CONFLICT_FIELDS = frozenset(
    {
        "project_alias",
        "project_id",
        "tenant_id",
        "parent_id",
        "account_id",
        "region",
        "profile",
        "context",
        "kubeconfig_path",
        "cluster_id",
        "cluster_name",
        "agent_name",
        "instance_id",
        "operation_id",
        "service_account_id",
        "run_id",
        "workflow_s3_uri",
        "sky_job_id",
        "controller_context",
    }
)

_CLEANUP_BACKEND_FIELDS = (
    "bucket",
    "endpoint",
    "region",
    "state_key",
    "addressing_style",
)
_CLEANUP_RESOURCE_FIELDS = (
    "resource_type",
    "provider_id",
    "requested_name",
    "project_id",
    "ownership",
    "ownership_source",
    "creation_window_start",
    "creation_window_end",
)


def _present(value: object) -> bool:
    return value not in (None, "", [], {})


def _clean(values: Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        str(key): value.strip() if isinstance(value, str) else value
        for key, value in dict(values or {}).items()
        if _present(value.strip() if isinstance(value, str) else value)
    }


def provisioning_operation_cleanup_identity(
    payload: Mapping[str, Any], *, state_paths: list[str] | None = None
) -> dict[str, Any]:
    """Project one provisioning journal into non-secret cleanup evidence.

    Operation journals intentionally contain runtime backend classifiers and may
    grow additional implementation fields. Receipts are a narrower recovery
    contract, so only immutable selectors needed to find state/resources cross
    that boundary. In particular, credential material and ``credential_source``
    never enter teardown identity.
    """

    backend = payload.get("backend")
    safe_backend = (
        {
            key: str(backend.get(key) or "")
            for key in _CLEANUP_BACKEND_FIELDS
            if _present(backend.get(key))
        }
        if isinstance(backend, Mapping)
        else {}
    )
    resources = payload.get("resources")
    safe_resources = [
        {
            key: str(item.get(key) or "")
            for key in _CLEANUP_RESOURCE_FIELDS
            if _present(item.get(key))
        }
        for item in (resources if isinstance(resources, list) else [])
        if isinstance(item, Mapping)
    ]
    return _clean(
        {
            "operation_id": str(payload.get("operation_id") or ""),
            "resource_type": str(payload.get("resource_type") or ""),
            "requested_name": str(payload.get("requested_name") or ""),
            "project_alias": str(payload.get("project_alias") or ""),
            "project_id": str(payload.get("project_id") or ""),
            "tenant_id": str(payload.get("tenant_id") or ""),
            "region": str(payload.get("region") or ""),
            "backend": safe_backend,
            "resources": safe_resources,
            "state_paths": list(state_paths or []),
        }
    )


def _merge_with_conflicts(
    target: dict[str, Any],
    incoming: Mapping[str, Any],
    *,
    target_source: dict[str, str],
    source: str,
) -> None:
    for key, value in _clean(incoming).items():
        saved = target.get(key)
        if (
            key in _CONFLICT_FIELDS
            and _present(saved)
            and _present(value)
            and saved != value
        ):
            raise CleanupIdentityError(
                f"cleanup identity conflict for {key}: "
                f"{target_source.get(key, 'another source')}={saved!r}, "
                f"{source}={value!r}; no mutation was attempted"
            )
        if not _present(saved):
            target[key] = value
            target_source[key] = source


def _matching_item(items: object, *, key: str, value: str) -> dict[str, Any]:
    candidates = items if isinstance(items, (list, tuple)) else []
    matches = [
        dict(item)
        for item in candidates
        if isinstance(item, Mapping) and str(item.get(key) or "") == value
    ]
    if len(matches) > 1:
        raise CleanupIdentityError(
            f"receipt contains ambiguous {key}={value!r} cleanup identities"
        )
    return matches[0] if matches else {}


def _flatten_receipt_identity(
    receipt: Mapping[str, Any],
    *,
    phase: str = "",
    resource: str = "",
    selectors: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    values = _clean(
        receipt.get("identity") if isinstance(receipt.get("identity"), Mapping) else {}
    )
    values.setdefault("project_alias", str(receipt.get("project_alias") or ""))
    values.setdefault("project_id", str(receipt.get("project_id") or ""))
    events = [item for item in receipt.get("events") or [] if isinstance(item, Mapping)]
    selected_cluster_id = str((selectors or {}).get("cluster_id") or "").strip()

    def event_matches(item: Mapping[str, Any]) -> bool:
        if phase and str(item.get("phase") or "") != phase:
            return False
        if resource and str(item.get("resource") or "") != resource:
            return False
        if selected_cluster_id:
            event_identity = item.get("identity")
            event_cluster_id = str(
                event_identity.get("cluster_id")
                if isinstance(event_identity, Mapping)
                else ""
            ).strip()
            if event_cluster_id and event_cluster_id != selected_cluster_id:
                return False
        return True

    matching = [item for item in events if event_matches(item)]
    latest = max(
        matching, key=lambda item: str(item.get("recorded_at") or ""), default={}
    )
    if latest:
        for key in ("project_alias", "project_id", "context"):
            if _present(latest.get(key)):
                values[key] = latest[key]
        event_identity = latest.get("identity")
        if isinstance(event_identity, Mapping):
            values.update(_clean(event_identity))
        precheck = latest.get("precheck")
        if isinstance(precheck, Mapping):
            for key in _CONFLICT_FIELDS:
                if _present(precheck.get(key)) and not _present(values.get(key)):
                    values[key] = precheck[key]

    if phase == "agent" and resource:
        item = _matching_item(values.get("agents"), key="agent_name", value=resource)
        values.update(_clean(item))
        operation = _matching_item(
            values.get("operations"), key="requested_name", value=resource
        )
        values.update(_clean(operation))
    elif phase in {"cluster", "controller"} and resource:
        exact_cluster_id = str((selectors or {}).get("cluster_id") or "").strip()
        item = _matching_item(
            values.get("clusters"),
            key="cluster_id" if exact_cluster_id else "context",
            value=exact_cluster_id or resource,
        )
        values.update(_clean(item))
        values.setdefault("controller_context", values.get("context", ""))
    elif phase == "workflow" and resource:
        item = _matching_item(values.get("workflows"), key="run_id", value=resource)
        values.update(_clean(item))
    elif phase == "storage_iam":
        storage_item = values.get("storage_iam")
        if isinstance(storage_item, Mapping):
            values.update(_clean(storage_item))
    return _clean(values), dict(latest)


@dataclass(frozen=True)
class CleanupIdentity:
    values: dict[str, Any]
    source: str
    field_sources: dict[str, str]
    receipt_id: str = ""
    receipt_event: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = "") -> Any:
        return self.values.get(key, default)

    @property
    def terminal_state(self) -> str:
        return str(self.receipt_event.get("terminal_state") or "").lower()

    @property
    def receipt_is_terminal(self) -> bool:
        return bool(self.terminal_state and self.terminal_state in TERMINAL_STATES)

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity_source": self.source,
            "receipt_id": self.receipt_id,
            "identity": dict(self.values),
            "identity_field_sources": dict(self.field_sources),
        }


def resolve_cleanup_identity(
    *,
    explicit: Mapping[str, Any] | None = None,
    receipt_id: str = "",
    live: Mapping[str, Any] | None = None,
    phase: str = "",
    resource: str = "",
) -> CleanupIdentity:
    """Resolve explicit > receipt > live while rejecting every overlap conflict."""

    exact = _clean(explicit)
    configured = _clean(live)
    receipt_values: dict[str, Any] = {}
    event: dict[str, Any] = {}
    if receipt_id:
        receipt = load_teardown_receipt(receipt_id)
        receipt_values, event = _flatten_receipt_identity(
            receipt, phase=phase, resource=resource, selectors=exact
        )

    values: dict[str, Any] = {}
    field_sources: dict[str, str] = {}
    # Merge low to high precedence; conflicts still fail instead of overriding.
    _merge_with_conflicts(
        values,
        configured,
        target_source=field_sources,
        source="live_configuration",
    )
    _merge_with_conflicts(
        values,
        receipt_values,
        target_source=field_sources,
        source=f"receipt:{receipt_id}" if receipt_id else "receipt",
    )
    _merge_with_conflicts(
        values,
        exact,
        target_source=field_sources,
        source="explicit_exact_arguments",
    )
    # Re-apply precedence for non-conflicting complementary fields.
    values.update(receipt_values)
    values.update(exact)
    for key in exact:
        field_sources[key] = "explicit_exact_arguments"
    for key in receipt_values:
        if key not in exact:
            field_sources[key] = f"receipt:{receipt_id}"
    source = (
        "explicit_exact_arguments"
        if exact
        else f"receipt:{receipt_id}"
        if receipt_id
        else "live_configuration"
        if configured
        else "unavailable"
    )
    return CleanupIdentity(values, source, field_sources, receipt_id, event)


def project_cleanup_identity_snapshot(alias: str) -> dict[str, Any]:
    """Collect the non-secret immutable identity that survives project forget."""

    from npa.clients.config import list_projects

    cleaned = str(alias or "").strip()
    stanza = dict((list_projects() or {}).get(cleaned) or {})
    project_id = str(stanza.get("project_id") or "")
    identity: dict[str, Any] = {
        "project_alias": cleaned,
        "project_id": project_id,
        "parent_id": project_id,
        "tenant_id": str(stanza.get("tenant_id") or ""),
        "region": str(stanza.get("region") or ""),
    }
    try:
        from npa.clients.nebius_auth import nebius_profile

        identity["profile"] = nebius_profile()
    except (OSError, RuntimeError, ValueError):
        identity["profile"] = ""
    terraform = stanza.get("terraform_state")
    if isinstance(terraform, Mapping):
        identity["terraform_backends"] = [
            {
                "bucket": str(terraform.get("bucket") or ""),
                "endpoint": str(terraform.get("endpoint") or ""),
            }
        ]
    agents: list[dict[str, Any]] = []
    for name, record in dict(stanza.get("agents") or {}).items():
        if not isinstance(record, Mapping):
            continue
        agents.append(
            {
                "agent_name": str(name),
                "instance_id": str(record.get("instance_id") or ""),
                "project_id": str(record.get("project_id") or project_id),
                "tenant_id": str(record.get("tenant_id") or identity["tenant_id"]),
                "region": str(record.get("region") or identity["region"]),
                "service_account_id": str(record.get("service_account_id") or ""),
            }
        )
    identity["agents"] = agents

    try:
        from npa.cluster.state import list_local_clusters

        identity["clusters"] = [
            {
                "context": item.name,
                "controller_context": item.name,
                "cluster_name": item.name,
                "cluster_id": item.cluster_id,
                "project_id": item.project_id,
                "region": item.region,
                "kubeconfig_path": item.kubeconfig_path,
            }
            for item in list_local_clusters()
            if not project_id or item.project_id == project_id
        ]
    except (OSError, RuntimeError, ValueError):
        identity["clusters"] = []

    try:
        from npa.provisioning_journal import list_operations

        operations = list_operations(project_alias=cleaned, project_id=project_id)
        identity["operations"] = []
        for operation in operations:
            payload = operation.read()
            payload = {
                **payload,
                "operation_id": operation.operation_id,
                "project_alias": str(payload.get("project_alias") or cleaned),
                "project_id": str(payload.get("project_id") or project_id),
            }
            identity["operations"].append(
                provisioning_operation_cleanup_identity(
                    payload,
                    state_paths=[str(path) for path in operation.state_copies()],
                )
            )
    except (OSError, RuntimeError, ValueError):
        identity["operations"] = []

    marker = stanza.get("storage_iam_verification_required")
    if isinstance(marker, Mapping):
        identity["storage_iam"] = {
            "service_account_id": str(marker.get("service_account_id") or ""),
            "service_account_name": str(marker.get("service_account_name") or ""),
            "project_id": str(marker.get("project_id") or project_id),
            "tenant_id": str(marker.get("tenant_id") or identity["tenant_id"]),
            "profile": str(marker.get("profile") or ""),
            "ownership": str(marker.get("ownership") or ""),
            "iam_key_ids": [
                str(item)
                for item in marker.get("access_key_ids", marker.get("iam_key_ids", []))
                if str(item).strip()
            ]
            if isinstance(marker.get("access_key_ids", marker.get("iam_key_ids")), list)
            else [],
        }

    workflows: list[dict[str, Any]] = []
    root = Path.home() / ".npa" / "workflow-submissions"
    if root.is_dir() and not root.is_symlink():
        for path in sorted(root.glob("*/*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (
                not isinstance(payload, Mapping)
                or str(payload.get("project") or "") != cleaned
            ):
                continue
            raw_workflow = payload.get("workflow")
            workflow = dict(raw_workflow) if isinstance(raw_workflow, Mapping) else {}
            raw_launch = payload.get("launch")
            launch = dict(raw_launch) if isinstance(raw_launch, Mapping) else {}
            workflows.append(
                {
                    "run_id": str(payload.get("run_id") or ""),
                    "workflow_s3_uri": str(workflow.get("run_prefix_uri") or ""),
                    "sky_job_id": str(launch.get("sky_job_id") or ""),
                    "submission_status": str(launch.get("status") or "planned"),
                }
            )
    identity["workflows"] = workflows
    return identity


__all__ = [
    "CleanupIdentity",
    "CleanupIdentityError",
    "project_cleanup_identity_snapshot",
    "provisioning_operation_cleanup_identity",
    "resolve_cleanup_identity",
]
