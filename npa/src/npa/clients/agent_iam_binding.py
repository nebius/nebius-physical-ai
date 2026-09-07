"""Exact-project permissions for VM-attached NPA agent service accounts.

Provisioning uses only a custom group within the selected project, an editor
permit on that same project, and the exact named agent account membership.
Creation callbacks journal provider-returned IDs before the next provider step.
Cleanup accepts creation provenance, never a familiar resource name alone.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from npa.clients import nebius

GROUP_NAME = "npa-agent-project-editors"
ROLE = "editor"
BINDING_KINDS = ("agent_group", "agent_permit", "agent_membership")
_RESOURCE_TYPES = {
    "agent_group": "group",
    "agent_permit": "access-permit",
    "agent_membership": "group-membership",
}


def _require(value: Any, message: str) -> None:
    if not value:
        raise nebius.NebiusError(message)


def _id(value: Any) -> str:
    _require(
        isinstance(value, str)
        and value
        and value == value.strip()
        and not value.startswith("-")
        and not any(char.isspace() for char in value),
        "Agent IAM requires an exact nonempty resource identity",
    )
    return value


def _metadata(
    payload: Any, *, expected_id: str = "", parent: str = "", name: str = ""
) -> dict:
    _require(
        isinstance(payload, dict) and isinstance(payload.get("metadata"), dict),
        "Agent IAM provider response lacks resource metadata",
    )
    metadata = payload["metadata"]
    identity = _id(metadata.get("id"))
    _require(
        not expected_id or identity == expected_id,
        "Agent IAM provider returned another resource",
    )
    _require(
        not parent or metadata.get("parent_id") == parent,
        "Agent IAM provider parent does not match scope",
    )
    _require(
        not name or metadata.get("name") == name,
        "Agent IAM provider name does not match scope",
    )
    return metadata


def _get(resource: str, resource_id: str) -> dict | None:
    try:
        return nebius._run_json(["iam", resource, "get", "--id", _id(resource_id)])
    except nebius.NebiusError as exc:
        if nebius.is_not_found(str(exc)) and not nebius.is_permission_denied(str(exc)):
            return None
        raise


def _validate_resource(kind: str, payload: dict, expected: Mapping[str, str]) -> None:
    parent = expected["project_id"] if kind == "agent_group" else expected["group_id"]
    _metadata(
        payload,
        expected_id=expected["id"],
        parent=parent,
        name=GROUP_NAME if kind == "agent_group" else "",
    )
    if kind != "agent_group":
        spec = payload.get("spec")
        _require(
            isinstance(spec, dict),
            "Agent IAM provider resource lacks its permission specification",
        )
        if kind == "agent_permit":
            _require(
                spec.get("resource_id") == expected["project_id"]
                and spec.get("role") == ROLE,
                "Agent IAM permit grants a different resource or role",
            )
        else:
            _require(
                spec.get("member_id") == expected["service_account_id"],
                "Agent IAM membership identifies another account",
            )


def _inventory(resource: str, group_id: str) -> list[dict]:
    """Read all pages explicitly; list-members does not support CLI --all."""
    operation = "list-members" if resource == "group-membership" else "list"
    field = "memberships" if resource == "group-membership" else "items"
    result: list[dict] = []
    seen_tokens: set[str] = set()
    seen_ids: set[str] = set()
    token = ""
    while True:
        argv = [
            "iam",
            resource,
            operation,
            "--parent-id",
            group_id,
            "--page-size",
            "1000",
        ]
        if token:
            argv.extend(["--page-token", token])
        payload = nebius._run_json(argv)
        # The successful CLI JSON response follows ProtoJSON: an empty
        # repeated field is omitted, so {} is an empty terminal page. Keep
        # that documented default distinct from null, malformed or error data.
        # https://protobuf.dev/programming-guides/json/#presence-and-default-values
        _require(
            isinstance(payload, dict) and set(payload) <= {field, "next_page_token"},
            "Agent IAM inventory has unknown or invalid response fields",
        )
        items = payload.get(field, [])
        _require(
            isinstance(items, list), "Agent IAM inventory has an invalid resource list"
        )
        for item in items:
            metadata = _metadata(item, parent=group_id)
            _require(
                metadata["id"] not in seen_ids,
                "Agent IAM inventory repeats a resource identity",
            )
            seen_ids.add(metadata["id"])
            spec = item.get("spec")
            _require(
                isinstance(spec, dict),
                "Agent IAM inventory has a malformed resource specification",
            )
            for key in (
                ("member_id",)
                if resource == "group-membership"
                else ("resource_id", "role")
            ):
                _id(spec.get(key))
            result.append(item)
        token = payload.get("next_page_token", "")
        _require(
            isinstance(token, str),
            "Agent IAM inventory has an invalid pagination token",
        )
        if not token:
            return result
        _require(
            token not in seen_tokens, "Agent IAM inventory repeats a pagination token"
        )
        seen_tokens.add(token)


def verify_agent_project_scope(project_id: str, tenant_id: str) -> None:
    """Verify the operator-selected hierarchy before creating an attached identity."""
    project_id, tenant_id = _id(project_id), _id(tenant_id)
    _metadata(_get("project", project_id), expected_id=project_id, parent=tenant_id)


def remove_created_agent_account(
    project_id: str, tenant_id: str, account_id: str
) -> None:
    """Reconcile a caller-journaled creation only after exact scope and absence proof."""
    from npa.lifecycle_intent import OperationIntent, require_intent

    require_intent(
        OperationIntent.DESTROY,
        OperationIntent.MUTATE,
        OperationIntent.ENSURE_PRESENT,
        primitive="remove_created_agent_account",
    )
    account_id = _id(account_id)
    current = _get("service-account", account_id)
    if current is None:
        return
    if tenant_id:
        verify_agent_project_scope(project_id, tenant_id)
    else:
        project = _metadata(_get("project", _id(project_id)), expected_id=project_id)
        _id(project.get("parent_id"))
    _metadata(
        current,
        expected_id=account_id,
        parent=project_id,
        name=nebius.AGENT_SERVICE_ACCOUNT_NAME,
    )
    try:
        nebius.delete_service_account(account_id)
    except nebius.NebiusError as exc:
        if not nebius.is_not_found(str(exc)) or nebius.is_permission_denied(str(exc)):
            raise
    _require(
        _get("service-account", account_id) is None,
        "Agent IAM account deletion is not verified absent",
    )


def ensure_agent_project_binding(
    *,
    project_id: str,
    tenant_id: str,
    service_account_id: str,
    on_resource_created: Callable[[str, dict[str, str]], None],
) -> dict[str, str]:
    """Create or verify the exact-project grant; never fall back to tenant IAM."""
    from npa.lifecycle_intent import forbid_destructive_provisioning

    forbid_destructive_provisioning("ensure_agent_project_binding")
    project_id, tenant_id, service_account_id = map(
        _id, (project_id, tenant_id, service_account_id)
    )
    _require(
        callable(on_resource_created),
        "Agent IAM creation requires a durable creation journal",
    )
    verify_agent_project_scope(project_id, tenant_id)
    account = _get("service-account", service_account_id)
    _metadata(
        account,
        expected_id=service_account_id,
        parent=project_id,
        name=nebius.AGENT_SERVICE_ACCOUNT_NAME,
    )
    context = {
        "project_id": project_id,
        "tenant_id": tenant_id,
        "service_account_id": service_account_id,
        "group_name": GROUP_NAME,
        "role": ROLE,
    }
    created: list[str] = []

    def capture(kind: str, payload: dict, **extra: str) -> dict[str, str]:
        # A returned immutable ID is recorded even if the following exact
        # scope/readback validation fails. Cleanup must still verify its scope.
        identity = _metadata(payload)["id"]
        record = {
            **context,
            **extra,
            "id": identity,
            "created_by": "npa",
            "ownership_source": "provider-create-response",
        }
        on_resource_created(kind, record)
        created.append(kind)
        return record

    try:
        group = nebius._run_json(
            [
                "iam",
                "group",
                "get-by-name",
                "--parent-id",
                project_id,
                "--name",
                GROUP_NAME,
            ]
        )
    except nebius.NebiusError as exc:
        if not nebius.is_not_found(str(exc)) or nebius.is_permission_denied(str(exc)):
            raise nebius.NebiusError(
                "Agent IAM group inventory is unreadable; no broader grant attempted"
            ) from exc
        group = nebius._run_json(
            ["iam", "group", "create", "--parent-id", project_id, "--name", GROUP_NAME]
        )
        capture("agent_group", group, name=GROUP_NAME)
    group_id = _metadata(group, parent=project_id, name=GROUP_NAME)["id"]
    context["group_id"] = group_id
    _validate_resource(
        "agent_group", _get("group", group_id), {**context, "id": group_id}
    )
    permits = _inventory("access-permit", group_id)
    _require(
        all(
            item["spec"]["resource_id"] == project_id and item["spec"]["role"] == ROLE
            for item in permits
        ),
        "Agent IAM group has unexpected permission grants; refusing membership",
    )
    _require(len(permits) <= 1, "Agent IAM group has ambiguous duplicate grants")
    members = _inventory("group-membership", group_id)
    matching = [
        item for item in members if item["spec"]["member_id"] == service_account_id
    ]
    _require(len(matching) <= 1, "Agent IAM group has ambiguous duplicate memberships")
    _require(
        permits or len(matching) == len(members),
        "Agent IAM cannot add a grant to a group containing other accounts",
    )
    if permits:
        permit_id = permits[0]["metadata"]["id"]
    else:
        permit = nebius._run_json(
            [
                "iam",
                "access-permit",
                "create",
                "--parent-id",
                group_id,
                "--resource-id",
                project_id,
                "--role",
                ROLE,
            ]
        )
        permit_id = capture("agent_permit", permit)["id"]
        _validate_resource("agent_permit", permit, {**context, "id": permit_id})
    _validate_resource(
        "agent_permit", _get("access-permit", permit_id), {**context, "id": permit_id}
    )
    if matching:
        membership_id = matching[0]["metadata"]["id"]
    else:
        membership = nebius._run_json(
            [
                "iam",
                "group-membership",
                "create",
                "--parent-id",
                group_id,
                "--member-id",
                service_account_id,
            ]
        )
        membership_id = capture("agent_membership", membership)["id"]
        _validate_resource(
            "agent_membership", membership, {**context, "id": membership_id}
        )
    _validate_resource(
        "agent_membership",
        _get("group-membership", membership_id),
        {**context, "id": membership_id},
    )
    return {
        "agent_iam_scope_id": project_id,
        "agent_iam_role": ROLE,
        "agent_iam_group_id": group_id,
        "agent_iam_permit_id": permit_id,
        "agent_iam_membership_id": membership_id,
        "agent_iam_state": "created" if created else "existing",
    }


def cleanup_agent_project_binding(
    project_id: str,
    resources: Mapping[str, Mapping[str, dict[str, str]]],
    *,
    on_removed: Callable[[str, str], Any],
) -> list[str]:
    """Remove exact owned binding resources after the last dependent agent.

    Caller establishes dependency absence. Every resource also needs a verified
    create record, exact provider scope and read-after-delete absence. Shared
    members/permits retain group permissions with a surfaced partial outcome.
    """
    from npa.lifecycle_intent import OperationIntent, require_intent

    require_intent(
        OperationIntent.DESTROY,
        OperationIntent.MUTATE,
        OperationIntent.ENSURE_PRESENT,
        primitive="cleanup_agent_project_binding",
    )
    owned: dict[str, dict[str, dict[str, str]]] = {}
    for kind in BINDING_KINDS:
        entries = resources.get(kind, {})
        _require(
            isinstance(entries, Mapping), "Agent IAM ownership journal is malformed"
        )
        owned[kind] = {}
        for identity, record in entries.items():
            _require(
                isinstance(record, dict)
                and record.get("id") == identity
                and record.get("project_id") == project_id
                and record.get("created_by") == "npa"
                and record.get("ownership_source") == "provider-create-response"
                and record.get("group_name") == GROUP_NAME
                and record.get("role") == ROLE,
                "Agent IAM resource lacks exact provider creation provenance",
            )
            for key in ("id", "tenant_id", "service_account_id"):
                _id(record.get(key))
            if kind != "agent_group":
                _id(record.get("group_id"))
            owned[kind][identity] = record
    deleted: list[str] = []
    failures: list[str] = []

    def remove(kind: str, identity: str, record: dict[str, str]) -> bool:
        resource = _RESOURCE_TYPES[kind]
        try:
            current = _get(resource, identity)
            if current is not None:
                _validate_resource(kind, current, record)
                project = _get("project", project_id)
                _metadata(project, expected_id=project_id, parent=record["tenant_id"])
                if kind != "agent_group":
                    group = _get("group", record["group_id"])
                    _metadata(
                        group,
                        expected_id=record["group_id"],
                        parent=project_id,
                        name=GROUP_NAME,
                    )
                nebius._run(["iam", resource, "delete", "--id", identity])
                _require(
                    _get(resource, identity) is None,
                    "Agent IAM deletion is not verified absent",
                )
            on_removed(kind, identity)
            deleted.append(kind)
            return True
        except (nebius.NebiusError, OSError, ValueError, KeyError):
            failures.append(
                kind + " deletion or ownership verification remains unresolved"
            )
            return False

    for identity, record in owned["agent_membership"].items():
        remove("agent_membership", identity, record)
    group_ids = {record["group_id"] for record in owned["agent_permit"].values()} | set(
        owned["agent_group"]
    )
    for group_id in group_ids:
        group_permits = {
            identity: record
            for identity, record in owned["agent_permit"].items()
            if record["group_id"] == group_id
        }
        try:
            group = _get("group", group_id)
            if group is not None:
                _metadata(
                    group, expected_id=group_id, parent=project_id, name=GROUP_NAME
                )
                members = _inventory("group-membership", group_id)
                permits = _inventory("access-permit", group_id)
                _require(
                    not members
                    and all(
                        item["metadata"]["id"] in group_permits for item in permits
                    ),
                    "Agent IAM group has shared or unowned members or permits",
                )
            for identity, record in group_permits.items():
                remove("agent_permit", identity, record)
            if group_id in owned["agent_group"]:
                if _get("group", group_id) is not None:
                    _require(
                        not _inventory("group-membership", group_id)
                        and not _inventory("access-permit", group_id),
                        "Agent IAM group is not empty",
                    )
                remove("agent_group", group_id, owned["agent_group"][group_id])
        except (nebius.NebiusError, OSError, ValueError, KeyError):
            failures.append("agent project group remains shared or unverified")
    if failures:
        raise nebius.NebiusError(
            "Agent IAM cleanup remains partial: " + "; ".join(failures)
        )
    return deleted
