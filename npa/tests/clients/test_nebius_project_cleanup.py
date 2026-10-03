from __future__ import annotations

import pytest

from npa.clients import nebius


def test_project_identity_not_found_is_verified_absence(monkeypatch) -> None:
    monkeypatch.setattr(
        nebius,
        "_run_json",
        lambda *_a, **_k: (_ for _ in ()).throw(nebius.NebiusError("NotFound")),
    )
    monkeypatch.setattr(nebius, "_iam_profile_args", lambda _profile: ([], "test"))

    assert nebius.get_project_identity("project-a", tenant_id="tenant-a") is None


def test_project_identity_falls_back_to_spec_region(monkeypatch) -> None:
    monkeypatch.setattr(nebius, "_iam_profile_args", lambda _profile: ([], "test"))
    monkeypatch.setattr(
        nebius,
        "_run_json",
        lambda *_a, **_k: {
            "metadata": {
                "id": "project-a",
                "parent_id": "tenant-a",
                "name": "demo",
            },
            "spec": {"region": "us-central1"},
            "status": {"region": None},
        },
    )

    identity = nebius.get_project_identity("project-a", tenant_id="tenant-a")

    assert identity is not None
    assert identity.region == "us-central1"


def test_default_network_identity_requires_one_exact_default_topology(
    monkeypatch,
) -> None:
    monkeypatch.setattr(nebius, "_iam_profile_args", lambda _profile: ([], "test"))

    def run(args):  # noqa: ANN001, ANN202
        if args[:3] == ["vpc", "network", "list"]:
            return {
                "items": [
                    {
                        "metadata": {
                            "id": "network-a",
                            "parent_id": "project-a",
                            "name": "default-network",
                        }
                    }
                ]
            }
        if args[:3] == ["vpc", "subnet", "list"]:
            return {
                "items": [
                    {
                        "metadata": {
                            "id": "subnet-a",
                            "parent_id": "project-a",
                            "name": "default-subnet-generated",
                        },
                        "spec": {"network_id": "network-a"},
                    }
                ]
            }
        return {
            "items": [
                {
                    "metadata": {
                        "id": "sg-a",
                        "parent_id": "project-a",
                        "name": "default-security-group-generated",
                    },
                    "spec": {"network_id": "network-a"},
                    "status": {"default": True},
                }
            ]
        }

    monkeypatch.setattr(nebius, "_run_json", run)

    identity = nebius.get_project_default_network_identity("project-a")

    assert identity is not None
    assert identity.network_id == "network-a"
    assert identity.subnet_id == "subnet-a"
    assert identity.security_group_id == "sg-a"


def test_default_network_identity_rejects_extra_or_nondefault_resources(
    monkeypatch,
) -> None:
    monkeypatch.setattr(nebius, "_iam_profile_args", lambda _profile: ([], "test"))
    monkeypatch.setattr(
        nebius,
        "_run_json",
        lambda args: (
            {
                "items": [
                    {
                        "metadata": {
                            "id": "resource-a",
                            "parent_id": "project-a",
                            "name": "shared-network",
                        }
                    },
                    {
                        "metadata": {
                            "id": "resource-b",
                            "parent_id": "project-a",
                            "name": "other",
                        }
                    },
                ]
            }
            if args[:3] == ["vpc", "network", "list"]
            else {"items": []}
        ),
    )

    with pytest.raises(nebius.NebiusError, match="not the unique"):
        nebius.get_project_default_network_identity("project-a")


def test_delete_default_network_orders_subnet_before_parent(monkeypatch) -> None:
    monkeypatch.setattr(nebius, "_iam_profile_args", lambda _profile: ([], "test"))
    calls: list[list[str]] = []
    monkeypatch.setattr(nebius, "_run", lambda args: calls.append(list(args)) or "")
    identity = nebius.ProjectDefaultNetworkIdentity(
        "network-a",
        "default-network",
        "subnet-a",
        "default-subnet-a",
        "sg-a",
        "default-security-group-a",
        "project-a",
    )

    nebius.delete_project_default_network(identity)

    assert calls == [
        ["vpc", "subnet", "delete", "--id", "subnet-a"],
        ["vpc", "network", "delete", "--id", "network-a"],
    ]


def test_project_dependency_inventory_requires_items_and_exact_ids(
    monkeypatch,
) -> None:
    monkeypatch.setattr(nebius, "_iam_profile_args", lambda _profile: ([], "test"))
    monkeypatch.setattr(nebius, "_run_json", lambda *_a, **_k: {"unexpected": []})

    with pytest.raises(nebius.NebiusError, match="schema-invalid"):
        nebius.list_project_dependencies("project-a")


@pytest.mark.parametrize("empty", [{}, {"items": None}, {"items": []}])
def test_project_dependency_inventory_accepts_only_known_empty_shapes(
    monkeypatch, empty
) -> None:
    monkeypatch.setattr(nebius, "_iam_profile_args", lambda _profile: ([], "test"))
    monkeypatch.setattr(nebius, "_run_json", lambda *_a, **_k: empty)
    monkeypatch.setattr(nebius, "_list_access_key_metadata", lambda *_a, **_k: [])

    inventory = nebius.list_project_dependencies("project-a")

    assert all(not identities for identities in inventory.values())


def test_delete_project_uses_only_supported_exact_id_adapter(monkeypatch) -> None:
    monkeypatch.setattr(nebius, "_iam_profile_args", lambda _profile: ([], "test"))
    calls: list[list[str]] = []
    monkeypatch.setattr(nebius, "_run", lambda args: calls.append(list(args)) or "")

    nebius.delete_project("project-a")

    assert calls == [["iam", "v2", "project", "delete", "--id", "project-a"]]


def test_project_dependency_inventory_keeps_projects_isolated(monkeypatch) -> None:
    monkeypatch.setattr(nebius, "_iam_profile_args", lambda _profile: ([], "test"))
    calls: list[list[str]] = []

    def run(args):  # noqa: ANN001, ANN202
        calls.append(list(args))
        return {"items": []}

    monkeypatch.setattr(nebius, "_run_json", run)
    monkeypatch.setattr(nebius, "_list_access_key_metadata", lambda project, **_k: [])

    inventory = nebius.list_project_dependencies("project-a")

    assert all(
        command[command.index("--parent-id") + 1] == "project-a" for command in calls
    )
    assert all(
        "--all" not in command
        for command in calls
        if command[0:2] in (["ai", "endpoint"], ["ai", "job"])
    )
    assert all(
        "--all" in command
        for command in calls
        if command[0:2] not in (["ai", "endpoint"], ["ai", "job"])
    )
    assert all(not identities for identities in inventory.values())
