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
