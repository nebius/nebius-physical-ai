from __future__ import annotations

import json
from pathlib import Path

import pytest

from npa.cleanup_identity import (
    CleanupIdentityError,
    provisioning_operation_cleanup_identity,
    resolve_cleanup_identity,
)
from npa.teardown_receipts import record_teardown_event


def _receipt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> str:
    monkeypatch.setenv("NPA_TEARDOWN_RECEIPT_DIR", str(tmp_path / "receipts"))
    path = record_teardown_event(
        phase="agent",
        resource="agent",
        terminal_state="verified_absent",
        project_alias="removed-alias",
        project_id="project-1",
        identity={
            "project_id": "project-1",
            "tenant_id": "tenant-1",
            "region": "region-1",
            "agents": [
                {
                    "agent_name": "agent",
                    "instance_id": "instance-1",
                    "project_id": "project-1",
                }
            ],
        },
    )
    return path.stem


def test_receipt_resolves_without_live_project_stanza(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    receipt_id = _receipt(monkeypatch, tmp_path)

    identity = resolve_cleanup_identity(
        receipt_id=receipt_id, phase="agent", resource="agent"
    )

    assert identity.source == f"receipt:{receipt_id}"
    assert identity.get("project_id") == "project-1"
    assert identity.get("instance_id") == "instance-1"
    assert identity.receipt_is_terminal is True
    assert identity.receipt_authorizes_noop is False


def test_agent_receipt_noop_authority_requires_complete_phase_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("NPA_TEARDOWN_RECEIPT_DIR", str(tmp_path / "receipts"))
    path = record_teardown_event(
        phase="agent",
        resource="agent",
        terminal_state="verified_deleted",
        project_id="project-1",
        identity={
            "project_id": "project-1",
            "agent_name": "agent",
            "instance_id": "instance-1",
        },
        action={"kind": "terraform_agent_destroy"},
        verification={
            "exact_instance_absent": True,
            "terraform_destroy_completed": True,
            "terraform_dependency_graph": [
                "compute_instance",
                "boot_disk",
                "network",
                "subnet",
                "security_group",
                "public_ip",
            ],
        },
    )

    identity = resolve_cleanup_identity(
        receipt_id=path.stem,
        phase="agent",
        resource="agent",
        explicit={"instance_id": "instance-1"},
    )

    assert identity.receipt_authorizes_noop is True


def test_receipt_event_sequence_dominates_wall_clock_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa import teardown_receipts

    monkeypatch.setenv("NPA_TEARDOWN_RECEIPT_DIR", str(tmp_path / "receipts"))
    path = record_teardown_event(
        phase="agent",
        resource="agent",
        terminal_state="verified_deleted",
        project_id="project-1",
        identity={
            "project_id": "project-1",
            "agent_name": "agent",
            "instance_id": "instance-1",
        },
        action={"kind": "terraform_agent_destroy"},
        verification={
            "exact_instance_absent": True,
            "terraform_destroy_completed": True,
            "terraform_dependency_graph": [
                "compute_instance",
                "boot_disk",
                "network",
                "subnet",
                "security_group",
                "public_ip",
            ],
        },
    )
    record_teardown_event(
        phase="agent",
        resource="agent",
        terminal_state="partial",
        project_id="project-1",
        identity={
            "project_id": "project-1",
            "agent_name": "agent",
            "instance_id": "instance-1",
        },
        action={"kind": "terraform_agent_destroy"},
        errors=["later retry is unresolved"],
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["events"][0]["recorded_at"] = "2099-01-01T00:00:00Z"
    payload["events"][1]["recorded_at"] = "2000-01-01T00:00:00Z"
    teardown_receipts._write_atomic(path, payload)

    identity = resolve_cleanup_identity(
        receipt_id=path.stem,
        phase="agent",
        resource="agent",
        explicit={"instance_id": "instance-1"},
    )

    assert identity.terminal_state == "partial"
    assert identity.receipt_authorizes_noop is False


def test_exact_identity_precedence_is_complementary_but_conflicts_fail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    receipt_id = _receipt(monkeypatch, tmp_path)
    identity = resolve_cleanup_identity(
        explicit={"project_id": "project-1", "profile": "operator"},
        receipt_id=receipt_id,
        live={"region": "region-1"},
        phase="agent",
        resource="agent",
    )
    assert identity.source == "explicit_exact_arguments"
    assert identity.get("profile") == "operator"
    assert identity.get("instance_id") == "instance-1"

    with pytest.raises(CleanupIdentityError, match="project_id"):
        resolve_cleanup_identity(
            explicit={"project_id": "wrong-project"},
            receipt_id=receipt_id,
            phase="agent",
            resource="agent",
        )


def test_exact_agent_generation_ignores_ambiguous_complementary_operations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("NPA_TEARDOWN_RECEIPT_DIR", str(tmp_path / "receipts"))
    path = record_teardown_event(
        phase="agent",
        resource="agent",
        terminal_state="partial",
        project_alias="demo",
        project_id="project-1",
        identity={
            "project_alias": "demo",
            "project_id": "project-1",
            "agent_name": "agent",
            "instance_id": "instance-1",
            "agents": [
                {
                    "agent_name": "agent",
                    "instance_id": "instance-1",
                    "project_id": "project-1",
                }
            ],
            "operations": [
                {
                    "operation_id": "setup-operation",
                    "requested_name": "agent",
                    "project_id": "project-1",
                },
                {
                    "operation_id": "teardown-operation",
                    "requested_name": "agent",
                    "project_id": "project-1",
                },
            ],
        },
    )

    identity = resolve_cleanup_identity(
        receipt_id=path.stem, phase="agent", resource="agent"
    )

    assert identity.get("instance_id") == "instance-1"
    assert identity.get("operation_id") == ""

    selected = resolve_cleanup_identity(
        explicit={"operation_id": "teardown-operation"},
        receipt_id=path.stem,
        phase="agent",
        resource="agent",
    )
    assert selected.get("operation_id") == "teardown-operation"


def test_provisioning_cleanup_projection_is_typed_and_never_copies_credentials() -> (
    None
):
    identity = provisioning_operation_cleanup_identity(
        {
            "operation_id": "operation-a",
            "resource_type": "agent",
            "requested_name": "agent",
            "project_alias": "target",
            "project_id": "project-target",
            "tenant_id": "tenant-target",
            "region": "eu-test1",
            "backend": {
                "bucket": "state-bucket",
                "endpoint": "https://storage.example.invalid",
                "region": "eu-test1",
                "state_key": "npa/target/agent.tfstate",
                "addressing_style": "path",
                "credential_source": "project_saved",
                "access_key": "must-not-persist",
                "secret_key": "must-not-persist",
                "future_unknown": "must-not-persist",
            },
            "resources": [
                {
                    "resource_type": "agent_instance",
                    "provider_id": "instance-target",
                    "requested_name": "agent-target",
                    "project_id": "project-target",
                    "ownership": "created_by_this_operation",
                    "ownership_source": "provider-create-response",
                    "labels": {"credential_hint": "must-not-persist"},
                    "future_unknown": "must-not-persist",
                }
            ],
        },
        state_paths=["/safe/local-state.tfstate"],
    )

    assert identity["backend"] == {
        "bucket": "state-bucket",
        "endpoint": "https://storage.example.invalid",
        "region": "eu-test1",
        "state_key": "npa/target/agent.tfstate",
        "addressing_style": "path",
    }
    assert identity["resources"] == [
        {
            "resource_type": "agent_instance",
            "provider_id": "instance-target",
            "requested_name": "agent-target",
            "project_id": "project-target",
            "ownership": "created_by_this_operation",
            "ownership_source": "provider-create-response",
        }
    ]
    assert "credential_source" not in str(identity)
    assert "must-not-persist" not in str(identity)
