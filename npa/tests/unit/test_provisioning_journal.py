from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import stat

import pytest
import yaml

from npa.clients.credentials import update_private_yaml
from npa.provisioning_journal import (
    OperationIdentityError,
    OperationJournalError,
    ProvisioningOperation,
    current_operation,
    emit_recovery_summary,
    list_operations,
    operation_context,
)


@pytest.fixture
def journal_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    root = tmp_path / "operations"
    monkeypatch.setenv("NPA_OPERATION_JOURNAL_DIR", str(root))
    return root


def _prepare(**overrides: str) -> ProvisioningOperation:
    values = {
        "command": "npa agent deploy",
        "project_alias": "prod",
        "project_id": "project-a",
        "tenant_id": "tenant-a",
        "region": "eu-north1",
        "backend": {"bucket": "state-a", "endpoint": "storage.example"},
        "resource_type": "agent",
        "requested_name": "agent",
        "ownership_source": "test",
        "resume_command": "npa agent deploy --project prod --name agent",
        "destroy_command": "npa agent destroy --project prod --name agent --yes",
    }
    values.update(overrides)
    return ProvisioningOperation.prepare(**values)


def test_retry_reuses_nonterminal_operation_and_terminal_run_gets_new_receipt(
    journal_root: Path,
) -> None:
    first = _prepare()
    first.transition("mutating")
    retry = _prepare()

    assert retry.operation_id == first.operation_id
    assert retry.read()["resume_count"] == 1

    retry.transition("resource-created")
    retry.transition("state-durable")
    retry.commit()
    later = _prepare()

    assert later.operation_id == f"{first.operation_id}-r1"
    assert later.read()["phase"] == "prepared"


def test_rolled_back_failure_is_terminal_but_never_success(
    journal_root: Path,
) -> None:
    operation = _prepare(resource_type="cluster", requested_name="gpu")
    operation.transition("mutating")
    operation.record_resource(
        resource_type="managed_kubernetes_cluster",
        requested_name="gpu",
        provider_id="cluster-first",
        ownership="created_by_this_operation",
        ownership_source="terraform-output",
        project_id="project-a",
    )
    operation.record_failure(RuntimeError("primary sanitized failure"))
    operation.record_rollback(
        attempted=True,
        completed=True,
        removed=operation.read()["resources"],
        preserved=[],
    )
    operation.transition("rolled-back")

    payload = operation.read()
    assert payload["phase"] == "rolled-back"
    assert payload["lifecycle"] == "failed"
    assert payload["result"] == "rolled_back"
    assert payload["last_error"] == "primary sanitized failure"
    rendered = emit_recovery_summary(operation)
    assert "Primary failure: RuntimeError: primary sanitized failure" in rendered
    assert "Rollback: attempted=True, completed=True, resources_removed=1" in rendered

    retry = _prepare(resource_type="cluster", requested_name="gpu")
    assert retry.operation_id == f"{operation.operation_id}-r1"
    assert retry.read()["resources"] == []


def test_rollback_journals_exact_per_resource_outcomes(journal_root: Path) -> None:
    operation = _prepare(resource_type="cluster", requested_name="gpu")
    resources = [
        {
            "resource_type": "managed_kubernetes_cluster",
            "requested_name": "gpu",
            "provider_id": "cluster-exact",
            "project_id": "project-a",
            "ownership": "created_by_this_operation",
            "ownership_source": "terraform-output",
        },
        {
            "resource_type": "managed_kubernetes_node_group",
            "requested_name": "gpu-ng",
            "provider_id": "nodegroup-exact",
            "project_id": "project-a",
            "ownership": "created_by_this_operation",
            "ownership_source": "terraform-output",
        },
    ]
    operation.record_rollback(
        attempted=True,
        completed=False,
        removed=[],
        preserved=resources,
        outcomes=[
            {**resources[0], "outcome": "removed"},
            {
                **resources[1],
                "outcome": "rollback_failed",
                "error": "provider refusal",
            },
        ],
        error="node group cleanup incomplete",
    )

    rollback = operation.read()["rollback"]
    assert rollback["attempted"] is True
    assert rollback["completed"] is False
    assert rollback["resource_outcomes"] == [
        {
            "resource_type": "managed_kubernetes_cluster",
            "requested_name": "gpu",
            "provider_id": "cluster-exact",
            "project_id": "project-a",
            "ownership": "created_by_this_operation",
            "ownership_source": "terraform-output",
            "outcome": "removed",
        },
        {
            "resource_type": "managed_kubernetes_node_group",
            "requested_name": "gpu-ng",
            "provider_id": "nodegroup-exact",
            "project_id": "project-a",
            "ownership": "created_by_this_operation",
            "ownership_source": "terraform-output",
            "outcome": "rollback_failed",
            "error": "provider refusal",
        },
    ]


def test_paid_resource_without_cluster_id_is_attempted_incomplete_not_false_absence(
    journal_root: Path,
) -> None:
    from npa.provisioning import _rollback_owned_cluster

    operation = _prepare(resource_type="cluster", requested_name="gpu")
    operation.transition("mutating")
    operation.record_resource(
        resource_type="managed_kubernetes_node_group",
        requested_name="gpu-ng",
        provider_id="nodegroup-exact",
        project_id="project-a",
        ownership="created_by_this_operation",
        ownership_source="terraform-output",
    )

    completed = _rollback_owned_cluster(
        operation,
        project_alias="prod",
        context="gpu",
        terraform_dir=None,
        kubeconfig=None,
        timeout=120,
    )

    rollback = operation.read()["rollback"]
    assert completed is False
    assert rollback["attempted"] is True
    assert rollback["completed"] is False
    assert rollback["resource_outcomes"][0]["provider_id"] == "nodegroup-exact"
    assert rollback["resource_outcomes"][0]["outcome"] == (
        "preserved_missing_cluster_identity"
    )
    assert operation.read()["phase"] == "rollback-incomplete"


def test_owned_cluster_rollback_passes_exact_operation_id(
    journal_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from npa.cli.cluster import terraform_lifecycle
    from npa.provisioning import _rollback_owned_cluster

    operation = _prepare(resource_type="cluster", requested_name="gpu")
    operation.transition("mutating")
    operation.record_resource(
        resource_type="managed_kubernetes_cluster",
        requested_name="gpu",
        provider_id="cluster-exact",
        project_id="project-a",
        ownership="created_by_this_operation",
        ownership_source="terraform-output",
    )
    observed: dict[str, object] = {}

    def fake_down_cmd(**kwargs: object) -> None:
        observed.update(kwargs)

    monkeypatch.setattr(terraform_lifecycle, "down_cmd", fake_down_cmd)

    assert _rollback_owned_cluster(
        operation,
        project_alias="prod",
        context="gpu",
        terraform_dir=None,
        kubeconfig=None,
        timeout=120,
    )
    assert observed["operation_id"] == operation.operation_id
    assert observed["cluster_id"] == "cluster-exact"
    assert operation.read()["phase"] == "rolled-back"


def test_long_requested_name_produces_a_valid_retry_path(journal_root: Path) -> None:
    requested_name = "project-scoped-bucket-with-a-long-provider-name"
    first = _prepare(requested_name=requested_name)
    first.commit()

    retry = _prepare(requested_name=requested_name)

    assert retry.path.parent.name == retry.operation_id
    assert retry.operation_id.endswith("-r1")


def test_list_operations_uses_operation_id_to_break_timestamp_ties(
    journal_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "npa.provisioning_journal.utc_now", lambda: "2026-08-08T00:00:00Z"
    )
    first = _prepare(requested_name="agent-a")
    second = _prepare(requested_name="agent-b")

    listed = list_operations(project_id="project-a", resource_type="agent")

    assert {item.operation_id for item in listed} == {
        first.operation_id,
        second.operation_id,
    }
    assert [item.operation_id for item in listed] == sorted(
        [first.operation_id, second.operation_id], reverse=True
    )


def test_journal_is_private_atomic_secret_free_and_identity_guarded(
    journal_root: Path,
) -> None:
    operation = _prepare()
    operation.record_config_mutation(
        store="credentials.yaml",
        fields=["tokens.HF_TOKEN"],
        secret_fields=["tokens.HF_TOKEN"],
    )
    operation.record_resource(
        resource_type="storage_bucket",
        requested_name="shared",
        ownership="adopted",
        ownership_source="verified-project-inventory",
        project_id="project-a",
    )

    payload = json.loads(operation.path.read_text(encoding="utf-8"))
    assert "hf_example_secret" not in json.dumps(payload)
    assert payload["resources"][0]["ownership"] == "adopted"
    assert stat.S_IMODE(operation.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(operation.path.parent.stat().st_mode) == 0o700

    with pytest.raises(OperationIdentityError):
        operation.update_identity(project_id="project-other")
    with pytest.raises(OperationJournalError, match="secret-bearing"):
        operation.record_resource(
            resource_type="test",
            requested_name="unsafe",
            ownership="created_by_this_operation",
            ownership_source="test",
            labels={"api_key": "must-not-land"},
        )


def test_immutable_plan_conflict_names_sanitized_nested_fields(
    journal_root: Path,
) -> None:
    operation = _prepare()
    original = {
        "project_alias": "prod",
        "project_id": "project-a",
        "tenant_id": "tenant-a",
        "region": "eu-north1",
        "topology": {"cluster_name": "a", "gpu_nodes": 1},
    }
    operation.record_preflight_plan(original)

    changed = {**original, "topology": {"cluster_name": "b", "gpu_nodes": 2}}
    with pytest.raises(OperationIdentityError) as caught:
        operation.record_preflight_plan(changed)

    message = str(caught.value)
    assert "topology.cluster_name" in message
    assert "topology.gpu_nodes" in message

    with pytest.raises(OperationJournalError, match="secret-bearing preflight plan"):
        operation.record_preflight_plan(
            {**original, "topology": {"api_token": "must-not-persist"}}
        )


def test_authoritative_region_can_be_corrected_only_before_resource_creation(
    journal_root: Path,
) -> None:
    operation = _prepare(region="eu-north1")
    operation.transition("mutating")
    operation.update_identity(region="us-central1", allow_region_correction=True)

    assert operation.read()["region"] == "us-central1"
    operation.record_resource(
        resource_type="compute_instance",
        requested_name="agent",
        ownership="created_by_this_operation",
        ownership_source="test",
    )
    with pytest.raises(OperationIdentityError):
        operation.update_identity(region="eu-north1", allow_region_correction=True)


def test_local_and_errored_state_survive_recovery_transition(
    journal_root: Path,
) -> None:
    operation = _prepare()
    operation.transition("mutating")
    state = operation.preserve_state_bytes(
        b'{"version":4,"resources":[{"type":"compute"}]}',
        name="errored",
    )
    operation.transition("recovery-required", error="NoSuchBucket")

    summary = operation.recovery_summary()
    assert state.is_file()
    assert stat.S_IMODE(state.stat().st_mode) == 0o600
    assert summary["local_state"] == [str(state)]
    assert summary["resume_command"].startswith("npa agent deploy")
    assert summary["destroy_command"].startswith("npa agent destroy")
    assert operation.read()["phase"] == "recovery-required"


def test_nested_operation_uses_parent_and_rejects_cross_project_context(
    journal_root: Path,
) -> None:
    parent = _prepare()
    child = _prepare(command="npa agent fresh-setup")
    other = _prepare(
        project_alias="other",
        project_id="project-b",
        requested_name="other-agent",
    )

    with operation_context(parent):
        assert current_operation() == parent
        with operation_context(child) as selected:
            assert selected == parent
        with pytest.raises(OperationIdentityError):
            with operation_context(other):
                pass

    assert current_operation() is None


def test_concurrent_private_store_updates_do_not_overwrite_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "credentials.yaml"

    def add(index: int) -> None:
        update_private_yaml(
            path,
            lambda current: {**current, f"field_{index}": f"value_{index}"},
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(add, range(24)))

    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert saved == {f"field_{index}": f"value_{index}" for index in range(24)}
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
