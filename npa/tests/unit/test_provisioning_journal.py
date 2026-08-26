from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import stat
import subprocess
import sys

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


def test_operation_owned_replacement_ids_are_preserved_as_generations(
    journal_root: Path,
) -> None:
    operation = _prepare()
    common = {
        "resource_type": "storage_service_account",
        "requested_name": "npa-storage",
        "project_id": "project-a",
        "ownership": "created_by_this_operation",
        "ownership_source": "provider-create-response",
    }
    operation.record_resource(provider_id="serviceaccount-old", **common)
    operation.record_resource(provider_id="serviceaccount-new", **common)

    resources = json.loads(operation.path.read_text(encoding="utf-8"))["resources"]
    assert [(item["provider_id"], item["generation"]) for item in resources] == [
        ("serviceaccount-old", 1),
        ("serviceaccount-new", 2),
    ]


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


def test_preflight_resume_allows_provider_progress_for_the_same_requested_shape(
    journal_root: Path,
) -> None:
    operation = _prepare()
    original = {
        "project_alias": "prod",
        "project_id": "project-a",
        "tenant_id": "tenant-a",
        "region": "us-central1",
        "topology": {
            "cluster_name": "gpu",
            "gpu_nodes": 1,
            "gpu_platform": "gpu-rtx6000",
            "gpu_preset": "1gpu-24vcpu-218gb",
            "gpu_disk_gib": 256,
            "existing_gpu_nodes": 0,
            "new_gpu_nodes": 1,
            "required_instances": 1,
            "required_disks": 1,
            "required_network_ssd_bytes": 256 * 1024**3,
            "required_network_ssd_gib": "256",
            "required_gpus": 1,
        },
    }
    operation.record_preflight_plan(original)

    converged = {
        **original,
        "topology": {
            **original["topology"],
            "existing_gpu_nodes": 1,
            "new_gpu_nodes": 0,
            "required_instances": 0,
            "required_disks": 0,
            "required_network_ssd_bytes": 0,
            "required_network_ssd_gib": "0",
            "required_gpus": 0,
        },
    }
    operation.record_preflight_plan(converged)

    assert operation.read()["preflight_plan"] == converged
    assert len(operation.read()["preflight_evaluations"]) == 2


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


def test_preserved_state_is_bound_to_hash_backend_and_operation(
    journal_root: Path,
) -> None:
    operation = _prepare()
    state = operation.preserve_state_bytes(b'{"version":4}', name="errored")
    record = operation.read()["local_state_copies"][0]
    assert record["operation_id"] == operation.operation_id
    assert record["sha256"]
    assert record["size_bytes"] == len(b'{"version":4}')
    assert record["backend_identity"] == {
        "bucket": "state-a",
        "endpoint": "storage.example",
    }
    assert operation.state_copies() == [state]

    state.write_bytes(b'{"version":4,"tampered":true}')
    with pytest.raises(OperationJournalError, match="failed SHA-256 validation"):
        operation.state_copies()


def test_represerving_same_named_state_atomically_replaces_stale_hash(
    journal_root: Path,
) -> None:
    operation = _prepare()
    state = operation.preserve_state_bytes(b'{"serial":1}', name="verified-remote")
    same = operation.preserve_state_bytes(b'{"serial":2}', name="verified-remote")

    assert same == state
    assert operation.state_copies() == [state]
    records = operation.read()["local_state_copies"]
    assert len(records) == 1
    assert records[0]["sha256"] == hashlib.sha256(b'{"serial":2}').hexdigest()


def test_missing_preserved_state_fails_closed_with_executable_recovery(
    journal_root: Path,
) -> None:
    operation = _prepare()
    state = operation.preserve_state_bytes(b'{"version":4}', name="errored")
    state.unlink()
    with pytest.raises(OperationJournalError) as exc_info:
        operation.state_copies()
    message = str(exc_info.value)
    assert "is missing; no state was adopted" in message
    assert "npa agent deploy --project prod --name agent" in message


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


def _context_holder(operation_id: str) -> str:
    return "\n".join(
        (
            "import sys",
            "from npa.provisioning_journal import load_operation, operation_context",
            f"operation = load_operation({operation_id!r})",
            "with operation_context(operation):",
            "    operation.transition('mutating')",
            "    print('READY', flush=True)",
            "    sys.stdin.readline()",
        )
    )


def test_project_lease_refuses_cross_process_setup_teardown_before_mutation(
    journal_root: Path,
) -> None:
    setup = _prepare()
    holder = subprocess.Popen(
        [sys.executable, "-c", _context_holder(setup.operation_id)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout is not None
    assert holder.stdout.readline().strip() == "READY"
    teardown = _prepare(
        command="npa destroy",
        resource_type="project",
        requested_name="prod",
        resume_command="npa destroy --project prod --yes",
    )
    try:
        with pytest.raises(OperationJournalError) as exc_info:
            with operation_context(teardown):
                pytest.fail("conflicting teardown entered its mutation window")
        message = str(exc_info.value)
        assert setup.operation_id in message
        assert "phase mutating" in message
        assert "Safe recovery: npa agent deploy --project prod --name agent" in message

        independent = _prepare(
            project_alias="other",
            project_id="project-b",
            requested_name="other-agent",
        )
        with operation_context(independent):
            independent.transition("mutating")
            independent.transition("rolled-back")
    finally:
        assert holder.stdin is not None
        holder.stdin.write("release\n")
        holder.stdin.flush()
        assert holder.wait() == 0


def test_project_lease_requires_explicit_resume_after_owner_crash(
    journal_root: Path,
) -> None:
    setup = _prepare()
    crash_source = _context_holder(setup.operation_id).replace(
        "    sys.stdin.readline()", "    __import__('os')._exit(23)"
    )
    crashed = subprocess.run(
        [sys.executable, "-c", crash_source],
        check=False,
        capture_output=True,
        text=True,
    )
    assert crashed.returncode == 23
    teardown = _prepare(
        command="npa destroy",
        resource_type="project",
        requested_name="prod",
        resume_command="npa destroy --project prod --yes",
    )
    with pytest.raises(OperationJournalError, match=setup.operation_id):
        with operation_context(teardown):
            pytest.fail("teardown must not adopt an interrupted setup")

    resumed = _prepare()
    assert resumed.operation_id == setup.operation_id
    with operation_context(resumed):
        resumed.transition("rolled-back")


def test_project_lease_allows_only_explicit_parent_child_reentrancy(
    journal_root: Path,
) -> None:
    parent = _prepare(command="npa destroy", resource_type="project-teardown")
    child = _prepare(
        command="npa agent destroy",
        resource_type="agent-teardown",
        requested_name="agent",
    )
    parent_script = f"""
import os, subprocess, sys
from npa.provisioning_journal import ProvisioningOperation, operation_context
with operation_context(ProvisioningOperation(sys.argv[1])):
    env = os.environ.copy()
    env['NPA_PARENT_LIFECYCLE_OPERATION'] = sys.argv[1]
    result = subprocess.run(
        [sys.executable, '-c', {repr(_context_holder(child.operation_id))}],
        input='release\\n', capture_output=True, text=True, env=env,
    )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    raise SystemExit(result.returncode)
"""
    completed = subprocess.run(
        [sys.executable, "-c", parent_script, parent.operation_id],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "READY" in completed.stdout


def test_active_teardown_blocks_workflow_submit_for_same_project(
    journal_root: Path,
) -> None:
    teardown = _prepare(
        command="npa destroy",
        resource_type="project-teardown",
        requested_name="prod",
        resume_command="npa destroy --project prod --all --yes",
    )
    submit = _prepare(
        command="npa workbench workflow submit",
        resource_type="workflow-submit",
        requested_name="paidf-run",
        resume_command=(
            "npa workbench workflow submit paidf.yaml --project prod "
            "--resume-run paidf-run"
        ),
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", _context_holder(teardown.operation_id)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout is not None
    assert holder.stdout.readline().strip() == "READY"
    try:
        with pytest.raises(OperationJournalError, match=teardown.operation_id):
            with operation_context(submit):
                pytest.fail("workflow submit entered during project teardown")
    finally:
        assert holder.stdin is not None
        holder.stdin.write("release\n")
        holder.stdin.flush()
        assert holder.wait() == 0


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
