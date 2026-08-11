from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import pytest
import yaml

from npa.clients import credentials, nebius, storage_setup
from npa.clients.project_credential_store import (
    AmbiguousLegacyCredentialError,
    forget_project_credentials,
    project_credential_record,
    write_project_credentials,
)
from npa.clients.storage_validation import (
    StorageConvergencePolicy,
    StorageFailureKind,
    StoragePhase,
    StorageProbeError,
    StorageProbeResult,
    StorageRetryability,
)
from npa.lifecycle_intent import OperationIntent, OperationIntentError, operation_intent


def _binding(state: nebius.IamBindingState) -> nebius.StorageIamBindingEvidence:
    return nebius.StorageIamBindingEvidence(
        state,
        nebius.STORAGE_RUNTIME_ROLE,
        "bucket-id",
        "group-id",
        nebius.STORAGE_BINDING_GROUP_NAME,
        "permit-id",
    )


def _bootstrap_result(binding_state: str) -> dict[str, str]:
    return {
        "service_account_id": "serviceaccount-stable",
        "nebius_api_key": "access-stable",
        "nebius_secret_key": "secret-stable",
        "s3_bucket": "bucket-stable",
        "s3_endpoint": "https://storage.example",
        "iam_binding_state": binding_state,
        "iam_binding_role": nebius.STORAGE_RUNTIME_ROLE,
        "iam_binding_scope_id": "bucket-id",
        "iam_binding_group_id": "group-id",
    }


def _deny() -> StorageProbeResult:
    error = StorageProbeError(
        StoragePhase.WRITE,
        StorageFailureKind.AUTHORIZATION,
        provider_code="AccessDenied",
        status_code=403,
        retryability=StorageRetryability.PROPAGATION,
        message="S3 write permission was denied.",
    )
    return StorageProbeResult(False, "forbidden", error.message, error=error)


def test_reused_key_new_grant_is_propagation_eligible_without_sleeping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "credentials.yaml"
    monkeypatch.setattr(credentials, "CREDENTIALS_PATH", path)
    monkeypatch.setattr(
        nebius, "bootstrap_environment", lambda *_a, **_k: _bootstrap_result("created")
    )
    probes = iter((_deny(), StorageProbeResult(True, "ok", "verified")))
    monkeypatch.setattr(storage_setup, "probe_storage_write", lambda **_k: next(probes))
    sleeps: list[float] = []

    result, probe = storage_setup.provision_storage(
        project_id="project-a",
        tenant_id="tenant-a",
        region="us-central1",
        bucket_name="bucket-stable",
        convergence_policy=StorageConvergencePolicy(
            max_attempts=2, initial_delay_seconds=0, jitter_ratio=0
        ),
        convergence_sleep=sleeps.append,
    )

    assert probe.ok and probe.attempts == 2
    assert sleeps == [0]
    assert result["service_account_id"] == "serviceaccount-stable"
    assert result["s3_bucket"] == "bucket-stable"


def test_reused_key_existing_grant_persistent_403_is_terminal_no_sleep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(credentials, "CREDENTIALS_PATH", tmp_path / "credentials.yaml")
    monkeypatch.setattr(
        nebius, "bootstrap_environment", lambda *_a, **_k: _bootstrap_result("existing")
    )
    monkeypatch.setattr(storage_setup, "probe_storage_write", lambda **_k: _deny())
    with pytest.raises(nebius.NebiusError, match="write permission was denied"):
        storage_setup.provision_storage(
            project_id="project-a",
            tenant_id="tenant-a",
            region="us-central1",
            bucket_name="bucket-stable",
            convergence_sleep=lambda _delay: pytest.fail(
                "terminal denial must not sleep"
            ),
        )


def test_narrow_binding_existing_is_accepted_without_mutation(monkeypatch) -> None:
    responses = iter(
        (
            {"metadata": {"id": "group-id"}},
            {
                "items": [
                    {
                        "metadata": {"id": "permit-id"},
                        "spec": {
                            "resource_id": "bucket-id",
                            "role": nebius.STORAGE_RUNTIME_ROLE,
                        },
                    }
                ]
            },
            {"memberships": [{"spec": {"member_id": "serviceaccount-a"}}]},
        )
    )
    monkeypatch.setattr(nebius, "_run_json", lambda *_a, **_k: next(responses))
    mutations: list[list[str]] = []
    monkeypatch.setattr(nebius, "_run", lambda argv, **_k: mutations.append(argv))
    evidence = nebius.ensure_storage_capability_binding(
        project_id="project-a",
        tenant_id="tenant-a",
        bucket_id="bucket-id",
        service_account_id="serviceaccount-a",
    )
    assert evidence.state is nebius.IamBindingState.EXISTING
    assert evidence.role == "storage.object-editor"
    assert mutations == []


def test_unknown_iam_inventory_fails_closed_before_key_or_probe(monkeypatch) -> None:
    monkeypatch.setattr(
        nebius,
        "_run_json",
        lambda *_a, **_k: (_ for _ in ()).throw(
            nebius.NebiusError("schema unreadable")
        ),
    )
    with pytest.raises(nebius.NebiusError, match="inventory is unreadable"):
        nebius.ensure_storage_capability_binding(
            project_id="project-a",
            tenant_id="tenant-a",
            bucket_id="bucket-id",
            service_account_id="serviceaccount-a",
        )


def test_bootstrap_reused_key_and_new_binding_reports_typed_propagation(
    monkeypatch,
) -> None:
    monkeypatch.setattr(nebius, "get_iam_token", lambda: "iam")
    monkeypatch.setattr(
        nebius, "ensure_service_account", lambda *_a, **_k: "sa-existing"
    )
    monkeypatch.setattr(nebius, "ensure_bucket", lambda *_a, **_k: None)
    monkeypatch.setattr(
        nebius,
        "get_bucket_by_name",
        lambda *_a, **_k: {"metadata": {"id": "bucket-id"}},
    )
    monkeypatch.setattr(nebius, "_existing_editors_binding", lambda *_a: None)
    monkeypatch.setattr(
        nebius,
        "ensure_storage_capability_binding",
        lambda **_k: _binding(nebius.IamBindingState.CREATED),
    )
    monkeypatch.setattr(
        nebius,
        "ensure_access_key",
        lambda *_a, **_k: (_ for _ in ()).throw(
            nebius.NebiusError("PermissionDenied: key create forbidden")
        ),
    )
    monkeypatch.setattr(
        nebius,
        "_saved_storage_credentials",
        lambda **_k: {
            "nebius_api_key": "reused-key",
            "nebius_secret_key": "reused-secret",
            "s3_bucket": "bucket-stable",
            "s3_endpoint": "https://storage.example",
        },
    )

    result = nebius.bootstrap_environment(
        "project-a", "tenant-a", "eu-test1", bucket_name="bucket-stable"
    )

    assert result["nebius_api_key"] == "reused-key"
    assert result["iam_binding_state"] == "created"
    assert result["iam_binding_role"] == nebius.STORAGE_RUNTIME_ROLE


def test_failed_editors_grant_has_typed_failed_evidence(monkeypatch) -> None:
    monkeypatch.setattr(
        nebius,
        "_run_json",
        lambda *_a, **_k: (_ for _ in ()).throw(
            nebius.NebiusError("PermissionDenied: redacted")
        ),
    )

    with pytest.raises(nebius.StorageIamBindingError) as raised:
        nebius.ensure_editors_membership("tenant-a", "sa-a")

    assert raised.value.evidence.state is nebius.IamBindingState.FAILED
    assert raised.value.evidence.compatibility_fallback is True
    for action in nebius.STORAGE_REQUIRED_S3_ACTIONS:
        assert action in str(raised.value)


def test_editors_fallback_only_for_provider_unsupported_role(monkeypatch) -> None:
    responses = iter(
        [
            nebius.NebiusError("NotFound"),
            {"metadata": {"id": "group-a"}},
            {"items": []},
            nebius.NebiusError("unknown role storage.object-editor"),
        ]
    )
    calls: list[list[str]] = []

    def run_json(argv, **_kwargs):
        calls.append(argv)
        value = next(responses)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(nebius, "_run_json", run_json)
    fallback = _binding(nebius.IamBindingState.CREATED)
    fallback = nebius.StorageIamBindingEvidence(
        fallback.state,
        "editor",
        "tenant-a",
        "editors-id",
        "editors",
        compatibility_fallback=True,
    )
    monkeypatch.setattr(nebius, "ensure_editors_membership", lambda *_a: fallback)

    result = nebius.ensure_storage_capability_binding(
        project_id="project-a",
        tenant_id="tenant-a",
        bucket_id="bucket-a",
        service_account_id="sa-a",
        allow_editors_fallback=True,
    )

    assert result.compatibility_fallback is True
    create_group = calls[1]
    assert create_group[create_group.index("--parent-id") + 1] == "tenant-a"
    assert create_group[
        create_group.index("--name") + 1
    ] == nebius.storage_binding_group_name("project-a")
    # Nebius CLI 0.12.254's group-create schema has no description field.
    assert "--description" not in create_group


def test_insufficient_binding_rejected_before_key(monkeypatch) -> None:
    monkeypatch.setattr(nebius, "get_iam_token", lambda: "iam")
    monkeypatch.setattr(nebius, "ensure_service_account", lambda *_a, **_k: "sa-a")
    monkeypatch.setattr(nebius, "ensure_bucket", lambda *_a, **_k: None)
    monkeypatch.setattr(
        nebius,
        "get_bucket_by_name",
        lambda *_a, **_k: {"metadata": {"id": "bucket-a"}},
    )
    monkeypatch.setattr(nebius, "_existing_editors_binding", lambda *_a: None)
    monkeypatch.setattr(
        nebius,
        "ensure_storage_capability_binding",
        lambda **_k: (_ for _ in ()).throw(nebius.NebiusError("insufficient role")),
    )
    key_calls: list[bool] = []
    monkeypatch.setattr(
        nebius, "ensure_access_key", lambda *_a, **_k: key_calls.append(True)
    )

    with pytest.raises(nebius.StorageIamBindingError) as raised:
        nebius.bootstrap_environment("project-a", "tenant-a", "eu-test1")

    assert raised.value.evidence.state is nebius.IamBindingState.FAILED
    assert key_calls == []


def test_destructive_intent_blocks_all_storage_creation_and_write_probe() -> None:
    with operation_intent(OperationIntent.DESTROY):
        with pytest.raises(OperationIntentError, match="provision_storage"):
            storage_setup.provision_storage(
                project_id="project-a",
                tenant_id="tenant-a",
                region="us-central1",
                bucket_name="bucket-a",
            )
        with pytest.raises(OperationIntentError, match="probe_storage_write"):
            storage_setup.probe_storage_write(  # type: ignore[attr-defined]
                bucket="bucket-a",
                endpoint_url="https://storage.example",
                access_key_id="a",
                secret_access_key="b",
            )


def test_two_projects_same_alias_changes_and_concurrent_writes_are_isolated(
    tmp_path: Path,
) -> None:
    path = tmp_path / "credentials.yaml"

    def save(index: int) -> None:
        project = f"project-{index % 2}"
        write_project_credentials(
            project,
            {
                "storage": {
                    "bucket": f"s3://bucket-{project}/",
                    "aws_access_key_id": f"access-{project}",
                    "aws_secret_access_key": f"secret-{project}",
                }
            },
            alias="shared" if index < 2 else f"renamed-{project}",
            path=path,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(save, range(40)))

    raw = yaml.safe_load(path.read_text())
    projects = raw["project_credentials"]["projects"]
    assert set(projects) == {"project-0", "project-1"}
    assert projects["project-0"]["storage"]["aws_access_key_id"] == "access-project-0"
    assert projects["project-1"]["storage"]["aws_access_key_id"] == "access-project-1"
    assert "shared" in projects["project-0"]["aliases"]
    assert "shared" in projects["project-1"]["aliases"]
    assert path.stat().st_mode & 0o077 == 0


def test_legacy_migration_requires_exact_proof_and_prunes_only_deleted_project(
    tmp_path: Path,
) -> None:
    path = tmp_path / "credentials.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "storage": {
                    "bucket": "s3://bucket-a/",
                    "aws_secret_access_key": "secret-a",
                },
                "storage_iam": {
                    "service_account_id": "sa-a",
                    "service_account_project_id": "project-a",
                    "service_account_managed_by": "npa",
                },
            }
        )
    )
    assert (
        project_credential_record("project-a", path=path)["storage"]["bucket"]
        == "s3://bucket-a/"
    )
    write_project_credentials(
        "project-b", {"storage": {"bucket": "s3://bucket-b/"}}, path=path
    )
    assert forget_project_credentials("project-a", path=path)
    remaining = yaml.safe_load(path.read_text())["project_credentials"]["projects"]
    assert set(remaining) == {"project-b"}
    assert "secret-a" not in path.read_text()


def test_ambiguous_legacy_credentials_fail_closed_and_remain_recoverable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "credentials.yaml"
    original = {
        "storage": {"bucket": "s3://ambiguous/", "aws_secret_access_key": "keep-me"}
    }
    path.write_text(yaml.safe_dump(original))
    with pytest.raises(AmbiguousLegacyCredentialError):
        project_credential_record("project-a", path=path)
    assert yaml.safe_load(path.read_text()) == original


def test_destructive_read_does_not_migrate_legacy_credentials(tmp_path: Path) -> None:
    path = tmp_path / "credentials.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "storage": {"bucket": "s3://legacy"},
                "storage_iam": {
                    "service_account_id": "sa-a",
                    "service_account_project_id": "project-a",
                },
            }
        ),
        encoding="utf-8",
    )
    before = path.read_bytes()

    with operation_intent(OperationIntent.DESTROY):
        assert (
            project_credential_record("project-a", path=path, migrate_legacy=False)
            == {}
        )

    assert path.read_bytes() == before


def test_verified_terminal_run_survives_missing_storage_and_alias(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.cli.workbench.workflow import _durable_workflow_status
    from npa.orchestration.npa_workflow import first_run_state
    from npa.orchestration.npa_workflow.first_run_state import state_path

    monkeypatch.setattr(first_run_state, "DEFAULT_ROOT", tmp_path / "runs")
    target = state_path(
        project_identity="project-a",
        workflow_identity="workflow-a",
        state_root=tmp_path / "runs",
    )
    target.parent.mkdir(parents=True)
    target.write_text(
        json.dumps(
            {
                "schema_version": first_run_state.STATE_SCHEMA,
                "project_identity": "project-a",
                "project_alias": "removed-alias",
                "workflow_identity": "workflow-a",
                "run_id": "run-a",
                "last_known_state": "SUCCEEDED",
                "last_verification_status": "VERIFIED",
            }
        ),
        encoding="utf-8",
    )

    result = _durable_workflow_status("run-a", project="project-a")

    assert result["status"] == "SUCCEEDED"
    assert result["artifact_verification"] == "unavailable"
    assert result["resolution_source"] == "project_run_terminal_ledger"


def test_storage_iam_receipt_keeps_replacement_generations(
    tmp_path: Path, monkeypatch
) -> None:
    from npa import teardown_receipts

    monkeypatch.setenv("NPA_TEARDOWN_RECEIPT_DIR", str(tmp_path / "receipts"))
    for account, key in (("sa-old", "key-old"), ("sa-new", "key-new")):
        teardown_receipts.record_teardown_event(
            phase="storage_iam",
            resource=account,
            terminal_state="completed",
            project_alias="demo",
            project_id="project-a",
            identity={
                "service_account_id": account,
                "service_account_name": "lerobot-training",
                "ownership": "npa",
                "iam_key_ids": [key],
            },
        )
    receipt = teardown_receipts.list_teardown_receipts(project_id="project-a")[0]
    generations = receipt["identity"]["storage_iam"]["generations"]
    assert {
        (item["service_account_id"], tuple(item["iam_key_ids"])) for item in generations
    } == {
        ("sa-old", ("key-old",)),
        ("sa-new", ("key-new",)),
    }
    assert "secret" not in json.dumps(receipt).lower()


def test_storage_iam_stable_absence_checks_ids_names_and_never_sleeps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa import project_destroy

    monkeypatch.setattr(
        nebius,
        "get_service_account_identity",
        lambda _account, **_kwargs: None,
    )
    monkeypatch.setattr(
        nebius,
        "get_service_account_id_by_name",
        lambda _project, _name, *, strict: None,
    )
    monkeypatch.setattr(project_destroy.time, "sleep", lambda _seconds: None)

    result = project_destroy._verify_storage_iam_stable_absence(
        project_id="project-a",
        account_ids=("sa-old", "sa-new"),
        names=("npa-storage-object-editors-project-a",),
    )

    assert result["stable_absence_observations"] == 2


def test_storage_iam_same_name_replacement_invalidates_absence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa import project_destroy

    monkeypatch.setattr(
        nebius,
        "get_service_account_identity",
        lambda _account, **_kwargs: None,
    )
    monkeypatch.setattr(
        nebius,
        "get_service_account_id_by_name",
        lambda _project, _name, *, strict: "sa-replacement",
    )
    monkeypatch.setattr(
        project_destroy.time,
        "sleep",
        lambda _seconds: pytest.fail("replacement failure must not sleep"),
    )

    with pytest.raises(RuntimeError, match="same-name replacement"):
        project_destroy._verify_storage_iam_stable_absence(
            project_id="project-a",
            account_ids=("sa-old",),
            names=("npa-storage-object-editors-project-a",),
        )
