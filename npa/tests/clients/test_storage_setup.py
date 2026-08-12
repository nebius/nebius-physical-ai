from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest
import yaml

from npa.clients import credentials, nebius, storage_setup
from npa.clients.storage_validation import (
    StorageConvergencePolicy,
    StorageFailureKind,
    StoragePhase,
    StorageProbeError,
    StorageProbeResult,
    StorageRetryability,
)

OK = StorageProbeResult(
    True, "ok", "verified", cleanup_attempted=True, cleanup_succeeded=True
)


@pytest.fixture
def credentials_path(tmp_path: Path, monkeypatch) -> Path:
    path = tmp_path / ".npa" / "credentials.yaml"
    monkeypatch.setattr(credentials, "CREDENTIALS_PATH", path)
    return path


def _result() -> dict[str, str]:
    return {
        "service_account_id": "serviceaccount-storage",
        "nebius_api_key": "NPA_ACCESS_CANARY",
        "nebius_secret_key": "NPA_SECRET_CANARY",
        "s3_bucket": "bucket-a",
        "s3_endpoint": "https://storage.example",
    }


def _created(callback) -> None:
    callback(
        "service_account",
        {"id": "serviceaccount-storage", "name": "lerobot-training"},
    )
    callback("bucket", {"name": "bucket-a"})
    callback(
        "access_key",
        {
            "id": "accesskey-storage",
            "name": "lerobot-access-key",
            "service_account_id": "serviceaccount-storage",
        },
    )


def test_success_records_each_creation_then_atomically_commits_private_credentials(
    credentials_path, monkeypatch
) -> None:
    snapshots: list[dict] = []

    def bootstrap(*_args, on_resource_created, **_kwargs):
        for kind, metadata in (
            (
                "service_account",
                {"id": "serviceaccount-storage", "name": "lerobot-training"},
            ),
            ("bucket", {"name": "bucket-a"}),
            (
                "access_key",
                {
                    "id": "accesskey-storage",
                    "name": "lerobot-access-key",
                    "service_account_id": "serviceaccount-storage",
                },
            ),
        ):
            on_resource_created(kind, metadata)
            snapshots.append(yaml.safe_load(credentials_path.read_text()))
        return _result()

    monkeypatch.setattr(nebius, "bootstrap_environment", bootstrap)
    monkeypatch.setattr(storage_setup, "probe_storage_write", lambda **_kwargs: OK)

    storage_setup.provision_storage(
        project_id="project-a",
        tenant_id="tenant-a",
        region="eu-north1",
        bucket_name="bucket-a",
        project_alias="demo",
    )

    assert (
        snapshots[0]["storage_setup"]["projects"]["project-a"]["resources"][
            "service_account"
        ]["id"]
        == "serviceaccount-storage"
    )
    assert (
        snapshots[1]["storage_setup"]["projects"]["project-a"]["resources"]["bucket"][
            "name"
        ]
        == "bucket-a"
    )
    assert (
        "accesskey-storage"
        in snapshots[2]["storage_setup"]["projects"]["project-a"]["resources"][
            "access_keys"
        ]
    )
    final = yaml.safe_load(credentials_path.read_text())
    assert final["storage_setup"]["projects"]["project-a"]["status"] == "complete"
    assert final["storage"]["aws_secret_access_key"] == "NPA_SECRET_CANARY"
    assert final["storage_iam"]["service_account_managed_by"] == "npa"
    assert stat.S_IMODE(credentials_path.stat().st_mode) == 0o600


def test_custom_created_storage_identity_keeps_exact_name_in_cleanup_proof(
    credentials_path, monkeypatch
) -> None:
    custom_name = "npa-run-scoped-storage-sa"

    def bootstrap(*_args, on_resource_created, service_account_name, **_kwargs):
        assert service_account_name == custom_name
        on_resource_created(
            "service_account",
            {"id": "serviceaccount-storage", "name": service_account_name},
        )
        return _result() | {
            "service_account_name": service_account_name,
            "service_account_project_id": "project-a",
            "service_account_managed_by": "npa",
        }

    monkeypatch.setattr(nebius, "bootstrap_environment", bootstrap)
    monkeypatch.setattr(storage_setup, "probe_storage_write", lambda **_kwargs: OK)
    storage_setup.provision_storage(
        project_id="project-a",
        tenant_id="tenant-a",
        region="eu-north1",
        bucket_name="bucket-a",
        service_account_name=custom_name,
        access_key_name="npa-run-scoped-key",
    )
    data = yaml.safe_load(credentials_path.read_text())
    assert {
        key: data["storage_iam"][key]
        for key in (
            "service_account_id",
            "service_account_name",
            "service_account_project_id",
            "service_account_managed_by",
        )
    } == {
        "service_account_id": "serviceaccount-storage",
        "service_account_name": custom_name,
        "service_account_project_id": "project-a",
        "service_account_managed_by": "npa",
    }


@pytest.mark.parametrize("boundary", ["service_account", "bucket", "access_key"])
def test_each_failure_boundary_rolls_back_only_this_attempt(
    credentials_path, monkeypatch, mocker, boundary
) -> None:
    def bootstrap(*_args, on_resource_created, **_kwargs):
        sequence = [
            (
                "service_account",
                {"id": "serviceaccount-storage", "name": "lerobot-training"},
            ),
            ("bucket", {"name": "bucket-a"}),
            (
                "access_key",
                {"id": "accesskey-storage", "name": "lerobot-access-key"},
            ),
        ]
        for kind, metadata in sequence:
            on_resource_created(kind, metadata)
            if kind == boundary:
                raise nebius.NebiusError("provider boundary failed")
        raise AssertionError("unreachable")

    monkeypatch.setattr(nebius, "bootstrap_environment", bootstrap)
    mocker.patch.object(
        nebius, "get_bucket_by_name", return_value={"metadata": {"id": "bucket-id"}}
    )
    delete_bucket = mocker.patch.object(nebius, "delete_bucket")
    delete_key = mocker.patch.object(nebius, "delete_access_key")
    delete_sa = mocker.patch.object(nebius, "delete_service_account")

    with pytest.raises(nebius.NebiusError, match="boundary failed"):
        storage_setup.provision_storage(
            project_id="project-a",
            tenant_id="tenant-a",
            region="eu-north1",
            bucket_name="bucket-a",
        )

    record = storage_setup.storage_setup_record("project-a")
    assert record["status"] == "rolled_back"
    assert record.get("resources", {}) == {}
    assert delete_sa.call_count == 1
    assert delete_bucket.call_count == (
        1 if boundary in {"bucket", "access_key"} else 0
    )
    assert delete_key.call_count == (1 if boundary == "access_key" else 0)


def test_failed_rollback_preserves_exact_resumable_provenance_without_secrets(
    credentials_path, monkeypatch, mocker
) -> None:
    canary = "NPA_ROLLBACK_SECRET_CANARY"

    def bootstrap(*_args, on_resource_created, **_kwargs):
        on_resource_created("bucket", {"name": "bucket-a"})
        raise nebius.NebiusError(f"secret={canary}")

    monkeypatch.setattr(nebius, "bootstrap_environment", bootstrap)
    mocker.patch.object(
        nebius, "get_bucket_by_name", return_value={"metadata": {"id": "bucket-id"}}
    )
    mocker.patch.object(
        nebius, "delete_bucket", side_effect=nebius.NebiusError("AccessDenied")
    )

    with pytest.raises(nebius.NebiusError) as caught:
        storage_setup.provision_storage(
            project_id="project-a",
            tenant_id="tenant-a",
            region="eu-north1",
            bucket_name="bucket-a",
            project_alias="demo",
        )

    raw = credentials_path.read_text()
    record = storage_setup.storage_setup_record("project-a")
    assert record["status"] == "partial"
    assert record["resources"]["bucket"]["name"] == "bucket-a"
    assert record["next_command"] == "npa provision-if-absent --project demo --skip-k8s"
    assert canary not in raw
    assert canary not in str(caught.value)


def test_failure_after_service_account_creation_keeps_cleanup_provenance_when_rollback_fails(
    credentials_path, monkeypatch, mocker
) -> None:
    def bootstrap(*_args, on_resource_created, **_kwargs):
        on_resource_created(
            "service_account",
            {"id": "serviceaccount-storage", "name": "lerobot-training"},
        )
        raise nebius.NebiusError("access-key configuration failed")

    monkeypatch.setattr(nebius, "bootstrap_environment", bootstrap)
    mocker.patch.object(
        nebius,
        "delete_service_account",
        side_effect=nebius.NebiusError("PermissionDenied"),
    )

    with pytest.raises(nebius.NebiusError, match="access-key configuration failed"):
        storage_setup.provision_storage(
            project_id="project-a",
            tenant_id="tenant-a",
            region="eu-north1",
            bucket_name="bucket-a",
        )

    record = storage_setup.storage_setup_record("project-a")
    account = record["resources"]["service_account"]
    assert record["status"] == "partial"
    assert account["id"] == "serviceaccount-storage"
    assert account["created_by"] == "npa"
    assert account["project_id"] == "project-a"


def test_creation_journal_serializes_only_non_secret_allowlisted_fields(
    credentials_path,
) -> None:
    transaction = storage_setup.StorageSetupTransaction(
        project_id="project-a",
        tenant_id="tenant-a",
        region="eu-north1",
        bucket_name="bucket-a",
    )
    transaction.begin()
    transaction.record_created(
        "access_key",
        {
            "id": "accesskey-a",
            "name": "lerobot-access-key",
            "service_account_id": "serviceaccount-a",
            "secret": "NPA_SECRET_MUST_NOT_BE_SAVED",
            "aws_secret_access_key": "NPA_AWS_SECRET_MUST_NOT_BE_SAVED",
        },
    )

    raw = credentials_path.read_text()
    assert "NPA_SECRET_MUST_NOT_BE_SAVED" not in raw
    assert "NPA_AWS_SECRET_MUST_NOT_BE_SAVED" not in raw
    assert "accesskey-a" in raw


def test_preexisting_resources_are_never_rolled_back(
    credentials_path, monkeypatch, mocker
) -> None:
    monkeypatch.setattr(
        nebius,
        "bootstrap_environment",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(nebius.NebiusError("failed")),
    )
    delete_key = mocker.patch.object(nebius, "delete_access_key")
    delete_bucket = mocker.patch.object(nebius, "delete_bucket")
    delete_sa = mocker.patch.object(nebius, "delete_service_account")

    with pytest.raises(nebius.NebiusError):
        storage_setup.provision_storage(
            project_id="project-a",
            tenant_id="tenant-a",
            region="eu-north1",
            bucket_name="bucket-a",
        )

    delete_key.assert_not_called()
    delete_bucket.assert_not_called()
    delete_sa.assert_not_called()


def test_preexisting_bucket_is_recorded_as_adopted_and_never_a_rollback_candidate(
    credentials_path, monkeypatch, mocker, tmp_path
) -> None:
    monkeypatch.setenv("NPA_OPERATION_JOURNAL_DIR", str(tmp_path / "operations"))
    monkeypatch.setattr(nebius, "bootstrap_environment", lambda *_a, **_k: _result())
    monkeypatch.setattr(storage_setup, "probe_storage_write", lambda **_kwargs: OK)
    delete_bucket = mocker.patch.object(nebius, "delete_bucket")

    storage_setup.provision_storage(
        project_id="project-a",
        tenant_id="tenant-a",
        region="eu-north1",
        bucket_name="bucket-a",
        project_alias="demo",
    )

    record = storage_setup.storage_setup_record("project-a")
    assert record["resources"]["bucket"]["created_by"] == "pre_existing"
    journals = list((tmp_path / "operations").glob("*/journal.json"))
    assert len(journals) == 1
    journal = json.loads(journals[0].read_text(encoding="utf-8"))
    bucket = next(
        resource
        for resource in journal["resources"]
        if resource["resource_type"] == "storage_bucket"
    )
    assert bucket["ownership"] == "adopted"
    delete_bucket.assert_not_called()


def test_malformed_owner_state_is_preserved_and_blocks_provider_changes(
    credentials_path, monkeypatch
) -> None:
    original = "storage: [unterminated\n"
    credentials_path.parent.mkdir(parents=True)
    credentials_path.write_text(original)
    called = False

    def bootstrap(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(nebius, "bootstrap_environment", bootstrap)

    with pytest.raises(storage_setup.StorageSetupStateError, match="malformed"):
        storage_setup.provision_storage(
            project_id="project-a",
            tenant_id="tenant-a",
            region="eu-north1",
            bucket_name="bucket-a",
        )

    assert called is False
    assert credentials_path.read_text() == original


def test_atomic_private_write_preserves_the_previous_document_on_replace_failure(
    credentials_path, monkeypatch
) -> None:
    original = "storage:\n  bucket: s3://existing/\n"
    credentials_path.parent.mkdir(parents=True)
    credentials_path.write_text(original)

    def fail_replace(_source, _destination):
        raise OSError("simulated atomic replace failure")

    monkeypatch.setattr("npa.clients.credentials.os.replace", fail_replace)

    with pytest.raises(OSError, match="replace failure"):
        credentials.write_private_yaml(credentials_path, {"storage": {"bucket": "new"}})

    assert credentials_path.read_text() == original
    assert list(credentials_path.parent.glob(f".{credentials_path.name}.*")) == []


def test_failed_write_probe_rolls_back_new_resources_before_credentials_commit(
    credentials_path, monkeypatch, mocker
) -> None:
    def bootstrap(*_args, on_resource_created, **_kwargs):
        _created(on_resource_created)
        return _result()

    monkeypatch.setattr(nebius, "bootstrap_environment", bootstrap)
    monkeypatch.setattr(
        storage_setup,
        "probe_storage_write",
        lambda **_kwargs: StorageProbeResult(
            False,
            "forbidden",
            "S3 write permission was denied.",
            error=StorageProbeError(
                StoragePhase.WRITE,
                StorageFailureKind.AUTHORIZATION,
                provider_code="AccessDenied",
                status_code=403,
                retryability=StorageRetryability.NEVER,
                message="S3 write permission was denied.",
            ),
        ),
    )
    mocker.patch.object(
        nebius, "get_bucket_by_name", return_value={"metadata": {"id": "bucket-id"}}
    )
    mocker.patch.object(nebius, "delete_bucket")
    mocker.patch.object(nebius, "delete_access_key")
    mocker.patch.object(nebius, "delete_service_account")

    with pytest.raises(nebius.NebiusError, match="write permission was denied"):
        storage_setup.provision_storage(
            project_id="project-a",
            tenant_id="tenant-a",
            region="eu-north1",
            bucket_name="bucket-a",
        )

    final = yaml.safe_load(credentials_path.read_text())
    assert final["storage_setup"]["projects"]["project-a"]["status"] == "rolled_back"
    assert "storage" not in final


def test_new_access_key_transient_403_converges_without_identity_drift_or_rollback(
    credentials_path, monkeypatch, mocker
) -> None:
    identities: list[tuple[str, str, str]] = []

    def bootstrap(*_args, on_resource_created, bucket_name, **_kwargs):
        on_resource_created(
            "service_account",
            {"id": "serviceaccount-storage", "name": "lerobot-training"},
        )
        on_resource_created("bucket", {"name": bucket_name})
        on_resource_created(
            "access_key",
            {
                "id": "accesskey-storage",
                "name": "lerobot-access-key",
                "service_account_id": "serviceaccount-storage",
            },
        )
        result = _result() | {"s3_bucket": bucket_name}
        identities.append(
            (
                result["service_account_id"],
                result["nebius_api_key"],
                result["s3_bucket"],
            )
        )
        return result

    transient = StorageProbeResult(
        False,
        "forbidden",
        "typed propagation candidate",
        error=StorageProbeError(
            StoragePhase.WRITE,
            StorageFailureKind.AUTHORIZATION,
            provider_code="AccessDenied",
            status_code=403,
            retryability=StorageRetryability.PROPAGATION,
            message="S3 write permission was denied.",
        ),
    )
    probes = iter((transient, OK))
    monkeypatch.setattr(nebius, "bootstrap_environment", bootstrap)
    monkeypatch.setattr(storage_setup, "probe_storage_write", lambda **_kwargs: next(probes))
    delete_key = mocker.patch.object(nebius, "delete_access_key")
    delete_bucket = mocker.patch.object(nebius, "delete_bucket")
    delete_sa = mocker.patch.object(nebius, "delete_service_account")
    sleeps: list[float] = []

    credentials_result, probe = storage_setup.provision_storage(
        project_id="project-a",
        tenant_id="tenant-a",
        region="eu-north1",
        bucket_name="bucket-stable",
        convergence_policy=StorageConvergencePolicy(
            max_attempts=3, initial_delay_seconds=2, jitter_ratio=0
        ),
        convergence_sleep=sleeps.append,
    )

    assert probe.ok and probe.attempts == 2
    assert sleeps == [2]
    assert identities == [
        ("serviceaccount-storage", "NPA_ACCESS_CANARY", "bucket-stable")
    ]
    assert credentials_result["s3_bucket"] == "bucket-stable"
    delete_key.assert_not_called()
    delete_bucket.assert_not_called()
    delete_sa.assert_not_called()
    record = storage_setup.storage_setup_record("project-a")
    assert record["status"] == "complete"
    assert record["resources"]["bucket"]["name"] == "bucket-stable"


def test_process_interruption_preserves_transaction_semantics(
    credentials_path, monkeypatch, mocker
) -> None:
    def interrupted(*_args, on_resource_created, **_kwargs):
        on_resource_created("bucket", {"name": "bucket-a"})
        raise KeyboardInterrupt

    monkeypatch.setattr(nebius, "bootstrap_environment", interrupted)
    mocker.patch.object(
        nebius, "get_bucket_by_name", return_value={"metadata": {"id": "bucket-id"}}
    )
    delete_bucket = mocker.patch.object(nebius, "delete_bucket")

    with pytest.raises(KeyboardInterrupt):
        storage_setup.provision_storage(
            project_id="project-a",
            tenant_id="tenant-a",
            region="eu-north1",
            bucket_name="bucket-a",
        )

    assert delete_bucket.call_args.args == ("bucket-id",)
    record = storage_setup.storage_setup_record("project-a")
    assert record["status"] == "rolled_back"
    assert record["next_command"] == "npa provision-if-absent --skip-k8s"


def test_retry_reconciles_interrupted_provenance_without_relabeling(
    credentials_path, monkeypatch
) -> None:
    interrupted = storage_setup.StorageSetupTransaction(
        project_id="project-a",
        tenant_id="tenant-a",
        region="eu-north1",
        bucket_name="bucket-a",
    )
    interrupted.begin()
    interrupted.record_created(
        "service_account",
        {"id": "serviceaccount-storage", "name": "lerobot-training"},
    )
    interrupted.record_created("bucket", {"name": "bucket-a"})

    def bootstrap(*_args, on_resource_created, **_kwargs):
        on_resource_created(
            "access_key",
            {"id": "accesskey-storage", "name": "lerobot-access-key"},
        )
        return _result()

    monkeypatch.setattr(nebius, "bootstrap_environment", bootstrap)
    monkeypatch.setattr(storage_setup, "probe_storage_write", lambda **_kwargs: OK)

    storage_setup.provision_storage(
        project_id="project-a",
        tenant_id="tenant-a",
        region="eu-north1",
        bucket_name="bucket-a",
    )

    record = storage_setup.storage_setup_record("project-a")
    assert record["status"] == "complete"
    assert (
        record["resources"]["service_account"]["attempt_id"] == interrupted.attempt_id
    )
    assert record["resources"]["bucket"]["attempt_id"] == interrupted.attempt_id
    assert json.loads(json.dumps(record))["resources"]["access_keys"][
        "accesskey-storage"
    ]


def test_retry_keeps_the_owned_partial_bucket_when_a_new_default_is_proposed(
    credentials_path, monkeypatch
) -> None:
    interrupted = storage_setup.StorageSetupTransaction(
        project_id="project-a",
        tenant_id="tenant-a",
        region="eu-north1",
        bucket_name="owned-partial-bucket",
    )
    interrupted.begin()
    interrupted.record_created("bucket", {"name": "owned-partial-bucket"})
    requested: list[str] = []

    def bootstrap(*_args, bucket_name, **_kwargs):
        requested.append(bucket_name)
        return _result() | {"s3_bucket": bucket_name}

    monkeypatch.setattr(nebius, "bootstrap_environment", bootstrap)
    monkeypatch.setattr(storage_setup, "probe_storage_write", lambda **_kwargs: OK)

    storage_setup.provision_storage(
        project_id="project-a",
        tenant_id="tenant-a",
        region="eu-north1",
        bucket_name="different-new-default",
    )

    assert requested == ["owned-partial-bucket"]
    record = storage_setup.storage_setup_record("project-a")
    assert record["resources"]["bucket"]["name"] == "owned-partial-bucket"
    assert record["resources"]["bucket"]["attempt_id"] == interrupted.attempt_id
