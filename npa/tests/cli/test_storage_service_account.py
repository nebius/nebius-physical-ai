"""Safe teardown for the storage service account NPA owns."""

from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from npa.cli.main import app


runner = CliRunner()


def _seed_owned_account(
    monkeypatch,
    tmp_path: Path,
    *,
    include_storage: bool = False,
    managed_by: str = "npa",
) -> Path:
    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "default_project": "prod",
                "projects": {
                    "prod": {"project_id": "project-a", "tenant_id": "tenant-a"}
                },
            }
        )
    )
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)

    credentials_path = tmp_path / "credentials.yaml"
    payload: dict[str, object] = {
        "nebius": {"service_account_id": "serviceaccount-storage"},
        "storage_iam": {
            "service_account_id": "serviceaccount-storage",
            "service_account_name": "lerobot-training",
            "service_account_project_id": "project-a",
            "service_account_managed_by": managed_by,
        }
    }
    if include_storage:
        payload["storage"] = {
            "bucket": "s3://still-configured",
            "aws_access_key_id": "AK",
            "aws_secret_access_key": "SK",
        }
    credentials_path.write_text(yaml.safe_dump(payload))
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", credentials_path)
    from npa.clients import nebius as nebius_module

    monkeypatch.setattr(
        nebius_module,
        "get_service_account_identity",
        lambda account_id, **_kwargs: nebius_module.ServiceAccountIdentity(
            account_id=account_id,
            name="lerobot-training",
            project_id="project-a",
            tenant_id="tenant-a",
            profile="",
        ),
    )
    monkeypatch.setattr(
        nebius_module,
        "get_service_account_id_by_name",
        lambda _project_id, _name, **_kwargs: "serviceaccount-storage",
    )
    return credentials_path


def _stub_iam(monkeypatch) -> list[str]:
    from npa.clients import nebius as nebius_module

    monkeypatch.setattr(
        nebius_module,
        "list_access_keys_for_service_account",
        lambda project_id, sa_id, **kwargs: [
            {"id": "accesskey-storage", "name": "lerobot-access-key", "state": "ACTIVE"}
        ],
    )
    deleted: list[str] = []
    monkeypatch.setattr(nebius_module, "delete_access_key", lambda key_id: deleted.append(key_id))
    monkeypatch.setattr(
        nebius_module, "delete_service_account", lambda sa_id: deleted.append(sa_id)
    )
    return deleted


def test_storage_service_account_dry_run_names_exact_owned_resources(monkeypatch, tmp_path) -> None:
    _seed_owned_account(monkeypatch, tmp_path)
    deleted = _stub_iam(monkeypatch)

    result = runner.invoke(
        app,
        ["storage", "service-account", "delete", "--project", "prod", "--dry-run"],
    )

    assert result.exit_code == 0, result.output
    assert "Would delete access key accesskey-storage" in result.output
    assert "Would delete NPA-owned service account lerobot-training" in result.output
    assert "serviceaccount-storage" in result.output
    assert deleted == []


def test_storage_service_account_yes_deletes_keys_then_account_and_prunes_marker(
    monkeypatch, tmp_path
) -> None:
    credentials_path = _seed_owned_account(monkeypatch, tmp_path)
    deleted = _stub_iam(monkeypatch)

    result = runner.invoke(
        app,
        ["storage", "service-account", "delete", "--project", "prod", "--yes"],
    )

    assert result.exit_code == 0, result.output
    assert deleted == ["accesskey-storage", "serviceaccount-storage"]
    assert "Deleted access key accesskey-storage" in result.output
    assert "Verified deletion: NPA-owned service account lerobot-training" in result.output
    assert not credentials_path.exists()


def test_bucket_then_owned_service_account_delete_preserves_provenance_until_iam_is_gone(
    monkeypatch, tmp_path
) -> None:
    import yaml

    from npa.cli import storage as storage_cli
    from npa.clients import config as config_module
    from npa.clients import nebius as nebius_module

    credentials_path = _seed_owned_account(monkeypatch, tmp_path, include_storage=True)
    deleted = _stub_iam(monkeypatch)
    monkeypatch.setattr(
        nebius_module,
        "get_bucket_by_name",
        lambda project_id, name: {
            "metadata": {"id": "bucket-storage", "name": name}
        },
    )
    monkeypatch.setattr(nebius_module, "delete_bucket", lambda bucket_id, *, ttl="": None)
    waited: list[tuple[str, str]] = []
    monkeypatch.setattr(
        storage_cli,
        "_wait_for_bucket_gone",
        lambda project_id, name, target, timeout: waited.append((project_id, name)),
    )

    bucket_result = runner.invoke(
        app,
        [
            "storage",
            "bucket",
            "delete",
            "--project",
            "prod",
            "--yes",
            "--wait",
        ],
    )

    assert bucket_result.exit_code == 0, bucket_result.output
    assert waited == [("project-a", "still-configured")]
    after_bucket = yaml.safe_load(credentials_path.read_text())
    assert "storage" not in after_bucket
    assert after_bucket["storage_iam"] == {
        "service_account_id": "serviceaccount-storage",
        "service_account_name": "lerobot-training",
        "service_account_project_id": "project-a",
        "service_account_managed_by": "npa",
    }
    marker = yaml.safe_load(config_module.CONFIG_PATH.read_text())["projects"]["prod"][
        "storage_iam_verification_required"
    ]
    assert marker["schema_version"] == "npa.storage-iam-residue.v2"
    assert marker["ownership_state"] == "owned"
    assert marker["bucket_cleanup_state"] == "complete"
    assert marker["service_account_id"] == "serviceaccount-storage"
    assert marker["access_key_ids"] == ["AK"]
    assert "SK" not in yaml.safe_dump(marker)
    assert "storage-IAM cleanup remains pending" in bucket_result.output
    assert (
        "npa storage service-account delete --project prod --dry-run"
        in bucket_result.output
    )

    iam_result = runner.invoke(
        app,
        ["storage", "service-account", "delete", "--project", "prod", "--yes"],
    )

    assert iam_result.exit_code == 0, iam_result.output
    assert deleted == ["accesskey-storage", "serviceaccount-storage"]
    assert not credentials_path.exists()
    config = yaml.safe_load(config_module.CONFIG_PATH.read_text())
    assert "storage_iam_verification_required" not in config["projects"]["prod"]


def test_v1_storage_iam_marker_migrates_without_losing_identity(
    monkeypatch, tmp_path
) -> None:
    from npa.clients import config as config_module

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "projects": {
                    "prod": {
                        "project_id": "project-a",
                        "storage_iam_verification_required": {
                            "schema_version": "npa.storage-iam-residue.v1",
                            "status": "present_owned",
                            "ownership": "npa",
                            "service_account_id": "serviceaccount-storage",
                            "access_key_ids": ["key-b", "key-a", "key-a"],
                        },
                    }
                }
            }
        )
    )
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)

    migrated = config_module.storage_iam_residue("prod")
    persisted = config_module.mark_storage_iam_residue(
        "prod", {"bucket_cleanup_state": "complete"}
    )

    assert migrated["ownership_state"] == "owned"
    assert migrated["schema_version"] == "npa.storage-iam-residue.v2"
    assert persisted["service_account_id"] == "serviceaccount-storage"
    assert persisted["access_key_ids"] == ["key-a", "key-b"]
    on_disk = yaml.safe_load(config_path.read_text())["projects"]["prod"][
        "storage_iam_verification_required"
    ]
    assert on_disk == persisted


def test_storage_service_account_refuses_unowned_legacy_identity(monkeypatch, tmp_path) -> None:
    credentials_path = _seed_owned_account(monkeypatch, tmp_path, managed_by="")
    deleted = _stub_iam(monkeypatch)

    result = runner.invoke(
        app,
        ["storage", "service-account", "delete", "--project", "prod", "--yes"],
    )

    assert result.exit_code == 2, result.output
    assert "No trustworthy NPA storage IAM ownership record" in result.output
    assert "not ownership proof" in result.output
    assert deleted == []
    assert credentials_path.exists()


def test_bucket_delete_preserves_nonowned_identity_evidence_but_iam_delete_refuses_it(
    monkeypatch, tmp_path
) -> None:
    import yaml

    from npa.clients import config as config_module
    from npa.clients import nebius as nebius_module

    credentials_path = _seed_owned_account(
        monkeypatch, tmp_path, include_storage=True, managed_by=""
    )
    deleted = _stub_iam(monkeypatch)
    monkeypatch.setattr(
        nebius_module,
        "get_bucket_by_name",
        lambda project_id, name: {
            "metadata": {"id": "bucket-storage", "name": name}
        },
    )
    monkeypatch.setattr(nebius_module, "delete_bucket", lambda bucket_id, *, ttl="": None)

    bucket_result = runner.invoke(
        app,
        ["storage", "bucket", "delete", "--project", "prod", "--yes"],
    )

    assert bucket_result.exit_code == 0, bucket_result.output
    after_bucket = yaml.safe_load(credentials_path.read_text())
    assert "storage" not in after_bucket
    assert after_bucket["nebius"]["service_account_id"] == "serviceaccount-storage"
    marker = yaml.safe_load(config_module.CONFIG_PATH.read_text())["projects"]["prod"][
        "storage_iam_verification_required"
    ]
    assert marker["ownership_state"] == "pending-verification"
    assert marker["service_account_id"] == "serviceaccount-storage"
    assert (
        "npa storage service-account reconcile --project prod --id "
        "serviceaccount-storage --dry-run"
    ) in bucket_result.output

    iam_result = runner.invoke(
        app,
        ["storage", "service-account", "delete", "--project", "prod", "--yes"],
    )

    assert iam_result.exit_code == 2, iam_result.output
    assert "not ownership proof" in iam_result.output
    assert "serviceaccount-storage" in iam_result.output
    assert deleted == []
    assert credentials_path.exists()


def test_agent_service_account_id_cannot_overwrite_storage_ownership(
    monkeypatch, tmp_path
) -> None:
    import yaml

    credentials_path = _seed_owned_account(monkeypatch, tmp_path)
    data = yaml.safe_load(credentials_path.read_text())
    data["nebius"]["service_account_id"] = "serviceaccount-agent"
    credentials_path.write_text(yaml.safe_dump(data))
    deleted = _stub_iam(monkeypatch)

    result = runner.invoke(
        app,
        ["storage", "service-account", "delete", "--project", "prod", "--yes"],
    )

    assert result.exit_code == 0, result.output
    assert deleted == ["accesskey-storage", "serviceaccount-storage"]
    remaining = yaml.safe_load(credentials_path.read_text())
    assert remaining["nebius"]["service_account_id"] == "serviceaccount-agent"
    assert "storage_iam" not in remaining


def test_complete_legacy_nebius_ownership_record_remains_deletable(
    monkeypatch, tmp_path
) -> None:
    import yaml

    credentials_path = _seed_owned_account(monkeypatch, tmp_path)
    data = yaml.safe_load(credentials_path.read_text())
    data["nebius"].update(data.pop("storage_iam"))
    credentials_path.write_text(yaml.safe_dump(data))
    deleted = _stub_iam(monkeypatch)

    result = runner.invoke(
        app,
        ["storage", "service-account", "delete", "--project", "prod", "--yes"],
    )

    assert result.exit_code == 0, result.output
    assert deleted == ["accesskey-storage", "serviceaccount-storage"]
    assert not credentials_path.exists()


def test_storage_service_account_refuses_while_bucket_credentials_remain(
    monkeypatch, tmp_path
) -> None:
    _seed_owned_account(monkeypatch, tmp_path, include_storage=True)
    deleted = _stub_iam(monkeypatch)

    result = runner.invoke(
        app,
        ["storage", "service-account", "delete", "--project", "prod", "--yes"],
    )

    assert result.exit_code != 0
    assert "still-configured" in result.output
    assert "storage bucket delete" in result.output
    assert deleted == []


def test_storage_service_account_requires_yes_in_noninteractive_mode(monkeypatch, tmp_path) -> None:
    _seed_owned_account(monkeypatch, tmp_path)
    deleted = _stub_iam(monkeypatch)

    result = runner.invoke(
        app,
        ["storage", "service-account", "delete", "--project", "prod"],
    )

    assert result.exit_code != 0
    assert "Re-run with --yes" in result.output
    assert deleted == []


def test_storage_service_account_rejects_mismatched_project_marker(monkeypatch, tmp_path) -> None:
    credentials_path = _seed_owned_account(monkeypatch, tmp_path)
    data = yaml.safe_load(credentials_path.read_text())
    data["storage_iam"]["service_account_project_id"] = "project-other"
    credentials_path.write_text(yaml.safe_dump(data))
    deleted = _stub_iam(monkeypatch)

    result = runner.invoke(
        app,
        ["storage", "service-account", "delete", "--project", "prod", "--yes"],
    )

    assert result.exit_code == 2, result.output
    assert "does not match project" in result.output
    assert deleted == []


def test_storage_service_account_failure_is_reported_and_marker_is_retriable(
    monkeypatch, tmp_path
) -> None:
    from npa.clients import nebius as nebius_module

    credentials_path = _seed_owned_account(monkeypatch, tmp_path)
    monkeypatch.setattr(
        nebius_module,
        "list_access_keys_for_service_account",
        lambda project_id, sa_id, **kwargs: [
            {"id": "accesskey-storage", "name": "lerobot-access-key", "state": "ACTIVE"}
        ],
    )
    monkeypatch.setattr(
        nebius_module,
        "delete_access_key",
        lambda key_id: (_ for _ in ()).throw(nebius_module.NebiusError("key busy")),
    )
    monkeypatch.setattr(
        nebius_module,
        "delete_service_account",
        lambda sa_id: (_ for _ in ()).throw(nebius_module.NebiusError("account busy")),
    )

    result = runner.invoke(
        app,
        ["storage", "service-account", "delete", "--project", "prod", "--yes"],
    )

    assert result.exit_code == 1
    assert "could not delete access key accesskey-storage" in result.output
    assert "could not delete NPA-owned service account serviceaccount-storage" in result.output
    assert credentials_path.exists()
    assert yaml.safe_load(credentials_path.read_text())["storage_iam"][
        "service_account_managed_by"
    ] == "npa"


def test_storage_service_account_list_not_found_reconciles_absent_account(
    monkeypatch, tmp_path
) -> None:
    from npa.clients import nebius as nebius_module

    credentials_path = _seed_owned_account(monkeypatch, tmp_path)
    deleted: list[str] = []
    identities = iter(
        [
            nebius_module.ServiceAccountIdentity(
                "serviceaccount-storage",
                "lerobot-training",
                "project-a",
                "tenant-a",
                "",
            ),
            None,
        ]
    )
    monkeypatch.setattr(
        nebius_module,
        "get_service_account_identity",
        lambda *_args, **_kwargs: next(identities),
    )
    monkeypatch.setattr(
        nebius_module,
        "list_access_keys_for_service_account",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            nebius_module.NebiusError("NotFound: service account is absent")
        ),
    )
    monkeypatch.setattr(
        nebius_module,
        "delete_service_account",
        lambda account_id: deleted.append(account_id),
    )

    result = runner.invoke(
        app,
        ["storage", "service-account", "delete", "--project", "prod", "--yes"],
    )

    assert result.exit_code == 0, result.output
    assert "already absent" in result.output
    assert deleted == []
    assert not credentials_path.exists()


def test_storage_service_account_access_key_not_found_is_idempotent(
    monkeypatch, tmp_path
) -> None:
    from npa.clients import nebius as nebius_module

    credentials_path = _seed_owned_account(monkeypatch, tmp_path)
    monkeypatch.setattr(
        nebius_module,
        "list_access_keys_for_service_account",
        lambda *args, **kwargs: [{"id": "accesskey-storage"}],
    )
    monkeypatch.setattr(
        nebius_module,
        "delete_access_key",
        lambda key_id: (_ for _ in ()).throw(
            nebius_module.NebiusError("NotFound: access key is absent")
        ),
    )
    deleted: list[str] = []
    monkeypatch.setattr(
        nebius_module,
        "delete_service_account",
        lambda account_id: deleted.append(account_id),
    )

    result = runner.invoke(
        app,
        ["storage", "service-account", "delete", "--project", "prod", "--yes"],
    )

    assert result.exit_code == 0, result.output
    assert "Access key accesskey-storage is already absent" in result.output
    assert deleted == ["serviceaccount-storage"]
    assert not credentials_path.exists()


def test_storage_service_account_does_not_delete_when_key_inventory_fails(
    monkeypatch, tmp_path
) -> None:
    from npa.clients import nebius as nebius_module

    credentials_path = _seed_owned_account(monkeypatch, tmp_path)
    monkeypatch.setattr(
        nebius_module,
        "list_access_keys_for_service_account",
        lambda project_id, sa_id, **kwargs: (_ for _ in ()).throw(
            nebius_module.NebiusError("permission denied")
        ),
    )
    deleted: list[str] = []
    monkeypatch.setattr(nebius_module, "delete_access_key", lambda key_id: deleted.append(key_id))
    monkeypatch.setattr(
        nebius_module, "delete_service_account", lambda sa_id: deleted.append(sa_id)
    )

    result = runner.invoke(
        app,
        ["storage", "service-account", "delete", "--project", "prod", "--yes"],
    )

    assert result.exit_code != 0
    assert "provider/auth verification failed while inspecting access keys" in result.output
    assert "nothing was deleted" in result.output
    assert deleted == []
    assert credentials_path.exists()


def test_storage_service_account_already_absent_is_idempotent(monkeypatch, tmp_path) -> None:
    from npa.clients import nebius as nebius_module

    credentials_path = _seed_owned_account(monkeypatch, tmp_path)
    monkeypatch.setattr(
        nebius_module,
        "list_access_keys_for_service_account",
        lambda project_id, sa_id, **kwargs: [],
    )
    monkeypatch.setattr(
        nebius_module,
        "delete_service_account",
        lambda sa_id: (_ for _ in ()).throw(nebius_module.NebiusError("NotFound")),
    )

    result = runner.invoke(
        app,
        ["storage", "service-account", "delete", "--project", "prod", "--yes"],
    )

    assert result.exit_code == 0, result.output
    assert "is already absent" in result.output
    assert not credentials_path.exists()


def test_storage_service_account_reports_local_marker_cleanup_failure(
    monkeypatch, tmp_path
) -> None:
    from npa.cli import storage as storage_cli

    credentials_path = _seed_owned_account(monkeypatch, tmp_path)
    _stub_iam(monkeypatch)
    monkeypatch.setattr(
        storage_cli, "_remove_storage_service_account_record", lambda account_id: False
    )

    result = runner.invoke(
        app,
        ["storage", "service-account", "delete", "--project", "prod", "--yes"],
    )

    assert result.exit_code == 1
    assert "service account is gone" in result.output
    assert "ownership record" in result.output
    assert credentials_path.exists()


def test_failed_setup_journal_is_trusted_for_safe_recovery(
    monkeypatch, tmp_path
) -> None:
    credentials_path = _seed_owned_account(monkeypatch, tmp_path)
    data = yaml.safe_load(credentials_path.read_text())
    data.pop("storage_iam")
    data.pop("nebius")
    data["storage_setup"] = {
        "version": 1,
        "projects": {
            "project-a": {
                "status": "partial",
                "phase": "rollback_incomplete",
                "resources": {
                    "service_account": {
                        "id": "serviceaccount-storage",
                        "name": "lerobot-training",
                        "created_by": "npa",
                        "project_id": "project-a",
                        "attempt_id": "attempt-a",
                    }
                },
            }
        },
    }
    credentials_path.write_text(yaml.safe_dump(data))
    deleted = _stub_iam(monkeypatch)

    dry_run = runner.invoke(
        app,
        ["storage", "service-account", "delete", "--project", "prod", "--dry-run"],
    )
    assert dry_run.exit_code == 0, dry_run.output
    assert "storage setup journal for project-a" in dry_run.output

    deletion = runner.invoke(
        app,
        ["storage", "service-account", "delete", "--project", "prod", "--yes"],
    )
    assert deletion.exit_code == 0, deletion.output
    assert "Verified deletion" in deletion.output
    assert deleted == ["accesskey-storage", "serviceaccount-storage"]
    assert not credentials_path.exists()


def test_missing_ownership_exact_present_is_consistent_across_dry_run_and_yes(
    monkeypatch, tmp_path
) -> None:
    credentials_path = _seed_owned_account(
        monkeypatch, tmp_path, managed_by=""
    )

    dry_run = runner.invoke(
        app,
        ["storage", "service-account", "delete", "--project", "prod", "--dry-run"],
    )
    result = runner.invoke(
        app, ["storage", "service-account", "delete", "--project", "prod", "--yes"]
    )

    assert dry_run.exit_code == result.exit_code == 2
    assert "not ownership proof" in dry_run.output
    assert "not ownership proof" in result.output
    assert "Verified absence" not in dry_run.output + result.output
    assert credentials_path.exists()


def test_missing_ownership_provider_failure_is_partial(
    monkeypatch, tmp_path
) -> None:
    _seed_owned_account(monkeypatch, tmp_path, managed_by="")
    from npa.clients import nebius as nebius_module

    monkeypatch.setattr(
        nebius_module,
        "get_service_account_identity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            nebius_module.NebiusError("Unauthenticated")
        ),
    )

    result = runner.invoke(
        app,
        ["storage", "service-account", "delete", "--project", "prod", "--yes"],
    )

    assert result.exit_code == 2
    assert "provider/auth verification failed" in result.output
    assert "nothing was deleted" in result.output


def test_owned_account_provider_verification_failure_preserves_provenance(
    monkeypatch, tmp_path
) -> None:
    credentials_path = _seed_owned_account(monkeypatch, tmp_path)
    from npa.clients import nebius as nebius_module

    monkeypatch.setattr(
        nebius_module,
        "get_service_account_identity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            nebius_module.NebiusError("PermissionDenied")
        ),
    )
    deleted: list[str] = []
    monkeypatch.setattr(
        nebius_module, "delete_service_account", lambda account_id: deleted.append(account_id)
    )

    result = runner.invoke(
        app,
        ["storage", "service-account", "delete", "--project", "prod", "--yes"],
    )

    assert result.exit_code == 2
    assert "provider/auth verification failed" in result.output
    assert deleted == []
    assert "storage_iam" in yaml.safe_load(credentials_path.read_text())


def test_verified_absence_dry_run_does_not_remove_ownership_before_real_pass(
    monkeypatch, tmp_path
) -> None:
    credentials_path = _seed_owned_account(monkeypatch, tmp_path)
    from npa.clients import nebius as nebius_module

    monkeypatch.setattr(
        nebius_module, "get_service_account_identity", lambda *_args, **_kwargs: None
    )

    dry_run = runner.invoke(
        app,
        ["storage", "service-account", "delete", "--project", "prod", "--dry-run"],
    )
    assert dry_run.exit_code == 0, dry_run.output
    assert "Verified absence" in dry_run.output
    assert yaml.safe_load(credentials_path.read_text())["storage_iam"][
        "service_account_managed_by"
    ] == "npa"

    real = runner.invoke(
        app, ["storage", "service-account", "delete", "--project", "prod", "--yes"]
    )
    assert real.exit_code == 0, real.output
    assert not credentials_path.exists()


def test_reconcile_dry_run_and_explicit_attestation_feed_guarded_delete(
    monkeypatch, tmp_path
) -> None:
    credentials_path = _seed_owned_account(monkeypatch, tmp_path, managed_by="")
    _stub_iam(monkeypatch)
    from npa.clients import config as config_module

    dry_run = runner.invoke(
        app,
        [
            "storage",
            "service-account",
            "reconcile",
            "--project",
            "prod",
            "--id",
            "serviceaccount-storage",
            "--dry-run",
            "--json",
        ],
    )
    assert dry_run.exit_code == 0, dry_run.output
    plan = yaml.safe_load(dry_run.output)
    assert plan["result"] == "reconciliation_planned"
    assert plan["service_account_id"] == "serviceaccount-storage"
    assert plan["will_delete"] is False
    assert yaml.safe_load(credentials_path.read_text())["storage_iam"][
        "service_account_managed_by"
    ] == ""
    config = yaml.safe_load(config_module.CONFIG_PATH.read_text())
    marker = config["projects"]["prod"]["storage_iam_verification_required"]
    assert marker["status"] == "present_unverified_ownership"
    assert marker["service_account_id"] == "serviceaccount-storage"

    missing_attestation = runner.invoke(
        app,
        [
            "storage",
            "service-account",
            "reconcile",
            "--project",
            "prod",
            "--id",
            "serviceaccount-storage",
            "--reason",
            "legacy NPA setup",
            "--yes",
        ],
    )
    assert missing_attestation.exit_code == 2

    sensitive_reason = runner.invoke(
        app,
        [
            "storage",
            "service-account",
            "reconcile",
            "--project",
            "prod",
            "--id",
            "serviceaccount-storage",
            "--reason",
            "hf_abcdefghijklmnop",
            "--attest-npa-created",
            "--yes",
        ],
    )
    assert sensitive_reason.exit_code == 2
    assert "credential material" in sensitive_reason.output
    assert "hf_abcdefghijklmnop" not in sensitive_reason.output
    assert "recovery" not in yaml.safe_load(credentials_path.read_text())["storage_iam"]

    reconciled = runner.invoke(
        app,
        [
            "storage",
            "service-account",
            "reconcile",
            "--project",
            "prod",
            "--id",
            "serviceaccount-storage",
            "--reason",
            "legacy NPA setup",
            "--attested-by",
            "operator@example",
            "--attest-npa-created",
            "--yes",
        ],
    )
    assert reconciled.exit_code == 0, reconciled.output
    journal = yaml.safe_load(credentials_path.read_text())["storage_iam"]
    assert journal["service_account_managed_by"] == "npa-recovery-attested"
    assert journal["recovery"]["attested_by"] == "operator@example"
    assert journal["recovery"]["reason"] == "legacy NPA setup"
    assert journal["recovery"]["provider_verified"] is True
    assert "secret" not in yaml.safe_dump(journal).lower()
    reconciled_marker = config_module.storage_iam_residue("prod")
    assert reconciled_marker["ownership_state"] == "owned"

    # Reconciliation is restart-safe, and the resulting journal is accepted by
    # the existing guarded deletion plan rather than a parallel delete path.
    repeated = runner.invoke(
        app,
        [
            "storage",
            "service-account",
            "reconcile",
            "--project",
            "prod",
            "--id",
            "serviceaccount-storage",
            "--reason",
            "legacy NPA setup",
            "--attested-by",
            "operator@example",
            "--attest-npa-created",
            "--yes",
        ],
    )
    assert repeated.exit_code == 0, repeated.output
    delete_plan = runner.invoke(
        app,
        ["storage", "service-account", "delete", "--project", "prod", "--dry-run"],
    )
    assert delete_plan.exit_code == 0, delete_plan.output
    assert "NPA creation provenance" in delete_plan.output


def test_unresolved_marker_blocks_project_forgetting_until_verified_absence(
    monkeypatch, tmp_path
) -> None:
    _seed_owned_account(monkeypatch, tmp_path, managed_by="")
    from npa.clients import nebius as nebius_module

    partial = runner.invoke(
        app,
        ["storage", "service-account", "delete", "--project", "prod", "--dry-run"],
    )
    assert partial.exit_code == 2
    blocked = runner.invoke(app, ["configure", "--forget-project", "prod"])
    assert blocked.exit_code == 2
    assert "unresolved storage IAM" in blocked.output

    monkeypatch.setattr(
        nebius_module, "get_service_account_identity", lambda *_args, **_kwargs: None
    )
    absent = runner.invoke(
        app,
        ["storage", "service-account", "delete", "--project", "prod", "--dry-run"],
    )
    assert absent.exit_code == 0, absent.output
    forgotten = runner.invoke(app, ["configure", "--forget-project", "prod"])
    assert forgotten.exit_code == 0, forgotten.output
    assert "Removed project 'prod'" in forgotten.output
