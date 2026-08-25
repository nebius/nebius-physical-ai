"""Safe teardown for the storage service account NPA owns."""

from __future__ import annotations

import json
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
    account_name: str = "lerobot-training",
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
            "service_account_name": account_name,
            "service_account_project_id": "project-a",
            "service_account_managed_by": managed_by,
        },
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
            name=account_name,
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


def _seed_retained_storage_account(
    monkeypatch,
    tmp_path: Path,
    *,
    current_key: str = "accesskey-run",
    created_keys: tuple[str, ...] = (),
    with_residue: bool = True,
) -> Path:
    """Seed project storage on a pre-existing account plus a separate agent ID."""

    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module
    from npa.clients import nebius as nebius_module

    project: dict[str, object] = {
        "project_id": "project-a",
        "tenant_id": "tenant-a",
    }
    if with_residue:
        project["storage_iam_verification_required"] = {
            "project_id": "project-a",
            "service_account_id": "serviceaccount-storage",
            "candidate_service_account_ids": ["serviceaccount-storage"],
            "access_key_ids": [current_key] if current_key else [],
            "created_access_key_ids": list(created_keys),
            "bucket_cleanup_state": "complete",
            "bucket_name": "gone",
            "ownership": "unverified",
            "ownership_state": "pending-verification",
        }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"default_project": "prod", "projects": {"prod": project}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)

    storage = (
        {
            "bucket": "s3://gone/",
            "aws_access_key_id": current_key,
            "aws_secret_access_key": "secret",
        }
        if current_key
        else {}
    )
    credentials_path = tmp_path / "credentials.yaml"
    credentials_path.write_text(
        yaml.safe_dump(
            {
                "nebius": {"service_account_id": "serviceaccount-agent"},
                "project_credentials": {
                    "schema_version": "npa.project-credentials.v2",
                    "current_project_id": "project-a",
                    "projects": {
                        "project-a": {
                            "project_id": "project-a",
                            "storage_selected": bool(storage),
                            "storage": storage,
                            "storage_iam": {
                                "active_service_account_id": "serviceaccount-storage",
                                "generations": [
                                    {
                                        "service_account_id": "serviceaccount-storage",
                                        "service_account_project_id": "project-a",
                                        "access_key_ids": list(created_keys),
                                        "binding": {"state": "existing"},
                                    }
                                ],
                            },
                            "nebius": {
                                "service_account_id": "serviceaccount-agent",
                                "service_account_project_id": "project-a",
                            },
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", credentials_path)
    monkeypatch.setenv("NPA_TEARDOWN_RECEIPT_DIR", str(tmp_path / "receipts"))
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
        lambda *_args, **_kwargs: "serviceaccount-storage",
    )
    return credentials_path


def test_exact_create_provenance_supports_custom_run_scoped_storage_account_name(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.cli import storage as storage_module

    name = "npa-pr218-run-scoped-sa"
    _seed_owned_account(monkeypatch, tmp_path, account_name=name)
    record, _note = storage_module._storage_service_account_record(
        account_id="serviceaccount-storage", project_id="project-a"
    )
    assert record is not None and record.name == name
    context = storage_module._resolve_storage_iam_context(
        project="prod", service_account_id="serviceaccount-storage"
    )
    observation = storage_module._observe_storage_iam(context)
    assert observation.outcome == "present"
    assert observation.account_name == name


def test_project_store_binding_ownership_beats_lossy_compatibility_view(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.cli import storage as storage_module
    from npa.clients import credentials as credentials_module

    path = tmp_path / "credentials.yaml"
    generation = {
        "service_account_id": "serviceaccount-storage",
        "service_account_name": "lerobot-training",
        "service_account_project_id": "project-a",
        "service_account_managed_by": "npa",
        "binding": {
            "group_id": "group-storage",
            "access_permit_id": "permit-storage",
            "group_managed_by": "npa",
            "access_permit_managed_by": "npa",
        },
    }
    path.write_text(
        yaml.safe_dump(
            {
                # Compatibility views intentionally omit narrow-binding
                # provenance and must never shadow the authoritative record.
                "storage_iam": {
                    key: generation[key]
                    for key in (
                        "service_account_id",
                        "service_account_name",
                        "service_account_project_id",
                        "service_account_managed_by",
                    )
                },
                "project_credentials": {
                    "schema_version": "npa.project-credentials.v2",
                    "projects": {
                        "project-a": {"storage_iam": {"generations": [generation]}}
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", path)

    record, note = storage_module._storage_service_account_record(
        account_id="serviceaccount-storage", project_id="project-a"
    )

    assert note == ""
    assert record is not None
    assert record.binding_managed_by_npa is True
    assert (record.group_id, record.access_permit_id) == (
        "group-storage",
        "permit-storage",
    )


def test_untrusted_storage_identity_never_falls_back_across_project_store(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.cli import storage as storage_module
    from npa.clients import credentials as credentials_module

    path = tmp_path / "credentials.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "storage_iam": {
                    "service_account_id": "serviceaccount-b",
                    "service_account_project_id": "project-b",
                },
                "project_credentials": {
                    "schema_version": "npa.project-credentials.v2",
                    "current_project_id": "project-b",
                    "projects": {
                        "project-a": {
                            "storage_iam": {
                                "active_service_account_id": "serviceaccount-a",
                                "generations": [
                                    {"service_account_id": "serviceaccount-a"}
                                ],
                            }
                        },
                        "project-b": {
                            "storage_iam": {
                                "active_service_account_id": "serviceaccount-b",
                                "generations": [
                                    {"service_account_id": "serviceaccount-b"}
                                ],
                            }
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", path)

    assert storage_module._untrusted_storage_account_ids("project-a") == {
        "serviceaccount-a"
    }
    assert storage_module._untrusted_storage_account_ids("project-missing") == set()


def test_bucket_tombstone_prefers_project_storage_identity_over_agent_generic(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.cli import storage as storage_module
    from npa.clients import config as config_module

    _seed_retained_storage_account(monkeypatch, tmp_path, with_residue=False)

    alias, marker = storage_module._begin_bucket_iam_cleanup_tombstone(
        "prod", "project-a", "gone"
    )

    assert alias == "prod"
    assert marker["service_account_id"] == "serviceaccount-storage"
    assert marker["candidate_service_account_ids"] == ["serviceaccount-storage"]
    assert marker["access_key_ids"] == ["accesskey-run"]
    assert marker.get("created_access_key_ids", []) == []
    saved = yaml.safe_load(config_module.CONFIG_PATH.read_text(encoding="utf-8"))
    assert saved["projects"]["prod"]["storage_iam_verification_required"] == marker


def test_retained_storage_account_with_absent_run_access_is_terminal(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.cli import storage as storage_module
    from npa.clients import config as config_module
    from npa.clients import nebius as nebius_module

    credentials_path = _seed_retained_storage_account(monkeypatch, tmp_path)
    storage_module._prune_storage_credentials("gone")
    monkeypatch.setattr(
        nebius_module,
        "list_access_keys_for_service_account",
        lambda *_args, **_kwargs: [],
    )
    deleted_accounts: list[str] = []
    monkeypatch.setattr(
        nebius_module,
        "delete_service_account",
        lambda account_id, **_kwargs: deleted_accounts.append(account_id),
    )

    dry_run = runner.invoke(
        app,
        [
            "storage",
            "service-account",
            "delete",
            "--project",
            "prod",
            "--dry-run",
            "--json",
        ],
    )
    assert dry_run.exit_code == 0, dry_run.output
    assert json.loads(dry_run.output)["result"] == "access_state_already_absent"

    result = runner.invoke(
        app,
        [
            "storage",
            "service-account",
            "delete",
            "--project",
            "prod",
            "--yes",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["result"] == "access_state_resolved"
    assert payload["service_account_preserved"] is True
    assert deleted_accounts == []
    config = yaml.safe_load(config_module.CONFIG_PATH.read_text(encoding="utf-8"))
    assert "storage_iam_verification_required" not in config["projects"]["prod"]
    credentials = yaml.safe_load(credentials_path.read_text(encoding="utf-8"))
    project_record = credentials["project_credentials"]["projects"]["project-a"]
    assert "storage" not in project_record
    assert "storage_iam" not in project_record
    assert project_record["nebius"]["service_account_id"] == "serviceaccount-agent"


def test_retained_storage_account_deletes_only_created_access_key(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.cli import storage as storage_module
    from npa.clients import nebius as nebius_module

    _seed_retained_storage_account(
        monkeypatch,
        tmp_path,
        created_keys=("accesskey-run",),
    )
    storage_module._prune_storage_credentials("gone")
    live_keys = {"accesskey-run"}

    def list_keys(_project_id, service_account_id, **_kwargs):
        return [
            {
                "id": key_id,
                "name": "lerobot-access-key",
                "service_account_id": service_account_id,
            }
            for key_id in sorted(live_keys)
        ]

    monkeypatch.setattr(
        nebius_module, "list_access_keys_for_service_account", list_keys
    )
    deleted_accounts: list[str] = []
    monkeypatch.setattr(
        nebius_module,
        "delete_access_key",
        lambda key_id, **_kwargs: live_keys.discard(key_id),
    )
    monkeypatch.setattr(
        nebius_module,
        "delete_service_account",
        lambda account_id, **_kwargs: deleted_accounts.append(account_id),
    )

    dry_run = runner.invoke(
        app,
        [
            "storage",
            "service-account",
            "delete",
            "--project",
            "prod",
            "--dry-run",
            "--json",
        ],
    )
    assert dry_run.exit_code == 0, dry_run.output
    assert json.loads(dry_run.output)["result"] == "access_cleanup_planned"
    assert live_keys == {"accesskey-run"}

    result = runner.invoke(
        app,
        [
            "storage",
            "service-account",
            "delete",
            "--project",
            "prod",
            "--yes",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["result"] == "access_state_resolved"
    assert live_keys == set()
    assert deleted_accounts == []


def _stub_iam(monkeypatch) -> list[str]:
    from npa.clients import nebius as nebius_module

    live_keys = {"accesskey-storage"}
    deleted_accounts: set[str] = set()
    read_identity = nebius_module.get_service_account_identity
    monkeypatch.setattr(
        nebius_module,
        "list_access_keys_for_service_account",
        lambda project_id, sa_id, **kwargs: [
            {
                "id": key_id,
                "name": "lerobot-access-key",
                "state": "ACTIVE",
                "service_account_id": sa_id,
            }
            for key_id in sorted(live_keys)
        ],
    )
    deleted: list[str] = []

    def delete_key(key_id, **_kwargs):  # noqa: ANN001, ANN202
        deleted.append(key_id)
        live_keys.discard(key_id)

    def delete_account(account_id, **_kwargs):  # noqa: ANN001, ANN202
        deleted.append(account_id)
        deleted_accounts.add(account_id)

    monkeypatch.setattr(
        nebius_module,
        "delete_access_key",
        delete_key,
    )
    monkeypatch.setattr(
        nebius_module,
        "delete_service_account",
        delete_account,
    )
    monkeypatch.setattr(
        nebius_module,
        "get_service_account_identity",
        lambda account_id, **_kwargs: (
            read_identity(account_id, **_kwargs)
            if account_id not in deleted_accounts
            else None
        ),
    )
    return deleted


def _seed_multi_project_owned_accounts(monkeypatch, tmp_path: Path) -> Path:
    """Add unrelated NPA-owned IAM evidence without changing the target default."""

    from npa.clients import config as config_module

    credentials_path = _seed_owned_account(monkeypatch, tmp_path)
    config = yaml.safe_load(config_module.CONFIG_PATH.read_text(encoding="utf-8"))
    config["projects"]["other"] = {
        "project_id": "project-b",
        "tenant_id": "tenant-b",
    }
    config_module.CONFIG_PATH.write_text(yaml.safe_dump(config), encoding="utf-8")
    credentials = yaml.safe_load(credentials_path.read_text(encoding="utf-8"))
    credentials["storage_setup"] = {
        "version": 1,
        "projects": {
            "project-b": {
                "status": "partial",
                "phase": "rollback_incomplete",
                "resources": {
                    "service_account": {
                        "id": "serviceaccount-unrelated",
                        "name": "lerobot-training",
                        "created_by": "npa",
                        "project_id": "project-b",
                        "attempt_id": "attempt-b",
                    },
                    "access_keys": {
                        "accesskey-unrelated": {
                            "service_account_id": "serviceaccount-unrelated"
                        }
                    },
                },
            }
        },
    }
    credentials_path.write_text(yaml.safe_dump(credentials), encoding="utf-8")
    return credentials_path


def test_explicit_storage_id_selects_exact_owned_project_and_preserves_unrelated(
    monkeypatch, tmp_path: Path
) -> None:
    credentials_path = _seed_multi_project_owned_accounts(monkeypatch, tmp_path)
    deleted = _stub_iam(monkeypatch)

    result = runner.invoke(
        app,
        [
            "storage",
            "service-account",
            "delete",
            "--project",
            "prod",
            "--id",
            "serviceaccount-storage",
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.output
    assert deleted == ["accesskey-storage", "serviceaccount-storage"]
    remaining = yaml.safe_load(credentials_path.read_text(encoding="utf-8"))
    unrelated = remaining["storage_setup"]["projects"]["project-b"]
    assert unrelated["resources"]["service_account"]["id"] == (
        "serviceaccount-unrelated"
    )
    assert unrelated["resources"]["access_keys"] == {
        "accesskey-unrelated": {"service_account_id": "serviceaccount-unrelated"}
    }


def test_storage_id_without_exact_ownership_proof_fails_closed(
    monkeypatch, tmp_path: Path
) -> None:
    _seed_multi_project_owned_accounts(monkeypatch, tmp_path)
    deleted = _stub_iam(monkeypatch)

    result = runner.invoke(
        app,
        [
            "storage",
            "service-account",
            "delete",
            "--project",
            "prod",
            "--id",
            "serviceaccount-missing",
            "--yes",
        ],
    )

    assert result.exit_code == 2
    assert (
        "No trustworthy NPA storage IAM ownership record proves exact" in result.output
    )
    assert deleted == []


def test_storage_id_with_mismatched_ownership_project_fails_closed(
    monkeypatch, tmp_path: Path
) -> None:
    credentials_path = _seed_multi_project_owned_accounts(monkeypatch, tmp_path)
    credentials = yaml.safe_load(credentials_path.read_text(encoding="utf-8"))
    credentials["storage_iam"]["service_account_project_id"] = "project-b"
    credentials_path.write_text(yaml.safe_dump(credentials), encoding="utf-8")
    deleted = _stub_iam(monkeypatch)

    result = runner.invoke(
        app,
        [
            "storage",
            "service-account",
            "delete",
            "--project",
            "prod",
            "--id",
            "serviceaccount-storage",
            "--yes",
        ],
    )

    assert result.exit_code == 2
    assert "not selected project project-a" in result.output
    assert deleted == []


def test_storage_id_with_mismatched_provider_parent_fails_closed(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.clients import nebius as nebius_module

    _seed_multi_project_owned_accounts(monkeypatch, tmp_path)
    monkeypatch.setattr(
        nebius_module,
        "get_service_account_identity",
        lambda account_id, **_kwargs: nebius_module.ServiceAccountIdentity(
            account_id=account_id,
            name="lerobot-training",
            project_id="project-b",
            tenant_id="tenant-b",
            profile="",
        ),
    )
    deleted = _stub_iam(monkeypatch)

    result = runner.invoke(
        app,
        [
            "storage",
            "service-account",
            "delete",
            "--project",
            "prod",
            "--id",
            "serviceaccount-storage",
            "--yes",
        ],
    )

    assert result.exit_code == 2
    assert "parent project" in result.output
    assert "parent tenant" in result.output
    assert deleted == []


def test_storage_access_key_relationship_mismatch_fails_before_delete(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.clients import nebius as nebius_module

    _seed_multi_project_owned_accounts(monkeypatch, tmp_path)
    monkeypatch.setattr(
        nebius_module,
        "list_access_keys_for_service_account",
        lambda *_args, **_kwargs: [
            {
                "id": "accesskey-wrong-owner",
                "service_account_id": "serviceaccount-unrelated",
            }
        ],
    )
    deleted: list[str] = []
    monkeypatch.setattr(
        nebius_module,
        "delete_access_key",
        lambda key_id, **_kwargs: deleted.append(key_id),
    )
    monkeypatch.setattr(
        nebius_module,
        "delete_service_account",
        lambda account_id, **_kwargs: deleted.append(account_id),
    )
    result = runner.invoke(
        app,
        [
            "storage",
            "service-account",
            "delete",
            "--project",
            "prod",
            "--id",
            "serviceaccount-storage",
            "--yes",
        ],
    )

    assert result.exit_code == 2
    assert "did not prove the selected service-account relationship" in result.output
    assert deleted == []


def test_storage_without_explicit_id_still_refuses_multi_project_ambiguity(
    monkeypatch, tmp_path: Path
) -> None:
    _seed_multi_project_owned_accounts(monkeypatch, tmp_path)
    deleted = _stub_iam(monkeypatch)

    result = runner.invoke(
        app,
        ["storage", "service-account", "delete", "--project", "prod", "--yes"],
    )

    assert result.exit_code == 2
    assert "Conflicting NPA storage IAM ownership records" in result.output
    assert deleted == []


def test_storage_service_account_dry_run_names_exact_owned_resources(
    monkeypatch, tmp_path
) -> None:
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
    assert (
        "Verified deletion: NPA-owned service account lerobot-training" in result.output
    )
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
        lambda project_id, name: {"metadata": {"id": "bucket-storage", "name": name}},
    )
    monkeypatch.setattr(
        nebius_module, "delete_bucket", lambda bucket_id, *, ttl="": None
    )
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


def test_storage_service_account_refuses_unowned_legacy_identity(
    monkeypatch, tmp_path
) -> None:
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
        lambda project_id, name: {"metadata": {"id": "bucket-storage", "name": name}},
    )
    monkeypatch.setattr(
        nebius_module, "delete_bucket", lambda bucket_id, *, ttl="": None
    )

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


def test_storage_service_account_requires_yes_in_noninteractive_mode(
    monkeypatch, tmp_path
) -> None:
    _seed_owned_account(monkeypatch, tmp_path)
    deleted = _stub_iam(monkeypatch)

    result = runner.invoke(
        app,
        ["storage", "service-account", "delete", "--project", "prod"],
    )

    assert result.exit_code != 0
    assert "Re-run with --yes" in result.output
    assert deleted == []


def test_storage_service_account_rejects_mismatched_project_marker(
    monkeypatch, tmp_path
) -> None:
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
            {
                "id": "accesskey-storage",
                "name": "lerobot-access-key",
                "state": "ACTIVE",
                "service_account_id": sa_id,
            }
        ],
    )
    monkeypatch.setattr(
        nebius_module,
        "delete_access_key",
        lambda key_id, **_kwargs: (_ for _ in ()).throw(
            nebius_module.NebiusError("key busy")
        ),
    )
    monkeypatch.setattr(
        nebius_module,
        "delete_service_account",
        lambda sa_id, **_kwargs: (_ for _ in ()).throw(
            nebius_module.NebiusError("account busy")
        ),
    )

    result = runner.invoke(
        app,
        ["storage", "service-account", "delete", "--project", "prod", "--yes"],
    )

    assert result.exit_code == 1
    assert "could not delete access key accesskey-storage" in result.output
    assert "service-account deletion was blocked" not in result.output
    assert credentials_path.exists()
    assert (
        yaml.safe_load(credentials_path.read_text())["storage_iam"][
            "service_account_managed_by"
        ]
        == "npa"
    )


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
            nebius_module.NebiusError(
                "NotFound: service account is absent",
                provider_status=nebius_module.ProviderStatus.NOT_FOUND,
            )
        ),
    )
    monkeypatch.setattr(
        nebius_module,
        "delete_service_account",
        lambda account_id, **_kwargs: deleted.append(account_id),
    )

    result = runner.invoke(
        app,
        ["storage", "service-account", "delete", "--project", "prod", "--yes"],
    )

    assert result.exit_code == 0, result.output
    assert "already absent" in result.output
    assert deleted == []
    assert not credentials_path.exists()


def test_storage_service_account_list_not_found_requires_confirmation_to_prune(
    monkeypatch, tmp_path
) -> None:
    from npa.clients import nebius as nebius_module

    credentials_path = _seed_owned_account(monkeypatch, tmp_path)
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
            nebius_module.NebiusError(
                "NotFound: service account is absent",
                provider_status=nebius_module.ProviderStatus.NOT_FOUND,
            )
        ),
    )

    result = runner.invoke(
        app,
        ["storage", "service-account", "delete", "--project", "prod", "--json"],
    )

    assert result.exit_code == 1
    assert json.loads(result.output)["result"] == "confirmation_required"
    assert credentials_path.exists()


def test_storage_service_account_access_key_not_found_is_idempotent(
    monkeypatch, tmp_path
) -> None:
    from npa.clients import nebius as nebius_module

    credentials_path = _seed_owned_account(monkeypatch, tmp_path)
    inventory_calls = 0

    def list_keys(_project, account, **kwargs):  # noqa: ANN001, ANN202
        nonlocal inventory_calls
        inventory_calls += 1
        if inventory_calls == 1:
            return [{"id": "accesskey-storage", "service_account_id": account}]
        return []

    monkeypatch.setattr(
        nebius_module, "list_access_keys_for_service_account", list_keys
    )
    monkeypatch.setattr(
        nebius_module,
        "delete_access_key",
        lambda key_id, **_kwargs: (_ for _ in ()).throw(
            nebius_module.NebiusError(
                "NotFound: access key is absent",
                provider_status=nebius_module.ProviderStatus.NOT_FOUND,
            )
        ),
    )
    deleted: list[str] = []
    monkeypatch.setattr(
        nebius_module,
        "delete_service_account",
        lambda account_id, **_kwargs: deleted.append(account_id),
    )
    monkeypatch.setattr(
        nebius_module,
        "get_service_account_identity",
        lambda account_id, **_kwargs: (
            None
            if account_id in deleted
            else nebius_module.ServiceAccountIdentity(
                account_id=account_id,
                name="lerobot-training",
                project_id="project-a",
                tenant_id="tenant-a",
                profile="",
            )
        ),
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
    monkeypatch.setattr(
        nebius_module,
        "delete_access_key",
        lambda key_id, **_kwargs: deleted.append(key_id),
    )
    monkeypatch.setattr(
        nebius_module,
        "delete_service_account",
        lambda sa_id, **_kwargs: deleted.append(sa_id),
    )

    result = runner.invoke(
        app,
        ["storage", "service-account", "delete", "--project", "prod", "--yes"],
    )

    assert result.exit_code != 0
    assert (
        "provider/auth verification failed while inspecting access keys"
        in result.output
    )
    assert "nothing was deleted" in result.output
    assert deleted == []
    assert credentials_path.exists()


def test_storage_service_account_already_absent_is_idempotent(
    monkeypatch, tmp_path
) -> None:
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
        lambda sa_id, **_kwargs: (_ for _ in ()).throw(
            nebius_module.NebiusError(
                "NotFound",
                provider_status=nebius_module.ProviderStatus.NOT_FOUND,
            )
        ),
    )
    identity_calls = 0

    def read_identity(account_id, **_kwargs):  # noqa: ANN001, ANN202
        nonlocal identity_calls
        identity_calls += 1
        if identity_calls == 1:
            return nebius_module.ServiceAccountIdentity(
                account_id=account_id,
                name="lerobot-training",
                project_id="project-a",
                tenant_id="tenant-a",
                profile="",
            )
        return None

    monkeypatch.setattr(
        nebius_module, "get_service_account_identity", read_identity
    )

    result = runner.invoke(
        app,
        ["storage", "service-account", "delete", "--project", "prod", "--yes"],
    )

    assert result.exit_code == 0, result.output
    assert "is already absent" in result.output
    assert not credentials_path.exists()


def test_absent_service_account_requires_confirmation_before_local_record_prune(
    monkeypatch, tmp_path
) -> None:
    from npa.clients import nebius as nebius_module

    credentials_path = _seed_owned_account(monkeypatch, tmp_path)
    monkeypatch.setattr(
        nebius_module,
        "get_service_account_identity",
        lambda *_args, **_kwargs: None,
    )

    result = runner.invoke(
        app,
        ["storage", "service-account", "delete", "--project", "prod", "--json"],
    )

    assert result.exit_code == 1
    assert json.loads(result.output)["result"] == "confirmation_required"
    assert credentials_path.exists()


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
    credentials_path = _seed_owned_account(monkeypatch, tmp_path, managed_by="")

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


def test_missing_ownership_provider_failure_is_partial(monkeypatch, tmp_path) -> None:
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


def test_bucket_first_interlock_fails_closed_when_credentials_are_unreadable(
    monkeypatch, tmp_path
) -> None:
    _seed_owned_account(monkeypatch, tmp_path)
    deleted = _stub_iam(monkeypatch)
    from npa.clients import credentials as credentials_module

    monkeypatch.setattr(
        credentials_module,
        "load_credentials",
        lambda **kwargs: (_ for _ in ()).throw(
            credentials_module.CredentialStoreError(
                "malformed", credentials_module.CREDENTIALS_PATH, "invalid YAML"
            )
        ),
    )

    result = runner.invoke(
        app,
        ["storage", "service-account", "delete", "--project", "prod", "--yes"],
    )

    assert result.exit_code == 2
    assert "delete-bucket-first interlock" in result.output
    assert "Nothing was deleted" in result.output
    assert deleted == []


def test_bucket_first_interlock_unreadable_json_is_typed_and_machine_readable(
    monkeypatch, tmp_path
) -> None:
    _seed_owned_account(monkeypatch, tmp_path)
    deleted = _stub_iam(monkeypatch)
    from npa.clients import credentials as credentials_module

    monkeypatch.setattr(
        credentials_module,
        "load_credentials",
        lambda **kwargs: (_ for _ in ()).throw(
            credentials_module.CredentialStoreError(
                "unreadable", credentials_module.CREDENTIALS_PATH, "permission denied"
            )
        ),
    )

    result = runner.invoke(
        app,
        [
            "storage",
            "service-account",
            "delete",
            "--project",
            "prod",
            "--yes",
            "--json",
        ],
    )

    assert result.exit_code == 2
    payload = json.loads(result.output)
    assert payload["result"] == "bucket_interlock_unresolved"
    assert payload["verification_unresolved"] is True
    assert payload["mutated"] is False
    assert deleted == []


def test_stale_residue_id_does_not_override_exact_owned_record(
    monkeypatch, tmp_path
) -> None:
    _seed_owned_account(monkeypatch, tmp_path)
    _stub_iam(monkeypatch)
    from npa.clients import config as config_module

    config = yaml.safe_load(config_module.CONFIG_PATH.read_text())
    config["projects"]["prod"]["storage_iam_verification_required"] = {
        "schema_version": "npa.storage-iam-residue.v2",
        "status": "verification_failed",
        "service_account_id": "serviceaccount-stale",
        "candidate_service_account_ids": ["serviceaccount-stale"],
    }
    config_module.CONFIG_PATH.write_text(yaml.safe_dump(config))

    result = runner.invoke(
        app,
        ["storage", "service-account", "delete", "--project", "prod", "--dry-run"],
    )

    assert result.exit_code == 0, result.output
    assert "serviceaccount-storage" in result.output
    assert "Conflicting exact" not in result.output
    assert (
        "storage_iam_verification_required"
        in yaml.safe_load(config_module.CONFIG_PATH.read_text())["projects"]["prod"]
    )

    deletion = runner.invoke(
        app,
        ["storage", "service-account", "delete", "--project", "prod", "--yes"],
    )

    assert deletion.exit_code == 0, deletion.output
    assert (
        "storage_iam_verification_required"
        not in yaml.safe_load(config_module.CONFIG_PATH.read_text())["projects"]["prod"]
    )


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
        nebius_module,
        "delete_service_account",
        lambda account_id, **_kwargs: deleted.append(account_id),
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
    assert (
        yaml.safe_load(credentials_path.read_text())["storage_iam"][
            "service_account_managed_by"
        ]
        == "npa"
    )

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
    assert (
        yaml.safe_load(credentials_path.read_text())["storage_iam"][
            "service_account_managed_by"
        ]
        == ""
    )
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


def test_exact_project_alias_free_absence_persists_global_receipt(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.cli import storage as storage_module
    from npa.teardown_receipts import list_teardown_receipts

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("NPA_TEARDOWN_RECEIPT_DIR", str(tmp_path / "receipts"))
    monkeypatch.setattr(
        storage_module,
        "_storage_service_account_record",
        lambda **_kwargs: (None, "no ownership record"),
    )
    monkeypatch.setattr(
        storage_module, "_untrusted_storage_account_ids", lambda _project="": set()
    )
    monkeypatch.setattr(
        "npa.clients.nebius.get_service_account_identity",
        lambda *_args, **_kwargs: None,
    )

    context = storage_module._resolve_storage_iam_context(
        project_id="project-exact",
        tenant_id="tenant-exact",
        service_account_id="service-account-exact",
    )
    observation = storage_module._observe_storage_iam(context)
    storage_module._persist_storage_iam_observation(observation)

    assert context.alias == ""
    assert context.identity_source == "explicit_exact_arguments"
    assert observation.outcome == "verified_absent"
    [receipt] = list_teardown_receipts()
    assert receipt["project_id"] == "project-exact"
    assert (
        receipt["identity"]["storage_iam"]["service_account_id"]
        == "service-account-exact"
    )
    assert receipt["events"][-1]["terminal_state"] == "verified_absent"


def test_receipt_context_carries_exact_account_without_configured_alias(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.cli.storage import _resolve_storage_iam_context
    from npa.teardown_receipts import record_teardown_event

    monkeypatch.setenv("NPA_TEARDOWN_RECEIPT_DIR", str(tmp_path / "receipts"))
    path = record_teardown_event(
        phase="storage_iam",
        resource="service-account-1",
        terminal_state="operator_action_remains",
        project_id="project-1",
        identity={
            "project_id": "project-1",
            "tenant_id": "tenant-1",
            "service_account_id": "service-account-1",
        },
    )

    context = _resolve_storage_iam_context(receipt=path.stem)

    assert context.alias == ""
    assert context.project_id == "project-1"
    assert context.tenant_id == "tenant-1"
    assert context.account_id == "service-account-1"
    assert context.receipt_id == path.stem


def test_receipt_only_storage_iam_verifies_parent_and_access_key_after_forget(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module
    from npa.clients import nebius as nebius_module
    from npa.teardown_receipts import record_teardown_event

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("NPA_TEARDOWN_RECEIPT_DIR", str(tmp_path / "receipts"))
    config_path = tmp_path / "config.yaml"
    config_path.write_text("projects: {}\n", encoding="utf-8")
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)
    monkeypatch.setattr(
        credentials_module, "CREDENTIALS_PATH", tmp_path / "missing-credentials.yaml"
    )
    path = record_teardown_event(
        phase="bucket",
        resource="bucket-a",
        terminal_state="verified_deleted",
        project_alias="forgotten",
        project_id="project-a",
        identity={"project_id": "project-a"},
        action={"kind": "bucket_delete", "scheduled": False},
        verification={"bucket_absent": True},
    )
    record_teardown_event(
        phase="storage_iam",
        resource="serviceaccount-storage",
        terminal_state="operator_action_remains",
        project_alias="forgotten",
        project_id="project-a",
        identity={
            "project_id": "project-a",
            "tenant_id": "tenant-a",
            "service_account_id": "serviceaccount-storage",
            "service_account_name": "lerobot-training",
            "ownership": "npa",
            "iam_key_ids": ["accesskey-storage"],
        },
    )
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
        "list_access_keys_for_service_account",
        lambda _project, account_id, **_kwargs: [
            {
                "id": "accesskey-storage",
                "name": "lerobot-access-key",
                "state": "ACTIVE",
                "service_account_id": account_id,
            }
        ],
    )

    result = runner.invoke(
        app,
        [
            "storage",
            "service-account",
            "delete",
            "--receipt",
            path.stem,
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["result"] == "delete_planned"
    assert payload["service_account_id"] == "serviceaccount-storage"
    assert payload["access_key_ids"] == ["accesskey-storage"]


def test_receipt_reuses_exact_durable_storage_iam_absence_without_credentials(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.cli import storage as storage_module
    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module
    from npa.teardown_receipts import record_teardown_event

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("NPA_TEARDOWN_RECEIPT_DIR", str(tmp_path / "receipts"))
    config_path = tmp_path / "config.yaml"
    config_path.write_text("projects: {}\n", encoding="utf-8")
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)
    monkeypatch.setattr(
        credentials_module, "CREDENTIALS_PATH", tmp_path / "missing-credentials.yaml"
    )
    receipt_path = record_teardown_event(
        phase="storage_iam",
        resource="serviceaccount-storage",
        terminal_state="verified_absent",
        project_id="project-a",
        identity={
            "project_id": "project-a",
            "tenant_id": "tenant-a",
            "service_account_id": "serviceaccount-storage",
            "service_account_name": "lerobot-training",
            "ownership": "npa",
            "iam_key_ids": ["accesskey-storage"],
        },
        action={"kind": "exact_provider_check"},
        verification={
            "provider_outcome": "verified_absent",
            "exact_service_account_absent": True,
        },
    )
    monkeypatch.setattr(
        "npa.clients.nebius.get_service_account_identity",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("provider must not run")
        ),
    )

    context = storage_module._resolve_storage_iam_context(receipt=receipt_path.stem)
    observation = storage_module._observe_storage_iam(context)

    assert observation.verified_absent
    assert observation.account_id == "serviceaccount-storage"


def test_newer_conflicting_storage_iam_receipt_blocks_old_absence(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.cli import storage as storage_module
    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module
    from npa.clients.nebius import NebiusError
    from npa.teardown_receipts import record_teardown_event

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("NPA_TEARDOWN_RECEIPT_DIR", str(tmp_path / "receipts"))
    config_path = tmp_path / "config.yaml"
    config_path.write_text("projects: {}\n", encoding="utf-8")
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)
    monkeypatch.setattr(
        credentials_module, "CREDENTIALS_PATH", tmp_path / "missing-credentials.yaml"
    )
    identity = {
        "project_id": "project-a",
        "tenant_id": "tenant-a",
        "service_account_id": "serviceaccount-storage",
        "service_account_name": "lerobot-training",
        "ownership": "npa",
    }
    receipt_path = record_teardown_event(
        phase="storage_iam",
        resource="serviceaccount-storage",
        terminal_state="verified_absent",
        project_id="project-a",
        identity=identity,
        action={"kind": "exact_provider_check"},
        verification={"provider_outcome": "verified_absent"},
    )
    record_teardown_event(
        phase="storage_iam",
        resource="serviceaccount-storage",
        terminal_state="verification_failed",
        project_id="project-a",
        identity=identity,
        action={"kind": "exact_provider_check"},
        verification={"provider_outcome": "verification_failed"},
        errors=["provider inventory unavailable"],
    )
    monkeypatch.setattr(
        "npa.clients.nebius.get_service_account_identity",
        lambda *_a, **_k: (_ for _ in ()).throw(NebiusError("authentication failed")),
    )

    context = storage_module._resolve_storage_iam_context(receipt=receipt_path.stem)
    observation = storage_module._observe_storage_iam(context)

    assert observation.outcome == "verification_failed"
    assert not observation.verified_absent


def test_newer_bucket_failure_blocks_old_receipt_cleanup_authority(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.cli.storage import _receipt_proves_bucket_cleanup
    from npa.teardown_receipts import record_teardown_event

    monkeypatch.setenv("NPA_TEARDOWN_RECEIPT_DIR", str(tmp_path / "receipts"))
    identity = {"project_id": "project-a", "bucket_name": "bucket-a"}
    receipt_path = record_teardown_event(
        phase="bucket",
        resource="bucket-a",
        terminal_state="verified_absent",
        project_id="project-a",
        identity=identity,
        action={"kind": "bucket_delete"},
        verification={"bucket_absent": True},
    )
    assert _receipt_proves_bucket_cleanup(receipt_path.stem, "project-a")

    record_teardown_event(
        phase="bucket",
        resource="bucket-a",
        terminal_state="verification_failed",
        project_id="project-a",
        identity=identity,
        action={"kind": "bucket_delete"},
        verification={"bucket_absent": False},
        errors=["provider verification unavailable"],
    )

    assert not _receipt_proves_bucket_cleanup(receipt_path.stem, "project-a")
