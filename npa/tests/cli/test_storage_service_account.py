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
                "projects": {"prod": {"project_id": "project-a"}},
            }
        )
    )
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)

    credentials_path = tmp_path / "credentials.yaml"
    payload: dict[str, object] = {
        "nebius": {
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
    assert "Deleted NPA-owned service account lerobot-training" in result.output
    assert not credentials_path.exists()


def test_storage_service_account_refuses_unowned_legacy_identity(monkeypatch, tmp_path) -> None:
    credentials_path = _seed_owned_account(monkeypatch, tmp_path, managed_by="")
    deleted = _stub_iam(monkeypatch)

    result = runner.invoke(
        app,
        ["storage", "service-account", "delete", "--project", "prod", "--yes"],
    )

    assert result.exit_code == 0, result.output
    assert "No NPA-owned storage service account" in result.output
    assert "not proof of ownership" in result.output
    assert deleted == []
    assert credentials_path.exists()


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
    data["nebius"]["service_account_project_id"] = "project-other"
    credentials_path.write_text(yaml.safe_dump(data))
    deleted = _stub_iam(monkeypatch)

    result = runner.invoke(
        app,
        ["storage", "service-account", "delete", "--project", "prod", "--yes"],
    )

    assert result.exit_code == 0, result.output
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
    assert yaml.safe_load(credentials_path.read_text())["nebius"][
        "service_account_managed_by"
    ] == "npa"


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
    assert "Could not inspect access keys" in result.output
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
