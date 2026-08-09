from __future__ import annotations

import json

from typer.testing import CliRunner

from npa.cli.main import app
from npa.clients import nebius


runner = CliRunner()


def _args(*extra: str) -> list[str]:
    return [
        "registry",
        "delete",
        "--project",
        "demo",
        "--project-id",
        "project-a",
        "--tenant-id",
        "tenant-a",
        "--id",
        "registry-a",
        "--name",
        "registry-demo",
        *extra,
    ]


def _project(mocker) -> None:  # noqa: ANN001
    mocker.patch(
        "npa.clients.nebius.get_project_identity",
        return_value=nebius.ProjectIdentity(
            "project-a", "demo", "tenant-a", "us-central1", "test"
        ),
    )


def test_registry_delete_requires_npa_project_ownership(mocker) -> None:
    _project(mocker)
    mocker.patch("npa.project_destroy._project_ownership_operation", return_value=None)
    delete = mocker.patch("npa.clients.nebius.delete_registry")

    result = runner.invoke(app, _args("--yes"))

    assert result.exit_code == 1
    assert "durable NPA project-creation proof" in result.output
    delete.assert_not_called()


def test_registry_delete_rejects_identity_mismatch(mocker) -> None:
    _project(mocker)
    mocker.patch(
        "npa.project_destroy._project_ownership_operation",
        return_value=mocker.Mock(operation_id="project-create-a"),
    )
    mocker.patch(
        "npa.clients.nebius.get_registry_identity",
        return_value=nebius.RegistryIdentity(
            "registry-a", "different-name", "project-a", "test"
        ),
    )
    delete = mocker.patch("npa.clients.nebius.delete_registry")

    result = runner.invoke(app, _args("--yes"))

    assert result.exit_code == 1
    assert "does not match" in result.output
    delete.assert_not_called()


def test_registry_delete_cleans_artifacts_and_verifies_absence(mocker) -> None:
    _project(mocker)
    mocker.patch(
        "npa.project_destroy._project_ownership_operation",
        return_value=mocker.Mock(operation_id="project-create-a"),
    )
    get_registry = mocker.patch(
        "npa.clients.nebius.get_registry_identity",
        side_effect=[
            nebius.RegistryIdentity("registry-a", "registry-demo", "project-a", "test"),
            None,
        ],
    )
    inventory = mocker.patch(
        "npa.clients.nebius.list_registry_image_ids",
        side_effect=[("image-root", "image-config"), ()],
    )
    remove_images = mocker.patch(
        "npa.clients.nebius.delete_all_registry_images",
        return_value=("image-root", "image-config"),
    )
    delete = mocker.patch("npa.clients.nebius.delete_registry")
    record = mocker.patch("npa.teardown_receipts.record_teardown_event")

    result = runner.invoke(app, _args("--yes", "--json"))

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["outcome"] == "verified_deleted"
    assert payload["images_removed"] == ["image-root", "image-config"]
    assert inventory.call_count == 2
    remove_images.assert_called_once_with("registry-a", profile=None)
    delete.assert_called_once_with("registry-a", profile=None)
    assert get_registry.call_count == 2
    assert record.call_count == 2


def test_registry_delete_is_idempotent_when_exact_id_is_absent(mocker) -> None:
    _project(mocker)
    mocker.patch(
        "npa.project_destroy._project_ownership_operation",
        return_value=mocker.Mock(operation_id="project-create-a"),
    )
    mocker.patch("npa.clients.nebius.get_registry_identity", return_value=None)
    delete = mocker.patch("npa.clients.nebius.delete_registry")

    result = runner.invoke(app, _args("--yes", "--json"))

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["outcome"] == "already_absent"
    delete.assert_not_called()
