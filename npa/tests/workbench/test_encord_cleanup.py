"""`encord cleanup`: tear down run-scoped Encord state by title prefix."""

from __future__ import annotations

import pytest

from encord_fakes import FakeDataset, FakeFolder, FakeUserClient, fake_uuid, folder_item
from npa.workbench.encord.cleanup import run_cleanup
from npa.workbench.encord.schemas import EncordToolError


def test_run_cleanup_deletes_prefix_scoped_state_and_reports_datasets() -> None:
    folder = FakeFolder(name="npa-e2e-run1")
    folder.folder_items = [folder_item(71, "p/a.mp4")]
    client = FakeUserClient(folders=[folder])
    collection = client.create_collection(
        top_level_folder_uuid=str(folder.uuid), name="npa-e2e-run1"
    )
    preset = client.create_preset(name="npa-curate-run1", filter_preset_json={})
    client.create_preset(name="customer-preset", filter_preset_json={})
    client.datasets[fake_uuid(72)] = FakeDataset(fake_uuid(72), "npa-e2e-run1")
    client.datasets[fake_uuid(73)] = FakeDataset(fake_uuid(73), "customer-data")

    summary = run_cleanup(title_prefix="npa-e2e-", user_client=client)
    assert summary["folders_deleted"] == ["npa-e2e-run1"]
    assert summary["items_deleted"] == 1
    assert folder.deleted is True and folder.deleted_item_uuids == [fake_uuid(71)]
    assert summary["collections_deleted"] == ["npa-e2e-run1"]
    assert client.deleted_collections == [str(collection.uuid)]
    assert summary["datasets_undeletable"] == ["npa-e2e-run1"]
    assert "customer-preset" not in summary["presets_deleted"]

    cleanup_preset_summary = run_cleanup(title_prefix="npa-curate-", user_client=client)
    assert cleanup_preset_summary["presets_deleted"] == ["npa-curate-run1"]
    assert client.deleted_presets == [str(preset.uuid)]


def test_run_cleanup_dry_run_deletes_nothing() -> None:
    folder = FakeFolder(name="npa-e2e-run2")
    client = FakeUserClient(folders=[folder])
    summary = run_cleanup(title_prefix="npa-e2e-", dry_run=True, user_client=client)
    assert summary["folders_deleted"] == ["npa-e2e-run2"]
    assert not getattr(folder, "deleted", False)


def test_run_cleanup_rejects_dangerously_short_prefix() -> None:
    with pytest.raises(EncordToolError, match="at least 4"):
        run_cleanup(title_prefix="np", user_client=FakeUserClient())


