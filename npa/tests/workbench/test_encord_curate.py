"""Headless Encord curation and durable selection evidence."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from npa.workbench.encord.curate import parse_filters, run_curate
from npa.workbench.encord.schemas import (
    EncordToolError,
    OutcomeCounts,
    PushItem,
    PushReceipt,
)

sys.path.insert(0, str(Path(__file__).parents[1]))
from encord_fakes import MemoryArtifactStore  # noqa: E402

URI = "s3://bucket/run/curate/curate_receipt.json"


class Collection:
    uuid = "collection-id"
    name = "keepers"
    top_level_folder_uuid = "folder-id"

    def __init__(self, items=()):
        self.items = list(items)
        self.applied = 0

    def list_items(self, **_):
        return list(self.items)

    def add_preset_items(self, _):
        self.applied += 1
        self.items = [SimpleNamespace(uuid="item-1")]


class Client:
    def __init__(self, *, populated=False, delete_fails=False):
        self.events = []
        self.folder = SimpleNamespace(
            uuid="folder-id",
            name="folder",
            list_items=lambda **_: [
                SimpleNamespace(uuid="item-1"),
                SimpleNamespace(uuid="item-2"),
            ],
        )
        self.collection = (
            Collection([SimpleNamespace(uuid="old")]) if populated else None
        )
        self.delete_fails = delete_fails

    def list_storage_folders(self, **_):
        return [self.folder]

    def list_collections(self, top_level_folder_uuid=None):
        if self.collection and (
            top_level_folder_uuid is None
            or self.collection.top_level_folder_uuid == top_level_folder_uuid
        ):
            return [self.collection]
        return []

    def get_collection(self, _):
        return self.collection

    def create_collection(self, **_):
        self.events.append("create_collection")
        self.collection = Collection()
        return self.collection

    def create_preset(self, **_):
        self.events.append("create_preset")
        return SimpleNamespace(uuid="preset-id")

    def delete_preset(self, _):
        self.events.append("delete_preset")
        if self.delete_fails:
            raise RuntimeError("delete failed")


def run(client=None, store=None, collection="keepers", **kwargs):
    return run_curate(
        folder="folder",
        filters=["width:128:16384"],
        collection=collection,
        output_path=URI,
        user_client=client or Client(),
        artifact_store=store or MemoryArtifactStore(),
        poll_seconds=2,
        **kwargs,
    )


@pytest.mark.parametrize(
    "filter_spec", ["height:nan:5", "width:inf:5", "width:5:4", "unknown:1:2"]
)
def test_invalid_filter_fails_before_any_remote_call(filter_spec):
    with pytest.raises(EncordToolError):
        run_curate(
            folder="folder",
            filters=[filter_spec],
            collection="keepers",
            output_path=URI,
        )


def test_filter_payload_uses_pinned_encord_shape():
    canonical, payload = parse_filters(
        ["width:128:4096,brightness:0.2:0.8,file-size:68:1000"]
    )
    assert canonical == ["width:128:4096", "brightness:0.2:0.8", "file-size:68:1000"]
    assert payload["global_filters"]["filters"] == [
        {
            "include": True,
            "values": [128.0, 4096.0],
            "domain": "data",
            "metric": "metric_width",
            "type": "metric",
        },
        {
            "include": True,
            "values": [0.2, 0.8],
            "domain": "data",
            "metric": "metric_brightness",
            "type": "metric",
        },
        {
            "include": True,
            "values": [68.0, 1000.0],
            "domain": "item",
            "metric": "metric_file_size",
            "type": "metric",
        },
    ]


def test_receipt_precedes_mutation_and_captures_exact_selection(monkeypatch):
    client = Client()
    store = MemoryArtifactStore()
    monkeypatch.setattr("npa.workbench.encord.curate.time.sleep", lambda _: None)
    original_create = client.create_collection

    def checked_create(**kwargs):
        assert store.read_json(URI)["phase"] == "checkpoint"
        return original_create(**kwargs)

    client.create_collection = checked_create
    receipt = run(client, store)
    assert receipt.phase == "final" and receipt.status == "completed"
    assert receipt.selected_item_uuids == ["item-1"]
    assert (receipt.items_total, receipt.items_selected) == (2, 1)
    assert receipt.preset_deleted
    assert client.events == ["create_collection", "create_preset", "delete_preset"]
    assert store.read_json(URI)["selected_item_uuids"] == ["item-1"]


def test_source_push_receipt_limits_selection(monkeypatch):
    client = Client()
    store = MemoryArtifactStore()
    source_uri = "s3://bucket/run/push/push_receipt.json"
    item = PushItem(
        source_uri="s3://bucket/input/item.mp4",
        bucket="bucket",
        object_key="input/item.mp4",
        category="videos",
        submitted_object_url="https://storage.example/bucket/input/item.mp4",
        item_uuid="item-1",
        registration_state="registered",
        identity_state="resolved",
        outcome="successful",
    )
    source = PushReceipt(
        phase="final",
        status="completed",
        revision=1,
        generated_at="2026-09-30T00:00:00+00:00",
        updated_at="2026-09-30T00:00:00+00:00",
        input_uri="s3://bucket/input/",
        encord_domain="https://api.encord.com",
        folder_name="folder",
        folder_uuid="folder-id",
        media_filter="videos-images",
        counts=OutcomeCounts.from_outcomes(["successful"]),
        receipt_uri=source_uri,
        receipt_store_kind="s3",
        items=[item],
    )
    store.create_json(source_uri, source.model_dump(by_alias=True))
    monkeypatch.setattr("npa.workbench.encord.curate.time.sleep", lambda _: None)
    assert run(client, store, source_receipt_uri=source_uri).selected_item_uuids == [
        "item-1"
    ]


def test_populated_collection_fails_with_final_receipt():
    client = Client(populated=True)
    store = MemoryArtifactStore()
    with pytest.raises(EncordToolError, match="already contains items"):
        run(client, store)
    assert client.events == []
    assert store.read_json(URI)["status"] == "failed"


def test_collection_from_another_folder_is_rejected():
    client = Client(populated=True)
    client.collection.uuid = "12345678-1234-1234-1234-123456789abc"
    client.collection.top_level_folder_uuid = "other-folder"
    store = MemoryArtifactStore()
    with pytest.raises(EncordToolError, match="different folder"):
        run(client, store, collection=client.collection.uuid)
    assert store.read_json(URI)["status"] == "failed"


def test_uncomputed_quality_metric_has_diagnostic_and_receipt(monkeypatch):
    client = Client()
    store = MemoryArtifactStore()
    monkeypatch.setattr(
        "npa.workbench.encord.curate._poll_selection", lambda *_: ([], False)
    )
    with pytest.raises(EncordToolError, match="Computed quality metrics may need"):
        run_curate(
            folder="folder",
            filters=["brightness:0.2:0.8"],
            collection="keepers",
            output_path=URI,
            user_client=client,
            artifact_store=store,
        )
    assert store.read_json(URI)["status"] == "failed"


def test_checkpoint_failure_stops_before_preset_mutation():
    client = Client()
    store = MemoryArtifactStore(fail_replace=2)
    with pytest.raises(EncordToolError, match="checkpoint failed"):
        run(client, store)
    assert client.events == ["create_collection"]
    assert store.read_json(URI)["phase"] == "checkpoint"


def test_preset_cleanup_failure_is_not_success(monkeypatch):
    client = Client(delete_fails=True)
    store = MemoryArtifactStore()
    monkeypatch.setattr("npa.workbench.encord.curate.time.sleep", lambda _: None)
    with pytest.raises(EncordToolError, match="preset cleanup failed"):
        run(client, store)
    assert store.read_json(URI)["status"] == "failed"
    assert not store.read_json(URI)["preset_deleted"]
