"""Shared fakes and constants for the Encord workbench tests.

The Encord SaaS and S3 are mocked at the seam: ``FakeUserClient`` stands in for
``encord.user_client.EncordUserClient`` and ``FakeStorage`` for
``npa.clients.storage.StorageClient``. Test modules import from here so the
fakes evolve in one place.
"""

from __future__ import annotations

import io
import json
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ENDPOINT = "https://storage.test.example"
ENVIRON = {"AWS_ENDPOINT_URL": ENDPOINT}
# Every durable artifact is an s3:// object; the fake storage captures the bytes.
RECEIPT_URI = "s3://bkt/out/receipt.json"
CURATE_RECEIPT_URI = "s3://bkt/out/curate_receipt.json"
REPORT_URI = "s3://bkt/out/roundtrip_report.json"


def fake_uuid(seed: int) -> str:
    return str(uuid.UUID(int=seed))


class FakePaginator:
    def __init__(self, keys: list[str]) -> None:
        self._keys = keys

    def paginate(self, *, Bucket: str, Prefix: str) -> list[dict[str, Any]]:
        return [{"Contents": [{"Key": key} for key in self._keys]}]


class FakeS3:
    def __init__(self, keys: list[str] | None, objects: dict[str, bytes]) -> None:
        self.keys = keys or []
        self.objects = objects
        self.copy_calls: list[dict[str, Any]] = []

    def get_paginator(self, name: str) -> FakePaginator:
        assert name == "list_objects_v2"
        return FakePaginator(self.keys)

    def copy_object(self, *, Bucket: str, Key: str, CopySource: dict[str, str]) -> None:
        self.copy_calls.append({"Bucket": Bucket, "Key": Key, "CopySource": CopySource})

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        return {"Body": io.BytesIO(self.objects[f"s3://{Bucket}/{Key}"])}


class FakeStorage:
    """StorageClient stand-in: raw .s3 plus an in-memory object store.

    ``upload_file`` captures the uploaded bytes by URI (the tool writes every
    artifact through it), so tests read receipts back with ``written``.
    """

    def __init__(self, keys: list[str] | None = None) -> None:
        self.objects: dict[str, bytes] = {}
        self.s3 = FakeS3(keys, self.objects)
        self.uploads: list[tuple[str, str]] = []

    def upload_file(self, local_file: str, bucket_uri: str) -> str:
        self.uploads.append((local_file, bucket_uri))
        self.objects[bucket_uri] = Path(local_file).read_bytes()
        return bucket_uri

    def written(self, uri: str) -> dict[str, Any]:
        """The JSON document the tool wrote to ``uri``."""

        return json.loads(self.objects[uri])


class FakePollResult:
    def __init__(
        self,
        status: str = "DONE",
        done: int = 0,
        errors: int = 0,
        items: list[tuple[str, str]] | None = None,
        unit_errors: list[tuple[list[str], str]] | None = None,
    ) -> None:
        self.status = SimpleNamespace(name=status)
        self.units_done_count = done
        self.units_error_count = errors
        self.items_with_names = [
            SimpleNamespace(item_uuid=item_uuid, name=name)
            for item_uuid, name in (items or [])
        ]
        self.unit_errors = [
            SimpleNamespace(object_urls=urls, error=error)
            for urls, error in (unit_errors or [])
        ]


class FakeFolder:
    def __init__(self, name: str = "folder-a", results: list[FakePollResult] | None = None) -> None:
        self.uuid = uuid.UUID(int=1)
        self.name = name
        self.results = results or [FakePollResult()]
        self.start_calls: list[dict[str, Any]] = []
        self._result_index = 0
        self.folder_items: list[Any] = []
        # Items that become visible only after a registration completes.
        self.post_registration_items: list[Any] = []

    def list_items(self, page_size: int = 100, include_client_metadata: bool = False):
        return iter(self.folder_items)

    def delete_storage_items(self, item_uuids, remove_unused_frames=True):
        self.deleted_item_uuids = [str(u) for u in item_uuids]

    def delete(self):
        self.deleted = True

    def add_private_data_to_folder_start(self, *, integration_id, private_files, ignore_errors):
        self.start_calls.append(
            {
                "integration_id": integration_id,
                "private_files": private_files,
                "ignore_errors": ignore_errors,
            }
        )
        return uuid.UUID(int=100 + len(self.start_calls))

    def add_private_data_to_folder_get_result(self, job_id, timeout_seconds):
        result = self.results[min(self._result_index, len(self.results) - 1)]
        self._result_index += 1
        # Registration makes the new items visible in the folder inventory.
        self.folder_items.extend(self.post_registration_items)
        self.post_registration_items = []
        return result


class FakeDataset:
    def __init__(self, dataset_hash: str = fake_uuid(7), title: str = "ds-a") -> None:
        self.dataset_hash = dataset_hash
        self.title = title
        self.linked: list[list[str]] = []
        self.data_rows: list[Any] = []

    def link_items(self, item_uuids):
        self.linked.append([str(u) for u in item_uuids])
        return item_uuids


class FakeItem:
    def __init__(
        self,
        item_uuid: str,
        name: str,
        signed_url: str | None,
        file_size: int = 10,
    ) -> None:
        self.uuid = item_uuid
        self.name = name
        self.item_type = "VIDEO"
        self.mime_type = "video/mp4"
        self.file_size = file_size
        self.client_metadata = {}
        self._signed_url = signed_url

    def get_signed_url(self, refetch: bool = False) -> str | None:
        return self._signed_url


class FakeCollection:
    def __init__(self, items: list[FakeItem]) -> None:
        self.uuid = uuid.UUID(int=9)
        self.name = "keepers"
        self._items = items
        # Items revealed by add_preset_items (the async server-side selection).
        # reveal_after_calls > 1 models metric-indexing lag: the one-shot
        # evaluation finds nothing until it is re-issued.
        self.pending: list[FakeItem] = []
        self.preset_calls: list[str] = []
        self.reveal_after_calls = 1

    def list_items(self, page_size: int | None = None):
        return iter(self._items)

    def add_preset_items(self, filter_preset) -> None:
        self.preset_calls.append(str(filter_preset))
        if len(self.preset_calls) >= self.reveal_after_calls:
            self._items = list(self.pending)


class FakeUserClient:
    def __init__(
        self,
        *,
        integrations=None,
        folders=None,
        datasets=None,
        collection=None,
        items=None,
    ) -> None:
        self.integrations = integrations or [
            SimpleNamespace(id=fake_uuid(3), title="nebius-s3")
        ]
        self.folders = folders or []
        self.datasets = datasets or {}
        self.collection = collection
        self.items = items or []
        self.created_folders: list[str] = []
        self.created_datasets: list[str] = []
        self.created_collections: list[FakeCollection] = []
        self.created_presets: list[Any] = []
        self.deleted_presets: list[str] = []
        self.deleted_collections: list[str] = []
        # What a curate-created collection's add_preset_items should reveal,
        # and after how many evaluation calls (metric-indexing lag).
        self.pending_curate_items: list[FakeItem] = []
        self.curate_reveal_after_calls = 1

    def get_cloud_integrations(self):
        return list(self.integrations)

    def list_storage_folders(self, *, search: str = "", page_size: int = 100):
        return iter([f for f in self.folders if search in str(f.name)])

    def get_storage_folder(self, folder_uuid):
        for folder in self.folders:
            if str(folder.uuid) == str(folder_uuid):
                return folder
        raise KeyError(folder_uuid)

    def create_storage_folder(self, name, description=""):
        folder = FakeFolder(name=name)
        self.created_folders.append(name)
        self.folders.append(folder)
        return folder

    def get_dataset(self, dataset_hash):
        return self.datasets[str(dataset_hash)]

    def get_datasets(self, *, title_eq: str = ""):
        return [
            {"dataset": SimpleNamespace(dataset_hash=h, title=d.title)}
            for h, d in self.datasets.items()
            if not title_eq or d.title == title_eq
        ]

    def create_dataset(self, title, storage_location, dataset_description="", create_backing_folder=True):
        dataset_hash = fake_uuid(50 + len(self.created_datasets))
        self.created_datasets.append(title)
        self.datasets[dataset_hash] = FakeDataset(dataset_hash, title)
        return {"dataset_hash": dataset_hash}

    def get_collection(self, collection_uuid):
        return self.collection

    def list_collections(self, **kwargs):
        existing = [self.collection] if self.collection else []
        return iter(existing + self.created_collections)

    def create_collection(self, *, top_level_folder_uuid, name, description=""):
        collection = FakeCollection([])
        collection.uuid = uuid.UUID(int=200 + len(self.created_collections))
        collection.name = name
        collection.top_level_folder_uuid = str(top_level_folder_uuid)
        collection.pending = list(self.pending_curate_items)
        collection.reveal_after_calls = self.curate_reveal_after_calls
        self.created_collections.append(collection)
        return collection

    def create_preset(self, *, name, filter_preset_json, description=""):
        preset = SimpleNamespace(
            uuid=uuid.UUID(int=300 + len(self.created_presets)),
            name=name,
            filter_preset_json=filter_preset_json,
        )
        self.created_presets.append(preset)
        return preset

    def delete_preset(self, preset_uuid):
        self.deleted_presets.append(str(preset_uuid))

    def list_presets(self, page_size: int | None = None):
        return iter(self.created_presets)

    def delete_collection(self, collection_uuid):
        self.deleted_collections.append(str(collection_uuid))

    def get_storage_items(self, item_uuids, sign_url=False):
        wanted = {str(u) for u in item_uuids}
        return [item for item in self.items if str(item.uuid) in wanted]




def folder_item(seed: int, key: str, *, metadata: bool = True, url: bool = True):
    """A folder-inventory item carrying exact identity (metadata and/or url)."""

    return SimpleNamespace(
        uuid=fake_uuid(seed),
        name=key,
        client_metadata=({"npa": {"source_uri": f"s3://bkt/{key}"}} if metadata else {}),
        url=f"{ENDPOINT}/bkt/{key}" if url else "",
    )


class FakeUploadFolder(FakeFolder):
    def __init__(self) -> None:
        super().__init__()
        self.uploads: list[tuple[str, str, str]] = []  # (kind, path, title)

    def upload_image(self, file_path, title=None, **kwargs):
        self.uploads.append(("image", str(file_path), str(title)))
        return uuid.UUID(int=200 + len(self.uploads))

    def upload_video(self, file_path, title=None, **kwargs):
        self.uploads.append(("video", str(file_path), str(title)))
        return uuid.UUID(int=200 + len(self.uploads))


class FakeDownloadStorage(FakeStorage):
    def __init__(self, keys=None) -> None:
        super().__init__(keys)
        self.downloads: list[str] = []

    def download_file(self, bucket_uri: str, local_path: str) -> str:
        self.downloads.append(bucket_uri)
        Path(local_path).write_bytes(b"media-bytes")
        return local_path


