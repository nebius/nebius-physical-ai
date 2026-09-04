"""Unit tests for the Encord workbench tool (SaaS and S3 mocked at the seam)."""

from __future__ import annotations

import io
import json
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from npa.workbench.encord.client import (
    CORD_STORAGE_LOCATION,
    default_user_client,
    resolve_collection,
    resolve_dataset,
    resolve_folder,
    resolve_integration,
    resolve_public_endpoint,
)
from npa.workbench.encord.curate import (
    build_filter_preset_json,
    curate_receipt_uri_for,
    parse_filter_specs,
    preset_name_for,
    run_curate,
)
from npa.workbench.encord.identity import (
    canonical_s3_uri,
    normalize_object_url,
    resolve_exact_identity,
)
from npa.workbench.encord.integrity import (
    compare_checksums,
    etag_checksum,
    hash_file,
    write_hashed_stream,
)
from npa.workbench.encord.cleanup import run_cleanup
from npa.workbench.encord.verify import run_verify
from npa.workbench.encord.pull import (
    _same_endpoint_source,
    enumerate_items,
    pull_manifest_uri_for,
    run_pull,
    transfer_item,
)
from npa.workbench.encord.push import (
    BATCH_SIZE,
    build_upload_json,
    discover_objects,
    object_url_for,
    push_receipt_uri_for,
    run_push,
)
from npa.workbench.encord.schemas import (
    EncordAuthError,
    EncordToolError,
    PushedItem,
)
from npa.workbench.encord.seed_demo import run_seed_demo
from npa.workbench.encord.storage import write_json

ENDPOINT = "https://storage.test.example"
ENVIRON = {"AWS_ENDPOINT_URL": ENDPOINT}
# Every durable artifact is an s3:// object; the fake storage captures the bytes.
RECEIPT_URI = "s3://bkt/out/receipt.json"
CURATE_RECEIPT_URI = "s3://bkt/out/curate_receipt.json"
REPORT_URI = "s3://bkt/out/roundtrip_report.json"


def _uuid(seed: int) -> str:
    return str(uuid.UUID(int=seed))


# --- fakes -------------------------------------------------------------------


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
    def __init__(self, dataset_hash: str = _uuid(7), title: str = "ds-a") -> None:
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
            SimpleNamespace(id=_uuid(3), title="nebius-s3")
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
        dataset_hash = _uuid(50 + len(self.created_datasets))
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


# --- url + discovery ---------------------------------------------------------


def test_object_url_for_is_path_style_and_encoded() -> None:
    url = object_url_for(ENDPOINT + "/", "bkt", "runs/a b/frame#1.png")
    assert url == f"{ENDPOINT}/bkt/runs/a%20b/frame%231.png"
    assert " " not in url


def test_receipt_and_manifest_uri_helpers() -> None:
    assert push_receipt_uri_for("s3://b/p/") == "s3://b/p/push_receipt.json"
    assert push_receipt_uri_for("s3://b/p/r.json") == "s3://b/p/r.json"
    assert pull_manifest_uri_for("s3://b/p/") == "s3://b/p/manifest.json"


def test_discover_objects_maps_suffixes_and_skips() -> None:
    storage = FakeStorage(["p/a.mp4", "p/b.PNG", "p/c.jpeg", "p/d.txt", "p/e.mcap"])
    entries, skipped = discover_objects(storage, "s3://bkt/p/", "videos-images")
    assert [(key, category) for key, category, _, _ in entries] == [
        ("p/a.mp4", "videos"),
        ("p/b.PNG", "images"),
        ("p/c.jpeg", "images"),
    ]
    assert skipped == ["p/d.txt", "p/e.mcap"]


def test_discover_objects_mcap_gating() -> None:
    storage = FakeStorage(["p/a.mp4", "p/e.mcap"])
    entries, _ = discover_objects(storage, "s3://bkt/p/", "mcap")
    assert [(key, category) for key, category, _, _ in entries] == [("p/e.mcap", "mcap")]
    entries, skipped = discover_objects(storage, "s3://bkt/p/", "all")
    pairs = [(key, category) for key, category, _, _ in entries]
    assert ("p/e.mcap", "mcap") in pairs and ("p/a.mp4", "videos") in pairs
    assert skipped == []


def test_discover_objects_empty_prefix_fails() -> None:
    with pytest.raises(EncordToolError, match="No supported media"):
        discover_objects(FakeStorage(["p/readme.md"]), "s3://bkt/p/", "videos-images")


def test_build_upload_json_carries_identity_metadata() -> None:
    items = [
        PushedItem(key="a.mp4", source_uri="s3://bkt/a.mp4", object_url="u1", category="videos"),
        PushedItem(key="b.png", source_uri="s3://bkt/b.png", object_url="u2", category="images"),
    ]
    payload = build_upload_json(items)
    assert payload["skip_duplicate_urls"] is True
    assert payload["videos"] == [
        {
            "objectUrl": "u1",
            "title": "a.mp4",
            "clientMetadata": {"npa": {"source_uri": "s3://bkt/a.mp4"}},
        }
    ]
    assert payload["images"][0]["clientMetadata"] == {
        "npa": {"source_uri": "s3://bkt/b.png"}
    }


# --- client resolution -------------------------------------------------------


def test_resolve_integration_by_title_and_id() -> None:
    client = FakeUserClient()
    ref = resolve_integration(client, "nebius-s3")
    assert (ref.id, ref.title, ref.created) == (_uuid(3), "nebius-s3", False)
    by_id = resolve_integration(client, _uuid(3))
    assert (by_id.id, by_id.title) == (_uuid(3), "nebius-s3")
    with pytest.raises(EncordToolError, match="No Encord cloud integration titled"):
        resolve_integration(client, "missing")
    with pytest.raises(EncordToolError, match="No Encord cloud integration with id"):
        resolve_integration(client, _uuid(4))
    with pytest.raises(EncordToolError, match="must not be empty"):
        resolve_integration(client, "  ")


def test_resolve_folder_creates_on_missing_title_only() -> None:
    client = FakeUserClient()
    ref = resolve_folder(client, "fresh")
    assert ref.created is True and client.created_folders == ["fresh"]
    assert ref.title == "fresh" and ref.id == str(ref.obj.uuid)
    again = resolve_folder(client, "fresh")
    assert again.created is False and again.obj is ref.obj
    with pytest.raises(KeyError):
        resolve_folder(client, _uuid(99))


def test_resolvers_never_guess_between_same_titled_objects() -> None:
    client = FakeUserClient(folders=[FakeFolder(name="dup"), FakeFolder(name="dup")])
    with pytest.raises(EncordToolError, match="Multiple Encord storage folders"):
        resolve_folder(client, "dup")
    assert client.created_folders == []
    client.datasets[_uuid(1)] = FakeDataset(_uuid(1), "dup-ds")
    client.datasets[_uuid(2)] = FakeDataset(_uuid(2), "dup-ds")
    with pytest.raises(EncordToolError, match="pass the dataset hash"):
        resolve_dataset(client, "dup-ds")


def test_resolve_dataset_title_create_and_pull_no_create() -> None:
    client = FakeUserClient()
    ref = resolve_dataset(client, "new-ds")
    assert ref.created is True and ref.title == "new-ds"
    assert ref.id in client.datasets
    assert resolve_dataset(client, "new-ds").created is False
    with pytest.raises(EncordToolError, match="No Encord dataset titled"):
        resolve_dataset(client, "absent", create=False)


def test_cord_storage_location_matches_the_sdk_enum() -> None:
    """The injected-client seam never imports the SDK; pin the value it stands for."""

    dataset_orm = pytest.importorskip("encord.orm.dataset")
    assert int(dataset_orm.StorageLocation.CORD_STORAGE) == CORD_STORAGE_LOCATION


def test_default_user_client_requires_secret_and_decodes_b64() -> None:
    with pytest.raises(EncordAuthError, match="No Encord credential"):
        default_user_client({})
    with pytest.raises(EncordAuthError, match="not valid base64"):
        default_user_client({"ENCORD_SSH_KEY_B64": "!!!not-base64!!!"})


def test_default_user_client_ignores_a_raw_pem_transport() -> None:
    """Exactly two transports exist; the truncation-prone raw PEM is not one."""

    raw_pem = "-----BEGIN OPENSSH PRIVATE KEY-----\nabc\n-----END OPENSSH PRIVATE KEY-----"
    with pytest.raises(EncordAuthError, match="No Encord credential"):
        default_user_client({"ENCORD_SSH_KEY": raw_pem})


def test_resolve_public_endpoint_prefers_env() -> None:
    assert resolve_public_endpoint({"AWS_ENDPOINT_URL": ENDPOINT + "/"}) == ENDPOINT
    with pytest.raises(EncordToolError, match="No S3 endpoint"):
        resolve_public_endpoint({})


# --- run_push ----------------------------------------------------------------


def _push_kwargs(storage: FakeStorage, client: FakeUserClient, **overrides):
    kwargs = dict(
        input_path="s3://bkt/p/",
        integration="nebius-s3",
        folder="fresh",
        output_path=RECEIPT_URI,
        user_client=client,
        storage_client=storage,
        environ=dict(ENVIRON),
    )
    kwargs.update(overrides)
    return kwargs


@pytest.fixture(autouse=True)
def _fast_identity_relist(monkeypatch: pytest.MonkeyPatch):
    import npa.workbench.encord.push as push_module

    monkeypatch.setattr(push_module, "IDENTITY_RELIST_DELAY_SECONDS", 0.0)


def _folder_item(seed: int, key: str, *, metadata: bool = True, url: bool = True):
    """A folder-inventory item carrying exact identity (metadata and/or url)."""

    return SimpleNamespace(
        uuid=_uuid(seed),
        name=key,
        client_metadata=(
            {"npa": {"source_uri": f"s3://bkt/{key}"}} if metadata else {}
        ),
        url=f"{ENDPOINT}/bkt/{key}" if url else "",
    )


def test_run_push_happy_path_links_dataset() -> None:
    storage = FakeStorage(["p/a.mp4", "p/b.png"])
    folder = FakeFolder(
        results=[
            FakePollResult(
                status="DONE",
                done=2,
                items=[(_uuid(21), "p/a.mp4"), (_uuid(22), "b.png")],
            )
        ]
    )
    # After registration the folder inventory exposes the items with their
    # npa.source_uri clientMetadata — the only lineage signal besides the URL.
    folder.post_registration_items = [_folder_item(21, "p/a.mp4"), _folder_item(22, "p/b.png")]
    client = FakeUserClient(folders=[])
    client.create_storage_folder = lambda name, description="": folder  # type: ignore[assignment]
    receipt = run_push(**_push_kwargs(storage, client, dataset="new-ds"))
    assert receipt.status == "done"
    assert receipt.units_done == 2 and receipt.units_error == 0
    assert receipt.dataset_created is True and receipt.linked_count == 2
    dataset = next(iter(client.datasets.values()))
    assert dataset.linked == [[_uuid(21), _uuid(22)]]
    # uuids attached to receipt rows by exact metadata identity, never names
    assert {item.item_uuid for item in receipt.items} == {_uuid(21), _uuid(22)}
    assert {item.identity_signal for item in receipt.items} == {"metadata"}
    assert receipt.items[0].source_uri == "s3://bkt/p/a.mp4"
    # objectUrls are path-style against the environ endpoint
    assert receipt.items[0].object_url == f"{ENDPOINT}/bkt/p/a.mp4"
    # registration payload carried the identity metadata
    first_entry = folder.start_calls[0]["private_files"]["videos"][0]
    assert first_entry["clientMetadata"] == {"npa": {"source_uri": "s3://bkt/p/a.mp4"}}
    # receipt written locally
    payload = storage.written(RECEIPT_URI)
    assert payload["schema"] == "npa.encord.push_receipt.v1"
    assert payload["status"] == "done"


def test_run_push_unit_errors_write_receipt_then_raise() -> None:
    storage = FakeStorage(["p/a.mp4"])
    bad_url = f"{ENDPOINT}/bkt/p/a.mp4"
    folder = FakeFolder(
        results=[FakePollResult(status="DONE", done=0, errors=1, unit_errors=[([bad_url], "403 from integration")])]
    )
    client = FakeUserClient(folders=[folder])
    with pytest.raises(EncordToolError, match="1 unit error"):
        run_push(**_push_kwargs(storage, client, folder=str(folder.uuid)))
    payload = storage.written(RECEIPT_URI)
    assert payload["status"] == "failed"
    assert payload["items"][0]["status"] == "error"
    assert "403" in payload["items"][0]["error"]


def test_run_push_timeout_is_fail_closed() -> None:
    storage = FakeStorage(["p/a.mp4"])
    folder = FakeFolder(results=[FakePollResult(status="PENDING")])
    client = FakeUserClient(folders=[folder])
    with pytest.raises(EncordToolError, match="timeout"):
        run_push(**_push_kwargs(storage, client, folder=str(folder.uuid)))
    payload = storage.written(RECEIPT_URI)
    assert payload["status"] == "timeout"


def test_run_push_mcap_is_experimental_error() -> None:
    storage = FakeStorage(["p/e.mcap"])
    folder = FakeFolder()
    client = FakeUserClient(folders=[folder])
    with pytest.raises(EncordToolError):
        run_push(**_push_kwargs(storage, client, folder=str(folder.uuid), media="mcap"))
    payload = storage.written(RECEIPT_URI)
    assert payload["items"][0]["status"] == "experimental_error"
    assert folder.start_calls == []  # nothing guessed onto the wire


def test_run_push_batches_at_500() -> None:
    keys = [f"p/{index:04d}.png" for index in range(BATCH_SIZE + 1)]
    storage = FakeStorage(keys)
    folder = FakeFolder(results=[FakePollResult(status="DONE", done=BATCH_SIZE + 1)])
    folder.post_registration_items = [
        _folder_item(1000 + index, key) for index, key in enumerate(keys)
    ]
    client = FakeUserClient(folders=[folder])
    run_push(**_push_kwargs(storage, client, folder=str(folder.uuid)))
    assert len(folder.start_calls) == 2
    first = folder.start_calls[0]["private_files"]
    assert len(first["images"]) == BATCH_SIZE


def test_run_push_unattributable_item_fails_closed() -> None:
    """Encord accepted the batch but no exact identity matched: never silent."""

    storage = FakeStorage(["p/a.mp4"])
    folder = FakeFolder(results=[FakePollResult(status="DONE", done=1)])
    client = FakeUserClient(folders=[folder])  # inventory never shows the item
    with pytest.raises(EncordToolError, match="unit error"):
        run_push(**_push_kwargs(storage, client, folder=str(folder.uuid)))
    payload = storage.written(RECEIPT_URI)
    assert payload["items"][0]["status"] == "error"
    assert "no exact metadata or object URL identity" in payload["items"][0]["error"]


def test_run_push_counts_a_failed_item_once_even_when_encord_names_it_differently() -> None:
    """A unit error Encord attributes to a differently spelled URL is one error."""

    storage = FakeStorage(["p/a.mp4"])
    # Encord echoes the failing objectUrl with different percent-encoding, so
    # the by-url match misses; the item then fails identity resolution instead.
    folder = FakeFolder(
        results=[
            FakePollResult(
                status="DONE", done=0, errors=1,
                unit_errors=[([f"{ENDPOINT}/bkt/p/a%2Emp4"], "403 from integration")],
            )
        ]
    )
    client = FakeUserClient(folders=[folder])
    with pytest.raises(EncordToolError, match="1 unit error"):
        run_push(**_push_kwargs(storage, client, folder=str(folder.uuid)))
    payload = storage.written(RECEIPT_URI)
    assert payload["units_error"] == 1 and payload["units_done"] == 0
    assert payload["units_done"] + payload["units_error"] == payload["files_discovered"]


def test_run_push_register_mode_requires_an_integration_before_any_io() -> None:
    storage = FakeStorage(["p/a.mp4"])
    client = FakeUserClient()
    with pytest.raises(EncordToolError, match="--integration is required"):
        run_push(**_push_kwargs(storage, client, integration=""))
    assert storage.uploads == [] and client.created_folders == []


def test_run_push_rejects_local_input(tmp_path: Path) -> None:
    with pytest.raises(EncordToolError, match="s3:// prefix"):
        run_push(
            input_path=str(tmp_path),
            integration="i",
            folder="f",
            output_path="s3://bkt/out/r.json",
            user_client=FakeUserClient(),
            storage_client=FakeStorage(),
            environ=dict(ENVIRON),
        )


def test_run_push_planned_receipt_lands_before_the_first_mutation() -> None:
    """The write-ahead receipt must exist before Encord is touched.

    Folder creation is the first possible mutation; by then the planned
    receipt — every discovered item, the requested folder, status planned —
    is already durable, so an uncatchable kill leaves a record of intent.
    """

    storage = FakeStorage(["p/a.mp4"])
    folder = FakeFolder(results=[FakePollResult(status="DONE", done=1)])
    folder.post_registration_items = [_folder_item(21, "p/a.mp4")]

    class MutationChecksReceipt(FakeUserClient):
        def create_storage_folder(self, name, description=""):
            planned = storage.written(RECEIPT_URI)
            assert planned["status"] == "planned"
            assert planned["folder_name"] == "fresh"
            assert planned["files_discovered"] == 1
            assert planned["items"][0]["source_uri"] == "s3://bkt/p/a.mp4"
            assert planned["items"][0]["item_uuid"] == ""
            self.created_folders.append(name)
            return folder

    receipt = run_push(**_push_kwargs(storage, MutationChecksReceipt(folders=[])))
    assert receipt.status == "done"
    # Exactly two writes: the write-ahead copy, then the final receipt.
    assert [uri for _, uri in storage.uploads] == [RECEIPT_URI, RECEIPT_URI]
    assert storage.written(RECEIPT_URI)["status"] == "done"


def test_discover_objects_carries_listing_facts_by_name() -> None:
    storage = FakeStorage(["p/a.mp4"])
    (entry,), _ = discover_objects(storage, "s3://bkt/p/", "videos-images")
    assert (entry.key, entry.category, entry.size, entry.etag) == ("p/a.mp4", "videos", 0, "")


def _stub_httpx_stream(monkeypatch: pytest.MonkeyPatch, body: bytes = b"payload") -> None:
    """Make httpx.stream yield one fixed body: the cross-origin download path."""

    import httpx

    class FakeResponse:
        def raise_for_status(self) -> None: ...

        def iter_bytes(self, chunk_size: int):
            yield body

    class FakeStream:
        def __init__(self, *args, **kwargs) -> None: ...

        def __enter__(self) -> FakeResponse:
            return FakeResponse()

        def __exit__(self, *args) -> None: ...

    monkeypatch.setattr(httpx, "stream", FakeStream)


# --- pull --------------------------------------------------------------------


def test_same_endpoint_source_path_and_virtual_hosted() -> None:
    assert _same_endpoint_source(
        f"{ENDPOINT}/bkt/p/a.mp4?X-Sig=abc", ENDPOINT
    ) == ("bkt", "p/a.mp4")
    host = ENDPOINT.removeprefix("https://")
    assert _same_endpoint_source(
        f"https://bkt.{host}/p/a%20b.mp4?X-Sig=abc", ENDPOINT
    ) == ("bkt", "p/a b.mp4")
    assert _same_endpoint_source("https://elsewhere.example/x", ENDPOINT) is None


def test_transfer_item_same_bucket_copies_server_side() -> None:
    storage = FakeStorage()
    item = FakeItem(_uuid(31), "a.mp4", f"{ENDPOINT}/bkt/p/a.mp4?sig=1", file_size=5)
    record = transfer_item(
        item, storage_client=storage, output_uri="s3://out/pull", endpoint_url=ENDPOINT
    )
    assert record.transfer == "copy"
    assert storage.s3.copy_calls[0]["CopySource"] == {"Bucket": "bkt", "Key": "p/a.mp4"}
    assert storage.s3.copy_calls[0]["Bucket"] == "out"
    assert record.media_uri.startswith("s3://out/pull/media/")


def test_transfer_item_composite_without_signed_url_is_error() -> None:
    record = transfer_item(
        FakeItem(_uuid(32), "group", None),
        storage_client=FakeStorage(),
        output_uri="s3://out/pull",
        endpoint_url=ENDPOINT,
    )
    assert record.transfer == "error" and "signed URL" in record.error


def test_transfer_item_downloads_cross_origin(monkeypatch: pytest.MonkeyPatch) -> None:

    _stub_httpx_stream(monkeypatch)
    storage = FakeStorage()
    record = transfer_item(
        FakeItem(_uuid(33), "far.mp4", "https://cdn.encord.example/x?sig=1"),
        storage_client=storage,
        output_uri="s3://out/pull",
        endpoint_url=ENDPOINT,
    )
    assert record.transfer == "download"
    assert storage.uploads and storage.uploads[0][1] == record.media_uri


def test_transfer_item_records_why_the_copy_fast_path_was_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A same-endpoint copy that fails falls back to download, visibly."""

    _stub_httpx_stream(monkeypatch)
    storage = FakeStorage()

    def failing_copy(**kwargs):
        raise RuntimeError("AccessDenied on CopySource")

    storage.s3.copy_object = failing_copy  # type: ignore[assignment]
    record = transfer_item(
        FakeItem(_uuid(34), "a.mp4", f"{ENDPOINT}/bkt/p/a.mp4?sig=1"),
        storage_client=storage,
        output_uri="s3://out/pull",
        endpoint_url=ENDPOINT,
    )
    assert record.transfer == "download"
    assert "AccessDenied" in record.copy_error


def test_enumerate_items_per_source() -> None:
    items = [FakeItem(_uuid(41), "a.mp4", "u")]
    collection = FakeCollection(items)
    dataset = FakeDataset()
    dataset.data_rows = [SimpleNamespace(backing_item_uuid=_uuid(41))]
    client = FakeUserClient(
        collection=collection, datasets={dataset.dataset_hash: dataset}, items=items
    )
    found = enumerate_items(client, source="collection", source_id=str(collection.uuid))
    # Collection items are re-fetched in bulk with signed URLs, like the other sources.
    assert found.source_name == "keepers" and found.items == items
    assert found.project is None and found.label_rows == ()
    found = enumerate_items(client, source="dataset", source_id=dataset.dataset_hash)
    assert found.source_id == dataset.dataset_hash and found.items == items


def test_run_pull_writes_manifest_and_fails_closed_on_errors() -> None:
    good = FakeItem(_uuid(51), "a.mp4", f"{ENDPOINT}/bkt/p/a.mp4", file_size=7)
    bad = FakeItem(_uuid(52), "b.mp4", None)
    collection = FakeCollection([good, bad])
    client = FakeUserClient(collection=collection, items=[good, bad])
    storage = FakeStorage()
    out_uri = "s3://out/pull"
    with pytest.raises(EncordToolError, match="failed for 1 of 2"):
        run_pull(
            source="collection",
            source_id=str(collection.uuid),
            output_path=out_uri,
            user_client=client,
            storage_client=storage,
            environ=dict(ENVIRON),
        )
    # manifest + per-item JSON were still uploaded before the raise
    uploaded = [uri for _, uri in storage.uploads]
    assert f"{out_uri}/manifest.json" in uploaded
    assert f"{out_uri}/items/{_uuid(51)}.json" in uploaded


def test_run_pull_keeps_every_record_when_one_signed_url_fetch_raises() -> None:
    class UnsignableItem(FakeItem):
        def get_signed_url(self, refetch: bool = False) -> str | None:
            raise RuntimeError("502 from Encord while signing")

    good = FakeItem(_uuid(54), "a.mp4", f"{ENDPOINT}/bkt/p/a.mp4", file_size=7)
    bad = UnsignableItem(_uuid(55), "b.mp4", None)
    collection = FakeCollection([good, bad])
    client = FakeUserClient(collection=collection, items=[good, bad])
    storage = FakeStorage()
    with pytest.raises(EncordToolError, match="failed for 1 of 2"):
        run_pull(
            source="collection",
            source_id=str(collection.uuid),
            output_path="s3://out/pull",
            user_client=client,
            storage_client=storage,
            environ=dict(ENVIRON),
        )
    manifest = storage.written("s3://out/pull/manifest.json")
    assert manifest["items_total"] == 2 and manifest["media_copied"] == 1
    failed = next(row for row in manifest["items"] if row["item_uuid"] == _uuid(55))
    assert failed["transfer"] == "error" and "502" in failed["error"]


def test_run_pull_happy_path_counts() -> None:
    good = FakeItem(_uuid(53), "a.mp4", f"{ENDPOINT}/bkt/p/a.mp4", file_size=7)
    collection = FakeCollection([good])
    client = FakeUserClient(collection=collection, items=[good])
    storage = FakeStorage()
    manifest = run_pull(
        source="collection",
        source_id=str(collection.uuid),
        output_path="s3://out/pull",
        user_client=client,
        storage_client=storage,
        environ=dict(ENVIRON),
    )
    assert manifest.items_total == 1
    assert manifest.media_copied == 1 and manifest.media_failed == 0
    assert manifest.media_bytes == 7
    assert manifest.manifest_uri == "s3://out/pull/manifest.json"


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


def test_run_push_upload_mode_copies_bytes_and_links() -> None:
    storage = FakeDownloadStorage(["p/a.mp4", "p/b.png"])
    folder = FakeUploadFolder()
    client = FakeUserClient(folders=[folder])
    receipt = run_push(
        input_path="s3://bkt/p/",
        integration="",  # unused in upload mode
        folder=str(folder.uuid),
        dataset="new-ds",
        transfer="upload",
        output_path=RECEIPT_URI,
        user_client=client,
        storage_client=storage,
        environ={},  # upload mode needs no public endpoint
    )
    assert receipt.status == "done" and receipt.transfer == "upload"
    assert receipt.units_done == 2 and receipt.units_error == 0
    # bytes moved: each object downloaded from S3 then uploaded by kind, titled by key
    assert storage.downloads == ["s3://bkt/p/a.mp4", "s3://bkt/p/b.png"]
    assert [(kind, title) for kind, _, title in folder.uploads] == [
        ("video", "p/a.mp4"),
        ("image", "p/b.png"),
    ]
    # no registration jobs, no objectUrls
    assert folder.start_calls == []
    assert all(item.object_url == "" for item in receipt.items)
    assert all(item.status == "uploaded" and item.item_uuid for item in receipt.items)
    dataset = next(iter(client.datasets.values()))
    assert len(dataset.linked[0]) == 2


def test_run_push_upload_mode_per_item_error_fails_closed() -> None:
    storage = FakeDownloadStorage(["p/a.mp4"])

    class BrokenFolder(FakeUploadFolder):
        def upload_video(self, file_path, title=None, **kwargs):
            raise RuntimeError("507 storage quota")

    folder = BrokenFolder()
    client = FakeUserClient(folders=[folder])
    with pytest.raises(EncordToolError, match="1 unit error"):
        run_push(
            input_path="s3://bkt/p/",
            integration="",
            folder=str(folder.uuid),
            transfer="upload",
            output_path=RECEIPT_URI,
            user_client=client,
            storage_client=storage,
            environ={},
        )
    payload = storage.written(RECEIPT_URI)
    assert payload["status"] == "failed"
    assert "507" in payload["items"][0]["error"]


def test_run_push_rejects_unknown_transfer() -> None:
    with pytest.raises(EncordToolError, match="Unknown --transfer"):
        run_push(
            input_path="s3://bkt/p/",
            integration="i",
            folder="f",
            transfer="teleport",
            output_path="s3://bkt/out/r.json",
            user_client=FakeUserClient(),
            storage_client=FakeStorage(["p/a.mp4"]),
            environ=dict(ENVIRON),
        )


def test_run_push_repush_links_existing_items_via_object_url() -> None:
    """skip_duplicate_urls adds nothing on a re-push; linking must still happen.

    Items registered before identity metadata existed resolve through their
    registered objectUrl — still exact, never a display name.
    """

    storage = FakeStorage(["p/a.mp4", "p/b.png"])
    folder = FakeFolder(results=[FakePollResult(status="DONE", done=0, items=[])])
    folder.folder_items = [
        _folder_item(61, "p/a.mp4", metadata=False),
        _folder_item(62, "p/b.png", metadata=False),
    ]
    client = FakeUserClient(folders=[folder])
    receipt = run_push(**_push_kwargs(storage, client, folder=str(folder.uuid), dataset="new-ds"))
    dataset = next(iter(client.datasets.values()))
    assert sorted(dataset.linked[0]) == [_uuid(61), _uuid(62)]
    assert receipt.linked_count == 2
    assert {item.item_uuid for item in receipt.items} == {_uuid(61), _uuid(62)}
    assert {item.identity_signal for item in receipt.items} == {"object_url"}


def test_run_push_shared_basenames_resolve_to_distinct_items() -> None:
    """The identity-regression case: same basename, different objects.

    Exact identity (metadata/objectUrl) attributes each receipt row to its own
    Encord item; a name-based scheme could not tell these apart.
    """

    storage = FakeStorage(["p/left/clip.mp4", "p/right/clip.mp4"])
    folder = FakeFolder(results=[FakePollResult(status="DONE", done=2, items=[])])
    folder.post_registration_items = [
        _folder_item(64, "p/left/clip.mp4"),
        _folder_item(65, "p/right/clip.mp4"),
    ]
    client = FakeUserClient(folders=[folder])
    receipt = run_push(
        **_push_kwargs(storage, client, folder=str(folder.uuid), dataset="new-ds")
    )
    dataset = next(iter(client.datasets.values()))
    assert sorted(dataset.linked[0]) == [_uuid(64), _uuid(65)]
    by_key = {item.key: item.item_uuid for item in receipt.items}
    assert by_key == {"p/left/clip.mp4": _uuid(64), "p/right/clip.mp4": _uuid(65)}


def test_run_push_identity_conflict_fails_the_item_closed() -> None:
    """Two folder items claiming one source is a conflict, never a guess."""

    storage = FakeStorage(["p/a.mp4"])
    folder = FakeFolder(results=[FakePollResult(status="DONE", done=1, items=[])])
    duplicate = _folder_item(67, "p/a.mp4")
    duplicate.uuid = _uuid(68)
    folder.post_registration_items = [_folder_item(67, "p/a.mp4"), duplicate]
    client = FakeUserClient(folders=[folder])
    with pytest.raises(EncordToolError, match="unit error"):
        run_push(**_push_kwargs(storage, client, folder=str(folder.uuid)))
    payload = storage.written(RECEIPT_URI)
    assert payload["items"][0]["status"] == "error"
    assert "identity signals conflict" in payload["items"][0]["error"]


def test_run_push_writes_receipt_when_linking_crashes() -> None:
    """Post-mutation failures must still leave a receipt behind."""

    storage = FakeStorage(["p/a.mp4"])
    folder = FakeFolder(
        results=[FakePollResult(status="DONE", done=1, items=[(_uuid(63), "p/a.mp4")])]
    )
    folder.post_registration_items = [_folder_item(63, "p/a.mp4")]
    client = FakeUserClient(folders=[folder])

    class ExplodingDataset(FakeDataset):
        def link_items(self, item_uuids):
            raise RuntimeError("503 from Encord")

    exploding = ExplodingDataset()
    client.datasets[exploding.dataset_hash] = exploding
    with pytest.raises(EncordToolError, match="503 from Encord"):
        run_push(
            **_push_kwargs(
                storage, client,
                folder=str(folder.uuid), dataset=exploding.dataset_hash,
            )
        )
    payload = storage.written(RECEIPT_URI)
    assert payload["status"] == "failed"
    assert "503 from Encord" in payload["error"]
    assert payload["units_done"] == 1  # registration evidence preserved


def test_run_push_writes_receipt_when_dataset_create_fails() -> None:
    """Folder creation is a mutation too, so setup failure must retain lineage."""

    class DatasetCreateFails(FakeUserClient):
        def create_dataset(self, *args, **kwargs):
            raise RuntimeError("dataset create unavailable")

    storage = FakeStorage(["p/a.mp4"])
    client = DatasetCreateFails()
    with pytest.raises(EncordToolError, match="dataset create unavailable"):
        run_push(
            **_push_kwargs(storage, client, folder="new-folder", dataset="new-dataset")
        )
    payload = storage.written(RECEIPT_URI)
    assert client.created_folders == ["new-folder"]
    assert payload["status"] == "failed"
    assert payload["folder_name"] == "new-folder"
    assert "dataset create unavailable" in payload["error"]


def test_run_pull_writes_manifest_when_labels_crash() -> None:
    class ExplodingProject:
        title = "proj"

        def list_label_rows_v2(self):
            return [SimpleNamespace(backing_item_uuid=_uuid(71))]

        def create_bundle(self, bundle_size: int):
            raise RuntimeError("bundle exploded")

    item = FakeItem(_uuid(71), "a.mp4", f"{ENDPOINT}/bkt/p/a.mp4", file_size=3)
    client = FakeUserClient(items=[item])
    client.get_project = lambda h: ExplodingProject()  # type: ignore[assignment]
    storage = FakeStorage()
    with pytest.raises(EncordToolError, match="bundle exploded"):
        run_pull(
            source="project",
            source_id=_uuid(70),
            output_path="s3://out/pull",
            user_client=client,
            storage_client=storage,
            environ=dict(ENVIRON),
        )
    uploaded = [uri for _, uri in storage.uploads]
    assert "s3://out/pull/manifest.json" in uploaded  # manifest landed anyway


def test_enumerate_items_survives_raising_backing_property() -> None:
    """SDK DataRow.backing_item_uuid RAISES on legacy rows; getattr can't guard it."""

    class LegacyRow:
        @property
        def backing_item_uuid(self):
            raise NotImplementedError("Storage API is not yet implemented")

    dataset = FakeDataset()
    dataset.data_rows = [LegacyRow(), SimpleNamespace(backing_item_uuid=_uuid(41))]
    items = [FakeItem(_uuid(41), "a.mp4", "u")]
    client = FakeUserClient(datasets={dataset.dataset_hash: dataset}, items=items)
    found = enumerate_items(client, source="dataset", source_id=dataset.dataset_hash)
    assert found.items == items  # legacy row skipped, run survives


def test_transfer_item_retries_download_after_expired_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    import npa.workbench.encord.pull as pull_module  # noqa: F401 - patched via httpx

    calls: list[str] = []

    class FakeResponse:
        def __init__(self, url: str) -> None:
            self._expired = "expired" in url

        def raise_for_status(self) -> None:
            if self._expired:
                request = httpx.Request("GET", "https://x")
                raise httpx.HTTPStatusError(
                    "403", request=request, response=httpx.Response(403, request=request)
                )

        def iter_bytes(self, chunk_size: int):
            yield b"payload"

    class FakeStream:
        def __init__(self, method, url, **kwargs) -> None:
            calls.append(url)
            self._response = FakeResponse(url)

        def __enter__(self) -> FakeResponse:
            return self._response

        def __exit__(self, *args) -> None: ...

    monkeypatch.setattr(httpx, "stream", FakeStream)

    class RefreshingItem(FakeItem):
        def get_signed_url(self, refetch: bool = False) -> str | None:
            return "https://cdn.example/fresh" if refetch else "https://cdn.example/expired"

    record = transfer_item(
        RefreshingItem(_uuid(72), "far.mp4", "unused"),
        storage_client=FakeStorage(),
        output_uri="s3://out/pull",
        endpoint_url=ENDPOINT,
    )
    assert record.transfer == "download"
    assert calls == ["https://cdn.example/expired", "https://cdn.example/fresh"]


def test_run_pull_empty_source_writes_manifest_then_raises() -> None:
    collection = FakeCollection([])
    client = FakeUserClient(collection=collection)
    storage = FakeStorage()
    with pytest.raises(EncordToolError, match="contains no storage items"):
        run_pull(
            source="collection",
            source_id=str(collection.uuid),
            output_path="s3://out/pull",
            user_client=client,
            storage_client=storage,
            environ=dict(ENVIRON),
        )
    assert [uri for _, uri in storage.uploads] == ["s3://out/pull/manifest.json"]


# --- curate --------------------------------------------------------------------


def _curate_kwargs(storage: FakeStorage, client: FakeUserClient, **overrides):
    kwargs = dict(
        folder="src",
        filters=["width:1:100000"],
        collection="keepers-new",
        output_path=CURATE_RECEIPT_URI,
        user_client=client,
        storage_client=storage,
        environ=dict(ENVIRON),
    )
    kwargs.update(overrides)
    return kwargs


def _curate_client(**client_kwargs) -> tuple[FakeUserClient, FakeFolder]:
    """A client whose 'src' folder holds one item (curate fails fast on empty)."""

    folder = FakeFolder(name="src")
    folder.folder_items = [FakeItem(_uuid(30), "seed.png", None)]
    return FakeUserClient(folders=[folder], **client_kwargs), folder


@pytest.fixture(autouse=False)
def _fast_polling(monkeypatch: pytest.MonkeyPatch):
    import npa.workbench.encord.curate as curate_module

    monkeypatch.setattr(curate_module, "POLL_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(curate_module, "REISSUE_INTERVAL_SECONDS", 0.0)


def test_curate_receipt_uri_helper() -> None:
    assert curate_receipt_uri_for("s3://b/p/") == "s3://b/p/curate_receipt.json"
    assert curate_receipt_uri_for("s3://b/p/r.json") == "s3://b/p/r.json"


def test_preset_name_is_run_scoped_and_never_collides_ad_hoc() -> None:
    assert preset_name_for("run-1") == "npa-curate-run-1"
    first, second = preset_name_for(""), preset_name_for("   ")
    assert first.startswith("npa-curate-adhoc-") and second.startswith("npa-curate-adhoc-")
    assert first != second  # two ad-hoc curates must not race on one title


def test_parse_filter_specs_repeatable_and_comma_separated() -> None:
    parsed = parse_filter_specs(["brightness:0.2:0.8,sharpness:0.3:1", "width:32:4096"])
    assert [(f.metric, f.min, f.max) for f in parsed] == [
        ("brightness", 0.2, 0.8),
        ("sharpness", 0.3, 1.0),
        ("width", 32.0, 4096.0),
    ]
    assert [f.encord_metric for f in parsed] == [
        "metric_brightness",
        "metric_sharpness",
        "metric_width",
    ]
    # Computed vs intrinsic drives the zero-selection diagnostic.
    assert [f.computed for f in parsed] == [True, True, False]


@pytest.mark.parametrize(
    ("specs", "match"),
    [
        ([], "At least one --filter"),
        (["  ", ""], "At least one --filter"),
        (["blur:0:1"], "Unknown filter metric 'blur'"),
        (["brightness:0.2"], "expected metric:min:max"),
        (["brightness:a:b"], "must be numbers"),
        (["brightness:0.9:0.1"], "min exceeds max"),
    ],
)
def test_parse_filter_specs_fails_closed(specs: list[str], match: str) -> None:
    with pytest.raises(EncordToolError, match=match):
        parse_filter_specs(specs)


def test_build_filter_preset_json_pins_the_live_verified_shape() -> None:
    payload = build_filter_preset_json(parse_filter_specs(["brightness:0.2:0.8"]))
    assert payload == {
        "global_filters": {
            "filters": [
                {
                    "include": True,
                    "values": [0.2, 0.8],
                    "domain": "data",
                    "metric": "metric_brightness",
                    "type": "metric",
                }
            ]
        }
    }


def test_run_curate_happy_path_creates_collection_and_preset(_fast_polling) -> None:
    client, folder = _curate_client()
    client.pending_curate_items = [
        FakeItem(_uuid(31), "a.png", None),
        FakeItem(_uuid(32), "b.png", None),
    ]
    storage = FakeStorage()
    receipt = run_curate(**_curate_kwargs(storage, client, workflow_run="run-1"))

    assert receipt.status == "done"
    assert receipt.items_selected == 2
    assert receipt.items_total == 1  # the folder held one item when evaluated
    assert receipt.collection_created is True
    assert receipt.collection_name == "keepers-new"
    assert receipt.folder_uuid == str(folder.uuid)
    assert receipt.preset_name == "npa-curate-run-1"
    # The preset payload sent to Encord is the pinned shape, verbatim.
    (preset,) = client.created_presets
    assert preset.filter_preset_json["global_filters"]["filters"][0]["metric"] == "metric_width"
    (collection,) = client.created_collections
    assert collection.top_level_folder_uuid == str(folder.uuid)
    assert collection.preset_calls == [str(preset.uuid)]
    # The run-scoped preset is transient scaffolding, deleted once evaluated.
    assert client.deleted_presets == [str(preset.uuid)]
    assert receipt.preset_deleted is True
    # Curate never creates folders.
    assert client.created_folders == []
    payload = storage.written(CURATE_RECEIPT_URI)
    assert payload["schema"] == "npa.encord.curate_receipt.v1"
    assert payload["items_selected"] == 2 and payload["items_total"] == 1


def test_run_curate_planned_receipt_lands_before_the_first_mutation(
    _fast_polling,
) -> None:
    """The write-ahead receipt must exist before Encord is touched."""

    storage = FakeStorage()

    class MutationChecksReceipt(FakeUserClient):
        def create_collection(self, **kwargs):
            planned = storage.written(CURATE_RECEIPT_URI)
            assert planned["status"] == "planned"
            assert planned["preset_name"] == "npa-curate-run-2"
            assert planned["filters"][0]["metric"] == "width"
            return super().create_collection(**kwargs)

    folder = FakeFolder(name="src")
    folder.folder_items = [FakeItem(_uuid(30), "seed.png", None)]
    client = MutationChecksReceipt(folders=[folder])
    client.pending_curate_items = [FakeItem(_uuid(31), "a.png", None)]
    receipt = run_curate(**_curate_kwargs(storage, client, workflow_run="run-2"))
    assert receipt.status == "done"
    assert [uri for _, uri in storage.uploads] == [CURATE_RECEIPT_URI, CURATE_RECEIPT_URI]


def test_run_curate_reuses_existing_collection(_fast_polling) -> None:
    existing = FakeCollection([])
    existing.name = "keepers-new"
    existing.pending = [FakeItem(_uuid(33), "c.png", None)]
    client, _ = _curate_client(collection=existing)
    receipt = run_curate(**_curate_kwargs(FakeStorage(), client))
    assert receipt.collection_created is False
    assert receipt.collection_uuid == str(existing.uuid)
    assert receipt.items_selected == 1
    assert client.created_collections == []


def test_run_curate_refuses_a_populated_collection(_fast_polling) -> None:
    """A stale selection would read as this run's; fail closed instead."""

    existing = FakeCollection([FakeItem(_uuid(36), "old.png", None)])
    existing.name = "keepers-new"
    client, _ = _curate_client(collection=existing)
    storage = FakeStorage()
    with pytest.raises(EncordToolError, match="already holds items"):
        run_curate(**_curate_kwargs(storage, client))
    assert client.created_presets == []
    assert existing.preset_calls == []
    assert storage.written(CURATE_RECEIPT_URI)["status"] == "failed"


def test_run_curate_reissues_evaluation_until_indexing_catches_up(_fast_polling) -> None:
    # Observed live: add_preset_items evaluates once, and items pushed moments
    # earlier are not metric-indexed yet — the re-issue loop must recover.
    client, _ = _curate_client()
    client.pending_curate_items = [FakeItem(_uuid(35), "late.png", None)]
    client.curate_reveal_after_calls = 3
    receipt = run_curate(**_curate_kwargs(FakeStorage(), client, poll_seconds=5.0))
    assert receipt.items_selected == 1
    (collection,) = client.created_collections
    assert len(collection.preset_calls) >= 3


def test_run_curate_zero_selection_fails_closed_with_receipt(_fast_polling) -> None:
    client, _ = _curate_client()
    client.pending_curate_items = []
    storage = FakeStorage()
    with pytest.raises(EncordToolError) as excinfo:
        run_curate(
            **_curate_kwargs(
                storage,
                client,
                filters=["brightness:0.2:0.8"],
                poll_seconds=0.05,
            )
        )
    message = str(excinfo.value)
    assert "selected 0 items" in message
    # The diagnostic names the computed-metric cause when one is in play.
    assert "quality metrics have been computed" in message
    payload = storage.written(CURATE_RECEIPT_URI)
    assert payload["status"] == "empty"
    assert payload["items_selected"] == 0 and payload["items_total"] == 1
    # The transient preset is gone even though the run failed.
    assert payload["preset_deleted"] is True and len(client.deleted_presets) == 1


def test_run_curate_deletes_the_preset_when_evaluation_raises() -> None:
    """A crash after create_preset must not leave the transient preset behind."""

    class ExplodingCollection(FakeCollection):
        def add_preset_items(self, filter_preset) -> None:
            raise RuntimeError("502 from Encord")

    class ExplodingClient(FakeUserClient):
        def create_collection(self, *, top_level_folder_uuid, name, description=""):
            collection = ExplodingCollection([])
            collection.name = name
            self.created_collections.append(collection)
            return collection

    folder = FakeFolder(name="src")
    folder.folder_items = [FakeItem(_uuid(30), "seed.png", None)]
    client = ExplodingClient(folders=[folder])
    storage = FakeStorage()
    with pytest.raises(EncordToolError, match="502 from Encord"):
        run_curate(**_curate_kwargs(storage, client))
    (preset,) = client.created_presets
    assert client.deleted_presets == [str(preset.uuid)]
    payload = storage.written(CURATE_RECEIPT_URI)
    assert payload["status"] == "failed" and payload["preset_deleted"] is True
    assert payload["preset_uuid"] == str(preset.uuid)


def test_run_curate_records_a_failed_preset_delete(_fast_polling) -> None:
    class StickyPresets(FakeUserClient):
        def delete_preset(self, preset_uuid):
            raise RuntimeError("403 preset delete")

    folder = FakeFolder(name="src")
    folder.folder_items = [FakeItem(_uuid(30), "seed.png", None)]
    client = StickyPresets(folders=[folder])
    client.pending_curate_items = [FakeItem(_uuid(31), "a.png", None)]
    receipt = run_curate(**_curate_kwargs(FakeStorage(), client))
    assert receipt.status == "done"
    assert receipt.preset_deleted is False  # cleanup by prefix is owed


def test_run_curate_empty_folder_fails_fast_before_any_scaffolding() -> None:
    client = FakeUserClient(folders=[FakeFolder(name="src")])  # no folder items
    storage = FakeStorage()
    with pytest.raises(EncordToolError, match="contains no storage items"):
        run_curate(**_curate_kwargs(storage, client))
    assert client.created_collections == []
    assert client.created_presets == []
    payload = storage.written(CURATE_RECEIPT_URI)
    assert payload["status"] == "failed"
    assert payload["items_total"] == 0
    assert "contains no storage items" in payload["error"]


def test_run_curate_unknown_metric_fails_before_any_encord_call() -> None:
    client = FakeUserClient(folders=[FakeFolder(name="src")])
    storage = FakeStorage()
    with pytest.raises(EncordToolError, match="Unknown filter metric"):
        run_curate(**_curate_kwargs(storage, client, filters=["blur:0:1"]))
    assert client.created_presets == []
    assert client.created_collections == []
    assert storage.objects == {}  # not even a planned receipt


def test_run_curate_missing_folder_writes_receipt_then_raises() -> None:
    client = FakeUserClient(folders=[])
    storage = FakeStorage()
    with pytest.raises(EncordToolError, match="No Encord storage folder"):
        run_curate(**_curate_kwargs(storage, client, folder="absent"))
    payload = storage.written(CURATE_RECEIPT_URI)
    assert payload["status"] == "failed"
    assert "No Encord storage folder" in payload["error"]
    assert client.created_folders == []


def test_resolve_folder_create_flag_fails_closed() -> None:
    client = FakeUserClient(folders=[])
    with pytest.raises(EncordToolError, match="No Encord storage folder"):
        resolve_folder(client, "absent", create=False)
    assert client.created_folders == []


def test_resolve_collection_create_paths() -> None:
    client = FakeUserClient()
    with pytest.raises(EncordToolError, match="No Encord collection"):
        resolve_collection(client, "fresh")
    ref = resolve_collection(client, "fresh", create_in_folder_uuid=_uuid(1))
    assert ref.created is True and ref.title == "fresh"
    assert ref.id == str(ref.obj.uuid)
    # Idempotent re-resolution finds the created collection instead.
    again = resolve_collection(client, "fresh")
    assert again.created is False and again.id == ref.id


# --- idempotency + isolation ---------------------------------------------------


def test_run_push_repush_is_a_no_op_on_the_wire() -> None:
    """Retry-safety is our invariant: nothing already present is re-sent."""

    storage = FakeStorage(["p/a.mp4"])
    folder = FakeFolder()
    folder.folder_items = [_folder_item(61, "p/a.mp4")]
    client = FakeUserClient(folders=[folder])
    receipt = run_push(**_push_kwargs(storage, client, folder=str(folder.uuid)))
    assert receipt.status == "done"
    assert receipt.units_done == 1
    assert folder.start_calls == []  # no registration round-trip at all
    assert receipt.items[0].item_uuid == _uuid(61)


def test_run_push_upload_mode_repush_skips_duplicate_byte_copies() -> None:
    storage = FakeDownloadStorage(["p/a.mp4"])
    folder = FakeUploadFolder()
    folder.folder_items = [_folder_item(62, "p/a.mp4", url=False)]
    client = FakeUserClient(folders=[folder])
    receipt = run_push(
        input_path="s3://bkt/p/",
        integration="",
        folder=str(folder.uuid),
        transfer="upload",
        output_path=RECEIPT_URI,
        user_client=client,
        storage_client=storage,
        environ={},
    )
    assert receipt.status == "done" and receipt.units_done == 1
    assert storage.downloads == []  # no bytes moved on re-push
    assert folder.uploads == []
    assert receipt.items[0].item_uuid == _uuid(62)
    assert receipt.items[0].status == "uploaded"


def test_register_mode_never_falls_through_to_upload() -> None:
    """Register failures must never copy customer bytes into the SaaS."""

    storage = FakeStorage(["p/a.mp4"])
    folder = FakeUploadFolder()  # records any upload_* call
    folder.results = [FakePollResult(status="ERROR")]
    client = FakeUserClient(folders=[folder])
    with pytest.raises(EncordToolError):
        run_push(**_push_kwargs(storage, client, folder=str(folder.uuid)))
    assert folder.uploads == []


# --- cleanup ---------------------------------------------------------------------


def test_run_cleanup_deletes_prefix_scoped_state_and_reports_datasets() -> None:
    folder = FakeFolder(name="npa-e2e-run1")
    folder.folder_items = [_folder_item(71, "p/a.mp4")]
    client = FakeUserClient(folders=[folder])
    collection = client.create_collection(
        top_level_folder_uuid=str(folder.uuid), name="npa-e2e-run1"
    )
    preset = client.create_preset(name="npa-curate-run1", filter_preset_json={})
    client.create_preset(name="customer-preset", filter_preset_json={})
    client.datasets[_uuid(72)] = FakeDataset(_uuid(72), "npa-e2e-run1")
    client.datasets[_uuid(73)] = FakeDataset(_uuid(73), "customer-data")

    summary = run_cleanup(title_prefix="npa-e2e-", user_client=client)
    assert summary["folders_deleted"] == ["npa-e2e-run1"]
    assert summary["items_deleted"] == 1
    assert folder.deleted is True and folder.deleted_item_uuids == [_uuid(71)]
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


# --- exact identity ------------------------------------------------------------


def test_canonical_s3_uri_never_aliases_keys() -> None:
    # A literal percent triplet in a key must not alias the slash form.
    assert canonical_s3_uri("bkt", "a%2Fb.png") != canonical_s3_uri("bkt", "a/b.png")
    assert canonical_s3_uri("bkt", "runs/a b.png") == "s3://bkt/runs/a%20b.png"
    for bad in ("", "a/", "/a", "a/../b", "a//b"):
        with pytest.raises(EncordToolError):
            canonical_s3_uri("bkt", bad)


def test_normalize_object_url_is_identity_preserving() -> None:
    a = normalize_object_url("HTTPS://Host.example/bkt/p/a%2eмp4")
    b = normalize_object_url("https://host.example/bkt/p/a.мp4")
    assert a == b  # unreserved escapes normalize
    # reserved escapes are preserved: %2F is not a path separator
    assert normalize_object_url("https://h/bkt/a%2Fb") != normalize_object_url(
        "https://h/bkt/a/b"
    )
    with pytest.raises(EncordToolError):
        normalize_object_url("https://user:pw@h/bkt/a")


def test_resolve_exact_identity_prefers_metadata_and_detects_conflicts() -> None:
    meta_item = SimpleNamespace(
        uuid=_uuid(81),
        client_metadata={"npa": {"source_uri": "s3://bkt/p/a.mp4"}},
        url="",
    )
    url_item = SimpleNamespace(
        uuid=_uuid(82), client_metadata={}, url=f"{ENDPOINT}/bkt/p/a.mp4"
    )
    resolution = resolve_exact_identity(
        source_uri="s3://bkt/p/a.mp4",
        submitted_object_url=f"{ENDPOINT}/bkt/p/a.mp4",
        candidates=[meta_item, url_item],
    )
    # Two different uuids both claiming the source is a conflict, not a pick.
    assert resolution.error_code == "identity_conflict"
    resolution = resolve_exact_identity(
        source_uri="s3://bkt/p/a.mp4",
        submitted_object_url=f"{ENDPOINT}/bkt/p/a.mp4",
        candidates=[meta_item],
    )
    assert resolution.resolved and resolution.signal == "metadata"
    resolution = resolve_exact_identity(
        source_uri="s3://bkt/p/other.mp4",
        submitted_object_url="",
        candidates=[meta_item, url_item],
    )
    assert resolution.error_code == "identity_unresolved"


# --- integrity + verify ---------------------------------------------------------


def test_etag_checksum_and_compare() -> None:
    assert etag_checksum('"a" ') == ("", "none")
    md5 = "d41d8cd98f00b204e9800998ecf8427e"
    assert etag_checksum(f'"{md5}"') == (md5, "md5")
    assert etag_checksum(f'"{md5}-3"') == ("", "none")  # multipart is not a digest
    assert compare_checksums(md5, "md5", md5.upper(), "md5") is True
    assert compare_checksums(md5, "md5", "deadbeef" * 4, "md5") is False
    assert compare_checksums(md5, "md5", "abc", "sha256") is None
    assert compare_checksums("", "none", md5, "md5") is None


def test_write_hashed_stream_digest(tmp_path: Path) -> None:
    import hashlib

    dest = tmp_path / "out.bin"
    digest = write_hashed_stream([b"abc", b"", b"def"], dest)
    assert dest.read_bytes() == b"abcdef"
    assert digest.size == 6
    assert digest.sha256 == hashlib.sha256(b"abcdef").hexdigest()
    assert hash_file(dest) == digest


def _verify_fixtures(
    storage: FakeStorage,
    *,
    receipt_overrides=None,
    pulled_overrides=None,
    drop_pulled=False,
    drop_pushed=False,
) -> tuple[str, str]:
    """Write a receipt + manifest pair into the fake object store; return URIs."""

    sha = "a" * 64
    receipt = {
        "schema": "npa.encord.push_receipt.v1",
        "generated_at": "t",
        "input_uri": "s3://bkt/p/",
        "endpoint_url": ENDPOINT,
        "encord_domain": "https://api.encord.com",
        "folder_name": "f",
        "media_filter": "videos-images",
        "status": "done",
        "items": []
        if drop_pushed
        else [
            {
                "key": "p/a.mp4",
                "source_uri": "s3://bkt/p/a.mp4",
                "category": "videos",
                "item_uuid": _uuid(90),
                "status": "uploaded",
                "source_size": 6,
                "source_checksum": sha,
                "source_checksum_kind": "sha256",
            }
        ],
    }
    receipt.update(receipt_overrides or {})
    pulled = {
        "item_uuid": _uuid(90),
        "name": "p/a.mp4",
        "transfer": "download",
        "observed_size": 6,
        "checksum": sha,
        "checksum_kind": "sha256",
    }
    pulled.update(pulled_overrides or {})
    manifest = {
        "schema": "npa.encord.pull_manifest.v1",
        "generated_at": "t",
        "encord_domain": "https://api.encord.com",
        "source_kind": "dataset",
        "source_id": "d",
        "output_uri": "s3://bkt/out/",
        "items": [] if drop_pulled else [pulled],
    }
    receipt_uri = "s3://bkt/push/push_receipt.json"
    manifest_uri = "s3://bkt/pull/manifest.json"
    write_json(receipt, result_uri=receipt_uri, filename="push_receipt.json", storage_client=storage)
    write_json(manifest, result_uri=manifest_uri, filename="manifest.json", storage_client=storage)
    return receipt_uri, manifest_uri


def _verify(storage: FakeStorage, receipt_uri: str, manifest_uri: str):
    return run_verify(
        receipt_uri=receipt_uri,
        manifest_uri=manifest_uri,
        output_path=REPORT_URI,
        storage_client=storage,
    )


def test_run_verify_passes_on_exact_match() -> None:
    storage = FakeStorage()
    report = _verify(storage, *_verify_fixtures(storage))
    assert report.status == "passed"
    assert report.expected == report.matched == 1
    assert report.checksum_verified == 1 and report.checksum_mismatched == 0
    assert report.defects == []
    payload = storage.written(REPORT_URI)
    assert payload["schema"] == "npa.encord.roundtrip_report.v1"


def test_run_verify_fails_closed_on_checksum_mismatch() -> None:
    storage = FakeStorage()
    uris = _verify_fixtures(storage, pulled_overrides={"checksum": "b" * 64})
    with pytest.raises(EncordToolError, match="1 checksum mismatched"):
        _verify(storage, *uris)
    payload = storage.written(REPORT_URI)
    assert payload["status"] == "failed"
    assert payload["items"][0]["checksum_state"] == "mismatched"


def test_run_verify_fails_closed_on_missing_item() -> None:
    storage = FakeStorage()
    uris = _verify_fixtures(storage, drop_pulled=True)
    with pytest.raises(EncordToolError, match="1 missing"):
        _verify(storage, *uris)


@pytest.mark.parametrize("receipt_status", ["planned", "failed", "timeout"])
def test_run_verify_fails_closed_when_the_push_never_completed(receipt_status: str) -> None:
    """A write-ahead or failed receipt is not evidence of a roundtrip."""

    storage = FakeStorage()
    uris = _verify_fixtures(storage, receipt_overrides={"status": receipt_status})
    with pytest.raises(EncordToolError, match=f"status is {receipt_status!r}, not 'done'"):
        _verify(storage, *uris)
    payload = storage.written(REPORT_URI)
    assert payload["status"] == "failed"
    assert any("never completed" in defect for defect in payload["defects"])
    # The per-item join still ran and matched: the defect is receipt-level.
    assert payload["matched"] == 1


def test_run_verify_fails_closed_on_zero_attributable_items() -> None:
    """0/0 matched must never read as passed."""

    storage = FakeStorage()
    uris = _verify_fixtures(storage, drop_pushed=True, drop_pulled=True)
    with pytest.raises(EncordToolError, match="no attributable items"):
        _verify(storage, *uris)
    payload = storage.written(REPORT_URI)
    assert payload["status"] == "failed"
    assert payload["expected"] == payload["matched"] == 0


def test_run_verify_fails_closed_when_an_item_has_no_evidence_at_all() -> None:
    """Multipart ETags on both sides and no size from Encord verify nothing."""

    storage = FakeStorage()
    uris = _verify_fixtures(
        storage,
        receipt_overrides={
            "items": [
                {
                    "key": "p/a.mp4",
                    "source_uri": "s3://bkt/p/a.mp4",
                    "category": "videos",
                    "item_uuid": _uuid(90),
                    "status": "registered",
                    "source_size": 6,
                    "source_checksum": "",
                    "source_checksum_kind": "none",
                }
            ]
        },
        pulled_overrides={
            "transfer": "copy",
            "observed_size": 0,
            "file_size": 0,
            "checksum": "",
            "checksum_kind": "none",
        },
    )
    with pytest.raises(EncordToolError, match="1 unverifiable"):
        _verify(storage, *uris)
    payload = storage.written(REPORT_URI)
    assert payload["status"] == "failed" and payload["unverifiable"] == 1
    assert payload["items"][0]["reasons"] == [
        "unverifiable: no comparable checksum and no size on both sides"
    ]


def test_run_verify_incomparable_kinds_are_unavailable_not_failures() -> None:
    # A multipart-source object vs a sha256 download: no comparison exists.
    storage = FakeStorage()
    uris = _verify_fixtures(
        storage, pulled_overrides={"checksum": "c" * 32, "checksum_kind": "md5"}
    )
    report = _verify(storage, *uris)
    assert report.status == "passed"
    assert report.checksum_unavailable == 1


# --- artifact storage ------------------------------------------------------------


def test_write_json_rejects_non_s3_destinations(tmp_path: Path) -> None:
    with pytest.raises(EncordToolError, match="expected an s3:// URI"):
        write_json(
            {"a": 1},
            result_uri=str(tmp_path / "receipt.json"),
            filename="receipt.json",
            storage_client=FakeStorage(),
        )


# --- seed-demo ------------------------------------------------------------------


def test_seed_demo_skips_when_operator_supplied_a_curated_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import npa.workflows.data_factory_input as dfi

    def explode(*args, **kwargs):
        raise AssertionError("seed must not fetch when skipping")

    monkeypatch.setattr(dfi, "_fetch_starter", explode)
    summary = run_seed_demo(
        media_uri="s3://bkt/run/seed/",
        dataset="npa-demo-src-run",
        active_source_id="my-curated-collection",
        storage_client=FakeStorage(),
    )
    assert summary["skipped"]


def test_seed_demo_requires_an_integration_before_fetching_anything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import npa.workflows.data_factory_input as dfi

    def explode(*args, **kwargs):
        raise AssertionError("seed must not fetch when its arguments are invalid")

    monkeypatch.setattr(dfi, "_fetch_starter", explode)
    storage = FakeStorage()
    with pytest.raises(EncordToolError, match="--integration is required"):
        run_seed_demo(
            media_uri="s3://bkt/run/seed/",
            dataset="npa-demo-src-run",
            active_source_id="npa-demo-src-run",
            storage_client=storage,
        )
    assert storage.uploads == []


def test_seed_demo_fetches_uploads_and_pushes_the_starter_clip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import npa.workbench.encord.push as push_module
    import npa.workflows.data_factory_input as dfi

    clip = tmp_path / "starter.mp4"
    clip.write_bytes(b"starter-bytes")
    monkeypatch.setattr(dfi, "_fetch_starter", lambda contract, **kw: (clip, "verified_hit"))
    monkeypatch.setattr(
        dfi,
        "load_starter_contract",
        lambda: {"integrity": {"sha256": "a" * 64}, "license": {"name": "CC-BY-4.0"}},
    )
    pushed: dict = {}

    def fake_run_push(**kwargs):
        pushed.update(kwargs)
        return SimpleNamespace(units_done=1)

    monkeypatch.setattr(push_module, "run_push", fake_run_push)
    storage = FakeStorage()
    summary = run_seed_demo(
        media_uri="s3://bkt/run/seed/",
        dataset="npa-demo-src-run",
        active_source_id="npa-demo-src-run",
        integration="nebius-s3",
        storage_client=storage,
    )
    assert storage.uploads[0][1] == "s3://bkt/run/seed/starter-clip.mp4"
    assert pushed["dataset"] == pushed["folder"] == "npa-demo-src-run"
    assert pushed["input_path"] == "s3://bkt/run/seed/"
    assert pushed["output_path"] == "s3://bkt/run/seed/push/"
    # The push default (register) holds here too: no bytes enter the SaaS
    # unless the spec or operator asks for upload.
    assert pushed["transfer"] == "register" and pushed["integration"] == "nebius-s3"
    assert summary["units_done"] == 1 and summary["attribution"] == "CC-BY-4.0"
    assert summary["transfer"] == "register"
