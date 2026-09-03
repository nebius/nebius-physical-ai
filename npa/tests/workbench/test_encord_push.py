"""`encord push`: discovery, register/upload transport, receipts, idempotency."""

from __future__ import annotations

from pathlib import Path

import pytest

from encord_fakes import (
    ENDPOINT,
    ENVIRON,
    RECEIPT_URI,
    FakeDataset,
    FakeDownloadStorage,
    FakeFolder,
    FakePollResult,
    FakeStorage,
    FakeUploadFolder,
    FakeUserClient,
    fake_uuid,
    folder_item,
)
from npa.workbench.encord.push import (
    BATCH_SIZE,
    build_upload_json,
    discover_objects,
    object_url_for,
    push_receipt_uri_for,
    run_push,
)
from npa.workbench.encord.schemas import EncordToolError, PushedItem

# --- url + discovery ---------------------------------------------------------

def test_object_url_for_is_path_style_and_encoded() -> None:
    url = object_url_for(ENDPOINT + "/", "bkt", "runs/a b/frame#1.png")
    assert url == f"{ENDPOINT}/bkt/runs/a%20b/frame%231.png"
    assert " " not in url


def test_receipt_uri_helper() -> None:
    assert push_receipt_uri_for("s3://b/p/") == "s3://b/p/push_receipt.json"
    assert push_receipt_uri_for("s3://b/p/r.json") == "s3://b/p/r.json"

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


def test_run_push_happy_path_links_dataset() -> None:
    storage = FakeStorage(["p/a.mp4", "p/b.png"])
    folder = FakeFolder(
        results=[
            FakePollResult(
                status="DONE",
                done=2,
                items=[(fake_uuid(21), "p/a.mp4"), (fake_uuid(22), "b.png")],
            )
        ]
    )
    # After registration the folder inventory exposes the items with their
    # npa.source_uri clientMetadata — the only lineage signal besides the URL.
    folder.post_registration_items = [folder_item(21, "p/a.mp4"), folder_item(22, "p/b.png")]
    client = FakeUserClient(folders=[])
    client.create_storage_folder = lambda name, description="": folder  # type: ignore[assignment]
    receipt = run_push(**_push_kwargs(storage, client, dataset="new-ds"))
    assert receipt.status == "done"
    assert receipt.units_done == 2 and receipt.units_error == 0
    assert receipt.dataset_created is True and receipt.linked_count == 2
    dataset = next(iter(client.datasets.values()))
    assert dataset.linked == [[fake_uuid(21), fake_uuid(22)]]
    # uuids attached to receipt rows by exact metadata identity, never names
    assert {item.item_uuid for item in receipt.items} == {fake_uuid(21), fake_uuid(22)}
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
        folder_item(1000 + index, key) for index, key in enumerate(keys)
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
    folder.post_registration_items = [folder_item(21, "p/a.mp4")]

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


# --- upload mode + crash paths ------------------------------------------------


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
        folder_item(61, "p/a.mp4", metadata=False),
        folder_item(62, "p/b.png", metadata=False),
    ]
    client = FakeUserClient(folders=[folder])
    receipt = run_push(**_push_kwargs(storage, client, folder=str(folder.uuid), dataset="new-ds"))
    dataset = next(iter(client.datasets.values()))
    assert sorted(dataset.linked[0]) == [fake_uuid(61), fake_uuid(62)]
    assert receipt.linked_count == 2
    assert {item.item_uuid for item in receipt.items} == {fake_uuid(61), fake_uuid(62)}
    assert {item.identity_signal for item in receipt.items} == {"object_url"}


def test_run_push_shared_basenames_resolve_to_distinct_items() -> None:
    """The identity-regression case: same basename, different objects.

    Exact identity (metadata/objectUrl) attributes each receipt row to its own
    Encord item; a name-based scheme could not tell these apart.
    """

    storage = FakeStorage(["p/left/clip.mp4", "p/right/clip.mp4"])
    folder = FakeFolder(results=[FakePollResult(status="DONE", done=2, items=[])])
    folder.post_registration_items = [
        folder_item(64, "p/left/clip.mp4"),
        folder_item(65, "p/right/clip.mp4"),
    ]
    client = FakeUserClient(folders=[folder])
    receipt = run_push(
        **_push_kwargs(storage, client, folder=str(folder.uuid), dataset="new-ds")
    )
    dataset = next(iter(client.datasets.values()))
    assert sorted(dataset.linked[0]) == [fake_uuid(64), fake_uuid(65)]
    by_key = {item.key: item.item_uuid for item in receipt.items}
    assert by_key == {"p/left/clip.mp4": fake_uuid(64), "p/right/clip.mp4": fake_uuid(65)}


def test_run_push_identity_conflict_fails_the_item_closed() -> None:
    """Two folder items claiming one source is a conflict, never a guess."""

    storage = FakeStorage(["p/a.mp4"])
    folder = FakeFolder(results=[FakePollResult(status="DONE", done=1, items=[])])
    duplicate = folder_item(67, "p/a.mp4")
    duplicate.uuid = fake_uuid(68)
    folder.post_registration_items = [folder_item(67, "p/a.mp4"), duplicate]
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
        results=[FakePollResult(status="DONE", done=1, items=[(fake_uuid(63), "p/a.mp4")])]
    )
    folder.post_registration_items = [folder_item(63, "p/a.mp4")]
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


# --- idempotency + isolation ---------------------------------------------------


def test_run_push_repush_is_a_no_op_on_the_wire() -> None:
    """Retry-safety is our invariant: nothing already present is re-sent."""

    storage = FakeStorage(["p/a.mp4"])
    folder = FakeFolder()
    folder.folder_items = [folder_item(61, "p/a.mp4")]
    client = FakeUserClient(folders=[folder])
    receipt = run_push(**_push_kwargs(storage, client, folder=str(folder.uuid)))
    assert receipt.status == "done"
    assert receipt.units_done == 1
    assert folder.start_calls == []  # no registration round-trip at all
    assert receipt.items[0].item_uuid == fake_uuid(61)


def test_run_push_upload_mode_repush_skips_duplicate_byte_copies() -> None:
    storage = FakeDownloadStorage(["p/a.mp4"])
    folder = FakeUploadFolder()
    folder.folder_items = [folder_item(62, "p/a.mp4", url=False)]
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
    assert receipt.items[0].item_uuid == fake_uuid(62)
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


