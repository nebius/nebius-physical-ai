from __future__ import annotations

import hashlib
import json
import sys
import base64
from pathlib import Path

import pytest

from npa.workbench.encord.pull import (
    _preallocate_media,
    _same_endpoint_source,
    _transfer_one,
    run_pull,
)
from npa.workbench.encord.schemas import (
    EncordToolError,
    OutcomeCounts,
    PushItem,
    PushReceipt,
)
from npa.workbench.encord.verify import verify_roundtrip

sys.path.insert(0, str(Path(__file__).parents[1]))

from encord_fakes import (  # noqa: E402
    BytesDownloader,
    FakeStorageClient,
    FakeStorageItem,
    MemoryArtifactStore,
)


class Collection:
    def __init__(self, items) -> None:
        self.items = items
        self.name = "collection"

    def list_items(self, **kwargs):
        del kwargs
        return self.items


class CollectionClient:
    def __init__(self, items) -> None:
        self.collection = Collection(items)

    def get_collection(self, value):
        del value
        return self.collection


class Dataset:
    title = "dataset"

    def __init__(self, rows) -> None:
        self.rows = rows

    def list_data_rows(self):
        return self.rows


class DatasetClient:
    def __init__(self, rows) -> None:
        self.dataset = Dataset(rows)

    def get_datasets(self, **kwargs):
        del kwargs
        from types import SimpleNamespace

        return [{"dataset": SimpleNamespace(dataset_hash="dataset-id")}]

    def get_dataset(self, *args, **kwargs):
        del args, kwargs
        return self.dataset


class SequenceDownloader:
    def __init__(self, values: dict[str, bytes | Exception]) -> None:
        self.values = values

    def download(self, url: str, destination: Path):
        value = self.values[url]
        if isinstance(value, Exception):
            raise value
        destination.write_bytes(value)
        from npa.workbench.encord.integrity import StreamDigest

        return StreamDigest(len(value), hashlib.sha256(value).hexdigest())


def item(uuid: str, name: str, url: str, size: int = 5) -> FakeStorageItem:
    return FakeStorageItem(
        uuid=uuid,
        name=name,
        signed_url=url,
        file_size=size,
        client_metadata={
            "npa": {"source_uri": f"s3://source-bucket/incoming/{name}"}
        },
    )


def test_pull_preallocates_one_row_per_discovered_item() -> None:
    rows = [
        item("uuid-1", "a.mp4", "https://external.example/a"),
        item("uuid-2", "b.mp4", "https://external.example/b"),
        item("uuid-3", "c.mp4", "https://external.example/c"),
    ]
    storage = FakeStorageClient()
    store = MemoryArtifactStore()
    with pytest.raises(EncordToolError):
        run_pull(
            source="collection",
            source_id="00000000-0000-0000-0000-000000000001",
            output_path="s3://result-bucket/run",
            user_client=CollectionClient(rows),
            storage_client=storage,
            artifact_store=store,
            downloader=SequenceDownloader(
                {
                    "https://external.example/a": b"aaaaa",
                    "https://external.example/b": OSError("injected download failure"),
                    "https://external.example/c": b"ccccc",
                }
            ),
            environ={},
        )
    payload = json.loads(store.payloads["s3://result-bucket/run/manifest.json"])
    assert [row["outcome"] for row in payload["items"]] == [
        "successful",
        "failed",
        "unattempted",
    ]
    assert payload["counts"] == {
        "attempted": 2,
        "discovered": 3,
        "failed": 1,
        "successful": 1,
        "unattempted": 1,
        "unresolved": 0,
    }


def test_download_records_sha256_while_streaming() -> None:
    storage = FakeStorageClient()
    descriptor = item("uuid-1", "clip.mp4", "https://external.example/clip")
    row = _preallocate_media(descriptor, "s3://result-bucket/run")
    _transfer_one(
        descriptor,
        row,
        storage_client=storage,
        downloader=BytesDownloader(b"video"),
        endpoint_url="",
    )
    assert row.outcome == "successful"
    assert row.source_checksum_kind == "sha256"
    assert row.destination_checksum_kind == "etag_opaque"
    assert row.source_checksum == hashlib.sha256(b"video").hexdigest()


def test_server_side_copy_heads_destination() -> None:
    storage = FakeStorageClient()
    storage.s3.objects[("source-bucket", "incoming/clip.mp4")] = b"video"
    descriptor = item(
        "uuid-1",
        "clip.mp4",
        "https://storage.test.example/source-bucket/incoming/clip.mp4?signature=x",
    )
    row = _preallocate_media(descriptor, "s3://result-bucket/run")
    _transfer_one(
        descriptor,
        row,
        storage_client=storage,
        downloader=BytesDownloader(b"unused"),
        endpoint_url="https://storage.test.example",
    )
    assert row.transfer == "copy"
    assert row.destination_exists
    assert any(event[0] == "copy" for event in storage.s3.events)
    assert storage.s3.events[-2][0] == "head" or storage.s3.events[-1][0] == "put"


@pytest.mark.parametrize(
    ("encoded_path", "expected_key"),
    [
        ("incoming/a%25b.mp4", "incoming/a%b.mp4"),
        ("incoming/a%2Fb.mp4", "incoming/a/b.mp4"),
        ("incoming/a%252Fb.mp4", "incoming/a%2Fb.mp4"),
        ("incoming/a%25252Fb.mp4", "incoming/a%252Fb.mp4"),
        ("incoming/a%20b.mp4", "incoming/a b.mp4"),
        ("incoming/a+b.mp4", "incoming/a+b.mp4"),
    ],
)
def test_same_endpoint_source_decodes_signed_path_exactly_once(
    encoded_path: str, expected_key: str
) -> None:
    found = _same_endpoint_source(
        f"https://storage.test.example/source-bucket/{encoded_path}"
        "?X-Amz-Signature=ignored%2Fquery",
        "https://storage.test.example",
    )
    assert found == ("source-bucket", expected_key)


def test_literal_percent_key_copy_and_wrong_object_verification_fail_closed() -> None:
    source_uri = "s3://source-bucket/incoming/a%252Fb.mp4"
    literal_key = "incoming/a%2Fb.mp4"
    slash_key = "incoming/a/b.mp4"
    descriptor = FakeStorageItem(
        uuid="uuid-literal",
        name="a%2Fb.mp4",
        signed_url=(
            "https://storage.test.example/source-bucket/incoming/a%252Fb.mp4"
            "?X-Amz-Signature=ignored"
        ),
        file_size=5,
        client_metadata={"npa": {"source_uri": source_uri}},
    )
    storage = FakeStorageClient()
    storage.s3.objects[("source-bucket", literal_key)] = b"RIGHT"
    storage.s3.objects[("source-bucket", slash_key)] = b"WRONG"
    store = MemoryArtifactStore()
    manifest = run_pull(
        source="collection",
        source_id="00000000-0000-0000-0000-000000000001",
        output_path="s3://result-bucket/run",
        user_client=CollectionClient([descriptor]),
        storage_client=storage,
        artifact_store=store,
        downloader=BytesDownloader(b"unused"),
        environ={"AWS_ENDPOINT_URL": "https://storage.test.example"},
    )
    copy_event = next(event for event in storage.s3.events if event[0] == "copy")
    assert copy_event[3] == {"Bucket": "source-bucket", "Key": literal_key}
    destination_uri = manifest.items[0].destination_uri
    destination_key = destination_uri.removeprefix("s3://result-bucket/")
    assert storage.s3.objects[("result-bucket", destination_key)] == b"RIGHT"
    assert manifest.items[0].source_uri == source_uri
    assert manifest.items[0].outcome == "successful"

    pushed = PushItem(
        source_uri=source_uri,
        bucket="source-bucket",
        object_key=literal_key,
        category="videos",
        submitted_object_url=(
            "https://storage.test.example/source-bucket/incoming/a%252Fb.mp4"
        ),
        source_size=5,
        source_checksum=manifest.items[0].source_checksum,
        source_checksum_kind=manifest.items[0].source_checksum_kind,
        item_uuid="uuid-literal",
        registration_state="existing",
        identity_state="resolved",
        outcome="successful",
    )
    receipt_uri = "s3://result-bucket/run/push_receipt.json"
    store.create_json(
        receipt_uri,
        PushReceipt(
            phase="final",
            status="completed",
            revision=1,
            generated_at="2026-08-30T00:00:00+00:00",
            updated_at="2026-08-30T00:00:00+00:00",
            input_uri="s3://source-bucket/incoming/",
            encord_domain="https://api.encord.com",
            folder_name="folder",
            media_filter="videos-images",
            counts=OutcomeCounts.from_outcomes(["successful"]),
            receipt_uri=receipt_uri,
            receipt_store_kind="s3",
            items=[pushed],
        ).model_dump(by_alias=True),
    )
    storage.s3.objects[("result-bucket", destination_key)] = b"WRONG"
    report_uri = "s3://result-bucket/run/roundtrip_report.json"
    with pytest.raises(EncordToolError, match="verification failed"):
        verify_roundtrip(
            receipt_uri=receipt_uri,
            manifest_uri="s3://result-bucket/run/manifest.json",
            output_path=report_uri,
            artifact_store=store,
            storage_client=storage,
        )
    report = json.loads(store.payloads[report_uri])
    assert report["passed"] is False
    assert report["items"][0]["relation"] == "integrity_failed"
    assert report["items"][0]["integrity_state"] == "not_comparable"


def test_signed_url_identity_conflict_never_falls_back_to_download() -> None:
    descriptor = FakeStorageItem(
        uuid="uuid-literal",
        name="a%2Fb.mp4",
        signed_url=(
            "https://storage.test.example/source-bucket/incoming/a/b.mp4"
            "?X-Amz-Signature=ignored"
        ),
        file_size=5,
        client_metadata={
            "npa": {
                "source_uri": "s3://source-bucket/incoming/a%252Fb.mp4"
            }
        },
    )
    storage = FakeStorageClient()
    row = _preallocate_media(descriptor, "s3://result-bucket/run")
    downloader = BytesDownloader(b"WRONG")
    _transfer_one(
        descriptor,
        row,
        storage_client=storage,
        downloader=downloader,
        endpoint_url="https://storage.test.example",
    )
    assert row.outcome == "failed"
    assert row.error_code == "signed_url_identity_conflict"
    assert downloader.calls == 0
    assert not any(event[0] in {"copy", "upload"} for event in storage.s3.events)


def test_media_write_then_metadata_failure_preserves_media_evidence() -> None:
    storage = FakeStorageClient()
    original = storage.s3.put_object

    def fail_metadata(*, Key, **kwargs):
        if "/items/" in Key:
            raise OSError("metadata unavailable")
        return original(Key=Key, **kwargs)

    storage.s3.put_object = fail_metadata  # type: ignore[method-assign]
    descriptor = item("uuid-1", "clip.mp4", "https://external.example/clip")
    row = _preallocate_media(descriptor, "s3://result-bucket/run")
    _transfer_one(
        descriptor,
        row,
        storage_client=storage,
        downloader=BytesDownloader(b"video"),
        endpoint_url="",
    )
    assert row.destination_exists
    assert row.metadata_state == "failed"
    assert row.outcome == "failed"


def test_same_size_corrupt_upload_fails_compatible_checksum_verification() -> None:
    storage = FakeStorageClient()
    descriptor = item("uuid-1", "clip.mp4", "https://external.example/clip")
    row = _preallocate_media(descriptor, "s3://result-bucket/run")
    destination = ("result-bucket", "run/media/uuid-1__clip.mp4")
    storage.s3.head_overrides[destination] = {
        "ChecksumSHA256": base64.b64encode(hashlib.sha256(b"wrong").digest()).decode(),
        "ChecksumType": "FULL_OBJECT",
    }
    _transfer_one(
        descriptor,
        row,
        storage_client=storage,
        downloader=BytesDownloader(b"video"),
        endpoint_url="",
    )
    assert row.outcome == "failed"
    assert row.error_code == "media_transfer_failed"
    assert "checksum did not verify" in row.error


def test_interrupted_enumeration_retains_every_yielded_row() -> None:
    first = item("uuid-1", "clip.mp4", "https://external.example/clip")

    def interrupted():
        yield first
        raise OSError("enumeration interrupted after one row")

    storage = FakeStorageClient()
    store = MemoryArtifactStore()
    with pytest.raises(EncordToolError, match="enumeration failed"):
        run_pull(
            source="collection",
            source_id="00000000-0000-0000-0000-000000000001",
            output_path="s3://result-bucket/run",
            user_client=CollectionClient(interrupted()),
            storage_client=storage,
            artifact_store=store,
            downloader=BytesDownloader(b"unused"),
            environ={},
        )
    durable = json.loads(store.payloads["s3://result-bucket/run/manifest.json"])
    assert durable["phase"] == "final"
    assert durable["status"] == "failed"
    assert durable["error_code"] == "enumeration_failed"
    assert [row["outcome"] for row in durable["items"]] == ["unattempted"]
    assert not any(event[0] in {"upload", "copy", "put"} for event in storage.s3.events)


def test_dataset_enumeration_is_also_incrementally_durable() -> None:
    from encord_fakes import FakeDataRow

    def interrupted():
        yield FakeDataRow("uuid-1")
        raise OSError("dataset enumeration interrupted")

    store = MemoryArtifactStore()
    with pytest.raises(EncordToolError, match="enumeration failed"):
        run_pull(
            source="dataset",
            source_id="dataset",
            output_path="s3://result-bucket/run",
            user_client=DatasetClient(interrupted()),
            storage_client=FakeStorageClient(),
            artifact_store=store,
            downloader=BytesDownloader(b"unused"),
            environ={},
        )
    durable = json.loads(store.payloads["s3://result-bucket/run/manifest.json"])
    assert durable["error_code"] == "enumeration_failed"
    assert durable["items"][0]["item_uuid"] == "uuid-1"
    assert durable["items"][0]["outcome"] == "unattempted"


def test_pull_checkpoint_failure_stops_after_first_media_mutation() -> None:
    descriptor = item("uuid-1", "clip.mp4", "https://external.example/clip")
    storage = FakeStorageClient()
    store = MemoryArtifactStore(fail_replace=2)
    with pytest.raises(EncordToolError, match="pull checkpoint failed"):
        run_pull(
            source="collection",
            source_id="00000000-0000-0000-0000-000000000001",
            output_path="s3://result-bucket/run",
            user_client=CollectionClient([descriptor]),
            storage_client=storage,
            artifact_store=store,
            downloader=BytesDownloader(b"video"),
            environ={},
        )
    assert ("result-bucket", "run/media/uuid-1__clip.mp4") in storage.s3.objects
    durable = json.loads(store.payloads["s3://result-bucket/run/manifest.json"])
    assert durable["phase"] == "checkpoint"
    assert durable["status"] == "running"
    assert durable["items"][0]["outcome"] == "unattempted"


def test_pull_final_write_failure_keeps_last_successful_checkpoint() -> None:
    descriptor = item("uuid-1", "clip.mp4", "https://external.example/clip")
    store = MemoryArtifactStore(fail_replace=3)
    with pytest.raises(EncordToolError, match="final manifest write failed"):
        run_pull(
            source="collection",
            source_id="00000000-0000-0000-0000-000000000001",
            output_path="s3://result-bucket/run",
            user_client=CollectionClient([descriptor]),
            storage_client=FakeStorageClient(),
            artifact_store=store,
            downloader=BytesDownloader(b"video"),
            environ={},
        )
    durable = json.loads(store.payloads["s3://result-bucket/run/manifest.json"])
    assert durable["phase"] == "checkpoint"
    assert durable["status"] == "running"
    assert durable["items"][0]["outcome"] == "successful"


class LabelRow:
    def __init__(self, uuid: str, *, fail: bool = False) -> None:
        self.backing_item_uuid = uuid
        self.label_hash = f"label-{uuid}"
        self.data_hash = f"data-{uuid}"
        self.fail = fail
        self.initialized = 0

    def initialise_labels(self) -> None:
        self.initialized += 1
        if self.fail:
            raise OSError("label initialization failed")

    def to_encord_dict(self):
        return {"label_hash": self.label_hash, "data_hash": self.data_hash}


class Project:
    title = "project"

    def __init__(self, labels) -> None:
        self.labels = labels

    def list_label_rows_v2(self, **kwargs):
        del kwargs
        return self.labels


class ProjectClient:
    def __init__(self, project, storage_items) -> None:
        self.project = project
        self.storage_items = {row.uuid: row for row in storage_items}

    def get_project(self, value):
        del value
        return self.project

    def get_storage_items(self, uuids, **kwargs):
        del kwargs
        return [self.storage_items[str(uuid)] for uuid in uuids]


def test_label_export_none_is_read_only_default() -> None:
    label = LabelRow("uuid-1")
    media = item("uuid-1", "clip.mp4", "https://external.example/clip")
    manifest = run_pull(
        source="project",
        source_id="00000000-0000-0000-0000-000000000001",
        output_path="s3://result-bucket/run",
        user_client=ProjectClient(Project([label]), [media]),
        storage_client=FakeStorageClient(),
        artifact_store=MemoryArtifactStore(),
        downloader=BytesDownloader(b"video"),
        environ={},
    )
    assert manifest.label_export == "none"
    assert manifest.label_export_remote_mutation is False
    assert label.initialized == 0


def test_label_write_failure_preserves_prior_and_remaining_rows() -> None:
    labels = [LabelRow("uuid-1"), LabelRow("uuid-2", fail=True), LabelRow("uuid-3")]
    media = [
        item("uuid-1", "a.mp4", "https://external.example/a"),
        item("uuid-2", "b.mp4", "https://external.example/b"),
        item("uuid-3", "c.mp4", "https://external.example/c"),
    ]
    store = MemoryArtifactStore()
    with pytest.raises(EncordToolError):
        run_pull(
            source="project",
            source_id="00000000-0000-0000-0000-000000000001",
            output_path="s3://result-bucket/run",
            label_export="initialize",
            user_client=ProjectClient(Project(labels), media),
            storage_client=FakeStorageClient(),
            artifact_store=store,
            downloader=SequenceDownloader(
                {
                    "https://external.example/a": b"aaaaa",
                    "https://external.example/b": b"bbbbb",
                    "https://external.example/c": b"ccccc",
                }
            ),
            environ={},
        )
    payload = json.loads(store.payloads["s3://result-bucket/run/manifest.json"])
    assert [row["outcome"] for row in payload["label_artifacts"]] == [
        "successful",
        "failed",
        "unattempted",
    ]
    assert payload["label_export_remote_mutation"] is True
