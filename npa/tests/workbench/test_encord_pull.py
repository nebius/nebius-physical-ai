"""`encord pull`: server-side copy vs download, per-item error rows, manifests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from encord_fakes import (
    ENDPOINT,
    ENVIRON,
    FakeCollection,
    FakeDataset,
    FakeItem,
    FakeStorage,
    FakeUserClient,
    fake_uuid,
    stub_httpx_stream,
)
from npa.workbench.encord.pull import (
    _same_endpoint_source,
    enumerate_items,
    pull_manifest_uri_for,
    run_pull,
    transfer_item,
)
from npa.workbench.encord.schemas import EncordToolError


def test_manifest_uri_helper() -> None:
    assert pull_manifest_uri_for("s3://b/p/") == "s3://b/p/manifest.json"


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
    item = FakeItem(fake_uuid(31), "a.mp4", f"{ENDPOINT}/bkt/p/a.mp4?sig=1", file_size=5)
    record = transfer_item(
        item, storage_client=storage, output_uri="s3://out/pull", endpoint_url=ENDPOINT
    )
    assert record.transfer == "copy"
    assert storage.s3.copy_calls[0]["CopySource"] == {"Bucket": "bkt", "Key": "p/a.mp4"}
    assert storage.s3.copy_calls[0]["Bucket"] == "out"
    assert record.media_uri.startswith("s3://out/pull/media/")


def test_transfer_item_composite_without_signed_url_is_error() -> None:
    record = transfer_item(
        FakeItem(fake_uuid(32), "group", None),
        storage_client=FakeStorage(),
        output_uri="s3://out/pull",
        endpoint_url=ENDPOINT,
    )
    assert record.transfer == "error" and "signed URL" in record.error


def test_transfer_item_downloads_cross_origin(monkeypatch: pytest.MonkeyPatch) -> None:

    stub_httpx_stream(monkeypatch)
    storage = FakeStorage()
    record = transfer_item(
        FakeItem(fake_uuid(33), "far.mp4", "https://cdn.encord.example/x?sig=1"),
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

    stub_httpx_stream(monkeypatch)
    storage = FakeStorage()

    def failing_copy(**kwargs):
        raise RuntimeError("AccessDenied on CopySource")

    storage.s3.copy_object = failing_copy  # type: ignore[assignment]
    record = transfer_item(
        FakeItem(fake_uuid(34), "a.mp4", f"{ENDPOINT}/bkt/p/a.mp4?sig=1"),
        storage_client=storage,
        output_uri="s3://out/pull",
        endpoint_url=ENDPOINT,
    )
    assert record.transfer == "download"
    assert "AccessDenied" in record.copy_error


def test_enumerate_items_per_source() -> None:
    items = [FakeItem(fake_uuid(41), "a.mp4", "u")]
    collection = FakeCollection(items)
    dataset = FakeDataset()
    dataset.data_rows = [SimpleNamespace(backing_item_uuid=fake_uuid(41))]
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
    good = FakeItem(fake_uuid(51), "a.mp4", f"{ENDPOINT}/bkt/p/a.mp4", file_size=7)
    bad = FakeItem(fake_uuid(52), "b.mp4", None)
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
    assert f"{out_uri}/items/{fake_uuid(51)}.json" in uploaded


def test_run_pull_keeps_every_record_when_one_signed_url_fetch_raises() -> None:
    class UnsignableItem(FakeItem):
        def get_signed_url(self, refetch: bool = False) -> str | None:
            raise RuntimeError("502 from Encord while signing")

    good = FakeItem(fake_uuid(54), "a.mp4", f"{ENDPOINT}/bkt/p/a.mp4", file_size=7)
    bad = UnsignableItem(fake_uuid(55), "b.mp4", None)
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
    failed = next(row for row in manifest["items"] if row["item_uuid"] == fake_uuid(55))
    assert failed["transfer"] == "error" and "502" in failed["error"]


def test_run_pull_happy_path_counts() -> None:
    good = FakeItem(fake_uuid(53), "a.mp4", f"{ENDPOINT}/bkt/p/a.mp4", file_size=7)
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


# --- failure paths ---------------------------------------------------------------


def test_run_pull_writes_manifest_when_labels_crash() -> None:
    class ExplodingProject:
        title = "proj"

        def list_label_rows_v2(self):
            return [SimpleNamespace(backing_item_uuid=fake_uuid(71))]

        def create_bundle(self, bundle_size: int):
            raise RuntimeError("bundle exploded")

    item = FakeItem(fake_uuid(71), "a.mp4", f"{ENDPOINT}/bkt/p/a.mp4", file_size=3)
    client = FakeUserClient(items=[item])
    client.get_project = lambda h: ExplodingProject()  # type: ignore[assignment]
    storage = FakeStorage()
    with pytest.raises(EncordToolError, match="bundle exploded"):
        run_pull(
            source="project",
            source_id=fake_uuid(70),
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
    dataset.data_rows = [LegacyRow(), SimpleNamespace(backing_item_uuid=fake_uuid(41))]
    items = [FakeItem(fake_uuid(41), "a.mp4", "u")]
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
        RefreshingItem(fake_uuid(72), "far.mp4", "unused"),
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


