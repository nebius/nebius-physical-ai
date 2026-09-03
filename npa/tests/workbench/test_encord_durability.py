from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from npa.workbench.encord.push import run_push
from npa.workbench.encord.schemas import EncordToolError
from npa.workbench.encord.storage import ObjectMetadata

sys.path.insert(0, str(Path(__file__).parents[1]))

from encord_fakes import (  # noqa: E402
    FakeDataset,
    FakeFolder,
    FakeStorageClient,
    FakeStorageItem,
    FakeUserClient,
    MemoryArtifactStore,
)

ENV = {"AWS_ENDPOINT_URL": "https://storage.test.example"}


class FailingCreateStore:
    def create_json(self, uri, payload):
        del uri, payload
        raise OSError("injected create failure")

    def replace_json(self, uri, payload, version):
        raise AssertionError((uri, payload, version))

    def read_json(self, uri):
        raise AssertionError(uri)

    def head(self, uri):
        return ObjectMetadata(uri=uri, exists=False)


class RefreshingFolder(FakeFolder):
    def __init__(self, exact: FakeStorageItem) -> None:
        super().__init__([])
        self.exact = exact

    def add_private_data_to_folder_get_result(self, *args, **kwargs):
        result = super().add_private_data_to_folder_get_result(*args, **kwargs)
        self.items = [self.exact]
        result.items_with_names = [
            SimpleNamespace(item_uuid=self.exact.uuid, name="clip.mp4")
        ]
        result.units_done_count = 1
        return result


class PostCreateFetchFailureClient(FakeUserClient):
    def get_datasets(self, **kwargs):
        del kwargs
        return []

    def get_dataset(self, *args, **kwargs):
        del args, kwargs
        if self.created_datasets:
            raise OSError("post-create fetch failed")
        return self.dataset


def storage_with(*keys: str) -> FakeStorageClient:
    storage = FakeStorageClient()
    for key in keys:
        storage.s3.objects[("source-bucket", key)] = f"bytes:{key}".encode()
    return storage


def test_output_preflight_failure_prevents_all_remote_mutation() -> None:
    storage = storage_with("incoming/clip.mp4")
    folder = FakeFolder()
    client = FakeUserClient(folder)
    with pytest.raises(EncordToolError, match="provisional"):
        run_push(
            input_path="s3://source-bucket/incoming/",
            integration="s3",
            folder="folder",
            output_path="s3://result-bucket/run",
            user_client=client,
            storage_client=storage,
            artifact_store=FailingCreateStore(),
            environ=ENV,
        )
    assert folder.registered == []
    assert folder.uploaded == []
    assert client.created_folders == client.created_datasets == 0


def test_register_checkpoint_failure_stops_linking() -> None:
    storage = storage_with("incoming/clip.mp4")
    exact = FakeStorageItem(
        uuid="00000000-0000-0000-0000-000000000061",
        name="clip.mp4",
        url="https://storage.test.example/source-bucket/incoming/clip.mp4",
        client_metadata={
            "npa": {"source_uri": "s3://source-bucket/incoming/clip.mp4"}
        },
    )
    folder = RefreshingFolder(exact)
    dataset = FakeDataset()
    store = MemoryArtifactStore(fail_replace=1)
    with pytest.raises(EncordToolError, match="checkpoint failed"):
        run_push(
            input_path="s3://source-bucket/incoming/",
            integration="s3",
            folder="folder",
            dataset="dataset",
            output_path="s3://result-bucket/run",
            user_client=FakeUserClient(folder, dataset),
            storage_client=storage,
            artifact_store=store,
            environ=ENV,
        )
    assert len(folder.registered) == 1
    assert dataset.linked == []


def test_upload_checkpoints_after_each_synchronous_item() -> None:
    storage = storage_with("incoming/a.mp4", "incoming/b.mp4")
    folder = FakeFolder()
    store = MemoryArtifactStore(fail_replace=1)
    with pytest.raises(EncordToolError, match="checkpoint failed"):
        run_push(
            input_path="s3://source-bucket/incoming/",
            integration="",
            folder="folder",
            output_path="s3://result-bucket/run",
            transfer="upload",
            user_client=FakeUserClient(folder),
            storage_client=storage,
            artifact_store=store,
            environ={},
        )
    assert len(folder.uploaded) == 1


def test_register_failure_never_calls_upload() -> None:
    storage = storage_with("incoming/clip.mp4")
    folder = FakeFolder()

    def fail_start(**kwargs):
        del kwargs
        raise RuntimeError("registration unavailable")

    folder.add_private_data_to_folder_start = fail_start  # type: ignore[method-assign]
    store = MemoryArtifactStore()
    with pytest.raises(EncordToolError):
        run_push(
            input_path="s3://source-bucket/incoming/",
            integration="s3",
            folder="folder",
            output_path="s3://result-bucket/run",
            user_client=FakeUserClient(folder),
            storage_client=storage,
            artifact_store=store,
            environ=ENV,
        )
    assert folder.uploaded == []
    receipt = json.loads(store.payloads["s3://result-bucket/run/push_receipt.json"])
    assert receipt["status"] == "failed"


def test_final_write_failure_is_nonzero_after_mutation() -> None:
    storage = storage_with("incoming/clip.mp4")
    folder = FakeFolder()
    store = MemoryArtifactStore(fail_replace=2)
    with pytest.raises(EncordToolError, match="final receipt write failed"):
        run_push(
            input_path="s3://source-bucket/incoming/",
            integration="",
            folder="folder",
            output_path="s3://result-bucket/run",
            transfer="upload",
            user_client=FakeUserClient(folder),
            storage_client=storage,
            artifact_store=store,
            environ={},
        )
    assert len(folder.uploaded) == 1
    durable = json.loads(store.payloads["s3://result-bucket/run/push_receipt.json"])
    assert durable["phase"] == "checkpoint"
    assert durable["status"] == "running"


def test_exact_existing_item_is_verified_after_dataset_link() -> None:
    storage = storage_with("incoming/clip.mp4")
    exact = FakeStorageItem(
        uuid="00000000-0000-0000-0000-000000000061",
        name="unrelated-display-name.mp4",
        url="https://storage.test.example/source-bucket/incoming/clip.mp4",
        client_metadata={
            "npa": {"source_uri": "s3://source-bucket/incoming/clip.mp4"}
        },
    )
    folder = FakeFolder([exact])
    dataset = FakeDataset()
    receipt = run_push(
        input_path="s3://source-bucket/incoming/",
        integration="s3",
        folder="folder",
        dataset="dataset",
        output_path="s3://result-bucket/run",
        user_client=FakeUserClient(folder, dataset),
        storage_client=storage,
        artifact_store=MemoryArtifactStore(),
        environ=ENV,
    )
    assert receipt.status == "completed"
    assert receipt.items[0].link_state == "linked"
    assert dataset.linked == [[exact.uuid]]


def test_created_dataset_identity_is_checkpointed_before_hydration() -> None:
    storage = storage_with("incoming/clip.mp4")
    folder = FakeFolder()
    client = PostCreateFetchFailureClient(folder)
    store = MemoryArtifactStore()
    with pytest.raises(EncordToolError, match="dataset_hydration_failed"):
        run_push(
            input_path="s3://source-bucket/incoming/",
            integration="s3",
            folder="folder",
            dataset="new-dataset",
            output_path="s3://result-bucket/run",
            user_client=client,
            storage_client=storage,
            artifact_store=store,
            environ=ENV,
        )
    durable = json.loads(store.payloads["s3://result-bucket/run/push_receipt.json"])
    assert client.created_datasets == 1
    assert durable["phase"] == "final"
    assert durable["status"] == "failed"
    assert durable["revision"] == 2
    assert durable["dataset_created"] is True
    assert durable["dataset_hash"] == "00000000-0000-0000-0000-000000000040"


def test_known_inventory_conflict_never_submits_registration() -> None:
    storage = storage_with("incoming/clip.mp4")
    exact_metadata = {
        "npa": {"source_uri": "s3://source-bucket/incoming/clip.mp4"}
    }
    folder = FakeFolder(
        [
            FakeStorageItem(uuid="uuid-1", name="one", client_metadata=exact_metadata),
            FakeStorageItem(uuid="uuid-2", name="two", client_metadata=exact_metadata),
        ]
    )
    store = MemoryArtifactStore()
    with pytest.raises(EncordToolError):
        run_push(
            input_path="s3://source-bucket/incoming/",
            integration="s3",
            folder="folder",
            output_path="s3://result-bucket/run",
            user_client=FakeUserClient(folder),
            storage_client=storage,
            artifact_store=store,
            environ=ENV,
        )
    assert folder.registered == []
    durable = json.loads(store.payloads["s3://result-bucket/run/push_receipt.json"])
    assert durable["items"][0]["error_code"] == "identity_conflict"


def test_sidecar_conflict_never_registers_or_links() -> None:
    storage = storage_with("incoming/clip.mp4")
    contradictory = FakeStorageItem(
        uuid="uuid-1",
        name="clip.mp4",
        url="https://storage.test.example/source-bucket/archive/clip.mp4",
        client_metadata={
            "npa": {"source_uri": "s3://source-bucket/archive/clip.mp4"}
        },
    )
    folder = FakeFolder([contradictory])
    dataset = FakeDataset()
    store = MemoryArtifactStore()
    sidecar_uri = "s3://result-bucket/assertions/identity.json"
    store.create_json(
        sidecar_uri,
        {
            "schema": "npa.encord.identity_sidecar.v1",
            "items": [
                {
                    "source_uri": "s3://source-bucket/incoming/clip.mp4",
                    "item_uuid": "uuid-1",
                }
            ],
        },
    )
    with pytest.raises(EncordToolError):
        run_push(
            input_path="s3://source-bucket/incoming/",
            integration="s3",
            folder="folder",
            dataset="dataset",
            output_path="s3://result-bucket/run",
            identity_sidecar_uri=sidecar_uri,
            user_client=FakeUserClient(folder, dataset),
            storage_client=storage,
            artifact_store=store,
            environ=ENV,
        )
    assert folder.registered == []
    assert dataset.linked == []


def test_literal_percent_encoded_key_never_links_slash_key_inventory() -> None:
    storage = storage_with("incoming/a%2Fb.mp4")
    slash_item = FakeStorageItem(
        uuid="uuid-slash",
        name="b.mp4",
        url="https://storage.test.example/source-bucket/incoming/a/b.mp4",
        client_metadata={
            "npa": {"source_uri": "s3://source-bucket/incoming/a/b.mp4"}
        },
    )
    folder = FakeFolder([slash_item])
    dataset = FakeDataset()
    store = MemoryArtifactStore()
    with pytest.raises(EncordToolError):
        run_push(
            input_path="s3://source-bucket/incoming/",
            integration="s3",
            folder="folder",
            dataset="dataset",
            output_path="s3://result-bucket/run",
            user_client=FakeUserClient(folder, dataset),
            storage_client=storage,
            artifact_store=store,
            environ=ENV,
        )
    durable = json.loads(store.payloads["s3://result-bucket/run/push_receipt.json"])
    assert durable["items"][0]["object_key"] == "incoming/a%2Fb.mp4"
    assert durable["items"][0]["source_uri"].endswith("incoming/a%252Fb.mp4")
    assert durable["items"][0]["item_uuid"] == ""
    assert dataset.linked == []
