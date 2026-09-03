"""Portable proof that a unique basename cannot establish Encord identity."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

from npa.clients.storage import StoragePreconditionFailed
from npa.workbench.encord.push import run_push
from npa.workbench.encord.schemas import EncordToolError


def test_unique_basename_with_different_full_key_is_never_linked(tmp_path: Path) -> None:
    del tmp_path
    storage = _Storage()
    storage.s3.objects[("source-bucket", "incoming/clip.mp4")] = b"video"
    wrong = SimpleNamespace(
        uuid="00000000-0000-0000-0000-000000000061",
        item_uuid="00000000-0000-0000-0000-000000000061",
        name="archive/clip.mp4",
        url="https://storage.test.example/source-bucket/archive/clip.mp4",
        client_metadata={},
    )
    folder = _Folder(wrong)
    dataset = _Dataset()
    client = _Client(folder, dataset)

    raised = False
    try:
        run_push(
            input_path="s3://source-bucket/incoming/",
            integration="s3",
            folder="folder",
            dataset="dataset",
            output_path="s3://result-bucket/run",
            user_client=client,
            storage_client=storage,
            environ={"AWS_ENDPOINT_URL": "https://storage.test.example"},
        )
    except EncordToolError:
        raised = True

    receipt = json.loads(
        storage.s3.objects[("result-bucket", "run/push_receipt.json")]
    )
    item = receipt["items"][0]
    unresolved = receipt.get("counts", {}).get("unresolved", 0)

    assert dataset.linked == []
    assert raised is True
    assert not (item.get("item_uuid") or "")
    assert item.get("identity_state") == "unresolved"
    assert receipt["status"] in {"partial", "failed"}
    assert unresolved == 1


class _Body:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return self.payload

    def iter_chunks(self, chunk_size: int):
        del chunk_size
        yield self.payload

    def close(self) -> None:
        pass


class _Paginator:
    def __init__(self, s3: "_S3") -> None:
        self.s3 = s3

    def paginate(self, *, Bucket: str, Prefix: str, **_):
        return [
            {
                "Contents": [
                    {"Key": key, "Size": len(payload)}
                    for (bucket, key), payload in self.s3.objects.items()
                    if bucket == Bucket and key.startswith(Prefix)
                ]
            }
        ]


class _S3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def get_paginator(self, name: str) -> _Paginator:
        assert name == "list_objects_v2"
        return _Paginator(self)

    def head_object(self, *, Bucket: str, Key: str, **_):
        payload = self.objects[(Bucket, Key)]
        return {"ContentLength": len(payload), "ETag": '"opaque"'}

    def get_object(self, *, Bucket: str, Key: str):
        return {"Body": _Body(self.objects[(Bucket, Key)])}

    def put_object(self, *, Bucket: str, Key: str, Body, **_):
        self.objects[(Bucket, Key)] = Body
        return {"ETag": '"written"'}


class _Storage:
    def __init__(self) -> None:
        self.s3 = _S3()
        self.versions: dict[str, str] = {}

    def upload_file(self, local_file: str, uri: str) -> str:
        bucket, key = _split(uri)
        self.s3.objects[(bucket, key)] = Path(local_file).read_bytes()
        return uri

    def put_bytes_conditional(
        self,
        payload: bytes,
        uri: str,
        *,
        if_match: str = "",
        if_none_match: bool = False,
        **_,
    ) -> str:
        if if_none_match and uri in self.versions:
            raise StoragePreconditionFailed("exists")
        if if_match and self.versions.get(uri) != if_match:
            raise StoragePreconditionFailed("stale")
        token = f'"v{len(self.versions) + 1}"'
        self.versions[uri] = token
        bucket, key = _split(uri)
        self.s3.objects[(bucket, key)] = payload
        return token

    def read_bytes_with_etag(self, uri: str):
        bucket, key = _split(uri)
        payload = self.s3.objects.get((bucket, key))
        return (payload, self.versions[uri]) if payload is not None else None


class _Folder:
    uuid = "00000000-0000-0000-0000-000000000010"
    name = "folder"

    def __init__(self, wrong) -> None:
        self.wrong = wrong

    def list_items(self, **_):
        return [self.wrong]

    def add_private_data_to_folder_start(self, **_):
        return UUID("00000000-0000-0000-0000-000000000020")

    def add_private_data_to_folder_get_result(self, *_, **__):
        return SimpleNamespace(
            status=SimpleNamespace(name="DONE"),
            units_done_count=0,
            units_error_count=0,
            items_with_names=[],
            unit_errors=[],
        )


class _Dataset:
    title = "dataset"

    def __init__(self) -> None:
        self.linked: list[list[str]] = []
        self.data_rows: list[SimpleNamespace] = []

    def list_data_rows(self):
        return list(self.data_rows)

    def link_items(self, item_uuids):
        values = [str(value) for value in item_uuids]
        self.linked.append(values)
        self.data_rows.extend(
            SimpleNamespace(backing_item_uuid=value) for value in values
        )


class _Client:
    def __init__(self, folder: _Folder, dataset: _Dataset) -> None:
        self.folder = folder
        self.dataset = dataset

    def get_cloud_integrations(self):
        return [SimpleNamespace(id="integration-id", title="s3")]

    def list_storage_folders(self, **_):
        return [self.folder]

    def get_datasets(self, **_):
        return [
            {
                "dataset": SimpleNamespace(
                    dataset_hash="00000000-0000-0000-0000-000000000040"
                )
            }
        ]

    def get_dataset(self, *_, **__):
        return self.dataset

    def get_storage_items(self, uuids, **_):
        wanted = {str(value) for value in uuids}
        return [self.folder.wrong] if self.folder.wrong.uuid in wanted else []


def _split(uri: str) -> tuple[str, str]:
    bucket, key = uri.removeprefix("s3://").split("/", 1)
    return bucket, key
