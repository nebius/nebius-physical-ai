"""Hermetic Encord and S3 fakes with ordered mutation evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping
from uuid import UUID

from npa.workbench.encord.integrity import StreamDigest
from npa.workbench.encord.storage import (
    ArtifactConflict,
    ArtifactVersion,
    ObjectMetadata,
)


class StreamingBody:
    def __init__(self, payload: bytes, *, fail: bool = False) -> None:
        self.payload = payload
        self.fail = fail
        self.closed = False

    def read(self) -> bytes:
        return self.payload

    def iter_chunks(self, chunk_size: int) -> Any:
        midpoint = min(len(self.payload), max(1, chunk_size))
        yield self.payload[:midpoint]
        if self.fail:
            raise OSError("injected stream failure")
        if midpoint < len(self.payload):
            yield self.payload[midpoint:]

    def close(self) -> None:
        self.closed = True


class FakePaginator:
    def __init__(self, s3: "FakeS3") -> None:
        self.s3 = s3

    def paginate(self, *, Bucket: str, Prefix: str, **_: Any) -> list[dict[str, Any]]:
        rows = [
            {"Key": key, "Size": len(payload)}
            for (bucket, key), payload in sorted(self.s3.objects.items())
            if bucket == Bucket and key.startswith(Prefix)
        ]
        midpoint = max(1, len(rows) // 2)
        return [{"Contents": rows[:midpoint]}, {"Contents": rows[midpoint:]}]


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.content_types: dict[tuple[str, str], str] = {}
        self.events: list[tuple[Any, ...]] = []
        self.missing_after_write: set[tuple[str, str]] = set()
        self.wrong_size_after_write: set[tuple[str, str]] = set()
        self.head_overrides: dict[tuple[str, str], dict[str, Any]] = {}

    def get_paginator(self, name: str) -> FakePaginator:
        assert name == "list_objects_v2"
        return FakePaginator(self)

    def head_object(self, *, Bucket: str, Key: str, **_: Any) -> dict[str, Any]:
        self.events.append(("head", Bucket, Key))
        identity = (Bucket, Key)
        if identity in self.missing_after_write or identity not in self.objects:
            error = RuntimeError("not found")
            error.response = {  # type: ignore[attr-defined]
                "Error": {"Code": "404"},
                "ResponseMetadata": {"HTTPStatusCode": 404},
            }
            raise error
        payload = self.objects[identity]
        size = len(payload) + (1 if identity in self.wrong_size_after_write else 0)
        response = {
            "ContentLength": size,
            "ContentType": self.content_types.get(identity, "application/octet-stream"),
            "ETag": f'"opaque-{hashlib.sha1(payload).hexdigest()}"',  # noqa: S324
        }
        response.update(self.head_overrides.get(identity, {}))
        return response

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        self.events.append(("get", Bucket, Key))
        return {"Body": StreamingBody(self.objects[(Bucket, Key)])}

    def put_object(self, *, Bucket: str, Key: str, Body: Any, **kwargs: Any) -> dict[str, str]:
        payload = Body if isinstance(Body, bytes) else Body.read()
        self.events.append(("put", Bucket, Key, kwargs))
        self.objects[(Bucket, Key)] = payload
        self.content_types[(Bucket, Key)] = str(kwargs.get("ContentType", ""))
        return {"ETag": f'"v-{len(self.events)}"'}

    def copy_object(self, *, Bucket: str, Key: str, CopySource: Mapping[str, str]) -> None:
        self.events.append(("copy", Bucket, Key, dict(CopySource)))
        self.objects[(Bucket, Key)] = self.objects[(CopySource["Bucket"], CopySource["Key"])]


class FakeStorageClient:
    def __init__(self) -> None:
        self.s3 = FakeS3()
        self.versions: dict[str, str] = {}

    def read_bytes_with_etag(self, uri: str) -> tuple[bytes, str] | None:
        bucket, key = _split(uri)
        if (bucket, key) not in self.s3.objects:
            return None
        return self.s3.objects[(bucket, key)], self.versions[uri]

    def put_bytes_conditional(
        self,
        payload: bytes,
        uri: str,
        *,
        if_match: str = "",
        if_none_match: bool = False,
        content_type: str = "application/octet-stream",
    ) -> str:
        if if_none_match and uri in self.versions:
            from npa.clients.storage import StoragePreconditionFailed

            raise StoragePreconditionFailed("exists")
        if if_match and self.versions.get(uri) != if_match:
            from npa.clients.storage import StoragePreconditionFailed

            raise StoragePreconditionFailed("stale")
        token = f'"version-{len(self.versions) + len(self.s3.events) + 1}"'
        bucket, key = _split(uri)
        self.s3.objects[(bucket, key)] = payload
        self.s3.content_types[(bucket, key)] = content_type
        self.versions[uri] = token
        self.s3.events.append(("conditional_put", uri, if_match, if_none_match))
        return token

    def upload_file(self, local_file: str, uri: str) -> str:
        bucket, key = _split(uri)
        self.s3.objects[(bucket, key)] = Path(local_file).read_bytes()
        self.s3.events.append(("upload", uri))
        return uri


class MemoryArtifactStore:
    def __init__(self, *, fail_replace: int = 0) -> None:
        self.payloads: dict[str, bytes] = {}
        self.tokens: dict[str, int] = {}
        self.events: list[tuple[str, str]] = []
        self.fail_replace = fail_replace
        self.replace_count = 0

    def create_json(self, uri: str, payload: Mapping[str, Any]) -> ArtifactVersion:
        self.events.append(("create", uri))
        if uri in self.payloads:
            raise ArtifactConflict("exists")
        self.payloads[uri] = _json(payload)
        self.tokens[uri] = 1
        return ArtifactVersion(kind="local_sha256", token="1")

    def replace_json(
        self, uri: str, payload: Mapping[str, Any], version: ArtifactVersion
    ) -> ArtifactVersion:
        self.events.append(("replace", uri))
        self.replace_count += 1
        if self.fail_replace == self.replace_count:
            raise OSError("injected checkpoint failure")
        if version.token != str(self.tokens.get(uri)):
            raise ArtifactConflict("stale")
        self.tokens[uri] += 1
        self.payloads[uri] = _json(payload)
        return ArtifactVersion(kind="local_sha256", token=str(self.tokens[uri]))

    def read_json(self, uri: str) -> Mapping[str, Any]:
        self.events.append(("read", uri))
        return json.loads(self.payloads[uri])

    def head(self, uri: str) -> ObjectMetadata:
        payload = self.payloads.get(uri)
        return ObjectMetadata(uri=uri, exists=payload is not None, size=len(payload or b""))


@dataclass
class FakeStorageItem:
    uuid: str
    name: str
    url: str = ""
    client_metadata: dict[str, Any] = field(default_factory=dict)
    file_size: int = 0
    mime_type: str = "video/mp4"
    item_type: str = "VIDEO"
    signed_url: str = ""

    def get_signed_url(self, refetch: bool = False) -> str:
        del refetch
        return self.signed_url or self.url


@dataclass
class FakeDataRow:
    backing_item_uuid: str | None
    uid: str = "data-1"
    title: str = "clip.mp4"


class FakeDataset:
    def __init__(self, title: str = "dataset") -> None:
        self.title = title
        self.data_rows: list[FakeDataRow] = []
        self.linked: list[list[str]] = []

    def list_data_rows(self) -> list[FakeDataRow]:
        return list(self.data_rows)

    def link_items(self, item_uuids: list[str]) -> list[FakeDataRow]:
        values = [str(value) for value in item_uuids]
        self.linked.append(values)
        existing = {str(row.backing_item_uuid) for row in self.data_rows}
        for value in values:
            if value not in existing:
                self.data_rows.append(FakeDataRow(value))
        return list(self.data_rows)


class FakeFolder:
    def __init__(self, items: list[FakeStorageItem] | None = None) -> None:
        self.uuid = "00000000-0000-0000-0000-000000000010"
        self.name = "folder"
        self.items = list(items or [])
        self.registered: list[dict[str, Any]] = []
        self.uploaded: list[str] = []
        self.registration_result: Any = SimpleNamespace(
            status=SimpleNamespace(name="DONE"),
            units_done_count=0,
            units_error_count=0,
            units_pending_count=0,
            units_cancelled_count=0,
            items_with_names=[],
            unit_errors=[],
            errors=[],
        )

    def list_items(self, **_: Any) -> list[FakeStorageItem]:
        return list(self.items)

    def add_private_data_to_folder_start(self, **kwargs: Any) -> UUID:
        self.registered.append(kwargs)
        return UUID("00000000-0000-0000-0000-000000000020")

    def add_private_data_to_folder_get_result(self, *_: Any, **__: Any) -> Any:
        return self.registration_result

    def upload_video(self, path: str, **_: Any) -> UUID:
        self.uploaded.append(path)
        return UUID("00000000-0000-0000-0000-000000000031")

    def upload_image(self, path: str, **_: Any) -> UUID:
        self.uploaded.append(path)
        return UUID("00000000-0000-0000-0000-000000000032")


class FakeUserClient:
    def __init__(self, folder: FakeFolder, dataset: FakeDataset | None = None) -> None:
        self.folder = folder
        self.dataset = dataset or FakeDataset()
        self.created_folders = 0
        self.created_datasets = 0

    def get_cloud_integrations(self) -> list[Any]:
        return [SimpleNamespace(id="00000000-0000-0000-0000-000000000001", title="s3")]

    def list_storage_folders(self, **_: Any) -> list[FakeFolder]:
        return [self.folder]

    def get_storage_folder(self, _: str) -> FakeFolder:
        return self.folder

    def create_storage_folder(self, *_: Any, **__: Any) -> FakeFolder:
        self.created_folders += 1
        return self.folder

    def get_datasets(self, **_: Any) -> list[dict[str, Any]]:
        return [
            {
                "dataset": SimpleNamespace(
                    dataset_hash="00000000-0000-0000-0000-000000000040"
                )
            }
        ]

    def get_dataset(self, *_: Any, **__: Any) -> FakeDataset:
        return self.dataset

    def create_dataset(self, *_: Any, **__: Any) -> Any:
        self.created_datasets += 1
        return SimpleNamespace(dataset_hash="00000000-0000-0000-0000-000000000040")

    def get_storage_items(self, uuids: list[str], **_: Any) -> list[FakeStorageItem]:
        wanted = {str(value) for value in uuids}
        return [item for item in self.folder.items if item.uuid in wanted]


class BytesDownloader:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.calls = 0

    def download(self, url: str, destination: Path) -> StreamDigest:
        del url
        self.calls += 1
        destination.write_bytes(self.payload)
        return StreamDigest(size=len(self.payload), sha256=hashlib.sha256(self.payload).hexdigest())


def _split(uri: str) -> tuple[str, str]:
    value = uri.removeprefix("s3://")
    return tuple(value.split("/", 1))  # type: ignore[return-value]


def _json(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode()
