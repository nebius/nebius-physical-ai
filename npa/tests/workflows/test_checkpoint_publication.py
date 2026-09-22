from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError

from npa.clients.storage import StoragePreconditionFailed
from npa.workflows.behavior_challenge import checkpoint_publication as publication
from npa.workflows.behavior_challenge.checkpoint_publication import (
    publish_immutable_checkpoint_files,
)


class MemoryStorage:
    def __init__(self, objects: dict[str, bytes], lock: threading.Lock) -> None:
        self.objects = objects
        self.lock = lock
        self.writes: list[str] = []
        self.corrupt_uri = ""

    def put_bytes_conditional(
        self, payload: bytes, uri: str, *, if_none_match: bool
    ) -> str:
        assert if_none_match is True
        with self.lock:
            if uri in self.objects:
                raise StoragePreconditionFailed("exists")
            self.objects[uri] = bytes(payload)
            self.writes.append(uri)
        return "etag"

    def download_file(self, uri: str, local_path: str) -> None:
        payload = self.objects[uri]
        if uri == self.corrupt_uri:
            payload += b"corrupt"
        Path(local_path).write_bytes(payload)

    @property
    def s3(self):
        return self


class MultipartStorage(MemoryStorage):
    def __init__(self, objects: dict[str, bytes], lock: threading.Lock) -> None:
        super().__init__(objects, lock)
        self.uploads: dict[str, dict] = {}
        self.aborted: list[str] = []
        self.part_sizes: list[int] = []

    def create_multipart_upload(self, *, Bucket: str, Key: str) -> dict:
        upload_id = f"upload-{len(self.uploads)}"
        self.uploads[upload_id] = {"bucket": Bucket, "key": Key, "parts": {}}
        return {"UploadId": upload_id}

    def upload_part(self, *, UploadId: str, PartNumber: int, Body: bytes, **_) -> dict:
        self.uploads[UploadId]["parts"][PartNumber] = bytes(Body)
        self.part_sizes.append(len(Body))
        return {"ETag": f"part-{PartNumber}"}

    def complete_multipart_upload(
        self, *, Bucket: str, Key: str, UploadId: str, IfNoneMatch: str, **_
    ) -> dict:
        assert IfNoneMatch == "*"
        uri = f"s3://{Bucket}/{Key}"
        with self.lock:
            if uri in self.objects:
                raise ClientError(
                    {
                        "Error": {"Code": "PreconditionFailed"},
                        "ResponseMetadata": {"HTTPStatusCode": 412},
                    },
                    "CompleteMultipartUpload",
                )
            parts = self.uploads[UploadId]["parts"]
            self.objects[uri] = b"".join(parts[index] for index in sorted(parts))
        self.uploads.pop(UploadId)
        return {"ETag": "complete"}

    def abort_multipart_upload(self, *, UploadId: str, **_) -> None:
        self.uploads.pop(UploadId, None)
        self.aborted.append(UploadId)


def _factory(storage):
    return lambda: storage


def test_publishes_nested_files_with_complete_readback(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.write_bytes(b"first checkpoint member")
    second.write_bytes(b"second checkpoint member")
    objects: dict[str, bytes] = {}
    storage = MemoryStorage(objects, threading.Lock())
    scratch_parent = tmp_path / "scratch"
    scratch_parent.mkdir()
    sentinel = scratch_parent / "keep"
    sentinel.write_text("unrelated")

    result = publish_immutable_checkpoint_files(
        {"state/first": first, "metadata/second": second},
        "s3://bucket/checkpoints/run-1",
        scratch_parent,
        expected={
            "state/first": {
                "bytes": first.stat().st_size,
                "sha256": hashlib.sha256(first.read_bytes()).hexdigest(),
            },
            "metadata/second": {
                "bytes": second.stat().st_size,
                "sha256": hashlib.sha256(second.read_bytes()).hexdigest(),
            },
        },
        storage_factory=_factory(storage),
    )

    assert result["state/first"]["provider_readback"] is True
    assert objects["s3://bucket/checkpoints/run-1/state/first"] == first.read_bytes()
    assert sentinel.read_text() == "unrelated"
    assert list(scratch_parent.iterdir()) == [sentinel]


@pytest.mark.parametrize(
    "name",
    [".", "../escape", "a/../escape", "/absolute", "a\\b", "a//b", "a?b", "a#b"],
)
def test_rejects_unsafe_names_before_writes(tmp_path: Path, name: str) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"checkpoint")
    storage = MemoryStorage({}, threading.Lock())
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    with pytest.raises(ValueError, match="canonical"):
        publish_immutable_checkpoint_files(
            {"valid": source, name: source},
            "s3://bucket/checkpoints/run-1",
            scratch,
            storage_factory=_factory(storage),
        )
    assert storage.writes == []


def test_rejects_file_directory_name_collision_before_writes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"checkpoint")
    storage = MemoryStorage({}, threading.Lock())

    with pytest.raises(ValueError, match="overlap"):
        publish_immutable_checkpoint_files(
            {"state": source, "state/member": source},
            "s3://bucket/checkpoints/run-1",
            tmp_path,
            storage_factory=_factory(storage),
        )
    assert storage.writes == []


@pytest.mark.parametrize(
    "uri",
    [
        "https://bucket/prefix",
        "s3://bucket",
        "s3://bucket/a/../b",
        "s3://bucket/a//b",
        "s3://user@bucket/prefix",
        "s3://bucket/prefix?query=1",
    ],
)
def test_rejects_unscoped_or_noncanonical_uris(tmp_path: Path, uri: str) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"checkpoint")
    storage = MemoryStorage({}, threading.Lock())

    with pytest.raises(ValueError, match="scoped s3"):
        publish_immutable_checkpoint_files(
            {"member": source},
            uri,
            tmp_path,
            storage_factory=_factory(storage),
        )
    assert storage.writes == []


def test_rejects_symlink_and_mismatched_expected_before_writes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"checkpoint")
    link = tmp_path / "link"
    link.symlink_to(source)
    storage = MemoryStorage({}, threading.Lock())

    with pytest.raises(ValueError, match="regular file"):
        publish_immutable_checkpoint_files(
            {"member": link},
            "s3://bucket/checkpoints/run-1",
            tmp_path,
            storage_factory=_factory(storage),
        )
    with pytest.raises(ValueError, match="expected identities"):
        publish_immutable_checkpoint_files(
            {"member": source},
            "s3://bucket/checkpoints/run-1",
            tmp_path,
            expected={"member": {"bytes": 1, "sha256": "0" * 64}},
            storage_factory=_factory(storage),
        )
    assert storage.writes == []


def test_fifo_is_rejected_without_blocking_or_writes(tmp_path: Path) -> None:
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    storage = MemoryStorage({}, threading.Lock())

    with pytest.raises(ValueError, match="regular file"):
        publish_immutable_checkpoint_files(
            {"member": fifo},
            "s3://bucket/checkpoints/run-1",
            tmp_path,
            storage_factory=_factory(storage),
        )
    assert storage.writes == []


def test_small_conditional_create_rejects_size_change_before_write(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"checkpoint grew")
    storage = MemoryStorage({}, threading.Lock())

    with pytest.raises(ValueError, match="changed before conditional creation"):
        publication._conditional_create(
            storage,
            source,
            "s3://bucket/checkpoints/run-1/member",
            expected={
                "bytes": len(b"checkpoint"),
                "sha256": hashlib.sha256(b"checkpoint").hexdigest(),
            },
            part_bytes=5 * 1024 * 1024,
        )
    assert storage.writes == []


def test_identical_retry_passes_and_conflicting_object_fails(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"checkpoint")
    objects: dict[str, bytes] = {}
    storage = MemoryStorage(objects, threading.Lock())
    arguments = {
        "files": {"member": source},
        "destination_uri": "s3://bucket/checkpoints/run-1",
        "scratch_parent": tmp_path,
        "storage_factory": _factory(storage),
    }

    first = publish_immutable_checkpoint_files(**arguments)
    assert publish_immutable_checkpoint_files(**arguments) == first
    objects["s3://bucket/checkpoints/run-1/member"] = b"different"
    with pytest.raises(ValueError, match="provider readback differs"):
        publish_immutable_checkpoint_files(**arguments)


def test_multipart_is_bounded_and_conflict_keeps_first_object(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.write_bytes(b"a" * (10 * 1024 * 1024 + 17))
    second.write_bytes(b"b" * (10 * 1024 * 1024 + 17))
    objects: dict[str, bytes] = {}
    storage = MultipartStorage(objects, threading.Lock())
    arguments = {
        "destination_uri": "s3://bucket/checkpoints/run-1",
        "scratch_parent": tmp_path,
        "storage_factory": _factory(storage),
        "workers": 1,
        "part_bytes": 5 * 1024 * 1024,
    }

    publish_immutable_checkpoint_files({"member": first}, **arguments)
    with pytest.raises(ValueError, match="provider readback differs"):
        publish_immutable_checkpoint_files({"member": second}, **arguments)
    assert objects["s3://bucket/checkpoints/run-1/member"] == first.read_bytes()
    assert max(storage.part_sizes) <= 5 * 1024 * 1024
    assert storage.aborted


def test_corrupt_readback_fails_and_removes_only_owned_scratch(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"checkpoint")
    uri = "s3://bucket/checkpoints/run-1/member"
    storage = MemoryStorage({}, threading.Lock())
    storage.corrupt_uri = uri
    sentinel = tmp_path / "unrelated"
    sentinel.write_text("keep")

    with pytest.raises(ValueError, match="provider readback differs"):
        publish_immutable_checkpoint_files(
            {"member": source},
            "s3://bucket/checkpoints/run-1",
            tmp_path,
            storage_factory=_factory(storage),
        )
    assert sentinel.read_text() == "keep"
    assert not any(
        path.name.startswith(".npa-checkpoint") for path in tmp_path.iterdir()
    )


def test_insufficient_readback_space_fails_before_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"checkpoint")
    storage = MemoryStorage({}, threading.Lock())
    monkeypatch.setattr(
        publication.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=10, used=10, free=0),
    )

    with pytest.raises(ValueError, match="scratch space is insufficient"):
        publish_immutable_checkpoint_files(
            {"member": source},
            "s3://bucket/checkpoints/run-1",
            tmp_path,
            storage_factory=_factory(storage),
        )
    assert storage.writes == []
    assert not any(
        path.name.startswith(".npa-checkpoint") for path in tmp_path.iterdir()
    )
