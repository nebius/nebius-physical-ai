"""Hermetic public-storage integrity and local path confinement regressions."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import os
import sys
from types import SimpleNamespace

import pytest

from npa.workflows.byof import openpi_gcs as gcs


class ChecksumDouble:
    """Test checksum lifecycle; live qualification exercises the SDK CRC32C."""

    def __init__(self):
        self.digest_state = hashlib.sha256()

    def update(self, data):
        self.digest_state.update(data)

    def digest(self):
        return self.digest_state.digest()[:4]


def checksum(data):
    digest = ChecksumDouble()
    digest.update(data)
    return base64.b64encode(digest.digest()).decode()


@pytest.fixture(autouse=True)
def crc32c_double(monkeypatch):
    monkeypatch.setitem(sys.modules, "google_crc32c", SimpleNamespace(Checksum=ChecksumDouble))


class Blob:
    def __init__(self, name="dataset/file.bin", data=b"public fixture", *, downloaded=None):
        self.name = name
        self.data = data
        self.downloaded = data if downloaded is None else downloaded
        self.size = len(data)
        self.crc32c = checksum(data)
        self.generation = 17
        self.updated = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.calls = []

    def download_to_file(self, handle, **kwargs):
        self.calls.append(kwargs)
        handle.write(self.downloaded)


class Client:
    def __init__(self, blobs):
        self.blobs = blobs
        self.list_calls = []
        self.bucket_calls = []

    def list_blobs(self, bucket, *, prefix):
        self.list_calls.append((bucket, prefix))
        return iter(self.blobs)

    def bucket(self, bucket):
        self.bucket_calls.append(bucket)
        return self

    def get_blob(self, name):
        return next((blob for blob in self.blobs if blob.name == name), None)


def test_sync_checksums_every_file_reuses_good_bytes_and_repairs_corruption(tmp_path):
    blobs = [Blob(), Blob("dataset/nested/second.bin", b"second public file")]
    client = Client(blobs)
    args = ["-m", "rsync", "-r", "-c", "gs://public-fixture/dataset", str(tmp_path)]
    gcs.run(args, client=client)
    assert (tmp_path / "file.bin").read_bytes() == blobs[0].data
    assert (tmp_path / "nested/second.bin").read_bytes() == blobs[1].data
    gcs.run(args, client=client)
    assert [len(blob.calls) for blob in blobs] == [1, 1]
    (tmp_path / "file.bin").write_bytes(b"x" * len(blobs[0].data))
    gcs.run(args, client=client)
    assert [len(blob.calls) for blob in blobs] == [2, 1]
    assert blobs[0].calls[0] == {
        "if_generation_match": 17, "raw_download": True, "checksum": "crc32c",
    }
    assert (tmp_path / "file.bin").stat().st_mode & 0o777 == 0o600
    assert not list(tmp_path.rglob(".npa-gcs-*"))


@pytest.mark.parametrize("downloaded", [b"truncated", b"x" * len(b"public fixture")])
def test_failed_download_preserves_prior_destination_and_removes_partial(tmp_path, downloaded):
    destination = tmp_path / "file.bin"
    destination.write_bytes(b"prior trusted bytes")
    blob = Blob(downloaded=downloaded)
    with pytest.raises(gcs.GCSReadError, match="checksum or byte count"):
        gcs.download(blob, destination)
    assert destination.read_bytes() == b"prior trusted bytes"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["file.bin"]


def test_generation_failure_leaves_no_download_or_partial(tmp_path):
    blob = Blob()

    def changed(handle, **kwargs):
        assert kwargs["if_generation_match"] == 17
        handle.write(b"partial")
        raise OSError("provider rejected changed object generation")

    blob.download_to_file = changed
    with pytest.raises(OSError, match="changed object generation"):
        gcs.download(blob, tmp_path / "file.bin")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name", [
    "dataset/../outside", "dataset//outside", "/dataset/outside",
    "dataset/./outside", "dataset/back\\slash", "dataset/line\nbreak",
    "other/outside", "dataset/",
])
def test_malicious_listing_cannot_write_outside_destination(tmp_path, name):
    blob = Blob(name)
    with pytest.raises(gcs.GCSReadError):
        gcs.run(["rsync", "-r", "-c", "gs://public-fixture/dataset", str(tmp_path)],
                client=Client([blob]))
    assert list(tmp_path.iterdir()) == []
    assert blob.calls == []


@pytest.mark.parametrize("alias", ["parent_symlink", "file_symlink", "file_hardlink", "fifo"])
def test_existing_alias_or_special_destination_is_never_followed(tmp_path, alias):
    outside = tmp_path / "outside"
    outside.mkdir()
    original = outside / "original"
    original.write_bytes(b"private fixture")
    destination = tmp_path / "download" / "file.bin"
    if alias == "parent_symlink":
        destination.parent.symlink_to(outside, target_is_directory=True)
    else:
        destination.parent.mkdir()
        if alias == "file_symlink":
            destination.symlink_to(original)
        elif alias == "file_hardlink":
            os.link(original, destination)
        else:
            os.mkfifo(destination)
    blob = Blob()
    with pytest.raises((gcs.GCSReadError, OSError)):
        gcs.download(blob, destination)
    assert original.read_bytes() == b"private fixture"
    assert not (outside / "file.bin").exists()
    assert blob.calls == []


@pytest.mark.parametrize("field,value", [("crc32c", None), ("crc32c", "invalid"),
                                        ("generation", 0), ("size", -1)])
def test_missing_or_invalid_integrity_metadata_fails_before_write(tmp_path, field, value):
    blob = Blob()
    setattr(blob, field, value)
    with pytest.raises(gcs.GCSReadError, match="integrity metadata"):
        gcs.download(blob, tmp_path / "file.bin")
    assert list(tmp_path.iterdir()) == []
    assert blob.calls == []


@pytest.mark.parametrize("source", [
    "https://example.invalid/object", "gs://user:password@public-fixture/object",
    "gs://public-fixture/../object", "gs://public-fixture//object",
    "gs://public-fixture/object?credential=fixture", "gs://public-fixture/object#fragment",
    "gs://public-fixture/obj*", "gs://public-fixture/obj\nect",
])
def test_invalid_source_fails_before_client_access(tmp_path, source):
    client = Client([])
    with pytest.raises(gcs.GCSReadError):
        gcs.run(["cp", source, str(tmp_path / "file")], client=client)
    assert client.bucket_calls == []
    assert client.list_calls == []


def test_listing_retains_runner_inventory_format_and_copy_selects_exact_object(tmp_path, capsys):
    blob = Blob()
    client = Client([blob])
    gcs.run(["ls", "-l", "-r", "gs://public-fixture/dataset/**"], client=client)
    assert capsys.readouterr().out == (
        "14  2026-01-01T00:00:00+00:00  gs://public-fixture/dataset/file.bin\n"
    )
    gcs.run(["cp", "gs://public-fixture/dataset/file.bin", str(tmp_path / "copy")],
            client=client)
    assert (tmp_path / "copy").read_bytes() == blob.data
    assert client.bucket_calls == ["public-fixture"]


def test_cloud_destination_and_unsupported_operations_fail_before_client_access():
    client = Client([])
    for args in (["cp", "gs://public-fixture/dataset/file", "s3://other/output"],
                 ["rm", "gs://public-fixture/dataset/file"]):
        with pytest.raises(gcs.GCSReadError):
            gcs.run(args, client=client)
    assert client.bucket_calls == []


def test_terminal_error_has_no_private_provider_output(monkeypatch, capsys):
    def fail(_arguments):
        raise OSError("private provider response")

    monkeypatch.setattr(gcs, "run", fail)
    assert gcs.main(["cp"]) == 1
    assert capsys.readouterr().err == "OpenPI public GCS download failed\n"
