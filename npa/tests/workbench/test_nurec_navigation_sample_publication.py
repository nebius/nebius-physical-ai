"""Prove scan HTML retries retain exact bytes and cannot replace native evidence."""

import io
from pathlib import Path

from botocore.exceptions import ClientError
import pytest

from npa.clients.storage import StorageClient
from npa.workbench.nurec import navigation_sample_publication as publication
from npa.workbench.nurec.navigation_publication import verify_publication

_URI = "s3://test-bucket/scan/report"


class Objects:
    def __init__(self):
        self.objects = {}
        self.failure = ""
        self.committed_failure = False
        self.writes = []

    def get_paginator(self, _name):
        return self

    def paginate(self, *, Bucket, Prefix):
        return [
            {
                "Contents": [
                    {"Key": key} for key in self.objects if key.startswith(Prefix)
                ]
            }
        ]

    def get_object(self, *, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[Key]), "ETag": "fixture-etag"}

    def put_object(self, *, Bucket, Key, Body, IfNoneMatch, **kwargs):
        assert IfNoneMatch == "*"
        if Key in self.objects:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        if self.failure and Key.endswith(self.failure) and not self.committed_failure:
            raise OSError("interrupted upload")
        self.objects[Key] = bytes(Body)
        self.writes.append(Key)
        if self.failure and Key.endswith(self.failure):
            raise OSError("interrupted response")
        return {"ETag": "fixture-etag"}


@pytest.fixture
def storage(monkeypatch):
    objects = Objects()
    client = object.__new__(StorageClient)
    client._s3 = objects
    monkeypatch.setattr(StorageClient, "from_environment", lambda: client)
    return objects


def _report(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "index.html").write_text("<html>measured fixture</html>")
    (root / "summary.json").write_text('{"passed":true}\n')
    return root


@pytest.mark.parametrize(
    "member", [publication._CLAIM, "index.html", "summary.json", publication._SEAL]
)
@pytest.mark.parametrize("committed", [False, True])
def test_s3_retry_finishes_only_identical_report_bytes(
    storage, tmp_path, member, committed
):
    source = _report(tmp_path)
    storage.failure, storage.committed_failure = member, committed
    with pytest.raises(OSError, match="interrupted"):
        publication.publish_report(source, _URI)
    retained = dict(storage.objects)
    storage.failure = ""
    publication.publish_report(source, _URI)
    publication.publish_report(source, _URI)
    assert all(storage.objects[key] == payload for key, payload in retained.items())
    assert len(storage.writes) == len(set(storage.writes)) == 4
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for key, value in storage.objects.items():
        (snapshot / Path(key).name).write_bytes(value)
    verify_publication(snapshot)


@pytest.mark.parametrize(
    "member", [publication._CLAIM, "index.html", "unexpected.json"]
)
def test_s3_conflict_does_not_modify_existing_report(storage, tmp_path, member):
    source = _report(tmp_path)
    publication.publish_report(source, _URI)
    storage.objects["scan/report/" + member] = b"conflicting bytes"
    before = dict(storage.objects)
    with pytest.raises(ValueError, match="differs|unexpected"):
        publication.publish_report(source, _URI)
    assert storage.objects == before


def test_local_partial_report_retry_and_conflict(tmp_path, monkeypatch):
    source, destination = _report(tmp_path), tmp_path / "result"
    original = publication._put_local

    def interrupted(path, payload):
        if path.name == "summary.json":
            raise OSError("interrupted copy")
        original(path, payload)

    monkeypatch.setattr(publication, "_put_local", interrupted)
    with pytest.raises(OSError, match="interrupted copy"):
        publication.publish_report(source, str(destination))
    monkeypatch.setattr(publication, "_put_local", original)
    publication.publish_report(source, str(destination))
    verify_publication(destination)
    (destination / "index.html").write_bytes(b"different report")
    with pytest.raises(ValueError, match="differs"):
        publication.publish_report(source, str(destination))
    assert (destination / "index.html").read_bytes() == b"different report"


def test_evaluation_occupancy_includes_partial_and_unclaimed_objects(storage, tmp_path):
    assert not publication.evaluation_exists(_URI)
    storage.objects["scan/report/claim.json"] = b"partial native output"
    assert publication.evaluation_exists(_URI)
    root = tmp_path / "empty-but-occupied"
    assert not publication.evaluation_exists(str(root))
    root.mkdir()
    assert publication.evaluation_exists(str(root))


def test_provider_listing_failure_cannot_become_absence(storage, monkeypatch):
    def failure(**kwargs):
        raise OSError("provider unavailable")

    monkeypatch.setattr(storage, "paginate", failure)
    with pytest.raises(OSError, match="provider unavailable"):
        publication.evaluation_exists(_URI)


@pytest.mark.parametrize("same", [False, True])
def test_racing_s3_write_requires_identical_bytes(storage, monkeypatch, same):
    from npa.clients.storage import StoragePreconditionFailed

    client = StorageClient.from_environment()
    payload = b"exact deterministic report"

    def racing_write(body, uri, **kwargs):
        storage.objects["scan/report/index.html"] = body if same else b"other report"
        raise StoragePreconditionFailed("another publisher committed")

    monkeypatch.setattr(client, "put_bytes_conditional", racing_write)
    if same:
        publication._put_remote(client, _URI + "/index.html", payload)
    else:
        with pytest.raises(ValueError, match="differs"):
            publication._put_remote(client, _URI + "/index.html", payload)
        assert storage.objects["scan/report/index.html"] == b"other report"


def test_unclaimed_s3_report_is_never_adopted(storage, tmp_path):
    storage.objects["scan/report/index.html"] = b"legacy report"
    before = dict(storage.objects)
    with pytest.raises(ValueError, match="unclaimed"):
        publication.publish_report(_report(tmp_path), _URI)
    assert storage.objects == before


@pytest.mark.parametrize("location", ["root", "parent", "member"])
def test_local_report_refuses_symlinks(tmp_path, location):
    source = _report(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = tmp_path / "result"
    if location == "root":
        destination.symlink_to(outside, target_is_directory=True)
    elif location == "parent":
        destination.symlink_to(outside, target_is_directory=True)
        destination /= "nested"
    else:
        destination.mkdir()
        records = publication._report_records(source)
        (destination / publication._CLAIM).write_bytes(records[publication._CLAIM])
        (outside / "index.html").write_bytes(records["index.html"])
        (destination / "index.html").symlink_to(outside / "index.html")
    before = sorted(path.name for path in outside.iterdir())
    with pytest.raises(ValueError, match="symlink"):
        publication.publish_report(source, str(destination))
    assert sorted(path.name for path in outside.iterdir()) == before
