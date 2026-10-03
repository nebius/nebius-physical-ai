"""Exercise conditional navigation publication races and permanent failure claims."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Barrier, Lock

from botocore.exceptions import ClientError
import pytest

from npa.clients.storage import StorageClient
from npa.workbench.nurec import navigation_publication as publication

_URI = "s3://unit-test-bucket/navigation/run"
_PREFIX = "navigation/run/"


class _Objects:
    def __init__(self):
        self.objects = {}
        self.lock = Lock()
        self.barrier = None
        self.failure = ""
        self.racing_object = ""

    def get_paginator(self, _name):
        return self

    def paginate(self, *, Bucket, Prefix):
        with self.lock:
            objects = [{"Key": key} for key in self.objects if key.startswith(Prefix)]
        if not objects and self.barrier is not None:
            self.barrier.wait()
        return [{"Contents": objects}]

    def put_object(self, *, Bucket, Key, Body, IfNoneMatch, **_kwargs):
        assert IfNoneMatch == "*"
        with self.lock:
            if Key in self.objects:
                raise ClientError(
                    {"Error": {"Code": "PreconditionFailed"}}, "PutObject"
                )
            if Key.endswith(self.failure) and self.failure:
                raise OSError("injected publication failure")
            data = Body.read() if hasattr(Body, "read") else Body
            self.objects[Key] = data
            if Key.endswith(publication._CLAIM) and self.racing_object:
                self.objects[_PREFIX + self.racing_object] = b"legacy writer"
        return {"ETag": "unit-test-etag"}


@pytest.fixture
def storage(monkeypatch):
    objects = _Objects()
    client = object.__new__(StorageClient)
    client._s3 = objects
    monkeypatch.setattr(StorageClient, "from_environment", lambda: client)
    return objects


def _result(root: Path, label: str) -> Path:
    root.mkdir()
    (root / "scene.usdz").write_bytes(label.encode())
    (root / "provenance.json").write_text(json.dumps({"scene": label}))
    return root


def _snapshot(storage, root):
    root.mkdir()
    for key, data in storage.objects.items():
        (root / key.removeprefix(_PREFIX)).write_bytes(data)
    return root


def test_successful_s3_publication_is_sealed_and_cannot_be_reused(storage, tmp_path):
    source = _result(tmp_path / "source", "first")
    publication.publish_immutable(source, _URI)
    publication.verify_publication(_snapshot(storage, tmp_path / "snapshot"))
    original = dict(storage.objects)
    (source / "scene.usdz").write_bytes(b"replacement")
    with pytest.raises(ValueError, match="fresh output prefix"):
        publication.publish_immutable(source, _URI)
    assert storage.objects == original


def test_preexisting_unclaimed_prefix_is_untouched(storage, tmp_path):
    storage.objects[_PREFIX + "scene.usdz"] = b"old scene"
    original = dict(storage.objects)
    with pytest.raises(ValueError, match="occupied"):
        publication.publish_immutable(_result(tmp_path / "source", "new"), _URI)
    assert storage.objects == original


def test_concurrent_writers_never_mix_scene_and_provenance(storage, tmp_path):
    storage.barrier = Barrier(2)
    sources = [_result(tmp_path / label, label) for label in ("first", "second")]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(publication.publish_immutable, source, _URI)
            for source in sources
        ]
        results = [future.exception() for future in futures]
    assert sum(result is None for result in results) == 1
    assert sum(isinstance(result, ValueError) for result in results) == 1
    scene = storage.objects[_PREFIX + "scene.usdz"].decode()
    assert json.loads(storage.objects[_PREFIX + "provenance.json"])["scene"] == scene
    publication.verify_publication(_snapshot(storage, tmp_path / "snapshot"))


@pytest.mark.parametrize(
    "failure", ["scene.usdz", "provenance.json", publication._SEAL]
)
def test_failed_publication_is_unsealed_and_cannot_be_retried(
    storage, tmp_path, failure
):
    source = _result(tmp_path / "source", "first")
    storage.failure = failure
    with pytest.raises(OSError, match="injected"):
        publication.publish_immutable(source, _URI)
    assert _PREFIX + publication._CLAIM in storage.objects
    assert _PREFIX + publication._SEAL not in storage.objects
    with pytest.raises(ValueError):
        publication.verify_publication(_snapshot(storage, tmp_path / "snapshot"))
    original = dict(storage.objects)
    storage.failure = ""
    with pytest.raises(ValueError, match="claimed"):
        publication.publish_immutable(source, _URI)
    assert storage.objects == original
    publication.publish_immutable(source, _URI + "-fresh")
    assert _PREFIX.rstrip("/") + "-fresh/" + publication._SEAL in storage.objects


def test_legacy_writer_between_listing_and_claim_is_not_overwritten(storage, tmp_path):
    storage.racing_object = "provenance.json"
    with pytest.raises(ValueError, match="occupied"):
        publication.publish_immutable(_result(tmp_path / "source", "new"), _URI)
    assert storage.objects[_PREFIX + "provenance.json"] == b"legacy writer"
    assert _PREFIX + "scene.usdz" not in storage.objects
    assert _PREFIX + publication._SEAL not in storage.objects


def test_conflicting_late_member_is_conditionally_rejected(
    storage, tmp_path, monkeypatch
):
    original = publication._claim

    def after_claim(*args):
        location = original(*args)
        storage.objects[_PREFIX + "scene.usdz"] = b"concurrent foreign scene"
        return location

    monkeypatch.setattr(publication, "_claim", after_claim)
    with pytest.raises(ValueError, match="occupied"):
        publication.publish_immutable(_result(tmp_path / "source", "new"), _URI)
    assert storage.objects[_PREFIX + "scene.usdz"] == b"concurrent foreign scene"
    assert _PREFIX + publication._SEAL not in storage.objects


@pytest.mark.parametrize(
    "corruption", ["scene", "provenance", "claim", "extra", "missing"]
)
def test_seal_rejects_changed_or_incomplete_members(tmp_path, corruption):
    source = _result(tmp_path / "source", "first")
    output = tmp_path / "output"
    publication.publish_immutable(source, str(output))
    if corruption == "missing":
        (output / "scene.usdz").unlink()
    else:
        name = {
            "scene": "scene.usdz",
            "provenance": "provenance.json",
            "claim": publication._CLAIM,
            "extra": "unexpected",
        }[corruption]
        (output / name).write_text("{}")
    with pytest.raises(ValueError):
        publication.verify_publication(output)


def test_failed_local_copy_cannot_be_reused(tmp_path, monkeypatch):
    source = _result(tmp_path / "source", "first")
    output = tmp_path / "output"

    def fail(*_args):
        raise OSError("injected copy failure")

    monkeypatch.setattr(publication.shutil, "copyfileobj", fail)
    with pytest.raises(OSError, match="injected"):
        publication.publish_immutable(source, str(output))
    assert (output / publication._CLAIM).is_file()
    assert not (output / publication._SEAL).exists()
    with pytest.raises(FileExistsError):
        publication.publish_immutable(source, str(output))
