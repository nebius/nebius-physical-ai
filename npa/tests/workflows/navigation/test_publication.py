"""Prove conditional navigation publication with hostile readback and writer races."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from threading import Lock

import pytest

from npa.clients.storage import StoragePreconditionFailed
from npa.workflows.navigation import artifacts


class ConditionalStorage:
    """Emulate provider-atomic object creation, with controllable readback faults."""

    def __init__(self):
        self.objects = {}
        self.lock = Lock()
        self.fail_after_claim = False
        self.replace_bundle = False

    def put_bytes_conditional(self, body, uri, *, if_none_match):
        assert if_none_match is True
        with self.lock:
            if uri in self.objects:
                raise StoragePreconditionFailed("already claimed")
            if self.fail_after_claim and "/attempts/" in uri:
                raise OSError("injected upload failure")
            self.objects[uri] = bytes(body)
        return hashlib.sha256(body).hexdigest()

    def read_bytes_with_etag(self, uri):
        with self.lock:
            body = self.objects.get(uri)
        return None if body is None else (body, hashlib.sha256(body).hexdigest())

    def download_directory(self, uri, target):
        root = Path(target)
        root.mkdir()
        for key, body in self.objects.copy().items():
            if key.startswith(uri.rstrip("/") + "/"):
                path = root / key[len(uri.rstrip("/")) + 1 :]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(body)
        if self.replace_bundle:
            (root / "policy.pt").write_bytes(b"coherently-replaced-checkpoint")
            artifacts.write_json(root / "checksums.json", artifacts._files(root))


@pytest.fixture
def publication(tmp_path, monkeypatch):
    storage = ConditionalStorage()
    monkeypatch.setattr(artifacts.StorageClient, "from_environment", lambda: storage)
    root = tmp_path / "result"
    root.mkdir()
    (root / "policy.pt").write_bytes(b"original-checkpoint-fixture")
    return storage, root, "s3://fixture-bucket/run/train"


def test_exact_readback_completes_and_resolves_attempt(publication, tmp_path):
    storage, root, uri = publication
    expected = artifacts._files(root)
    artifacts.publish(root, uri)
    completion = json.loads(storage.objects[uri + "/completion.json"])
    assert completion["files"] == expected
    assert completion == json.loads(storage.objects[uri + "/claim.json"])
    assert "/attempts/" in next(
        key for key in storage.objects if key.endswith("/policy.pt")
    )
    readback = artifacts.materialize(uri, tmp_path / "consumer")
    assert artifacts._files(readback) == expected


def test_duplicate_or_late_publish_never_overwrites_completed_stage(publication):
    storage, root, uri = publication
    artifacts.publish(root, uri)
    original = dict(storage.objects)
    (root / "policy.pt").write_bytes(b"late-conflict")
    with pytest.raises(StoragePreconditionFailed):
        artifacts.publish(root, uri)
    assert storage.objects == original


def test_two_concurrent_claims_have_exactly_one_winner(publication, tmp_path):
    storage, first, uri = publication
    parent = tmp_path / "other"
    parent.mkdir()
    second = parent / "result"
    second.mkdir()
    (second / "policy.pt").write_bytes(b"competing-checkpoint")

    def write(root):
        try:
            artifacts.publish(root, uri)
            return "published"
        except StoragePreconditionFailed:
            return "refused"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(write, [first, second]))
    assert sorted(results) == ["published", "refused"]
    assert len([key for key in storage.objects if key.endswith("policy.pt")]) == 1


def test_upload_failure_never_completes_or_reuses_claim(publication, tmp_path):
    storage, root, uri = publication
    storage.fail_after_claim = True
    with pytest.raises(OSError, match="upload failure"):
        artifacts.publish(root, uri)
    assert uri + "/completion.json" not in storage.objects
    with pytest.raises(ValueError, match="incomplete"):
        artifacts.materialize(uri, tmp_path / "consumer")
    storage.fail_after_claim = False
    with pytest.raises(StoragePreconditionFailed):
        artifacts.publish(root, uri)


def test_coherent_bundle_and_manifest_replacement_cannot_pass_readback(publication):
    storage, root, uri = publication
    storage.replace_bundle = True
    with pytest.raises(ValueError, match="original local manifest"):
        artifacts.publish(root, uri)
    assert uri + "/completion.json" not in storage.objects


def test_corrupt_completed_attempt_cannot_be_consumed(publication, tmp_path):
    storage, root, uri = publication
    artifacts.publish(root, uri)
    storage.replace_bundle = True
    with pytest.raises(ValueError, match="original local manifest"):
        artifacts.materialize(uri, tmp_path / "consumer")
