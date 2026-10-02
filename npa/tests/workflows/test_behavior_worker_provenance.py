"""Exercise publication with oversized inputs and bounded diagnostic reads."""

import hashlib
import json
from pathlib import Path

import pytest

from npa.workflows.behavior_challenge import campaign_runner, worker_provenance

from test_behavior_campaign_runner import MemoryStorage


RECEIPT = "s3://example-bucket/run/workers/worker-0.json"
STEM = RECEIPT.removesuffix(".json")


def _index(storage):
    return json.loads(storage.objects[STEM + "/provenance-index.json"][0])


def test_checkpoint_archive_is_never_read_or_reuploaded(tmp_path, monkeypatch):
    archive = tmp_path / "checkpoint.zip"
    with archive.open("wb") as stream:
        stream.truncate(12 * 1024**3)
    (tmp_path / "startup.json").write_text('{"status":"ready"}')
    (tmp_path / "outside.log").symlink_to(archive)
    (tmp_path / "checkpoint").mkdir()
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        assert path != archive, "A staged model archive must not be read as evidence"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    storage = MemoryStorage()
    campaign_runner._publish_worker_provenance(storage, tmp_path, RECEIPT)
    index = _index(storage)
    assert set(index["files"]) == {"startup.json"}
    assert index["excluded_inputs"] == [
        {"name": "checkpoint.zip", "bytes": 12 * 1024**3, "reason": "not_diagnostic"}
    ]
    assert index["excluded_contents_verified"] is False
    assert len(storage.objects) == 2


def test_large_diagnostic_is_reconstructable_and_repeat_safe(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_provenance, "PART_BYTES", 8)
    payload = b"original diagnostic spanning several storage parts"
    (tmp_path / "startup.log").write_bytes(payload)
    storage = MemoryStorage()
    campaign_runner._publish_worker_provenance(storage, tmp_path, RECEIPT)
    original_objects = dict(storage.objects)
    campaign_runner._publish_worker_provenance(storage, tmp_path, RECEIPT)
    assert storage.objects == original_objects
    row = _index(storage)["files"]["startup.log"]
    parts = [storage.objects[p["uri"]][0] for p in row["parts"]]
    assert all(len(part) <= 8 for part in parts)
    assert b"".join(parts) == payload
    assert row["bytes"] == len(payload)
    assert row["sha256"] == hashlib.sha256(payload).hexdigest()
    assert [p["sha256"] for p in row["parts"]] == [
        hashlib.sha256(part).hexdigest() for part in parts
    ]


def test_changed_diagnostic_never_gets_a_complete_index(tmp_path):
    path = tmp_path / "startup.log"
    path.write_bytes(b"before")
    storage = MemoryStorage()

    def mutate_after_put(storage, payload, uri):
        campaign_runner._put_original(storage, payload, uri)
        path.write_bytes(b"changed while publishing")

    with pytest.raises(ValueError, match="changed during publication"):
        worker_provenance.publish_worker_provenance(
            storage, tmp_path, RECEIPT, mutate_after_put
        )
    assert STEM + "/provenance-index.json" not in storage.objects


def test_empty_diagnostic_retains_original_object(tmp_path):
    (tmp_path / "empty.log").touch()
    storage = MemoryStorage()
    campaign_runner._publish_worker_provenance(storage, tmp_path, RECEIPT)
    assert storage.objects[STEM + "/provenance/empty.log"][0] == b""
    assert _index(storage)["files"]["empty.log"]["bytes"] == 0
