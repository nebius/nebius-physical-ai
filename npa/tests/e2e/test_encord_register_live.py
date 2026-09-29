"""Exercise register discovery's conditional byte hashing on a real S3 object."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import pytest

from npa.clients.storage import StorageClient
from npa.workbench.encord.push import discover_objects
from npa.workbench.encord.storage import S3ObjectStorageGateway

ROOT = Path(__file__).resolve().parents[3]


def _hash_targets():
    path = os.environ.get("NPA_E2E_ENCORD_HASH_CONFIG", "")
    if not path:
        pytest.skip("Set NPA_E2E_ENCORD_HASH_CONFIG to selected S3 bucket/prefix")
    config = json.loads(Path(path).read_text())
    if not config.get("bucket") or not config.get("prefix", "").strip("/"):
        raise ValueError("Explicit bucket and nonempty prefix are required")
    return config["bucket"], config["prefix"].strip("/") + "/" + uuid4().hex


def test_unconfigured_register_hash_test_skips_without_storage(monkeypatch):
    monkeypatch.delenv("NPA_E2E_ENCORD_HASH_CONFIG", raising=False)
    with pytest.raises(pytest.skip.Exception):
        _hash_targets()


def _trace_discovery(storage, bucket, key, uri):
    events = []

    def record(params, model, **kwargs):
        del kwargs
        if params.get("Bucket") == bucket and params.get("Key") == key:
            events.append(
                {"operation": model.name, "conditional": bool(params.get("IfMatch"))}
            )

    event_name = "before-parameter-build.s3"
    storage.s3.meta.events.register(event_name, record)
    try:
        items, skipped = discover_objects(
            S3ObjectStorageGateway(storage), uri, "videos-images", require_checksum=True
        )
    finally:
        storage.s3.meta.events.unregister(event_name, record)
    return items, skipped, events


@pytest.mark.e2e
def test_live_register_discovery_hashes_without_provider_sha256(tmp_path):
    bucket, prefix = _hash_targets()
    payload = (
        ROOT / "npa/tests/browser/cypress/fixtures/browser-compatible.mp4"
    ).read_bytes()
    storage = StorageClient.from_environment()
    key = prefix + "/clip.mp4"
    storage.s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=payload,
        ContentType="video/mp4",
        ChecksumAlgorithm="CRC32",
    )
    uri = f"s3://{bucket}/{prefix}/"
    metadata = S3ObjectStorageGateway(storage).head(uri + "clip.mp4")
    assert metadata.checksum_kind not in {"sha256", "s3_checksum_sha256"}
    items, skipped, events = _trace_discovery(storage, bucket, key, uri)
    assert not skipped and len(items) == 1
    assert items[0].source_checksum_kind == "sha256"
    assert items[0].source_checksum == hashlib.sha256(payload).hexdigest()
    assert items[0].source_size == len(payload)
    assert [event["operation"] for event in events] == [
        "HeadObject",
        "GetObject",
        "HeadObject",
    ]
    assert events[1]["conditional"] is True
    _write_evidence(tmp_path, uri, payload, items[0], events)


def _write_evidence(tmp_path, uri, payload, item, events):
    evidence = Path(os.environ.get("NPA_E2E_ENCORD_EVIDENCE_DIR", str(tmp_path)))
    evidence.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (evidence / ("register-hash-" + uuid4().hex + ".json")).open("x") as output:
        os.chmod(output.name, 0o600)
        json.dump(
            {
                "source_uri": uri + "clip.mp4",
                "bytes": len(payload),
                "sha256": item.source_checksum,
                "events": events,
            },
            output,
            indent=2,
        )
