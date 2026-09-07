# Prove immutable archive and verified restore against explicitly selected real S3 storage.
"""Manual CPU live test using completed public/synthetic native CLIP result trees.

The operator preflights the exact workload bucket and supplies an owner-private
JSON config with source_paths, archive_prefix and evidence_dir. The trajectory
dataset must never be used as archive storage. Object creation intents and
readback receipts stay in evidence_dir; only those exact owned objects are deleted.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import uuid
from urllib.parse import urlsplit

import pytest

from npa.clients.storage import StorageClient

pytestmark = pytest.mark.e2e


def _write(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class _RecordedStorage:
    def __init__(self, client, evidence):
        self.client, self.evidence = client, evidence
        self.owned = {}

    def put_bytes_conditional(self, payload, uri, *, if_none_match):
        if uri not in self.owned:
            assert self.client.read_bytes_with_etag(uri) is None
            self.owned[uri] = {"intent_sha256": hashlib.sha256(payload).hexdigest(), "created": False}
            _write(self.evidence, self.owned)
        token = self.client.put_bytes_conditional(payload, uri, if_none_match=if_none_match)
        self.owned[uri]["created"] = True
        _write(self.evidence, self.owned)
        return token

    def read_bytes_with_etag(self, uri):
        return self.client.read_bytes_with_etag(uri)


def test_actual_s3_archive_restore_and_corruption(monkeypatch):
    if not (config_path := os.environ.get("NPA_RAY_CLIP_ARCHIVE_LIVE_CONFIG")):
        pytest.skip("requires private source provenance and preflighted owned workload storage")
    path = Path(config_path)
    assert path.stat().st_uid == os.getuid() and path.stat().st_mode & 0o077 == 0
    config = json.loads(path.read_text())
    evidence = Path(config["evidence_dir"])
    evidence.mkdir(mode=0o700, parents=True, exist_ok=True)
    assert evidence.stat().st_mode & 0o077 == 0
    directory = Path(__file__).parents[2] / "workflows/workbench/ray-clip-development"
    monkeypatch.syspath_prepend(str(directory))
    names = ("archive", "archive_inventory", "validation")
    saved = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    module = importlib.import_module("archive")
    client = StorageClient.from_environment()
    storage = _RecordedStorage(client, evidence / "objects.json")
    receipts = []
    try:
        for index, source_path in enumerate(config["source_paths"]):
            source = Path(source_path)
            original = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in source.rglob("*") if p.is_file()}
            prefix = config["archive_prefix"].rstrip("/") + "/" + uuid.uuid4().hex
            receipt = module.archive(source, prefix, storage)
            assert module.archive(source, prefix, storage) == receipt
            for repetition in range(2):
                restored = evidence / f"restored-{index}-{repetition}"
                assert module.restore(prefix, restored, receipt["manifest_sha256"], storage) == receipt
                observed = {str(p.relative_to(restored)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in restored.rglob("*") if p.is_file()}
                assert observed == original
            # A different complete tree must conflict at the same immutable prefix.
            conflict = evidence / f"conflict-{index}"
            shutil.copytree(source, conflict)
            report = conflict / "report.json"
            value = json.loads(report.read_text())
            value["archive_conflict_probe"] = True
            _write(report, value)
            files = {str(p.relative_to(conflict)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in conflict.rglob("*") if p.is_file() and p.name not in {"SHA256SUMS", "sha256.json"}}
            if (conflict / "sha256.json").exists():
                _write(conflict / "sha256.json", files)
                files["sha256.json"] = hashlib.sha256((conflict / "sha256.json").read_bytes()).hexdigest()
            (conflict / "SHA256SUMS").write_text("".join(f"{v}  {k}\n" for k, v in sorted(files.items())))
            with pytest.raises(ValueError, match="conflicting"):
                module.archive(conflict, prefix, storage)
            # Deliberately corrupt only a proven test-owned object, with its current ETag.
            uri = prefix + "/files/preview.png"
            assert storage.owned[uri]["created"]
            _, token = client.read_bytes_with_etag(uri)
            client.put_bytes_conditional(b"deliberate test corruption", uri, if_match=token)
            failed = evidence / f"corrupt-{index}"
            with pytest.raises(ValueError, match="corrupt"):
                module.restore(prefix, failed, receipt["manifest_sha256"], storage)
            assert not failed.exists() and not list(evidence.glob(".clip-restore-*"))
            with pytest.raises(ValueError, match="conflicting"):
                module.archive(source, prefix, storage)
            receipts.append({**receipt, "identical_retry": True, "verified_restores": 2,
                             "conflict_refused": True, "corruption_refused": True})
        _write(evidence / "result.json", {"receipts": receipts})
    finally:
        for uri, record in storage.owned.items():
            if record["created"]:
                parsed = urlsplit(uri)
                client.s3.delete_object(Bucket=parsed.netloc, Key=parsed.path[1:])
                assert client.read_bytes_with_etag(uri) is None
                record["deleted_and_absent"] = True
                _write(storage.evidence, storage.owned)
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(saved)
