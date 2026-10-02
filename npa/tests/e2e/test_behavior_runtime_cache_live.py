"""Exercise repeated runtime preparation with real S3 transport and CPU files."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tarfile
from urllib.parse import urlsplit
import uuid

import pytest

from npa.clients.config import resolve_project_storage
from npa.clients.storage import StorageClient
from npa.workflows.behavior_challenge.runtime_cache import prepare_runtime

pytestmark = pytest.mark.e2e


def _storage_scope() -> tuple[StorageClient, str, str]:
    config = os.environ.get("NPA_BEHAVIOR_RUNTIME_CACHE_LIVE_CONFIG")
    if not os.environ.get("NPA_BEHAVIOR_RUNTIME_CACHE_LIVE_CONFIG"):
        pytest.skip("requires operator-selected private storage for CPU cache checks")
    settings = json.loads(Path(config).read_text())
    target = urlsplit(settings["prefix"])
    storage = resolve_project_storage(
        settings["project"], include_shared_credentials=False, include_environment=False
    )
    bucket = storage.checkpoint_bucket.removeprefix("s3://").split("/", 1)[0]
    assert target.scheme == "s3" and target.netloc == bucket
    assert target.path.strip("/") and not target.query and not target.fragment
    assert ".." not in target.path.split("/")
    client = StorageClient(
        endpoint_url=storage.endpoint_url,
        aws_access_key_id=storage.aws_access_key_id,
        aws_secret_access_key=storage.aws_secret_access_key,
    )
    prefix = target.path.strip("/") + "/" + uuid.uuid4().hex
    return client, bucket, prefix


def _archive() -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, data in (
            (".local/python/bin/python", b"cache identity fixture; never executed"),
            ("work/runtime.txt", b"real S3 round trip"),
        ):
            member = tarfile.TarInfo(name)
            member.size = len(data)
            member.mode = 0o755
            archive.addfile(member, io.BytesIO(data))
    return output.getvalue()


def _manifest(archive_uri: str, archive: bytes) -> bytes:
    return json.dumps(
        {
            "schema": "npa.behavior.private-runtime.v1",
            "allowed_directories": ["work", ".local/python"],
            "archives": [
                {
                    "uri": archive_uri,
                    "sha256": hashlib.sha256(archive).hexdigest(),
                    "bytes": len(archive),
                }
            ],
        }
    ).encode()


def _exercise_cache(client: StorageClient, uri: str, manifest: bytes, home: Path):
    digest = hashlib.sha256(manifest).hexdigest()
    workspace = home / "work"
    first = prepare_runtime(client, client, uri, digest, home, workspace)
    executable = home / ".local/python/bin/python"
    original = executable.stat()
    second = prepare_runtime(client, client, uri, digest, home, workspace)
    assert second == first
    assert executable.stat().st_ino == original.st_ino
    assert executable.stat().st_mtime_ns == original.st_mtime_ns
    executable.write_bytes(b"changed base")
    with pytest.raises(ValueError, match="identity differs"):
        prepare_runtime(client, client, uri, digest, home, workspace)
    shutil.rmtree(home / ".local/python")
    assert prepare_runtime(client, client, uri, digest, home, workspace) == first
    assert executable.read_bytes() == b"cache identity fixture; never executed"


def test_repeated_preparation_reuses_exact_base_and_rejects_changes(tmp_path):
    """Verify real downloads, unchanged reuse, tamper rejection, and fresh restore."""
    client, bucket, prefix = _storage_scope()
    archive_key, manifest_key = prefix + "/base.tar.gz", prefix + "/manifest.json"
    archive = _archive()
    manifest = _manifest(f"s3://{bucket}/{archive_key}", archive)
    downloads = []
    client.s3.meta.events.register(
        "before-parameter-build.s3.GetObject",
        lambda params, **_: downloads.append(params["Key"]),
    )
    try:
        client.s3.put_object(Bucket=bucket, Key=archive_key, Body=archive)
        client.s3.put_object(Bucket=bucket, Key=manifest_key, Body=manifest)
        _exercise_cache(client, f"s3://{bucket}/{manifest_key}", manifest, tmp_path)
        assert downloads.count(archive_key) == 1
        assert downloads.count(manifest_key) == 4
    finally:
        for key in (archive_key, manifest_key):
            client.s3.delete_object(Bucket=bucket, Key=key)
