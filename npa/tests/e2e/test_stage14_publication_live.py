"""Exercise Stage14 publication authority against real task-scoped S3 objects.

These are transport controls, not a policy or full Sim2Real pipeline proof.
Set NPA_INTEGRATION_E2E=1, NPA_STAGE14_PUBLICATION_LIVE=1 and NPA_E2E_PROJECT
to an authorized configured project. Only unique fixture prefixes are mutated.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path

import pytest

from npa.clients.config import resolve_project_storage
from npa.clients.storage import StorageClient, StoragePreconditionFailed
from npa.lifecycle_intent import OperationIntent, operation_intent
from npa.workflows.sim2real.publication import (
    MutablePublicationTransaction,
    PublicationConflict,
    read_verified_committed_publication_bytes,
    recover_interrupted_publication,
    remote_object_snapshot,
    remote_object_version,
    resolve_committed_publication_snapshot,
    upload_immutable_file,
    verify_committed_publication_object,
)

pytestmark = pytest.mark.e2e


def _publication_client(project: str):
    with operation_intent(OperationIntent.OBSERVE):
        settings = resolve_project_storage(
            project, include_shared_credentials=False, include_environment=False
        )
    if not all(
        (
            settings.checkpoint_bucket,
            settings.endpoint_url,
            settings.aws_access_key_id,
            settings.aws_secret_access_key,
        )
    ):
        raise RuntimeError("Exact project storage is not fully configured")
    client = StorageClient(
        endpoint_url=settings.endpoint_url,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
    )
    bucket = settings.checkpoint_bucket.removeprefix("s3://").split("/", 1)[0]
    return client, bucket


@pytest.fixture
def publication_store():
    project = os.environ.get("NPA_E2E_PROJECT", "").strip()
    if os.environ.get("NPA_STAGE14_PUBLICATION_LIVE") != "1" or not project:
        pytest.skip("Enable Stage14 live controls with an exact configured project")
    client, bucket = _publication_client(project)
    prefix = f"cursor-explore-stage14-controls/{uuid.uuid4().hex}/"
    client.probe_list_access(f"s3://{bucket}/{prefix}")
    try:
        yield client, f"s3://{bucket}/{prefix.rstrip('/')}"
    finally:
        pages = client.s3.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=prefix
        )
        for page in pages:
            for item in page.get("Contents", []):
                key = item["Key"]
                assert key.startswith(prefix)
                client.s3.delete_object(Bucket=bucket, Key=key)


def _generation(client, root: str, directory: Path):
    generation = uuid.uuid4().hex
    objects = []
    for filename in ("sim2real-report.json", "sim2real.rrd", "sim2real.mcap"):
        path = directory / filename
        path.write_bytes(f"explicit publication transport control: {filename}".encode())
        immutable = f"{root}/reports/generations/{generation}/{filename}"
        upload_immutable_file(client, path, immutable)
        objects.append((path, f"{root}/reports/{filename}", immutable))
    path = directory / "component.json"
    path.write_bytes(b'{"scope":"explicit publication transport control"}')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    immutable = f"{root}/components/history/stage_14/{digest}.json"
    upload_immutable_file(client, path, immutable)
    objects.append((path, f"{root}/components/stage_14.json", immutable))
    return generation, objects


def _transaction(client, root, generation):
    journal = f"{root}/reports/.sim2real-publication.json"
    return MutablePublicationTransaction(
        client,
        lock_uri=journal,
        lock_snapshot=remote_object_snapshot(client, journal),
        transaction_id=generation,
    )


def _queue_generation(client, transaction, objects):
    for path, alias, immutable in objects:
        transaction.replace_file(
            path, alias, remote_object_version(client, alias), immutable_uri=immutable
        )


def _assert_committed_bytes(client, objects):
    snapshot = resolve_committed_publication_snapshot(client, objects[0][1])
    assert snapshot.journaled
    for path, alias, immutable in objects:
        assert verify_committed_publication_object(client, snapshot, alias) == immutable
        assert (
            read_verified_committed_publication_bytes(client, snapshot, alias)
            == path.read_bytes()
        )


def test_live_streamed_write_and_delete_reject_stale_versions(
    publication_store, tmp_path
):
    client, root = publication_store
    uri = f"{root}/conditional-control"
    path = tmp_path / "payload"
    path.write_bytes(b"first transport control")
    etag = client.put_file_conditional(str(path), uri, if_none_match=True)
    first = remote_object_snapshot(client, uri)
    assert first is not None and first.etag == etag
    path.write_bytes(b"second transport control")
    with pytest.raises(StoragePreconditionFailed):
        client.put_file_conditional(str(path), uri, if_none_match=True)
    assert remote_object_snapshot(client, uri) == first
    second_etag = client.put_file_conditional(str(path), uri, if_match=etag)
    second = remote_object_snapshot(client, uri)
    assert second is not None and second.payload == path.read_bytes()
    with pytest.raises(StoragePreconditionFailed):
        client.put_file_conditional(str(path), uri, if_match=etag)
    with pytest.raises(StoragePreconditionFailed):
        client.delete_file_conditional(uri, if_match=etag)
    assert remote_object_snapshot(client, uri) == second
    client.delete_file_conditional(uri, if_match=second_etag)
    assert remote_object_snapshot(client, uri) is None


def test_live_committed_generation_rejects_stale_writer_and_replaced_bytes(
    publication_store, tmp_path
):
    client, root = publication_store
    generation, objects = _generation(client, root, tmp_path)
    winner = _transaction(client, root, generation)
    stale = _transaction(client, root, generation)
    with winner:
        _queue_generation(client, winner, objects)
    _assert_committed_bytes(client, objects)
    journal = remote_object_snapshot(client, winner.lock_uri)
    with pytest.raises(PublicationConflict):
        with stale:
            _queue_generation(client, stale, objects)
    assert remote_object_snapshot(client, winner.lock_uri) == journal
    path, alias, immutable = objects[0]
    snapshot = resolve_committed_publication_snapshot(client, alias)
    before_alias = remote_object_snapshot(client, alias)
    bucket, key = immutable.removeprefix("s3://").split("/", 1)
    client.s3.put_object(Bucket=bucket, Key=key, Body=b"hostile replaced bytes")
    with pytest.raises(PublicationConflict):
        verify_committed_publication_object(client, snapshot, alias)
    with pytest.raises(PublicationConflict):
        upload_immutable_file(client, path, immutable)
    assert remote_object_snapshot(client, alias) == before_alias
    assert remote_object_snapshot(client, winner.lock_uri) == journal


def test_live_interrupted_publication_recovers_with_real_object_readback(
    publication_store, tmp_path, monkeypatch
):
    client, root = publication_store
    generation, objects = _generation(client, root, tmp_path)
    transaction = _transaction(client, root, generation)
    original = client.put_file_conditional
    writes = []

    def interrupt_second_alias(local_file, uri, **kwargs):
        writes.append(uri)
        if len(writes) == 2:
            raise RuntimeError("injected publication interruption")
        return original(local_file, uri, **kwargs)

    monkeypatch.setattr(client, "put_file_conditional", interrupt_second_alias)
    with pytest.raises(RuntimeError, match="injected publication interruption"):
        with transaction:
            _queue_generation(client, transaction, objects)
    assert len(writes) == 2
    with pytest.raises(PublicationConflict):
        resolve_committed_publication_snapshot(client, objects[0][1])
    monkeypatch.setattr(client, "put_file_conditional", original)
    assert recover_interrupted_publication(
        client,
        lock_uri=transaction.lock_uri,
        lock_snapshot=remote_object_snapshot(client, transaction.lock_uri),
    )
    _assert_committed_bytes(client, objects)
    assert not recover_interrupted_publication(
        client,
        lock_uri=transaction.lock_uri,
        lock_snapshot=remote_object_snapshot(client, transaction.lock_uri),
    )
