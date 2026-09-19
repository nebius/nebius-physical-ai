"""Live correctness evidence for concurrent S3 directory transfer.

Run with ``NPA_INTEGRATION_E2E=1`` and ``NPA_E2E_PROJECT`` set to an
explicitly configured project alias with real, writable S3 storage. Proves
``StorageClient.upload_directory``/``download_directory`` round-trip bytes
correctly for both the many-small-files case and a single file above the
multipart threshold, against real Nebius object storage, and deletes every
seeded object afterward. The wall-clock speedup itself is a separate,
non-committed diagnostic comparison (base vs candidate against the same
real bucket), since comparing two source trees in one pytest run is a
benchmark, not a regression test.
"""

from __future__ import annotations

import hashlib
import os
import uuid

import pytest

from npa.clients.config import resolve_project_storage
from npa.clients.storage import StorageClient

pytestmark = pytest.mark.e2e

_SMALL_FILE_COUNT = 5
_SMALL_FILE_BYTES = 16 * 1024
# s3transfer's default multipart threshold is 8 MiB; exceed it so this
# exercises the multipart path the adaptive per-file transfer config sizes.
_MULTIPART_FILE_BYTES = 12 * 1024 * 1024


def _sha256(path) -> str:
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@pytest.fixture
def live_bucket():
    project = os.environ.get("NPA_E2E_PROJECT", "").strip()
    if not project:
        pytest.skip("Set NPA_E2E_PROJECT to an explicitly configured test project")
    storage = resolve_project_storage(project)
    if not all(
        (
            storage.checkpoint_bucket,
            storage.endpoint_url,
            storage.aws_access_key_id,
            storage.aws_secret_access_key,
        )
    ):
        pytest.fail("Selected live project needs complete object storage configuration")
    bucket = storage.checkpoint_bucket.removeprefix("s3://").split("/", 1)[0]
    prefix = "storage-transfer-live-test/" + uuid.uuid4().hex + "/"
    client = StorageClient(
        endpoint_url=storage.endpoint_url,
        aws_access_key_id=storage.aws_access_key_id,
        aws_secret_access_key=storage.aws_secret_access_key,
    )
    try:
        yield client, bucket, prefix
    finally:
        paginator = client.s3.get_paginator("list_objects_v2")
        keys = [
            obj["Key"]
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix)
            for obj in page.get("Contents", [])
        ]
        for key in keys:
            client.s3.delete_object(Bucket=bucket, Key=key)
        remaining = client.s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
        assert not remaining.get("Contents"), (
            "Live test fixture objects were not fully deleted"
        )


def _round_trip(client, bucket, prefix, tmp_path, *, scenario: str, sizes: list[int]):
    local = tmp_path / scenario
    local.mkdir()
    expected = {}
    for index, size in enumerate(sizes):
        path = local / f"artifact-{index:03d}.bin"
        path.write_bytes(os.urandom(size))
        expected[path.name] = _sha256(path)

    uri = f"s3://{bucket}/{prefix}{scenario}/"
    client.upload_directory(str(local), uri)

    destination = tmp_path / f"{scenario}-download"
    client.download_directory(uri, str(destination))

    actual = {path.name: _sha256(path) for path in destination.iterdir()}
    assert actual == expected, f"{scenario}: readback hash mismatch"
    assert len(actual) == len(sizes)


def test_many_small_files_round_trip_with_matching_hashes(live_bucket, tmp_path):
    client, bucket, prefix = live_bucket
    _round_trip(
        client,
        bucket,
        prefix,
        tmp_path,
        scenario="small-files",
        sizes=[_SMALL_FILE_BYTES] * _SMALL_FILE_COUNT,
    )


def test_one_multipart_sized_file_round_trips_with_matching_hash(live_bucket, tmp_path):
    client, bucket, prefix = live_bucket
    _round_trip(
        client,
        bucket,
        prefix,
        tmp_path,
        scenario="multipart-file",
        sizes=[_MULTIPART_FILE_BYTES],
    )
