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

from .s3_fixture_cleanup import delete_owned_prefix

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
        delete_owned_prefix(client.s3, bucket, prefix)


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


def test_exact_slash_ending_object_wins_over_its_children(live_bucket, tmp_path):
    """Prove Nebius listing order preserves exact-key precedence without HEAD.

    A slash-ending object is legal S3 data, even when child objects exist.
    ``download_path`` skips HEAD for this spelling and resolves the exact key
    through its bounded listing before considering a directory download.
    """
    client, bucket, prefix = live_bucket
    exact_key = f"{prefix}exact-object/"
    payload = b"exact slash-ending object bytes\n"
    client.s3.put_object(Bucket=bucket, Key=exact_key + "child.bin", Body=b"child")
    client.s3.put_object(Bucket=bucket, Key=exact_key, Body=payload)
    client.s3.put_object(Bucket=bucket, Key=exact_key + "sibling.bin", Body=b"sibling")

    target = tmp_path / "exact-object.bin"
    result = client.download_path(f"s3://{bucket}/{exact_key}", str(target))

    assert result == str(target)
    assert target.is_file()
    assert target.read_bytes() == payload
