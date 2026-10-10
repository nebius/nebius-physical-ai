from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from npa.workbench.nurec.source_staging import (
    NcoreSourceStagingError,
    stage_source_archive,
)


class _Paginator:
    def __init__(self, storage):
        self.storage = storage

    def paginate(self, *, Bucket, Prefix):
        yield {
            "Contents": [
                {"Key": key, "Size": len(body), "ETag": '"etag"'}
                for (bucket, key), body in self.storage.objects.items()
                if bucket == Bucket and key.startswith(Prefix)
            ]
        }


class _S3:
    def __init__(self, storage):
        self.storage = storage

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return _Paginator(self.storage)

    def put_object(self, *, Bucket, Key, Body, ContentType, IfNoneMatch):
        assert ContentType == "application/zip"
        assert IfNoneMatch == "*"
        assert (Bucket, Key) not in self.storage.objects
        self.storage.objects[(Bucket, Key)] = Body.read()
        if self.storage.lose_response:
            raise OSError("response lost")
        return {"ETag": '"etag"'}

    def head_object(self, *, Bucket, Key):
        assert (Bucket, Key) in self.storage.objects
        return {"ETag": '"etag"'}

    def delete_object(self, *, Bucket, Key, IfMatch):
        assert IfMatch == '"etag"'
        self.storage.objects.pop((Bucket, Key), None)
        return {"ResponseMetadata": {"HTTPStatusCode": 204}}


class _Storage:
    def __init__(self, *, lose_response=False):
        self.objects = {}
        self.lose_response = lose_response
        self.s3 = _S3(self)

    def download_file(self, uri, destination):
        bucket, key = uri.removeprefix("s3://").split("/", 1)
        Path(destination).write_bytes(self.objects[(bucket, key)])

    def put_bytes_conditional(self, payload, uri, *, if_none_match, content_type):
        assert if_none_match is True
        assert content_type == "application/json"
        bucket, key = uri.removeprefix("s3://").split("/", 1)
        assert (bucket, key) not in self.objects
        self.objects[(bucket, key)] = payload
        return '"etag"'

    def read_bytes_with_etag(self, uri):
        bucket, key = uri.removeprefix("s3://").split("/", 1)
        body = self.objects.get((bucket, key))
        return None if body is None else (body, '"etag"')


@pytest.mark.parametrize("lose_response", [False, True])
def test_source_staging_conditionally_writes_and_reads_back(
    lose_response: bool, tmp_path: Path
) -> None:
    source = tmp_path / "source.zip"
    source.write_bytes(b"real-pinned-source")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    storage = _Storage(lose_response=lose_response)
    receipt_path = tmp_path / "evidence/staging.json"

    receipt = stage_source_archive(
        source,
        "s3://private/run/source/source.zip",
        expected_sha256=expected,
        receipt_path=receipt_path,
        scratch_dir=tmp_path / "scratch",
        storage_client=storage,
    )

    assert receipt["status"] == "pass"
    assert receipt["response_loss_recovered"] is lose_response
    assert receipt["source_sha256"] == expected
    assert receipt["before_objects"] == 0
    assert receipt["after_objects"] == 2
    assert receipt["attribution_sha256"]
    assert json.loads(receipt_path.read_text()) == receipt


def test_source_staging_rejects_a_nonfresh_prefix(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    source.write_bytes(b"real-pinned-source")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    storage = _Storage()
    storage.objects[("private", "run/source/existing")] = b"existing"

    with pytest.raises(NcoreSourceStagingError, match="not fresh"):
        stage_source_archive(
            source,
            "s3://private/run/source/source.zip",
            expected_sha256=expected,
            receipt_path=tmp_path / "staging.json",
            scratch_dir=tmp_path / "scratch",
            storage_client=storage,
        )
