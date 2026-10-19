from __future__ import annotations

import json
from pathlib import Path

import pytest

from npa.workbench.nurec.colmap import ColmapConversionRequest, NcoreConversionError
from npa.workbench.nurec.source_control import (
    NcoreSourceControlError,
    run_wrong_source_control,
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


class _Storage:
    def __init__(self):
        self.objects = {}
        self.s3 = _S3(self)


def _request(tmp_path: Path) -> ColmapConversionRequest:
    return ColmapConversionRequest(
        input_path="s3://private/source.zip",
        output_path="s3://private/run/wrong-source/",
        expected_archive_sha256="a" * 64,
        cache_dir=tmp_path / "cache",
        scratch_dir=tmp_path / "scratch",
    )


def test_wrong_source_control_binds_pre_extract_failure_and_zero_outputs(
    tmp_path: Path,
) -> None:
    storage = _Storage()
    seen = []

    def converter(request, *, storage_client):
        assert storage_client is storage
        seen.append(request)
        raise NcoreConversionError(
            "source archive SHA-256 differs from the required digest",
            phase="source_digest_pre_extract",
        )

    receipt_path = tmp_path / "evidence/control.json"
    receipt = run_wrong_source_control(
        _request(tmp_path),
        expected_archive_sha256="a" * 64,
        receipt_path=receipt_path,
        storage_client=storage,
        converter=converter,
    )

    assert seen[0].expected_archive_sha256 == "0" + "a" * 63
    assert receipt["status"] == "pass"
    assert receipt["failure_phase"] == "source_digest_pre_extract"
    assert receipt["native_started"] is False
    assert receipt["after_output_objects"] == 0
    assert json.loads(receipt_path.read_text()) == receipt
    assert receipt_path.stat().st_mode & 0o777 == 0o600


def test_wrong_source_control_rejects_nonfresh_output_prefix(tmp_path: Path) -> None:
    storage = _Storage()
    storage.objects[("private", "run/wrong-source/existing")] = b"owned-elsewhere"

    with pytest.raises(NcoreSourceControlError, match="not fresh"):
        run_wrong_source_control(
            _request(tmp_path),
            expected_archive_sha256="a" * 64,
            receipt_path=tmp_path / "control.json",
            storage_client=storage,
            converter=lambda *_args, **_kwargs: None,
        )
