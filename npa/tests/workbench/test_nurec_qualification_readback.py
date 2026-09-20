from __future__ import annotations

import json
from pathlib import Path

import pytest

from npa.workbench.nurec.qualification_readback import (
    NcoreQualificationReadbackError,
    readback_qualification,
)


class _Paginator:
    def __init__(self, storage):
        self.storage = storage

    def paginate(self, *, Bucket, Prefix):
        assert Bucket == "private"
        assert Prefix == "run/"
        self.storage.list_calls += 1
        records = [
            {
                "Key": Prefix + path,
                "Size": len(payload),
                "ETag": f'"etag-{index}"',
            }
            for index, (path, payload) in enumerate(
                sorted(self.storage.files.items()), start=1
            )
        ]
        if self.storage.change_after_download and self.storage.list_calls > 1:
            records.append({"Key": Prefix + "late.json", "Size": 1, "ETag": '"late"'})
        yield {"Contents": records}


class _S3:
    def __init__(self, storage):
        self.storage = storage

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return _Paginator(self.storage)


class _Storage:
    def __init__(self, *, change_after_download=False):
        self.files = {
            "evidence/runtime.json": b'{"status":"pass"}',
            "reports/final.json": b'{"has_rrd":true}',
        }
        self.change_after_download = change_after_download
        self.list_calls = 0
        self.s3 = _S3(self)

    def download_directory(self, uri, destination):
        assert uri == "s3://private/run/"
        root = Path(destination)
        for name, payload in self.files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        return destination


def test_complete_readback_binds_stable_s3_and_local_bytes(tmp_path: Path) -> None:
    destination = tmp_path / "readback"
    receipt_path = tmp_path / "readback.json"
    receipt = readback_qualification(
        "s3://private/run/",
        destination,
        receipt_path,
        storage_client=_Storage(),
    )
    assert receipt["status"] == "pass"
    assert receipt["object_count"] == 2
    assert [item["path"] for item in receipt["local_inventory"]] == [
        "evidence/runtime.json",
        "reports/final.json",
    ]
    assert all(
        path.stat().st_mode & 0o777 == 0o600
        for path in destination.rglob("*")
        if path.is_file()
    )
    assert receipt_path.stat().st_mode & 0o777 == 0o600
    assert json.loads(receipt_path.read_text()) == receipt


def test_complete_readback_rejects_inventory_drift(tmp_path: Path) -> None:
    with pytest.raises(
        NcoreQualificationReadbackError, match="changed during readback"
    ):
        readback_qualification(
            "s3://private/run/",
            tmp_path / "readback",
            tmp_path / "readback.json",
            storage_client=_Storage(change_after_download=True),
        )
    assert not (tmp_path / "readback.json").exists()


def test_complete_readback_requires_new_destination(tmp_path: Path) -> None:
    destination = tmp_path / "readback"
    destination.mkdir()
    with pytest.raises(NcoreQualificationReadbackError, match="must be new"):
        readback_qualification(
            "s3://private/run/",
            destination,
            tmp_path / "readback.json",
            storage_client=_Storage(),
        )
