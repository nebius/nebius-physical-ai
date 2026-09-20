from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from npa.clients.storage import StoragePreconditionFailed
from npa.workbench.nurec import s3_probe


class _Paginator:
    def __init__(self, storage):
        self.storage = storage

    def paginate(self, *, Bucket, Prefix):
        del Bucket
        yield {
            "Contents": [
                {"Key": key, "Size": len(value[0]), "ETag": value[1]}
                for key, value in sorted(self.storage.objects.items())
                if key.startswith(Prefix)
            ]
        }


class Storage:
    def __init__(self, *, allow_overwrite: bool = False):
        self.s3 = self
        self.objects: dict[str, tuple[bytes, str]] = {}
        self.allow_overwrite = allow_overwrite

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return _Paginator(self)

    def put_bytes_conditional(self, payload, uri, *, if_none_match):
        assert if_none_match is True
        key = uri.split("/", 3)[-1]
        if key in self.objects and not self.allow_overwrite:
            raise StoragePreconditionFailed("expected fixture conflict")
        etag = '"' + hashlib.md5(payload, usedforsecurity=False).hexdigest() + '"'
        self.objects[key] = (bytes(payload), etag)
        return etag

    def read_bytes_with_etag(self, uri):
        key = uri.split("/", 3)[-1]
        return self.objects.get(key)

    def delete_object(self, *, Bucket, Key, IfMatch):
        del Bucket
        assert self.objects[Key][1] == IfMatch
        self.objects.pop(Key, None)
        return {"ResponseMetadata": {"HTTPStatusCode": 204}}


def test_probe_proves_conditional_round_trip_enumeration_and_cleanup(
    tmp_path: Path,
) -> None:
    storage = Storage()
    output = tmp_path / "probe.json"

    receipt = s3_probe.probe_s3_handoff(
        "s3://example-bucket/run-owned/future/", output, storage_client=storage
    )

    assert receipt["status"] == "ok"
    assert receipt["payload_bytes"] == 257
    assert receipt["conditional_create"] is True
    assert receipt["exact_readback"] is True
    assert receipt["conditional_nonoverwrite_rejected"] is True
    assert receipt["enumerated_exactly_once"] is True
    assert receipt["absent_after_delete"] is True
    assert storage.objects == {}
    assert output.stat().st_mode & 0o777 == 0o600
    assert "example-bucket" not in output.read_text()


def test_probe_failure_still_deletes_the_probe_object(tmp_path: Path) -> None:
    storage = Storage(allow_overwrite=True)

    with pytest.raises(s3_probe.NcoreS3ProbeError, match="non-overwrite control"):
        s3_probe.probe_s3_handoff(
            "s3://example-bucket/run-owned/future/",
            tmp_path / "probe.json",
            storage_client=storage,
        )

    assert storage.objects == {}
    assert not (tmp_path / "probe.json").exists()


def test_probe_response_loss_after_committed_put_still_cleans_up(
    tmp_path: Path,
) -> None:
    class LostResponse(Storage):
        def put_bytes_conditional(self, payload, uri, *, if_none_match):
            super().put_bytes_conditional(payload, uri, if_none_match=if_none_match)
            raise RuntimeError("simulated response loss")

    storage = LostResponse()

    with pytest.raises(
        s3_probe.NcoreS3ProbeError, match="object-store probe operation failed"
    ):
        s3_probe.probe_s3_handoff(
            "s3://example-bucket/run-owned/future/",
            tmp_path / "probe.json",
            storage_client=storage,
        )

    assert storage.objects == {}


@pytest.mark.parametrize(
    "uri",
    [
        "example-bucket/prefix/",
        "s3://example-bucket/",
        "s3://example-bucket/not-a-prefix",
        "s3://example-bucket/run/../other/",
    ],
)
def test_probe_rejects_unsafe_or_non_s3_prefix(uri: str, tmp_path: Path) -> None:
    with pytest.raises(s3_probe.NcoreS3ProbeError):
        s3_probe.probe_s3_handoff(
            uri, tmp_path / "probe.json", storage_client=Storage()
        )
