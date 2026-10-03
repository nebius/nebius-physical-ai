"""Verified publication bytes are reusable, but never authorized by metadata alone."""

import asyncio
import hashlib
import io
from dataclasses import replace
from types import SimpleNamespace

from botocore.exceptions import ClientError
from fastapi import HTTPException, Request
import pytest

from npa.cli import agent_stage_runtime as runtime
from npa.workflows.sim2real import publication

from .test_stage_report_read_identity import (
    _BUCKET,
    _install_runtime_dependencies,
    _report_inventory,
    _report_store,
    _run_details,
)
from .test_verified_viewer_downloads import _viewer_context
from .test_stage14_twentieth_agent_controls import _import_rendered_backend


class _VersionedStore:
    def __init__(self, context):
        self.context = context
        self.recording_reads = 0

    def head_object(self, *, Bucket, Key):
        data = self.context.objects[Key]
        return {"ContentLength": len(data), "ETag": hashlib.sha256(data).hexdigest()}

    def get_object(self, *, Bucket, Key, **conditions):
        if Key.endswith(".sim2real-publication.json"):
            data = self.context.journal
        else:
            data = self.context.objects[Key]
            etag = hashlib.sha256(data).hexdigest()
            if conditions.get("IfMatch", etag) != etag:
                raise ClientError(
                    {
                        "Error": {"Code": "PreconditionFailed"},
                        "ResponseMetadata": {"HTTPStatusCode": 412},
                    },
                    "GetObject",
                )
            if Key.endswith(".rrd"):
                self.recording_reads += 1
        return {"Body": io.BytesIO(data)}


def test_repeated_listing_reuses_verified_version_bound_recording(monkeypatch):
    data, targets, inventory = _report_inventory(b"{}")
    uri = next(t["immutable_uri"] for t in targets if t["uri"].endswith(".rrd"))
    journal = publication._journal_bytes(
        transaction_id="a" * 64, attempt_id="b" * 32, state="committed", objects=targets
    )
    store = _VersionedStore(SimpleNamespace(objects=data, journal=journal))
    _install_runtime_dependencies(monkeypatch, store, inventory)
    for _ in range(3):
        runtime._committed_publication_artifacts(store, _BUCKET, "run-a", inventory)
    assert store.recording_reads == 1
    key = uri.removeprefix(f"s3://{_BUCKET}/")
    data[key] = b"X" * len(data[key])
    with pytest.raises(runtime.PublicationConflict, match="bytes"):
        runtime._committed_publication_artifacts(store, _BUCKET, "run-a", inventory)
    assert store.recording_reads == 2


async def _response_bytes(response):
    return b"".join([chunk async for chunk in response.body_iterator])


def test_range_reads_reuse_verified_bytes_and_reject_changed_object(
    monkeypatch, tmp_path
):
    module = _import_rendered_backend(
        monkeypatch, tmp_path, module_name="npa_rendered_verified_range_cache"
    )
    context = _viewer_context(module, tmp_path, True)
    store = _VersionedStore(context)
    for requested, expected in (
        ("bytes=0-3", context.good[:4]),
        ("bytes=4-7", context.good[4:8]),
    ):
        request = Request({"type": "http", "headers": [(b"range", requested.encode())]})
        response_context = module._artifact_content_response_context(
            "run-a", "", context.artifact, False
        )
        response = module._artifact_stream_response(
            store, request, "demo-bucket", context.artifact, response_context
        )
        assert response.status_code == 206
        assert asyncio.run(_response_bytes(response)) == expected
    assert store.recording_reads == 1
    context.objects[context.artifact.key] = b"X" * len(context.good)
    with pytest.raises(HTTPException) as rejected:
        module._artifact_stream_response(
            store, request, "demo-bucket", context.artifact, response_context
        )
    assert rejected.value.status_code == 409
    assert store.recording_reads == 2


@pytest.mark.parametrize("tamper", [False, True])
def test_healthy_large_report_retains_verified_stage_authority(monkeypatch, tamper):
    import json

    good = json.dumps(
        {"stages": {"diagnostic": {"status": "failed"}}, "padding": "x" * 70_000}
    ).encode()
    forged = good.replace(b'"failed"', b'"passed"')
    assert len(good) == len(forged)
    store, inventory = _report_store(good, forged, tamper)
    _install_runtime_dependencies(monkeypatch, store, inventory)
    if tamper:
        with pytest.raises(HTTPException) as rejected:
            _run_details()
        assert rejected.value.status_code == 409
    else:
        response = _run_details()
        stage = next(row for row in response["stages"] if row["id"] == "diagnostic")
        assert stage["status"] == "failed" and stage["authority"] == "authoritative"
    assert all(body.closed for body in store.bodies)


def test_local_report_summary_budget_is_not_a_generation_conflict():
    class UnreadStore:
        def get_object(self, **_kwargs):
            raise AssertionError("known oversized reports must not be read")

    assert (
        runtime._read_bounded_json_object(
            UnreadStore(), "demo-bucket", "report", expected_size=70_000
        )
        is None
    )


@pytest.mark.parametrize("mixed", [False, True])
def test_invalid_inventory_root_is_translated_to_conflict(monkeypatch, mixed):
    store, inventory = _report_store(b"{}", b"", False)
    if mixed:
        item = inventory[0]
        inventory[0] = replace(
            item, key="other/" + item.key, s3_uri="s3://demo-bucket/other/" + item.key
        )
    else:
        inventory = [
            replace(item, key="unknown/object", relative_key="") for item in inventory
        ]
    _install_runtime_dependencies(monkeypatch, store, inventory)
    with pytest.raises(HTTPException) as rejected:
        _run_details()
    assert rejected.value.status_code == 409
