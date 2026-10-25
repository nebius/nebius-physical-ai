"""Verified publication bytes are reusable, but never authorized by metadata alone."""

import asyncio
import hashlib
import io
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import uuid

from botocore.exceptions import ClientError
from fastapi import HTTPException, Request
import pytest

from npa.cli import agent_stage_runtime as runtime
from npa.workflows.sim2real import publication
from npa.agent_backend.publication_reader import PublicationObject

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
        self.meta = SimpleNamespace(
            endpoint_url="https://storage.example.test", region_name="test"
        )
        self._request_signer = SimpleNamespace(
            _credentials=SimpleNamespace(access_key="synthetic-" + uuid.uuid4().hex)
        )

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


@pytest.fixture(autouse=True)
def isolated_runtime_cache():
    runtime._clear_verified_publication_cache()
    yield
    runtime._clear_verified_publication_cache()


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


def _cache_target(store, *, run="run-a", value=b"verified"):
    key = f"{run}/reports/generations/frozen/sim2real.rrd"
    store.context.objects[key] = value
    return PublicationObject(
        canonical_uri=f"s3://demo-bucket/{run}/reports/sim2real.rrd",
        immutable_uri=f"s3://demo-bucket/{key}",
        sha256=hashlib.sha256(value).hexdigest(),
        size_bytes=len(value),
    )


def _cached_bytes(module=runtime):
    if module._PUBLICATION_CACHE_DIRECTORY is None:
        return 0
    root = Path(module._PUBLICATION_CACHE_DIRECTORY.name)
    return sum(path.stat().st_size for path in root.iterdir() if path.is_file())


def test_cache_has_independent_entry_and_byte_lru_budgets(monkeypatch):
    store = _VersionedStore(SimpleNamespace(objects={}))
    monkeypatch.setattr(runtime, "_PUBLICATION_CACHE_MAX_ENTRIES", 2)
    monkeypatch.setattr(runtime, "_PUBLICATION_CACHE_MAX_BYTES", 12)
    for index in range(8):
        target = _cache_target(store, run=f"run-{index}", value=b"verified")
        body, _ = runtime._verified_publication_object_body(
            store, "demo-bucket", target
        )
        assert body.read() == b"verified"
        body.close()
        assert len(runtime._PUBLICATION_CACHE_ENTRIES) <= 2
        assert _cached_bytes() <= 12
    assert len(runtime._PUBLICATION_CACHE_ENTRIES) == 1
    assert len(runtime._PUBLICATION_CACHE_LOCKS) == 16


def test_access_generation_and_replacement_reclaim_superseded_bytes(monkeypatch):
    store = _VersionedStore(SimpleNamespace(objects={}))
    for generation in range(8):
        monkeypatch.setattr(
            runtime, "_AGENT_RUN_CURSOR_GENERATION", generation, raising=False
        )
        value = bytes([generation]) * 131072
        target = _cache_target(store, value=value)
        body, _ = runtime._verified_publication_object_body(
            store, "demo-bucket", target
        )
        assert body.read() == value
        body.close()
        assert len(runtime._PUBLICATION_CACHE_ENTRIES) == 1
        assert _cached_bytes() == len(value)
    runtime._clear_verified_publication_cache()
    assert _cached_bytes() == 0


def test_eviction_does_not_corrupt_an_open_verified_reader(monkeypatch):
    store = _VersionedStore(SimpleNamespace(objects={}))
    monkeypatch.setattr(runtime, "_PUBLICATION_CACHE_MAX_ENTRIES", 1)
    left = _cache_target(store, run="left", value=b"original")
    held, _ = runtime._verified_publication_object_body(store, "demo-bucket", left)
    right = _cache_target(store, run="right", value=b"successor")
    current, _ = runtime._verified_publication_object_body(store, "demo-bucket", right)
    try:
        assert current.read() == b"successor"
        assert held.read() == b"original"
        assert len(runtime._PUBLICATION_CACHE_ENTRIES) == 1
    finally:
        current.close()
        held.close()


def _fail_cache_install(monkeypatch, phase, opened):
    identity = runtime._publication_cache_identity
    original_open = runtime._publication_cache_open

    def identity_failure(path):
        if path.suffix == ".blob":
            raise OSError("synthetic postrename identity failure")
        return identity(path)

    def tracked_open(*args):
        stream = original_open(*args)
        opened.append(stream)
        if phase == "authentication":

            def unreadable(*_args):
                raise OSError("synthetic authentication read failure")

            stream.read = unreadable
        elif phase == "authentication-conflict":
            stream.read = lambda *_args: b""
        return stream

    if phase == "identity":
        monkeypatch.setattr(runtime, "_publication_cache_identity", identity_failure)
    elif phase == "open":
        fdopen = runtime._publication_os.fdopen

        def tracked_fdopen(*args):
            stream = fdopen(*args)
            opened.append(stream)
            return stream

        def failed_stat(*_args):
            raise OSError("synthetic postrename open failure")

        monkeypatch.setattr(runtime._publication_os, "fdopen", tracked_fdopen)
        monkeypatch.setattr(runtime._publication_os, "fstat", failed_stat)
    else:
        monkeypatch.setattr(runtime, "_publication_cache_open", tracked_open)


@pytest.mark.parametrize(
    "phase", ["identity", "open", "authentication", "authentication-conflict"]
)
def test_late_install_failure_reclaims_owned_leaf_and_preserves_readers(
    monkeypatch, phase
):
    store = _VersionedStore(SimpleNamespace(objects={}))
    monkeypatch.setattr(runtime, "HTTPException", HTTPException)
    monkeypatch.setattr(runtime, "_PUBLICATION_CACHE_MAX_ENTRIES", 1)
    monkeypatch.setattr(runtime, "_PUBLICATION_CACHE_MAX_BYTES", 20)
    target = _cache_target(store, run="held", value=b"P" * 16)
    held, _ = runtime._verified_publication_object_body(store, "demo-bucket", target)
    opened = []
    _fail_cache_install(monkeypatch, phase, opened)
    try:
        for index in range(3):
            target = _cache_target(store, run=f"failed-{index}", value=b"V" * 16)
            expected = (
                runtime.PublicationConflict
                if phase == "authentication-conflict"
                else HTTPException
            )
            with pytest.raises(expected) as rejected:
                runtime._verified_publication_object_body(store, "demo-bucket", target)
            if phase != "authentication-conflict":
                assert rejected.value.status_code == 503
            assert not runtime._PUBLICATION_CACHE_ENTRIES
            assert _cached_bytes() == 0
            assert all(stream.closed for stream in opened)
        runtime._clear_verified_publication_cache()
        assert _cached_bytes() == 0 and held.read() == b"P" * 16
        assert store.recording_reads == 4
    finally:
        held.close()


def test_unidentified_client_and_oversize_object_are_verified_without_retention(
    monkeypatch,
):
    store = _VersionedStore(SimpleNamespace(objects={}))
    target = _cache_target(store, value=b"verified")
    monkeypatch.setattr(store, "meta", None)
    assert runtime._publication_client_scope(store) is None
    for _ in range(2):
        body, _ = runtime._verified_publication_object_body(
            store, "demo-bucket", target
        )
        assert body.read() == b"verified"
        body.close()
    assert store.recording_reads == 2 and not runtime._PUBLICATION_CACHE_ENTRIES
    monkeypatch.setattr(
        store,
        "meta",
        SimpleNamespace(
            endpoint_url="https://storage.example.test", region_name="test"
        ),
    )
    monkeypatch.setattr(runtime, "_PUBLICATION_CACHE_MAX_BYTES", 4)
    body, _ = runtime._verified_publication_object_body(store, "demo-bucket", target)
    assert body.read() == b"verified"
    body.close()
    assert not runtime._PUBLICATION_CACHE_ENTRIES and _cached_bytes() == 0


def test_failed_install_cleanup_retains_ownership_and_primary_error(monkeypatch):
    store = _VersionedStore(SimpleNamespace(objects={}))
    target = _cache_target(store, value=b"V" * 16)
    monkeypatch.setattr(runtime, "HTTPException", HTTPException)
    monkeypatch.setattr(runtime, "_PUBLICATION_CACHE_MAX_ENTRIES", 1)
    monkeypatch.setattr(runtime, "_PUBLICATION_CACHE_MAX_BYTES", 20)
    unlink = Path.unlink

    def failed_cleanup(path, **kwargs):
        if path.suffix == ".blob":
            raise OSError("synthetic cleanup failure")
        return unlink(path, **kwargs)

    with monkeypatch.context() as fault:
        _fail_cache_install(fault, "identity", [])
        fault.setattr(Path, "unlink", failed_cleanup)
        with pytest.raises(HTTPException) as rejected:
            runtime._verified_publication_object_body(store, "demo-bucket", target)
        assert rejected.value.status_code == 503
        assert (
            str(rejected.value.__context__) == "synthetic postrename identity failure"
        )
        assert len(runtime._PUBLICATION_CACHE_ENTRIES) == 1
        assert _cached_bytes() == 16
    runtime._clear_verified_publication_cache()
    assert not runtime._PUBLICATION_CACHE_ENTRIES and _cached_bytes() == 0


def test_cache_local_tamper_is_not_served(monkeypatch):
    store = _VersionedStore(SimpleNamespace(objects={}))
    target = _cache_target(store)
    body, _ = runtime._verified_publication_object_body(store, "demo-bucket", target)
    body.close()
    path, _identity = next(iter(runtime._PUBLICATION_CACHE_ENTRIES.values()))
    path.chmod(0o600)
    path.write_bytes(b"tampered")
    body, _ = runtime._verified_publication_object_body(store, "demo-bucket", target)
    assert body.read() == b"verified"
    body.close()
    assert store.recording_reads == 2


def test_access_invalidation_during_read_cannot_repopulate_cache(monkeypatch):
    store = _VersionedStore(SimpleNamespace(objects={}))
    target = _cache_target(store)
    original = store.get_object

    def invalidating_read(**kwargs):
        response = original(**kwargs)
        runtime._clear_verified_publication_cache()
        return response

    monkeypatch.setattr(store, "get_object", invalidating_read)
    body, _ = runtime._verified_publication_object_body(store, "demo-bucket", target)
    assert body.read() == b"verified"
    body.close()
    assert not runtime._PUBLICATION_CACHE_ENTRIES and _cached_bytes() == 0


def test_actual_access_invalidator_unlinks_verified_cache(monkeypatch, tmp_path):
    module = _import_rendered_backend(
        monkeypatch, tmp_path, module_name="npa_rendered_cache_invalidation"
    )
    context = _viewer_context(module, tmp_path, True)
    store = _VersionedStore(context)
    target = _cache_target(store)
    body, _ = module._verified_publication_object_body(store, "demo-bucket", target)
    body.close()
    assert _cached_bytes(module) == len(b"verified")
    module._invalidate_agent_artifact_discovery()
    assert not module._PUBLICATION_CACHE_ENTRIES and _cached_bytes(module) == 0


@pytest.mark.parametrize("retained", [True, False])
def test_cache_disk_failure_is_service_unavailable_not_integrity_conflict(
    monkeypatch, retained
):
    store = _VersionedStore(SimpleNamespace(objects={}))
    target = _cache_target(store)
    monkeypatch.setattr(runtime, "HTTPException", HTTPException)
    if not retained:
        monkeypatch.setattr(store, "meta", None)

    def unavailable(*_args, **_kwargs):
        raise OSError("synthetic no-space failure")

    monkeypatch.setattr(
        runtime._publication_tempfile, "NamedTemporaryFile", unavailable
    )
    monkeypatch.setattr(runtime._publication_tempfile, "TemporaryFile", unavailable)
    with pytest.raises(HTTPException) as rejected:
        runtime._verified_publication_object_body(store, "demo-bucket", target)
    assert rejected.value.status_code == 503
    assert not runtime._PUBLICATION_CACHE_ENTRIES


def test_legacy_report_keeps_the_original_small_summary_budget(monkeypatch):
    sizes = []

    class Body(io.BytesIO):
        def read(self, size=-1):
            sizes.append(size)
            return super().read(size)

    body = Body(b'{"padding":"' + b"x" * 70000 + b'"}')
    store = SimpleNamespace(get_object=lambda **_kwargs: {"Body": body})
    snapshot = SimpleNamespace(journaled=False)
    monkeypatch.setattr(
        runtime,
        "_assert_legacy_publication_snapshot_still_unjournaled",
        lambda *_args: None,
    )
    assert (
        runtime._read_publication_bound_json_object(
            store,
            "demo-bucket",
            "run-a/reports/sim2real-report.json",
            publication_snapshot=snapshot,
        )
        is None
    )
    assert sizes == [runtime._MAX_STAGE_EVIDENCE_BYTES + 1] and body.closed
