"""Concurrency, ordering, and resource-cleanup coverage for run-manifest discovery.

Exercises ``list_runs`` and ``discover_workflow_run_state`` in
``npa.orchestration.skypilot.workflow_state``, which fetch every candidate
run's manifest below a bucket prefix concurrently through one shared boto3
client rather than one client built per candidate.
"""

from __future__ import annotations

import io
import json
import threading
from typing import Any

import pytest

from npa.orchestration.skypilot.workflow_state import (
    WorkflowS3Config,
    WorkflowStateError,
    discover_workflow_run_state,
    get_text,
    list_runs,
)

_BUCKET = "bucket"


class _FakePaginator:
    def __init__(self, client: "_FakeManifestClient") -> None:
        self._client = client

    def paginate(self, *, Bucket: str, Prefix: str):
        contents = [
            {"Key": key}
            for (bucket, key) in sorted(self._client.objects)
            if bucket == Bucket and key.startswith(Prefix)
        ]
        yield {"Contents": contents}


class _FakeManifestClient:
    """A minimal S3 double tracking client-usage invariants under test."""

    def __init__(self, objects: dict[tuple[str, str], bytes]) -> None:
        self.objects = objects
        self.get_object_calls: list[tuple[str, str]] = []
        self.bodies: list[io.BytesIO] = []
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.barrier: threading.Barrier | None = None

    def get_paginator(self, name: str) -> _FakePaginator:
        assert name == "list_objects_v2"
        return _FakePaginator(self)

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        with self._lock:
            self.get_object_calls.append((Bucket, Key))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.barrier is not None:
                self.barrier.wait()
            data = self.objects[(Bucket, Key)]
        finally:
            with self._lock:
                self.active -= 1
        body = io.BytesIO(data)
        self.bodies.append(body)
        return {"Body": body}


def _manifest_bytes(*, run_id: str, workflow_name: str, updated_at: str) -> bytes:
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "workflow_name": workflow_name,
        "stages": {},
        "updated_at": updated_at,
    }
    return json.dumps(payload).encode("utf-8")


def _state_parent(prefix: str = "runs") -> WorkflowS3Config:
    """Build a run-parent config; the actual client comes from the patched factory."""

    return WorkflowS3Config(
        bucket=_BUCKET,
        prefix=prefix,
        endpoint_url="https://storage.example",
        aws_access_key_id="access",
        aws_secret_access_key="secret",
    )


def _patch_client_factory(
    monkeypatch: pytest.MonkeyPatch, client: _FakeManifestClient
) -> list[int]:
    """Patch ``boto3.client`` and return a list whose length is the call count."""

    calls: list[int] = []

    def _factory(*_args: object, **_kwargs: object) -> _FakeManifestClient:
        calls.append(1)
        return client

    monkeypatch.setattr(
        "npa.orchestration.skypilot.workflow_state.boto3.client", _factory
    )
    return calls


def test_list_runs_builds_exactly_one_client_for_the_whole_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objects = {
        (_BUCKET, f"runs/run-{i}/manifest.json"): _manifest_bytes(
            run_id=f"run-{i}",
            workflow_name="wf",
            updated_at=f"2026-01-01T00:00:{i:02d}Z",
        )
        for i in range(20)
    }
    client = _FakeManifestClient(objects)
    calls = _patch_client_factory(monkeypatch, client)

    runs = list_runs(state_parent=_state_parent(), limit=50)

    assert len(runs) == 20
    assert len(calls) == 1, (
        "list_runs must build exactly one client, not one per candidate"
    )


def test_discover_workflow_run_state_builds_exactly_one_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objects = {
        (_BUCKET, f"runs/run-{i}/manifest.json"): _manifest_bytes(
            run_id=f"run-{i}", workflow_name="wf", updated_at="2026-01-01T00:00:00Z"
        )
        for i in range(20)
    }
    client = _FakeManifestClient(objects)
    calls = _patch_client_factory(monkeypatch, client)

    found = discover_workflow_run_state(state_parent=_state_parent(), run_id="run-7")

    assert found is not None
    assert found.prefix == "runs/run-7"
    assert len(calls) == 1


def test_list_runs_filters_component_manifests_and_sorts_newest_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objects = {
        (_BUCKET, "runs/run-a/manifest.json"): _manifest_bytes(
            run_id="run-a", workflow_name="wf", updated_at="2026-01-01T00:00:00Z"
        ),
        (_BUCKET, "runs/run-b/manifest.json"): _manifest_bytes(
            run_id="run-b", workflow_name="wf", updated_at="2026-01-02T00:00:00Z"
        ),
        # A component manifest: no workflow_name/stages, must be excluded.
        (_BUCKET, "runs/npa-src/manifest.json"): json.dumps(
            {"schema_version": 1, "run_id": "npa-src", "package": "npa"}
        ).encode("utf-8"),
    }
    client = _FakeManifestClient(objects)
    _patch_client_factory(monkeypatch, client)

    runs = list_runs(state_parent=_state_parent(), limit=50)

    assert [r["run_id"] for r in runs] == ["run-b", "run-a"]


def test_list_runs_stops_early_without_fetching_the_whole_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    total = 30
    objects = {
        (_BUCKET, f"runs/run-{i:02d}/manifest.json"): _manifest_bytes(
            run_id=f"run-{i:02d}",
            workflow_name="wf",
            updated_at=f"2026-01-01T00:{i:02d}:00Z",
        )
        for i in range(total)
    }
    client = _FakeManifestClient(objects)
    _patch_client_factory(monkeypatch, client)

    runs = list_runs(state_parent=_state_parent(), limit=3)

    assert len(runs) == 3
    # Bounded overshoot: at most one extra in-flight batch beyond the limit,
    # never the whole bucket.
    assert len(client.get_object_calls) < total


def test_discover_workflow_run_state_prefers_newest_and_tie_breaks_by_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objects = {
        (_BUCKET, "runs/attempt-1/manifest.json"): _manifest_bytes(
            run_id="retried-run", workflow_name="wf", updated_at="2026-01-01T00:00:00Z"
        ),
        (_BUCKET, "runs/attempt-2/manifest.json"): _manifest_bytes(
            run_id="retried-run", workflow_name="wf", updated_at="2026-01-02T00:00:00Z"
        ),
        (_BUCKET, "runs/unrelated/manifest.json"): _manifest_bytes(
            run_id="other-run", workflow_name="wf", updated_at="2026-01-03T00:00:00Z"
        ),
    }
    client = _FakeManifestClient(objects)
    _patch_client_factory(monkeypatch, client)

    found = discover_workflow_run_state(
        state_parent=_state_parent(), run_id="retried-run"
    )

    assert found is not None
    assert found.prefix == "runs/attempt-2"


def test_discover_workflow_run_state_tie_breaks_equal_updated_at_by_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objects = {
        (_BUCKET, "runs/attempt-a/manifest.json"): _manifest_bytes(
            run_id="retried-run", workflow_name="wf", updated_at="2026-01-01T00:00:00Z"
        ),
        (_BUCKET, "runs/attempt-z/manifest.json"): _manifest_bytes(
            run_id="retried-run", workflow_name="wf", updated_at="2026-01-01T00:00:00Z"
        ),
    }
    client = _FakeManifestClient(objects)
    _patch_client_factory(monkeypatch, client)

    found = discover_workflow_run_state(
        state_parent=_state_parent(), run_id="retried-run"
    )

    assert found is not None
    assert found.prefix == "runs/attempt-z"


def test_manifest_fetches_close_every_response_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objects = {
        (_BUCKET, f"runs/run-{i}/manifest.json"): _manifest_bytes(
            run_id=f"run-{i}", workflow_name="wf", updated_at="2026-01-01T00:00:00Z"
        )
        for i in range(6)
    }
    client = _FakeManifestClient(objects)
    _patch_client_factory(monkeypatch, client)

    list_runs(state_parent=_state_parent(), limit=50)

    assert len(client.bodies) == 6
    assert all(body.closed for body in client.bodies)


def test_manifest_fetches_bound_in_flight_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.orchestration.skypilot import workflow_state as module

    objects = {
        (_BUCKET, f"runs/run-{i:02d}/manifest.json"): _manifest_bytes(
            run_id=f"run-{i:02d}", workflow_name="wf", updated_at="2026-01-01T00:00:00Z"
        )
        for i in range(40)
    }
    client = _FakeManifestClient(objects)
    _patch_client_factory(monkeypatch, client)

    list_runs(state_parent=_state_parent(), limit=50)

    assert len(client.get_object_calls) == 40
    assert client.max_active <= module._MANIFEST_FETCH_WORKERS


def test_manifest_fetches_run_concurrently_not_serially(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All manifest fetches in one batch must be in flight at once.

    A ``threading.Barrier`` sized to the worker count only releases once
    every party has called ``wait()``. If fetches ran serially, the first
    would block at the barrier forever and this test would fail with a
    deterministic ``BrokenBarrierError`` on timeout instead of a flaky
    wall-clock measurement.
    """

    from npa.orchestration.skypilot import workflow_state as module

    workers = module._MANIFEST_FETCH_WORKERS
    objects = {
        (_BUCKET, f"runs/run-{i}/manifest.json"): _manifest_bytes(
            run_id=f"run-{i}", workflow_name="wf", updated_at="2026-01-01T00:00:00Z"
        )
        for i in range(workers)
    }
    client = _FakeManifestClient(objects)
    client.barrier = threading.Barrier(workers, timeout=5)
    _patch_client_factory(monkeypatch, client)

    runs = list_runs(state_parent=_state_parent(), limit=50)

    assert len(runs) == workers


def test_get_text_closes_body_and_reuses_explicit_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objects = {(_BUCKET, "runs/x/manifest.json"): b'{"ok": true}'}
    client = _FakeManifestClient(objects)
    calls = _patch_client_factory(monkeypatch, client)
    state = WorkflowS3Config(
        bucket=_BUCKET,
        prefix="runs/x",
        endpoint_url="https://storage.example",
        aws_access_key_id="access",
        aws_secret_access_key="secret",
    )

    text = get_text(state, "manifest.json", client=client)

    assert text == '{"ok": true}'
    assert len(client.bodies) == 1
    assert client.bodies[0].closed
    # An explicitly passed client must not trigger a fresh client build.
    assert len(calls) == 0


def test_get_text_wraps_client_construction_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("no credentials on this host")

    monkeypatch.setattr("npa.orchestration.skypilot.workflow_state.boto3.client", _boom)
    state = WorkflowS3Config(
        bucket=_BUCKET,
        prefix="runs/x",
        endpoint_url="https://storage.example",
        aws_access_key_id="access",
        aws_secret_access_key="secret",
    )

    with pytest.raises(WorkflowStateError, match="not found or unreadable"):
        get_text(state, "manifest.json")
