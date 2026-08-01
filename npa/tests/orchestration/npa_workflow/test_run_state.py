from __future__ import annotations

import json

from npa.orchestration.npa_workflow.run_state import RunManifest, RunStateStore


def test_run_state_store_roundtrip() -> None:
    store: dict[tuple[str, str], bytes] = {}

    def writer(bucket: str, key: str, body: bytes) -> None:
        store[(bucket, key)] = body

    def reader(bucket: str, key: str) -> str:
        return store[(bucket, key)].decode("utf-8")

    state_store = RunStateStore(
        bucket="bucket",
        prefix="runs/demo",
        reader=reader,
        writer=writer,
    )
    manifest = RunManifest(
        workflow="demo",
        run_id="demo-1",
        api_version="npa.workflow/v0.0.1",
        status="running",
    )
    state_store.write_manifest(manifest)
    state_store.append_step(manifest, {"state": "augment", "status": "ok"})
    loaded = state_store.read_manifest()
    assert loaded is not None
    assert loaded.run_id == "demo-1"
    assert loaded.steps[0]["state"] == "augment"
    status_payload = json.loads(store[("bucket", "runs/demo/npa-workflow/status.json")])
    assert status_payload["status"] == "running"


def test_runtime_run_state_roundtrip_is_separate_from_the_manifest() -> None:
    """The runtime ledger is an additional document; RunManifest is untouched."""

    from npa.orchestration.npa_workflow.run_state import (
        RUNTIME_SCHEMA_VERSION,
        RuntimeRunState,
        runtime_key,
    )

    store: dict[tuple[str, str], bytes] = {}

    def reader(bucket: str, key: str) -> str:
        # Contract of the reader seam (and of the real S3 path): a missing object
        # raises FileNotFoundError. Anything else must propagate, so a transient
        # storage error can never be mistaken for "no ledger" and silently make
        # --resume resubmit every wave.
        try:
            return store[(bucket, key)].decode("utf-8")
        except KeyError as exc:
            raise FileNotFoundError(f"s3://{bucket}/{key}") from exc

    state_store = RunStateStore(
        bucket="bucket",
        prefix="runs/demo",
        reader=reader,
        writer=lambda bucket, key, body: store.__setitem__((bucket, key), body),
    )

    assert state_store.read_runtime_state() is None  # nothing written yet

    runtime_state = RuntimeRunState(
        workflow="demo", run_id="demo-1", api_version="npa.workflow/v0.0.1"
    )
    runtime_state.record_wave({"key": "001|serial|:a:-", "status": "running", "job_id": "7"})
    state_store.write_runtime_state(runtime_state)
    # Same key updated in place, not appended twice.
    runtime_state.record_wave({"key": "001|serial|:a:-", "status": "succeeded", "job_id": "7"})
    runtime_state.decisions.append({"decision": "promote_checkpoint"})
    runtime_state.watermarks["ingest"] = {"objects": 2}
    state_store.write_runtime_state(runtime_state)

    loaded = state_store.read_runtime_state()
    assert loaded is not None
    assert loaded.schema_version == RUNTIME_SCHEMA_VERSION
    assert loaded.run_prefix_uri == "s3://bucket/runs/demo"
    assert [wave["status"] for wave in loaded.waves] == ["succeeded"]
    assert loaded.completed_wave("001|serial|:a:-") is not None
    assert loaded.completed_wave("002|serial|:b:-") is None
    assert loaded.decisions[0]["decision"] == "promote_checkpoint"
    assert loaded.watermarks["ingest"]["objects"] == 2
    # Written next to, not instead of, the run manifest.
    assert ("bucket", runtime_key("runs/demo")) in store
    assert ("bucket", "runs/demo/npa-workflow/manifest.json") not in store


def test_completed_wave_ignores_failed_attempts() -> None:
    from npa.orchestration.npa_workflow.run_state import RuntimeRunState

    state = RuntimeRunState(workflow="demo", run_id="demo-1")
    state.record_wave({"key": "001", "status": "failed", "attempt": 1})
    assert state.completed_wave("001") is None
    state.record_wave({"key": "001", "status": "succeeded", "attempt": 2})
    assert state.completed_wave("001")["attempt"] == 2


def test_read_runtime_state_propagates_unexpected_storage_errors() -> None:
    """A transient read error must not look like "no ledger" (resume safety)."""

    import pytest

    from npa.orchestration.npa_workflow.run_state import RunStateStore as Store

    def angry_reader(bucket: str, key: str) -> str:
        raise PermissionError(f"denied s3://{bucket}/{key}")

    store = Store(bucket="bucket", prefix="runs/demo", reader=angry_reader, writer=lambda *_: None)
    with pytest.raises(PermissionError):
        store.read_runtime_state()
