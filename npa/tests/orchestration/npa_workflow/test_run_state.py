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
