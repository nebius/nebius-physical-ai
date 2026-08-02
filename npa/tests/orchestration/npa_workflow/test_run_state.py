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


# ── Resource-honest manifests for submitted runs ─────────────────────────────
# A cluster submit used to leave no `npa.workflow.run.v1` manifest at all, and the
# manifest the local path did write carried no resource profile -- so a run that
# requested N accelerators was indistinguishable from a CPU-only run, and the
# insights `gpus` metric (which reads steps[].resources_profile.accelerators) had
# no producer anywhere in the system.


class _FakeStep:
    def __init__(self, state, resources="", profile=None, tool_ref="", iteration=None):
        self.state = state
        self.resources = resources
        self.resources_profile = profile or {}
        self.tool_ref = tool_ref
        self.iteration = iteration
        self.group = ""
        self.loop_label = ""


def test_plan_step_records_carry_the_resource_profile() -> None:
    from npa.orchestration.npa_workflow.run_state import SUBMITTED_STATUS, plan_step_records

    records = plan_step_records(
        [
            _FakeStep("train", "trainer-gpu", {"accelerators": "RTXPRO6000:4", "cpus": 16}, "workbench.rl.policy_train"),
            _FakeStep("aggregate", "control-cpu", {"cpus": 4}),
        ]
    )
    assert records[0]["resources_profile"]["accelerators"] == "RTXPRO6000:4"
    assert records[0]["tool_ref"] == "workbench.rl.policy_train"
    assert records[0]["status"] == SUBMITTED_STATUS
    assert records[1]["resources_profile"] == {"cpus": 4}
    assert "tool_ref" not in records[1]


def test_persist_submitted_manifest_writes_a_resource_honest_manifest() -> None:
    from npa.orchestration.npa_workflow import run_state as rs

    written: dict[tuple[str, str], bytes] = {}

    class _Store(rs.RunStateStore):
        def _write(self, key, payload):  # type: ignore[override]
            written[(self.bucket, key)] = json.dumps(payload).encode("utf-8")

    original = rs.store_for_config
    rs.store_for_config = lambda config, *, run_id: _Store(  # type: ignore[assignment]
        bucket=str(config.get("bucket")), prefix=str(config.get("prefix") or run_id)
    )
    try:
        uri = rs.persist_submitted_manifest(
            {"bucket": "bkt", "prefix": "runs/demo-1"},
            run_id="demo-1",
            workflow="demo",
            api_version="npa.workflow/v0.0.1",
            steps=[_FakeStep("train", "trainer-gpu", {"accelerators": "H100:2"})],
        )
    finally:
        rs.store_for_config = original

    assert uri == "s3://bkt/runs/demo-1"
    payload = json.loads(written[("bkt", "runs/demo-1/npa-workflow/manifest.json")])
    assert payload["schema_version"] == "npa.workflow.run.v1"
    assert payload["status"] == "submitted"
    assert payload["steps"][0]["resources_profile"]["accelerators"] == "H100:2"


def test_persist_submitted_manifest_without_a_bucket_is_a_no_op() -> None:
    from npa.orchestration.npa_workflow.run_state import persist_submitted_manifest

    assert persist_submitted_manifest({}, run_id="r", workflow="w", steps=[]) == ""


def test_dispatch_step_records_carry_resources_for_any_executor() -> None:
    """The runtime tier's executor record must still describe the step's resources."""
    from npa.orchestration.npa_workflow.interpreter import PlanStep, _dispatch_step

    step = PlanStep(
        state="retrain",
        resources="trainer-gpu",
        resources_profile={"accelerators": "RTXPRO6000:2", "cpus": 16},
    )

    class _WaveExecutor:
        def execute(self, plan_step):
            return {"state": plan_step.state, "status": "ok", "job_id": "42"}

    record = _dispatch_step(step, _WaveExecutor())
    assert record["resources_profile"]["accelerators"] == "RTXPRO6000:2"
    assert record["resources"] == "trainer-gpu"
    assert record["job_id"] == "42"
