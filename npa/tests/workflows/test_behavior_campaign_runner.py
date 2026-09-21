"""Exercise durable campaign recovery against original JSON and decoded video."""

from contextlib import contextmanager, suppress
import json
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pytest

from npa.clients.storage import StoragePreconditionFailed
from npa.workflows.behavior_challenge import campaign, campaign_runner, serving_identity
from npa.workflows.behavior_challenge.case_store import CaseAlreadyStarted, CaseStore
from npa.workflows.behavior_challenge.campaign_status import panel_status


class MemoryStorage:
    def __init__(self):
        self.objects = {}
        self.revision = 0

    def read_bytes_with_etag(self, uri):
        return self.objects.get(uri)

    def put_bytes_conditional(
        self, payload, uri, *, if_match="", if_none_match=False, **_
    ):
        current = self.objects.get(uri)
        if (if_none_match and current) or (
            if_match and (not current or current[1] != if_match)
        ):
            raise StoragePreconditionFailed("conditional conflict")
        self.revision += 1
        version = str(self.revision)
        self.objects[uri] = (payload, version)
        return version

    def download_file(self, uri, destination):
        Path(destination).write_bytes(self.objects[uri][0])


@pytest.fixture
def fixture(tmp_path):
    registry = ["picking_up_trash"] + [f"task_{index}" for index in range(99)]
    artifact = {"sha256": "a" * 64, "bytes": 1}
    identity = campaign.freeze_policy_identity(
        "fixture", {"checkpoint": artifact, "serving": artifact}
    )
    panel = campaign.declare_panel(identity, registry, registry[:1], "development")
    storage = MemoryStorage()
    store = CaseStore(storage, "s3://example-bucket/campaign", panel["panel_id"])
    return panel, campaign.partition_panel(panel, 5), store, tmp_path


def _write_original(case, output):
    stem = f"{case['task']}_{case['instance_id']}_0"
    (output / "json").mkdir(exist_ok=True)
    (output / "videos").mkdir(exist_ok=True)
    distances = {key: 1.0 for key in ("base", "left", "right")}
    metrics = {key: case[key] for key in ("task", "instance_id", "rollout_id")}
    metrics.update(
        steps=2,
        success=False,
        q_score={"final": 0.0},
        agent_distance=distances,
        normalized_agent_distance=distances,
        time={"simulator_steps": 2, "simulator_time": 2 / 30, "normalized_time": 10.0},
    )
    (output / f"json/{stem}.json").write_text(json.dumps(metrics, indent=3) + "\n")
    with av.open(str(output / f"videos/{stem}.mp4"), "w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        for value in (0, 128):
            frame = av.VideoFrame.from_ndarray(
                np.full((16, 16, 3), value, dtype=np.uint8), format="rgb24"
            )
            container.mux(stream.encode(frame))
        container.mux(stream.encode())


def test_status_reads_mixed_case_states_without_mutating_or_scoring(fixture):
    panel, _, store, _ = fixture
    claimed, started, complete = panel["cases"][:3]
    store.claim(claimed, "worker-a")
    store.start(store.claim(started, "worker-b"))
    version = store.start(store.claim(complete, "worker-c"))
    store.complete(version, {**complete, "q_score": 1.0})
    before = dict(store.storage.objects)

    result = panel_status(store.storage, panel, "s3://example-bucket/campaign")

    assert result["counts"] == {
        "unclaimed": 7,
        "claimed": 1,
        "started": 1,
        "complete": 1,
    }
    assert not result["all_case_records_complete"]
    assert not result["artifact_bytes_verified"]
    assert not result["live_worker_state_verified"]
    assert "q_score" not in json.dumps(result)
    assert store.storage.objects == before


def test_status_requires_aggregation_even_with_all_complete_records(fixture):
    panel, _, store, _ = fixture
    for case in panel["cases"]:
        store.complete(store.start(store.claim(case, "worker")), case)

    result = panel_status(store.storage, panel, "s3://example-bucket/campaign")

    assert result["all_case_records_complete"]
    assert result["aggregation_required_before_comparison"]
    assert not result["artifact_bytes_verified"]


def test_status_does_not_turn_storage_errors_into_unclaimed_cases(fixture, monkeypatch):
    panel, _, store, _ = fixture

    def denied(_uri):
        raise PermissionError("access denied")

    monkeypatch.setattr(store.storage, "read_bytes_with_etag", denied)
    with pytest.raises(PermissionError, match="access denied"):
        panel_status(store.storage, panel, "s3://example-bucket/campaign")


def test_each_case_gets_a_fresh_managed_policy_context(fixture):
    panel, partition, store, workspace = fixture
    events = []

    @contextmanager
    def prepare(case, output):
        del output
        events.append(("start", case["instance_id"]))
        try:
            yield
        finally:
            events.append(("stop", case["instance_id"]))

    def execute(case, output):
        events.append(("evaluate", case["instance_id"]))
        _write_original(case, output)

    campaign_runner.run_partition(
        panel, partition, 0, store, workspace, execute, prepare_case=prepare
    )

    assert events == [
        ("start", 311),
        ("evaluate", 311),
        ("stop", 311),
        ("start", 316),
        ("evaluate", 316),
        ("stop", 316),
    ]


def test_resume_reuses_completed_case_and_never_repeats_started_case(fixture):
    panel, partition, store, workspace = fixture
    calls = []

    def interrupted(case, output):
        calls.append(case["instance_id"])
        if len(calls) == 2:
            raise RuntimeError("worker terminated")
        _write_original(case, output)

    with pytest.raises(RuntimeError, match="terminated"):
        campaign_runner.run_partition(
            panel, partition, 0, store, workspace, interrupted
        )
    with pytest.raises(CaseAlreadyStarted):
        campaign_runner.run_partition(
            panel, partition, 0, store, workspace, interrupted
        )
    assert calls == [311, 316]
    assert store.read(panel["cases"][0]).record["state"] == "complete"
    assert store.read(panel["cases"][5]).record["state"] == "started"


def test_upload_interruption_recovers_original_then_continues(fixture, monkeypatch):
    panel, partition, store, workspace = fixture
    original = campaign_runner._record_originals
    calls = []

    def execute(case, output):
        calls.append(case["instance_id"])
        _write_original(case, output)

    def unavailable(*_):
        raise ConnectionError("storage disconnected")

    monkeypatch.setattr(campaign_runner, "_record_originals", unavailable)
    with pytest.raises(ConnectionError):
        campaign_runner.run_partition(panel, partition, 0, store, workspace, execute)
    monkeypatch.setattr(campaign_runner, "_record_originals", original)
    records = campaign_runner.run_partition(
        panel, partition, 0, store, workspace, execute
    )
    assert calls == [311, 316]
    assert len(records) == 2
    assert all(record["q_score"] == 0.0 for record in records)
    again = campaign_runner.run_partition(
        panel, partition, 0, store, workspace / "new-pod", execute
    )
    assert again == records
    assert calls == [311, 316]


def test_uploaded_originals_recover_after_completion_write_failure(
    fixture, monkeypatch
):
    panel, partition, store, workspace = fixture
    complete = store.complete
    calls = []

    def execute(case, output):
        calls.append(case["instance_id"])
        _write_original(case, output)

    def disconnected(*_):
        raise ConnectionError("completion lost")

    monkeypatch.setattr(store, "complete", disconnected)
    with pytest.raises(ConnectionError):
        campaign_runner.run_partition(panel, partition, 0, store, workspace, execute)
    monkeypatch.setattr(store, "complete", complete)
    records = campaign_runner.run_partition(
        panel, partition, 0, store, workspace / "fresh", execute
    )
    assert len(records) == 2
    assert calls == [311, 316]


def test_reuse_rejects_changed_original_video(fixture):
    panel, partition, store, workspace = fixture
    campaign_runner.run_partition(
        panel, partition, 0, store, workspace, _write_original
    )
    video_uri = next(uri for uri in store.storage.objects if uri.endswith(".mp4"))
    store.storage.objects[video_uri] = (b"tampered video", "changed")
    with pytest.raises(ValueError, match="SHA-256"):
        campaign_runner.run_partition(
            panel, partition, 0, store, workspace / "recovery", _write_original
        )


def test_unfinished_evaluator_files_cannot_be_promoted(fixture):
    panel, partition, store, workspace = fixture

    def crash_after_outputs(case, output):
        _write_original(case, output)
        raise RuntimeError("evaluator failed")

    with suppress(RuntimeError):
        campaign_runner.run_partition(
            panel, partition, 0, store, workspace, crash_after_outputs
        )
    with pytest.raises(CaseAlreadyStarted):
        campaign_runner.run_partition(
            panel, partition, 0, store, workspace, _write_original
        )


def test_partial_panel_cannot_be_aggregated(fixture):
    panel, partition, store, workspace = fixture
    campaign_runner.run_partition(
        panel, partition, 0, store, workspace, _write_original
    )
    with pytest.raises(ValueError, match="Complete panel"):
        campaign_runner.aggregate_stored_panel(panel, store, workspace / "aggregate")


def test_serving_identity_binds_configuration_and_adapter_bytes(tmp_path, monkeypatch):
    args = SimpleNamespace(policy_kind="rlc", policy_execution_variant="native")
    native = serving_identity.serving_artifact(args)
    args.policy_execution_variant = "adaptive-short-chunk"
    assert serving_identity.serving_artifact(args) != native
    args.policy_execution_variant = "native"
    monkeypatch.setattr(serving_identity, "_SERVING_FILES", ("policy.py",))
    assert serving_identity.serving_artifact(args) != native


def test_serving_identity_binds_comet_task_name():
    args = SimpleNamespace(
        policy_kind="comet12",
        policy_execution_variant="native",
        policy_task_name="turning_on_radio",
    )
    first = serving_identity.serving_artifact(args)
    args.policy_task_name = "picking_up_trash"
    assert serving_identity.serving_artifact(args) != first


def test_provenance_failure_preserves_primary_evaluator_error(
    fixture, monkeypatch, caplog
):
    panel, partition, store, workspace = fixture

    @contextmanager
    def evaluator(*_):
        raise RuntimeError("original evaluator failure")
        yield

    def unavailable(*_):
        raise ConnectionError("storage failure")

    monkeypatch.setattr(campaign_runner, "_prepared_evaluator", evaluator)
    monkeypatch.setattr(campaign_runner, "_publish_worker_provenance", unavailable)
    args = SimpleNamespace(
        worker_index=0, worker_receipt_uri="s3://example-bucket/run/worker.json"
    )
    with pytest.raises(RuntimeError, match="original evaluator failure"):
        campaign_runner._execute_partition(args, panel, partition, store, workspace)
    assert "Provenance upload also failed" in caplog.text


def test_parallel_workers_have_separate_original_provenance(tmp_path):
    storage = MemoryStorage()
    for index in range(2):
        workspace = tmp_path / str(index)
        workspace.mkdir()
        (workspace / "policy.log").write_text(f"worker {index} original log")
        campaign_runner._publish_worker_provenance(
            storage, workspace, f"s3://example-bucket/run/workers/worker-{index}.json"
        )
    assert len(storage.objects) == 2
    assert {value[0] for value in storage.objects.values()} == {
        b"worker 0 original log",
        b"worker 1 original log",
    }
