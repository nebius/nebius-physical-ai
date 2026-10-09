"""Exercise the real NPA renderer and runtime through the private team scheduler adapter."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import yaml

from npa.workbench.team.engine import execute_run
from npa.workbench.team.errors import BackendError, ConflictError
from npa.workbench.team.ledger import TeamLedger
from npa.workbench.team.models import Allocation, SubmitRequest
from npa.workbench.team.service import binding_snapshot
from npa.workbench.team.sky_backend import SkyBackend, _read_log_paths
from npa.workbench.team.workflow_policy import worker_context


class Scheduler:
    """A process-boundary test transport recording only authorized SkyPilot calls."""

    def __init__(self):
        self.calls = []
        self.payloads = []
        self.fail_launch = False

    def __call__(self, command, **kwargs):
        request = json.loads(kwargs["input"])
        self.calls.append(request)
        operation = request["operation"]
        if operation == "launch":
            self.payloads.append(list(yaml.safe_load_all(request["yaml"])))
            if self.fail_launch:
                return SimpleNamespace(returncode=1, stdout='{"error":"unavailable"}')
            response = {"request_id": "request-1"}
        elif operation == "result":
            response = {"job_ids": [42]}
        elif operation == "queue":
            response = {
                "jobs": [
                    {
                        "job_id": 42,
                        "task_id": 0,
                        "task_name": "hello",
                        "job_name": "team",
                        "status": "SUCCEEDED",
                    }
                ]
            }
        elif operation == "cancel":
            response = {"cancel_requested": True}
        else:
            raise AssertionError(operation)
        assert "AWS_ACCESS_KEY_ID" not in kwargs["env"]
        assert "NEBIUS_TOKEN" not in kwargs["env"]
        return SimpleNamespace(returncode=0, stdout=json.dumps(response))


def _backend(config, actor, binding, workflow):
    ledger = TeamLedger(config.state_dir)
    request = SubmitRequest(
        workspace="robotics", cluster="east", idempotency_key="first", workflow=workflow
    )
    record, _ = ledger.create(actor, request, binding_snapshot(binding))
    ledger.transition(record["id"], ("accepted",), "running")
    scheduler = Scheduler()
    backend = SkyBackend(config, binding, ledger, record["id"], runner=scheduler)
    return record, backend, scheduler


def test_canonical_workflow_runtime_binds_every_launch(
    config, actor, binding, workflow, monkeypatch
):
    monkeypatch.delenv("NPA_SRC_S3_URI", raising=False)
    record, backend, scheduler = _backend(config, actor, binding, workflow)
    objects = {}
    storage = SimpleNamespace(
        client=SimpleNamespace(
            put_object=lambda **kwargs: objects.update({kwargs["Key"]: kwargs["Body"]})
        )
    )
    report = execute_run(
        config, binding, record, backend, lambda: None, storage=storage
    )
    assert report.status == "succeeded", report.error
    assert len(scheduler.payloads) == 1
    task = next(item for item in scheduler.payloads[0] if "resources" in item)
    assert task["resources"]["region"] == worker_context(binding)
    assert (
        task["config"]["kubernetes"]["pod_config"]["spec"][
            "automountServiceAccountToken"
        ]
        is False
    )
    assert task["envs"]["AWS_ACCESS_KEY_ID"] == "test-alice"
    assert "test-bob" not in json.dumps(scheduler.payloads)
    resumed = execute_run(
        config, binding, record, backend, lambda: None, resume=True, storage=storage
    )
    assert resumed.status == "succeeded"
    assert len(scheduler.payloads) == 1
    assert any(key.endswith("workflow.yaml") for key in objects)


@pytest.mark.parametrize("source", ["", "s3://team-test-alice/personal/runtime/npa"])
def test_runtime_source_is_scoped_and_never_inherited(
    config, actor, binding, workflow, monkeypatch, source
):
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://server-private/runtime")
    allocation = Allocation.model_validate(
        {**binding.allocation.model_dump(), "source_s3_uri": source}
    )
    binding = replace(binding, allocation=allocation)
    record, backend, scheduler = _backend(config, actor, binding, workflow)
    storage = SimpleNamespace(client=SimpleNamespace(put_object=lambda **kwargs: None))
    report = execute_run(
        config, binding, record, backend, lambda: None, storage=storage
    )
    assert report.status == "succeeded", report.error
    task = next(item for item in scheduler.payloads[0] if "resources" in item)
    assert task["envs"].get("NPA_SRC_S3_URI", "") == source
    assert "server-private" not in json.dumps(scheduler.payloads)
    assert json.loads(record["binding"]).get("source_s3_uri", "") == source


@pytest.mark.parametrize(
    "source",
    ["s3://team-test-bob/personal/runtime", "s3://team-test-alice/personal/../runtime"],
)
def test_runtime_source_cannot_escape_personal_scope(binding, source):
    values = binding.allocation.model_dump()
    values["source_s3_uri"] = source
    with pytest.raises(ValueError, match="allocated input scope"):
        Allocation.model_validate(values)


def test_lost_launch_ack_never_blindly_retries(
    config, actor, binding, workflow, tmp_path
):
    record, backend, scheduler = _backend(config, actor, binding, workflow)
    scheduler.fail_launch = True
    path = tmp_path / "wave.yaml"
    path.write_text(yaml.safe_dump({"resources": {}, "run": "echo hello"}))
    with pytest.raises(BackendError):
        backend.submit(path, "durable-wave")
    with pytest.raises(ConflictError):
        backend.submit(path, "durable-wave")
    assert backend.reconcile("durable-wave").outcome == "unknown"
    assert backend.cancel_run() is False
    assert len(scheduler.payloads) == 1


def test_cancellation_wins_against_new_launch(
    config, actor, binding, workflow, tmp_path
):
    record, backend, scheduler = _backend(config, actor, binding, workflow)
    backend.ledger.transition(record["id"], ("running",), "cancelling")
    path = tmp_path / "wave.yaml"
    path.write_text(yaml.safe_dump({"resources": {}, "run": "echo hello"}))
    with pytest.raises(ConflictError):
        backend.submit(path, "late-wave")
    assert scheduler.calls == []


def test_exact_wave_cannot_cancel_other_run(config, actor, binding, workflow):
    record, backend, scheduler = _backend(config, actor, binding, workflow)
    with pytest.raises(ConflictError):
        backend.cancel(job_id="another-wave")
    assert scheduler.calls == []


def test_log_reader_keeps_paths_and_symlinks_inside_private_destination(tmp_path):
    destination = tmp_path / "sky_logs"
    job = destination / "job"
    job.mkdir(parents=True)
    (job / "run.log").write_text("real worker output")
    assert _read_log_paths({"7": str(job)}, destination) == ["real worker output"]
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.log").write_text("must not escape")
    with pytest.raises(BackendError, match="escaped"):
        _read_log_paths({"7": str(outside)}, destination)
    (job / "linked.log").symlink_to(outside / "secret.log")
    with pytest.raises(BackendError, match="symlink escaped"):
        _read_log_paths({"7": str(job)}, destination)
