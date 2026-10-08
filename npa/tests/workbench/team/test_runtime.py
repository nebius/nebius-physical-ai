"""Exercise the real NPA renderer and runtime through the private team scheduler adapter."""

import json
from types import SimpleNamespace

import pytest
import yaml

from npa.workbench.team.engine import execute_run
from npa.workbench.team.errors import BackendError, ConflictError
from npa.workbench.team.ledger import TeamLedger
from npa.workbench.team.models import SubmitRequest
from npa.workbench.team.service import binding_snapshot
from npa.workbench.team.sky_backend import SkyBackend
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
