"""Prove orphan recovery refuses unrelated identities before native cancellation."""

import copy
import hashlib
import json

import pytest
import yaml

from npa.orchestration.skypilot import controller_recovery as recovery
from npa.orchestration.skypilot.controller_recovery_probe import verify_job


@pytest.fixture
def original(tmp_path):
    image = "example.invalid/worker@sha256:" + "a" * 64
    config = yaml.safe_dump(
        {
            "kubernetes": {"allowed_contexts": ["synthetic-context"]},
            "jobs": {"controller": {"resources": {"region": "synthetic-context"}}},
        }
    )
    resources = {
        "infra": "kubernetes/synthetic-context",
        "image_id": {"synthetic-context": "docker:" + image},
    }
    dag = yaml.safe_dump(
        {
            "resources": resources,
            "envs": {
                "NPA_WORKFLOW_RUN_ID": "synthetic-run",
                "NPA_WORKFLOW_ATTEMPT_ID": "synthetic-attempt",
            },
        }
    )
    record = {
        "schema": "npa.workflow.controller-recovery.v1",
        "run_id": "synthetic-run",
        "project": "synthetic-project",
        "workflow_s3_uri": "s3://example-bucket/run/npa-workflow",
        "context": "synthetic-context",
        "namespace": "default",
        "controller": "synthetic-controller",
        "controller_uid": "synthetic-controller-uid",
        "container": "controller",
        "job_id": 1,
        "name": "synthetic-job",
        "user_hash": "synthetic-caller",
        "workspace": "default",
        "attempt_id": "synthetic-attempt",
        "submitted_at": 1.25,
        "image": image,
        "dag_yaml_content_sha256": hashlib.sha256(dag.encode()).hexdigest(),
        "config_file_content_sha256": hashlib.sha256(config.encode()).hexdigest(),
    }
    job = {
        "spot_job_id": 1,
        "name": record["name"],
        "workspace": "default",
        "user_hash": record["user_hash"],
        "dag_yaml_content": dag,
        "config_file_content": config,
    }
    task = {
        "spot_job_id": 1,
        "submitted_at": 1.25,
        "full_resources": json.dumps(resources),
        "task_name": record["name"],
        "status": "RUNNING",
    }
    manifest = {"run_id": record["run_id"], "status": "FAILED"}
    runtime = {
        "run_id": record["run_id"],
        "status": "failed",
        "waves": [
            {
                "job_id": "1",
                "job_name": record["name"],
                "logical_launch_id": record["attempt_id"],
                "launch_sequence": 1,
            }
        ],
    }
    controller = {
        "metadata": {
            "name": record["controller"],
            "namespace": "default",
            "uid": record["controller_uid"],
        },
        "spec": {"containers": [{"name": "controller"}]},
        "status": {"phase": "Running"},
    }
    worker = {
        "metadata": {
            "name": "synthetic-worker",
            "namespace": "default",
            "annotations": {
                "skypilot-managed-job-id": "1",
                "skypilot-managed-job-name": record["name"],
            },
        },
        "spec": {"containers": [{"image": image}]},
        "status": {"phase": "Running", "containerStatuses": [{"imageID": image}]},
    }
    path = tmp_path / "recovery.json"
    path.write_text(json.dumps(record))
    path.chmod(0o600)
    return record, job, task, manifest, runtime, [controller, worker], path


def test_stale_durable_failure_does_not_hide_live_original_controller(
    original, monkeypatch
):
    record, job, task, manifest, runtime, pods, path = original
    monkeypatch.setattr(
        recovery, "_read_ledger", lambda r: recovery.verify_ledger(r, manifest, runtime)
    )
    monkeypatch.setattr(recovery, "_inventory", lambda r: pods)
    monkeypatch.setattr(recovery, "_save_durable", lambda *a: None)
    requests = []

    def probe(expected, *, cancel=False):
        verify_job([job], [task], expected)
        requests.append(cancel)
        return {"status": task["status"], "cancel_requested": cancel}

    monkeypatch.setattr(recovery, "_probe", probe)
    result = recovery.reconcile_controller(path, cancel=True)
    assert requests == [False, True]
    assert result["cancel_requested"] and not result["cleanup_verified"]
    assert len(list(path.parent.glob("recovery-observations/*.json"))) == 2
    task["status"] = "CANCELLED"
    pods.pop()
    result = recovery.reconcile_controller(path, cancel=True)
    assert result["cleanup_verified"] and requests == [False, True, False]


@pytest.mark.parametrize(
    "field,value",
    [
        ("user_hash", "another-caller"),
        ("name", "another-job"),
        ("workspace", "another-workspace"),
        ("spot_job_id", 2),
        ("dag_yaml_content", "changed"),
        ("config_file_content", "changed"),
    ],
)
def test_refuses_changed_original_caller_configuration_and_job(original, field, value):
    record, job, task, *_ = original
    job[field] = value
    with pytest.raises(ValueError):
        verify_job([job], [task], record)


def test_refuses_shared_controller(original):
    record, job, task, *_ = original
    with pytest.raises(ValueError, match="shared"):
        verify_job([job, {**job, "spot_job_id": 2}], [task], record)


@pytest.mark.parametrize("change", ["attempt", "duplicate", "run", "never_launched"])
def test_requires_independent_durable_attempt_binding(original, change):
    record, _, _, manifest, runtime, *_ = original
    if change == "attempt":
        runtime["waves"][0]["logical_launch_id"] = "unrelated-attempt"
    elif change == "duplicate":
        runtime["waves"].append(copy.deepcopy(runtime["waves"][0]))
    elif change == "run":
        manifest["run_id"] = "unrelated-run"
    else:
        runtime["waves"][0]["launch_sequence"] = 0
    with pytest.raises(recovery.ControllerRecoveryError):
        recovery.verify_ledger(record, manifest, runtime)


@pytest.mark.parametrize("change", ["uid", "image", "job"])
def test_worker_and_controller_identity_are_required(original, change):
    record, _, _, _, _, pods, _ = original
    if change == "uid":
        pods[0]["metadata"]["uid"] = "replacement-controller"
    elif change == "image":
        pods[1]["spec"]["containers"][0]["image"] = "example.invalid/worker:latest"
    else:
        pods[1]["metadata"]["annotations"]["skypilot-managed-job-id"] = "2"
    with pytest.raises(recovery.ControllerRecoveryError):
        recovery.verify_pods(record, pods)


def test_diagnostics_failure_prevents_cancel(original, monkeypatch):
    _, _, _, _, _, pods, path = original
    monkeypatch.setattr(recovery, "_read_ledger", lambda r: None)
    monkeypatch.setattr(recovery, "_inventory", lambda r: pods)
    calls = []
    monkeypatch.setattr(
        recovery, "_probe", lambda r, **kw: calls.append(kw) or {"status": "RUNNING"}
    )
    monkeypatch.setattr(
        recovery,
        "_save_evidence",
        lambda *a: (_ for _ in ()).throw(OSError("disk full")),
    )
    with pytest.raises(OSError):
        recovery.reconcile_controller(path, cancel=True)
    assert calls == [{}]


def test_world_readable_or_linked_record_is_refused(original):
    path = original[-1]
    path.chmod(0o644)
    with pytest.raises(recovery.ControllerRecoveryError):
        recovery._private_record(path)
    link = path.with_name("linked.json")
    link.symlink_to(path)
    with pytest.raises(OSError):
        recovery._private_record(link)


def test_durable_write_failure_prevents_native_cancellation(original, monkeypatch):
    _, _, _, _, _, pods, path = original
    monkeypatch.setattr(recovery, "_read_ledger", lambda r: None)
    monkeypatch.setattr(recovery, "_inventory", lambda r: pods)
    calls = []
    monkeypatch.setattr(
        recovery, "_probe", lambda r, **kw: calls.append(kw) or {"status": "RUNNING"}
    )
    monkeypatch.setattr(
        recovery,
        "_save_durable",
        lambda *a: (_ for _ in ()).throw(OSError("unavailable")),
    )
    with pytest.raises(OSError):
        recovery.reconcile_controller(path, cancel=True)
    assert calls == [{}]


def test_durable_receipt_requires_exact_readback():
    from io import BytesIO
    from types import SimpleNamespace
    from unittest.mock import Mock

    client = Mock()
    state = SimpleNamespace(
        prefix="synthetic", bucket="example-bucket", client=lambda: client
    )
    client.get_object.return_value = {"Body": BytesIO(b"changed")}
    with pytest.raises(recovery.ControllerRecoveryError, match="read-back"):
        recovery._save_durable(state, "a" * 64, {"status": "RUNNING"})
    assert client.put_object.call_args.kwargs["IfNoneMatch"] == "*"


def test_cli_storage_failure_is_sanitized_and_never_claims_cleanup(monkeypatch):
    from botocore.exceptions import ClientError
    import typer
    from typer.testing import CliRunner
    from npa.cli.workbench.workflow import controller_recovery as cli

    app = typer.Typer()
    cli.register(app)
    error = ClientError(
        {"Error": {"Code": "Denied", "Message": "private storage coordinate"}},
        "PutObject",
    )
    monkeypatch.setattr(
        cli, "reconcile_controller", lambda *a, **kw: (_ for _ in ()).throw(error)
    )
    result = CliRunner().invoke(app, ["unused.json", "--cancel", "--json"])
    assert result.exit_code == 1
    assert json.loads(result.stdout) == {
        "status": "VERIFICATION_UNAVAILABLE",
        "cleanup_verified": False,
    }
    assert "private storage coordinate" not in result.output
