from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import subprocess

import pytest

from npa.clients import config as client_config
from npa.clients.serverless import (
    AuthError,
    EndpointNotFoundError,
    JobIdentityError,
    JobSubmissionIndeterminateError,
    ServerlessClient,
    ServerlessClientError,
    TransientServerlessError,
)
from npa.orchestration.npa_workflow.supervisor import (
    AttemptIdentity,
    CheckpointValidation,
    ServerlessRecoverySpec,
    ServerlessSupervisorAdapter,
    SupervisorLedger,
)
from npa.orchestration.npa_workflow.run_state import RunStateStore
from npa.serverless_common.supervision import (
    ServerlessSupervisionConfig,
    ServerlessSupervisionError,
    supervise_serverless_job,
)


def MemoryStore():
    objects = {}

    def read(_bucket, key):
        if key not in objects:
            raise FileNotFoundError(key)
        return objects[key]

    return RunStateStore(
        bucket="unit-bucket", prefix="runs/unit", reader=read,
        writer=lambda _bucket, key, body: objects.__setitem__(key, body),
        artifact_lister=lambda _bucket, prefix: [key for key in objects if key.startswith(prefix)],
    )


def _job(state: str, **metadata: str) -> str:
    return json.dumps({
        "metadata": {
            "id": "provider-unit", "name": "training-unit", "parent_id": "project-unit",
            **metadata,
        },
        "status": {"state": state},
    })


def _create(client: ServerlessClient, **overrides):
    return client.create_job(**{
        "project_id": "project-unit", "name": "training-unit",
        "image": "registry.example/tool@sha256:" + "a" * 64,
        "command": "train", "gpu_type": "gpu-unit", "gpu_count": 1,
        "output_path": "s3://unit-bucket/result/", "durable": True,
        "extra_env": {"AWS_SECRET_ACCESS_KEY": "unit-private-value"},
        **overrides,
    })


def _identity() -> AttemptIdentity:
    return AttemptIdentity(
        runtime="serverless", run_id="training-unit", attempt=1,
        logical_attempt_id="training-unit:1", provider_job_id="provider-unit",
        provider_job_name="training-unit", workflow_sha256="w", source_sha256="s",
        image_digest="i",
    )


def _supervise(client, store, output_checker, *, sleeper=lambda _: None):
    return supervise_serverless_job(
        adapter=ServerlessSupervisorAdapter(
            ServerlessRecoverySpec(project_id="project-unit"), client=client,
        ),
        ledger=SupervisorLedger(store), identity=_identity(),
        config=ServerlessSupervisionConfig(
            expected_workflow_sha256="w", expected_source_sha256="s",
            expected_image_digest="i", declared_outputs=("s3://unit-bucket/result/model.pt",),
            poll_interval_seconds=0,
        ), output_checker=output_checker, sleeper=sleeper,
    )


def test_timeout_image_pull_reconnect_transient_running_complete_same_job(tmp_path):
    """Exercise real client parsing + durable launch + supervisor, mocking only CLI I/O."""
    calls = []
    state = "IMAGE_PULLING"
    fail_observation = False
    checkpoint = tmp_path / "model.pt"

    def provider(args, **kwargs):
        nonlocal fail_observation
        action = args[3]
        calls.append(action)
        if action == "create":
            records = list(client_config.CONFIG_PATH.parent.glob("runtime/**/*.json"))
            assert len(records) == 1
            assert json.loads(records[0].read_text())["state"] == "creating"
            raise subprocess.TimeoutExpired(args, kwargs["timeout"])
        if action == "get" and args[5] == "training-unit":
            return subprocess.CompletedProcess(args, 1, "", "not found")
        if fail_observation:
            fail_observation = False
            return subprocess.CompletedProcess(args, 1, "", "503 temporarily unavailable")
        return subprocess.CompletedProcess(args, 0, _job(state), "")

    original = ServerlessClient(subprocess_runner=provider)
    launched = _create(original)
    assert launched.id == "provider-unit"
    assert launched.status == "queued"
    assert launched.provider_state == "IMAGE_PULLING"
    assert original.classify_queue_state(launched) == "starting"

    # A new process/client uses the owner-only journal and never calls create.
    fresh = ServerlessClient(subprocess_runner=provider)
    assert _create(fresh).id == launched.id
    fail_observation = True
    store = MemoryStore()

    phases = iter(["IMAGE_PULLING", "RUNNING", "COMPLETED"])

    def advance(_seconds):
        nonlocal state
        state = next(phases)
        if state == "COMPLETED":
            checkpoint.write_bytes(b"unit checkpoint artifact")

    recovered, final = _supervise(fresh, store, lambda _: checkpoint.exists(), sleeper=advance)
    assert recovered.provider_job_id == launched.id == final.id
    assert recovered.attempt == 1
    assert final.status == "succeeded"
    assert calls.count("create") == 1
    assert "cancel" not in calls
    events = SupervisorLedger(store).events()
    assert any(e["observation"]["reason_code"] == "SERVERLESS_TRANSPORT" for e in events)
    assert any(e["observation"]["evidence"].get("provider_state") == "IMAGE_PULLING" for e in events)
    assert any(e["observation"]["state"] == "running" for e in events)
    assert any(e["recovery"]["action"] == "reuse_completed_wave" for e in events)
    for path in client_config.CONFIG_PATH.parent.glob("runtime/**/*"):
        if path.is_file():
            assert path.stat().st_mode & 0o077 == 0
            assert "unit-private-value" not in path.read_text()


def test_inconclusive_create_persists_identity_until_job_becomes_visible():
    calls = []
    visible = False

    def provider(args, **kwargs):
        calls.append(args[3])
        if args[3] == "create":
            raise subprocess.TimeoutExpired(args, kwargs["timeout"])
        if not visible:
            return subprocess.CompletedProcess(args, 1, "", "not found")
        return subprocess.CompletedProcess(args, 0, _job("IMAGE_PULLING"), "")

    with pytest.raises(JobSubmissionIndeterminateError) as failure:
        _create(ServerlessClient(subprocess_runner=provider))
    assert failure.value.job_name == "training-unit"
    with pytest.raises(JobSubmissionIndeterminateError):
        _create(ServerlessClient(subprocess_runner=provider))
    visible = True
    assert _create(ServerlessClient(subprocess_runner=provider)).id == "provider-unit"
    assert calls.count("create") == 1


def test_concurrent_reconnects_create_once():
    calls = []

    def provider(args, **_kwargs):
        calls.append(args[3])
        return subprocess.CompletedProcess(args, 0, _job("RUNNING"), "")

    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = list(pool.map(lambda _: _create(ServerlessClient(subprocess_runner=provider)), range(4)))
    assert {job.id for job in jobs} == {"provider-unit"}
    assert calls.count("create") == 1


def test_reconnect_rejects_changed_launch_and_replaced_provider_identity():
    returned = _job("RUNNING")
    client = ServerlessClient(subprocess_runner=lambda args, **_: subprocess.CompletedProcess(args, 0, returned, ""))
    _create(client)
    with pytest.raises(JobIdentityError, match="different launch contract"):
        _create(client, command="different-training")
    returned = _job("RUNNING", id="different-provider")
    with pytest.raises(JobIdentityError):
        _create(client)


@pytest.mark.parametrize("state, expected", [
    ("PROVISIONING", "queued"), ("STARTING", "queued"), ("IMAGE_PULLING", "queued"),
    ("image-pulling", "queued"), ("RUNNING", "running"), ("COMPLETED", "succeeded"),
    ("CANCELLING", "cancelling"), ("CANCELLED", "cancelled"), ("FAILED", "failed"),
    ("ERROR", "failed"), ("DELETING", "unknown"), ("STATE_UNSPECIFIED", "unknown"),
    ("FUTURE_PHASE", "unknown"),
])
def test_provider_phase_normalization_preserves_unknown(state, expected):
    client = ServerlessClient(subprocess_runner=lambda args, **_: subprocess.CompletedProcess(args, 0, _job(state), ""))
    assert client.get_job("provider-unit", "project-unit").status == expected


@pytest.mark.parametrize("error, exception", [
    ("403 permission denied", AuthError), ("401 unauthenticated", AuthError),
    ("503 unavailable", TransientServerlessError), ("unrecognized provider failure", ServerlessClientError),
])
def test_id_lookup_does_not_hide_auth_or_provider_failure_with_name_fallback(error, exception):
    calls = []

    def provider(args, **_kwargs):
        calls.append(args[3])
        return subprocess.CompletedProcess(args, 1, "", error)

    with pytest.raises(exception):
        ServerlessClient(subprocess_runner=provider).get_job("provider-unit", "project-unit")
    assert calls == ["get"]


@pytest.mark.parametrize("metadata", [{"id": ""}, {"parent_id": "other-project"}, {"name": "other-job"}])
def test_timeout_recovery_rejects_missing_id_or_wrong_scope(metadata):
    def provider(args, **kwargs):
        if args[3] == "create":
            raise subprocess.TimeoutExpired(args, kwargs["timeout"])
        return subprocess.CompletedProcess(args, 0, _job("RUNNING", **metadata), "")

    with pytest.raises(JobIdentityError):
        _create(ServerlessClient(subprocess_runner=provider))


@pytest.mark.parametrize("state, error, reason", [
    ("FUTURE_PHASE", "", "UNCLASSIFIED"),
    ("ERROR", "", "SERVERLESS_PROVIDER_ERROR"),
    ("FAILED", "", "PAYLOAD_EXIT_NONZERO"),
    ("", "403 permission denied", "AUTHORIZATION"),
    ("", "unrecognized provider failure", "SERVERLESS_PROVIDER_ERROR"),
])
def test_supervision_keeps_unknown_auth_provider_and_payload_failures_distinct(state, error, reason):
    calls = []

    def provider(args, **_kwargs):
        calls.append(args[3])
        return subprocess.CompletedProcess(args, int(bool(error)), _job(state), error)

    with pytest.raises(ServerlessSupervisionError, match=reason):
        _supervise(ServerlessClient(subprocess_runner=provider), MemoryStore(), lambda _: False)
    assert calls == ["get"]


def test_provider_state_details_retain_failure_and_finish_information():
    payload = json.loads(_job("ERROR"))
    payload["status"].update(state_details={"code": "PROVIDER_FAILURE", "message": "internal failure"}, finished_at="2026-01-01T01:00:00Z")
    client = ServerlessClient(subprocess_runner=lambda args, **_: subprocess.CompletedProcess(args, 0, json.dumps(payload), ""))
    job = client.get_job("provider-unit", "project-unit")
    assert job.provider_state == "ERROR"
    assert job.pending_reason == "PROVIDER_FAILURE"
    assert job.log_tail == "internal failure"
    assert job.ended_at == "2026-01-01T01:00:00Z"


def test_missing_durable_provider_identity_never_queries_or_relaunches():
    client = ServerlessClient(subprocess_runner=lambda *_args, **_kwargs: pytest.fail("provider must not be queried without exact identity"))
    observation = ServerlessSupervisorAdapter(
        ServerlessRecoverySpec(project_id="project-unit"), client=client,
    ).observe(replace(_identity(), provider_job_id=""))
    assert observation.reason_code == "AMBIGUOUS_ATTEMPT_IDENTITY"
    assert not observation.exact_identity


def test_resource_provisioning_message_is_not_capacity_failure():
    payload = json.loads(_job("PROVISIONING"))
    payload["status"]["state_details"] = {"message": "Preparing resources for the job"}
    client = ServerlessClient(subprocess_runner=lambda args, **_: subprocess.CompletedProcess(args, 0, json.dumps(payload), ""))
    observation = ServerlessSupervisorAdapter(
        ServerlessRecoverySpec(project_id="project-unit"), client=client,
    ).observe(_identity())
    assert observation.state.value == "queued"
    assert observation.reason_code == ""


def test_replacement_create_requires_fresh_preflight(mocker):
    client = mocker.Mock()
    client.get_job.side_effect = EndpointNotFoundError("absent")
    check = mocker.Mock(side_effect=ValueError("execution target access changed"))
    adapter = ServerlessSupervisorAdapter(
        ServerlessRecoverySpec(project_id="project-unit"), client=client, launch_preflight=check,
    )
    with pytest.raises(ValueError, match="execution target access changed"):
        adapter.launch_recovery(_identity(), checkpoint=CheckpointValidation())
    check.assert_called_once()
    client.create_job.assert_not_called()


def test_global_id_lookup_corroborates_missing_parent_with_exact_scoped_identity():
    calls = []

    def provider(args, **_):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, _job("RUNNING", parent_id=""), "")

    info = ServerlessClient(subprocess_runner=provider).get_job("provider-unit", "project-unit")
    assert info.id == "provider-unit"
    assert info.project_id == "project-unit"
    assert [args[3] for args in calls] == ["get", "get-by-name"]
    assert calls[1][calls[1].index("--parent-id") + 1] == "project-unit"
    assert calls[1][calls[1].index("--name") + 1] == "training-unit"


@pytest.mark.parametrize("metadata", [{"id": "different-provider"}, {"name": "different-job"}, {"parent_id": "different-project"}])
def test_global_id_corroboration_rejects_another_scoped_job(metadata):
    calls = []

    def provider(args, **_):
        calls.append(args[3])
        raw = _job("RUNNING", parent_id="") if len(calls) == 1 else _job("RUNNING", **metadata)
        return subprocess.CompletedProcess(args, 0, raw, "")

    with pytest.raises(JobIdentityError):
        ServerlessClient(subprocess_runner=provider).get_job("provider-unit", "project-unit")
    assert calls == ["get", "get-by-name"]


@pytest.mark.parametrize("error,exception", [("404 not found", JobIdentityError), ("403 permission denied", AuthError), ("503 unavailable", TransientServerlessError)])
def test_global_id_corroboration_preserves_missing_auth_and_transient_evidence(error, exception):
    calls = []

    def provider(args, **_):
        calls.append(args[3])
        if len(calls) == 1:
            return subprocess.CompletedProcess(args, 0, _job("RUNNING", parent_id=""), "")
        return subprocess.CompletedProcess(args, 1, "", error)

    with pytest.raises(exception):
        ServerlessClient(subprocess_runner=provider).get_job("provider-unit", "project-unit")
    assert calls == ["get", "get-by-name"]


def test_global_id_lookup_without_parent_or_name_refuses_without_create_or_fallback():
    calls = []

    def provider(args, **_):
        calls.append(args[3])
        return subprocess.CompletedProcess(args, 0, _job("RUNNING", name="", parent_id=""), "")

    with pytest.raises(JobIdentityError, match="missing identity/project evidence"):
        ServerlessClient(subprocess_runner=provider).get_job("provider-unit", "project-unit")
    assert calls == ["get"]


@pytest.mark.parametrize("error,exception", [("403 permission denied invalid argument", AuthError), ("503 unavailable invalid job id", TransientServerlessError)])
def test_global_id_auth_or_transport_error_wins_over_structural_fallback(error, exception):
    calls = []

    def provider(args, **_):
        calls.append(args[3])
        return subprocess.CompletedProcess(args, 1, "", error)

    with pytest.raises(exception):
        ServerlessClient(subprocess_runner=provider).get_job("provider-unit", "project-unit")
    assert calls == ["get"]
