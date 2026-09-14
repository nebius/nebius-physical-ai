"""Live status must include jobs written after the initial runtime manifest."""

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from typer.testing import CliRunner

from npa.cli.main import app

import pytest

from npa.cli.workbench.workflow import _durable_workflow_status
from npa.orchestration.npa_workflow.run_resolution import RunResolution
from npa.orchestration.npa_workflow.run_state import RunManifest, RunStateStore, runtime_manifest_view
from npa.orchestration.skypilot.workflow_state import WorkflowS3Config
from npa.orchestration.skypilot.workflow import workflow_task_statuses as _real_task_query


def _wave(name, job_id, status, *, attempt=1, iteration=None):
    return {
        "key": f"001|serial|:{name}:{iteration if iteration is not None else '-'}",
        "states": [name], "job_id": job_id, "status": status,
        "attempt": attempt,
    }


@pytest.fixture()
def observed_status(mocker):
    manifest = RunManifest("demo", "run-test", "npa.workflow/v0.0.1", status="running")
    resolution = RunResolution(
        run_id="run-test", project="test", found=True,
        source="durable_runtime_ledger", manifest=manifest.to_dict(),
        runtime_state={"status": "running", "waves": []},
        state=WorkflowS3Config(
            bucket="bucket", prefix="run-test/npa-workflow",
            endpoint_url="https://storage.example.test",
            aws_access_key_id="test", aws_secret_access_key="test",
        ),
    )
    mocker.patch("npa.orchestration.npa_workflow.run_resolution.resolve_run", return_value=resolution)
    jobs = mocker.patch("npa.orchestration.skypilot.workflow.workflow_status")
    jobs.side_effect = lambda job_id, **kwargs: SimpleNamespace(
        status=next(w["status"].upper() for w in reversed(resolution.runtime_state["waves"])
                    if w["job_id"] == job_id), error="",
    )
    mocker.patch("npa.orchestration.skypilot.workflow.workflow_task_statuses", return_value=[])
    mocker.patch("npa.orchestration.skypilot.workflow.workflow_controller_logs",
                 return_value=SimpleNamespace(returncode=0, stdout="", stderr=""))
    mocker.patch("npa.cli.workbench.workflow._stalled_job_blockers", return_value=[])
    mocker.patch("npa.orchestration.npa_workflow.run_state.RunStateStore")
    mocker.patch("npa.orchestration.npa_workflow.supervisor.SupervisorLedger.latest", return_value=None)
    return resolution, jobs


def test_empty_manifest_queries_each_observed_job(observed_status):
    resolution, jobs = observed_status
    resolution.job_id = "12"
    resolution.runtime_state["waves"] = [
        _wave("prepare", "11", "succeeded"), _wave("train", "12", "running"),
    ]

    payload = _durable_workflow_status("run-test")

    assert [call.args[0] for call in jobs.call_args_list] == ["11", "12"]
    assert payload["status"] == "RUNNING"
    assert payload["verification_status"] == "VERIFIED"
    assert payload["stages"]["prepare"]["managed_job_id"] == "11"
    assert payload["stages"]["train"]["managed_job_id"] == "12"
    assert resolution.manifest["steps"] == []


@pytest.mark.parametrize("runtime_status", ["running", "failed"])
def test_completed_observed_jobs_do_not_prove_workflow_completion(observed_status, runtime_status):
    resolution, _jobs = observed_status
    resolution.runtime_state.update(status=runtime_status, waves=[_wave("prepare", "11", "succeeded")])

    payload = _durable_workflow_status("run-test")

    assert payload["status"] == runtime_status.upper()
    assert payload["verification_status"] == "VERIFIED"
    assert payload["last_known"]["state"] == runtime_status.upper()
    assert payload["stages"]["prepare"]["state"] == "SUCCEEDED"
    assert payload["workflow_lifecycle"]["driver_liveness"] == "unknown"
    assert "--resume-run" not in json.dumps(payload)


def test_finished_runtime_preserves_success_after_retry(observed_status):
    resolution, jobs = observed_status
    resolution.runtime_state.update(status="succeeded", waves=[
        _wave("train", "11", "failed"), _wave("train", "12", "succeeded", attempt=2),
    ])

    payload = _durable_workflow_status("run-test")

    assert payload["status"] == "SUCCEEDED"
    assert [call.args[0] for call in jobs.call_args_list] == ["12"]
    assert list(payload["stages"]) == ["train"]
    assert len(payload["stages"]["train"]["managed_job_attempts"]) == 2
    assert resolution.runtime_state["waves"][0]["status"] == "failed"


def test_runtime_read_failure_cannot_claim_live_verification(observed_status):
    resolution, jobs = observed_status
    resolution.runtime_state_error = "runtime ledger could not be read"

    payload = _durable_workflow_status("run-test")

    assert payload["status"] == "VERIFICATION_UNAVAILABLE"
    assert "runtime ledger could not be read" in payload["live_verification"]["reason"]
    jobs.assert_not_called()


def test_partial_manifest_preserves_metadata_and_distinct_iterations():
    manifest = RunManifest("demo", "run-test", "npa.workflow/v0.0.1", steps=[
        {"state": "train", "iteration": 0, "status": "submitted", "custom": "preserved"},
    ])
    view = runtime_manifest_view(manifest, [
        _wave("train", "11", "failed", iteration=0),
        _wave("train", "12", "succeeded", iteration=0, attempt=2),
        _wave("train", "13", "running", iteration=1),
    ])

    assert len(view.steps) == 2
    assert view.steps[0] == manifest.steps[0]
    assert view.steps[1]["iteration"] == 1
    view.steps[0]["status"] = "changed"
    assert manifest.steps[0]["status"] == "submitted"


@pytest.mark.parametrize("stage_name", ["prepare", "finalize"])
@pytest.mark.parametrize("durable_time", ["2001-01-01T00:00:00Z", "2026-01-01T00:00:00Z"])
def test_handoff_preserves_lifecycle_without_fabricating_progress(observed_status, stage_name, durable_time):
    resolution, jobs = observed_status
    resolution.manifest["updated_at"] = durable_time
    resolution.runtime_state.update(status="running", updated_at=durable_time,
                                    waves=[_wave(stage_name, "11", "succeeded")])
    payload = _durable_workflow_status("run-test")
    assert payload["status"] == "RUNNING"
    assert payload["live_status"] == "SUCCEEDED"
    assert payload["verification_status"] == "VERIFIED"
    assert payload["workflow_lifecycle"] == {
        "manifest_status": "RUNNING", "runtime_status": "RUNNING",
        "manifest_evidence": {"status": "running", "updated_at": durable_time, "source": "authoritative_manifest"},
        "completion_recorded": False, "driver_liveness": "unknown",
        "source": "durable_runtime_ledger", "updated_at": durable_time,
    }
    assert payload["last_known"]["observed_at"] == durable_time
    assert payload["last_known"]["source"] == "durable_runtime_ledger"
    assert payload["last_heartbeat_at"] == ""
    assert payload["stages"][stage_name]["last_progress_at"] == ""
    assert payload["active_stage_name"] == ""
    assert payload["live_verification"]["attempted_at"] != durable_time
    assert "--resume-run" not in json.dumps(payload)
    assert [call.args[0] for call in jobs.call_args_list] == ["11"]


@pytest.mark.parametrize(("manifest_status", "runtime_status", "expected"), [
    ("running", "cancelled", "CANCELLED"),
    ("failed", "running", "FAILED"),
    ("cancelled", "running", "CANCELLED"),
    ("succeeded", "running", "SUCCEEDED"),
    ("running", "succeeded", "SUCCEEDED"),
    ("succeeded", "failed", "EVIDENCE_INCONSISTENT"),
    ("failed", "succeeded", "EVIDENCE_INCONSISTENT"),
    ("cancelled", "failed", "EVIDENCE_INCONSISTENT"),
])
def test_workflow_outcomes_are_separate_from_successful_jobs(observed_status, manifest_status, runtime_status, expected):
    resolution, _ = observed_status
    resolution.manifest["status"] = manifest_status
    resolution.runtime_state.update(status=runtime_status, waves=[_wave("prepare", "11", "succeeded")])
    payload = _durable_workflow_status("run-test")
    assert payload["status"] == expected
    assert payload["stages"]["prepare"]["state"] == "SUCCEEDED"
    assert payload["workflow_lifecycle"]["driver_liveness"] == "unknown"


@pytest.mark.parametrize("runtime_status", [None, "", "unknown", "invalid", {"status": "running"}])
def test_invalid_workflow_status_cannot_become_healthy_handoff(observed_status, runtime_status):
    resolution, _ = observed_status
    resolution.runtime_state.update(status=runtime_status, waves=[_wave("prepare", "11", "succeeded")])
    payload = _durable_workflow_status("run-test")
    assert payload["status"] == "VERIFICATION_UNAVAILABLE"
    assert payload["automation_may_trust_state"] is False


def _status_cli(mocker, *args):
    mocker.patch("npa.orchestration.npa_workflow.first_run_state.update_run_observation")
    return CliRunner().invoke(app, ["workbench", "workflow", "status", "run-test", "--project", "test", *args])


def test_handoff_json_command_exits_successfully(observed_status, mocker):
    resolution, _ = observed_status
    resolution.runtime_state["waves"] = [_wave("prepare", "11", "succeeded")]
    result = _status_cli(mocker, "--json")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "RUNNING"
    assert payload["workflow_lifecycle"]["completion_recorded"] is False


def test_watch_continues_across_handoff_to_real_completion(observed_status, mocker):
    resolution, jobs = observed_status
    resolution.runtime_state["waves"] = [_wave("prepare", "11", "succeeded")]
    def advance(_interval):
        resolution.runtime_state.update(status="succeeded", waves=[
            _wave("prepare", "11", "succeeded"), _wave("finalize", "12", "succeeded"),
        ])
    sleep = mocker.patch("npa.cli.workbench.workflow.time.sleep", side_effect=advance)
    result = _status_cli(mocker, "--watch")
    assert result.exit_code == 0, result.output
    assert "status: RUNNING" in result.stdout
    assert "status: SUCCEEDED" in result.stdout
    sleep.assert_called_once()
    assert [call.args[0] for call in jobs.call_args_list] == ["11", "11", "12"]


def _snapshot_ordering(observed_status, mocker, ordering, job_state="RUNNING"):
    resolution, jobs = observed_status
    durable_success = ordering == "durable-stage-lag"
    resolution.manifest.update(updated_at="2001-01-01T00:00:00Z", steps=[
        {"state": "prepare", "status": "succeeded"},
        {"state": "generate", "status": "succeeded" if durable_success else "submitted"},
    ])
    resolution.runtime_state.update(updated_at="2001-01-01T00:00:00Z", waves=[
        _wave("prepare", "11", "succeeded"),
        _wave("generate", "12", "succeeded" if durable_success else "running"),
    ])
    order = []

    def job_snapshot(job_id, **kwargs):
        order.append(("job", job_id))
        succeeded = job_id == "11" or resolution.runtime_state["status"] == "succeeded"
        return SimpleNamespace(status="SUCCEEDED" if succeeded else job_state, error="")

    def task_snapshot(job_id, **kwargs):
        order.append(("task", job_id))
        succeeded = job_id == "11" or not durable_success or resolution.runtime_state["status"] == "succeeded"
        return [{"task_id": 0, "task_name": "prepare" if job_id == "11" else "generate",
                 "status": "SUCCEEDED" if succeeded else job_state}]

    jobs.side_effect = job_snapshot
    mocker.patch("npa.orchestration.skypilot.workflow.workflow_task_statuses", side_effect=task_snapshot)
    return resolution, order


@pytest.mark.parametrize("ordering", ["task-finalization-lag", "durable-stage-lag"])
@pytest.mark.parametrize("job_state", ["PENDING", "STARTING", "RUNNING", "RECOVERING", "CANCELLING"])
def test_distinct_nonterminal_snapshots_preserve_incomplete_lifecycle(observed_status, mocker, ordering, job_state):
    _, order = _snapshot_ordering(observed_status, mocker, ordering, job_state)
    payload = _durable_workflow_status("run-test")
    assert order == [("job", "11"), ("task", "11"), ("job", "12"), ("task", "12")]
    assert payload["status"] == "RUNNING"
    assert payload["verification_status"] == "VERIFIED"
    assert payload["workflow_lifecycle"]["completion_recorded"] is False
    assert payload["workflow_lifecycle"]["driver_liveness"] == "unknown"
    assert payload["last_known"]["observed_at"] == "2001-01-01T00:00:00Z"
    assert payload["last_heartbeat_at"] == ""
    stage = payload["stages"]["generate"]
    assert stage["state"] == "SUCCEEDED"
    assert stage["managed_job_id"] == "12"
    assert stage["raw_job_scheduler_state"] == job_state
    assert stage["raw_task_scheduler_state"] == ("SUCCEEDED" if ordering == "task-finalization-lag" else job_state)
    assert stage["outcome_provenance"] == ("scheduler_final_attempt" if ordering == "task-finalization-lag" else "authoritative_stage_record")
    assert stage["last_progress_at"] == ""
    assert "--resume-run" not in json.dumps(payload)


@pytest.mark.parametrize("ordering", ["task-finalization-lag", "durable-stage-lag"])
@pytest.mark.parametrize("watch", [False, True])
@pytest.mark.parametrize("json_output", [False, True])
def test_snapshot_ordering_cli_continues_to_recorded_completion(observed_status, mocker, ordering, watch, json_output):
    resolution, order = _snapshot_ordering(observed_status, mocker, ordering)

    def complete(_interval):
        resolution.runtime_state["status"] = "succeeded"
        resolution.runtime_state["waves"][-1]["status"] = "succeeded"
        resolution.manifest["status"] = "completed"

    sleep = mocker.patch("npa.cli.workbench.workflow.time.sleep", side_effect=complete)
    args = (["--json"] if json_output else []) + (["--watch"] if watch else [])
    result = _status_cli(mocker, *args)
    assert result.exit_code == 0, result.output
    if json_output:
        payloads = _json_documents(result.stdout)
        assert [payload["status"] for payload in payloads] == (["RUNNING", "SUCCEEDED"] if watch else ["RUNNING"])
        assert payloads[0]["workflow_lifecycle"]["completion_recorded"] is False
        if watch:
            assert payloads[-1]["workflow_lifecycle"]["completion_recorded"] is True
    else:
        assert "status: RUNNING" in result.stdout
        assert ("status: SUCCEEDED" in result.stdout) is watch
    assert sleep.call_count == int(watch)
    assert order == [("job", "11"), ("task", "11"), ("job", "12"), ("task", "12")] * (1 + int(watch))


def _json_documents(output):
    documents = []
    decoder = json.JSONDecoder()
    while output.strip():
        document, end = decoder.raw_decode(output.lstrip())
        documents.append(document)
        output = output.lstrip()[end:]
    return documents


def test_nonterminal_job_snapshot_does_not_regress_durable_success(observed_status):
    resolution, jobs = observed_status
    resolution.runtime_state.update(updated_at="2001-01-01T00:00:00Z",
                                    waves=[_wave("prepare", "11", "succeeded")])
    resolution.manifest["steps"] = [{"state": "prepare", "status": "succeeded"}]
    jobs.side_effect = None
    jobs.return_value = SimpleNamespace(status="RUNNING", error="")
    # The former "regressed" refusal case has no contradictory terminal result:
    # the aggregate snapshot may precede the durable stage's success.
    payload = _durable_workflow_status("run-test")
    assert payload["status"] == "RUNNING"
    assert payload["verification_status"] == "VERIFIED"
    assert payload["stages"]["prepare"]["state"] == "SUCCEEDED"
    assert payload["stages"]["prepare"]["raw_job_scheduler_state"] == "RUNNING"
    assert payload["stages"]["prepare"]["raw_task_scheduler_state"] == ""
    assert payload["workflow_lifecycle"]["completion_recorded"] is False
    assert payload["workflow_lifecycle"]["driver_liveness"] == "unknown"
    assert payload["last_known"]["observed_at"] == "2001-01-01T00:00:00Z"
    assert payload["last_heartbeat_at"] == ""


@pytest.mark.parametrize("job_state", ["FAILED", "FAILED_SETUP", "CANCELLED"])
@pytest.mark.parametrize("watch", [False, True])
def test_successful_task_cannot_hide_terminal_job_disagreement(observed_status, mocker, job_state, watch):
    resolution, _ = _snapshot_ordering(observed_status, mocker, "task-finalization-lag", job_state)

    def record_failure(_interval):
        resolution.runtime_state["waves"][-1]["status"] = job_state.lower()
        mocker.patch("npa.orchestration.skypilot.workflow.workflow_task_statuses", side_effect=lambda job_id, **kwargs: [
            {"task_id": 0, "task_name": "prepare" if job_id == "11" else "generate",
             "status": "SUCCEEDED" if job_id == "11" else job_state}])

    sleep = mocker.patch("npa.cli.workbench.workflow.time.sleep", side_effect=record_failure)
    result = _status_cli(mocker, "--json", *(["--watch"] if watch else []))
    assert result.exit_code == (1 if watch else 0), result.output
    payloads = _json_documents(result.stdout)
    payload = payloads[0]
    assert payload["status"] == "UNKNOWN"
    stage = payload["stages"]["generate"]
    assert stage["outcome_conflict"] is True
    assert stage["raw_job_scheduler_state"] == job_state
    assert stage["raw_task_scheduler_state"] == "SUCCEEDED"
    assert sleep.call_count == int(watch)
    if watch:
        assert payloads[-1]["status"] == ("FAILED" if job_state.startswith("FAILED") else "CANCELLED")


@pytest.mark.parametrize("mode", ["unknown", "absent", "error", "auth", "exception", "malformed", "runtime-error"])
@pytest.mark.parametrize("watch", [False, True])
def test_successful_task_cannot_hide_unavailable_job_verification(observed_status, mocker, mode, watch):
    resolution, _ = _snapshot_ordering(observed_status, mocker, "task-finalization-lag")
    _, jobs = observed_status
    jobs.side_effect = None
    jobs.return_value = SimpleNamespace(status="RUNNING", error="")
    if mode in {"unknown", "malformed"}:
        jobs.return_value.status = "UNKNOWN" if mode == "unknown" else "invalid"
    elif mode in {"absent", "error", "auth"}:
        jobs.return_value.error = {"absent": "recorded exact job absent", "error": "query unavailable",
                                  "auth": "permission denied"}[mode]
    elif mode == "exception":
        jobs.side_effect = RuntimeError("query failed")
    else:
        resolution.runtime_state_error = "runtime unavailable"
    sleep = mocker.patch("npa.cli.workbench.workflow.time.sleep", side_effect=AssertionError("must not retry"))
    result = _status_cli(mocker, "--json", *(["--watch"] if watch else []))
    assert result.exit_code == 2, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "VERIFICATION_UNAVAILABLE"
    assert payload["automation_may_trust_state"] is False
    sleep.assert_not_called()


def _actual_task_query(mocker, response):
    from npa.orchestration.skypilot import workflow

    mocker.patch.object(workflow, "workflow_task_statuses", _real_task_query)
    mocker.patch.object(workflow, "resolve_config", return_value=SimpleNamespace(
        sky_bin=Path("/synthetic/sky"), global_config_path=None, isolated_config_dir=None))
    mocker.patch.object(workflow, "ensure_skypilot_version", side_effect=lambda path: path)
    mocker.patch.object(workflow, "sky_environment", return_value={})
    mocker.patch.object(workflow, "_stable_sky_cwd", return_value="/synthetic")
    return mocker.patch.object(workflow.subprocess, "run", return_value=response)


@pytest.mark.parametrize("detail", [
    "permission denied token=synthetic-private-value",
    "permission denied Authorization: Bearer synthetic-private-value",
    "permission denied https://example.test/request?token=synthetic-private-value",
])
def test_direct_strict_task_query_errors_redact_credentials(mocker, detail):
    _actual_task_query(mocker, subprocess.CompletedProcess([], 1, "", detail))
    with pytest.raises(RuntimeError, match="SkyPilot task queue query failed") as error:
        _real_task_query("11", raise_on_error=True)
    assert "permission denied" in str(error.value)
    assert "synthetic-private-value" not in str(error.value)


@pytest.mark.parametrize(("returncode", "stdout", "stderr"), [
    (1, "", "permission denied token=synthetic-private"),
    (2, "", "query failed"),
    (0, "{broken-json", ""),
    (0, "[]", "permission denied"),
    (0, '{"jobs": []}', "error: queue unavailable"),
    (0, "[]\n[]", ""),
])
@pytest.mark.parametrize("surface", ["callback", "json", "watch"])
def test_actual_task_query_failure_cannot_verify_durable_success(observed_status, mocker, returncode, stdout, stderr, surface):
    _snapshot_ordering(observed_status, mocker, "durable-stage-lag")
    query = _actual_task_query(mocker, subprocess.CompletedProcess([], returncode, stdout, stderr))
    sleep = mocker.patch("npa.cli.workbench.workflow.time.sleep", side_effect=AssertionError("must not retry"))
    if surface == "callback":
        payload = _durable_workflow_status("run-test")
    else:
        result = _status_cli(mocker, "--json", *(["--watch"] if surface == "watch" else []))
        assert result.exit_code == 2, result.output
        payload = json.loads(result.stdout)
    assert query.call_count == 2
    assert payload["status"] == "VERIFICATION_UNAVAILABLE"
    assert payload["automation_may_trust_state"] is False
    assert payload["live_verification"]["reason"]
    assert "synthetic-private" not in json.dumps(payload)
    sleep.assert_not_called()


@pytest.mark.parametrize(("returncode", "stdout", "stderr"), [
    (0, "[]", ""), (0, '{"jobs": []}', ""),
    (0, "Warning: optional display unavailable\n[]", ""),
    (1, "", "No in-progress managed jobs."),
])
@pytest.mark.parametrize("surface", ["callback", "json", "watch"])
def test_actual_empty_task_snapshot_preserves_handoff(observed_status, mocker, returncode, stdout, stderr, surface):
    resolution, _ = _snapshot_ordering(observed_status, mocker, "durable-stage-lag")
    query = _actual_task_query(mocker, subprocess.CompletedProcess([], returncode, stdout, stderr))

    def complete(_interval):
        resolution.runtime_state["status"] = "succeeded"
        resolution.manifest["status"] = "completed"

    sleep = mocker.patch("npa.cli.workbench.workflow.time.sleep", side_effect=complete)
    if surface == "callback":
        payloads = [_durable_workflow_status("run-test")]
    else:
        result = _status_cli(mocker, "--json", *(["--watch"] if surface == "watch" else []))
        assert result.exit_code == 0, result.output
        payloads = _json_documents(result.stdout)
    assert payloads[0]["status"] == "RUNNING"
    assert payloads[0]["verification_status"] == "VERIFIED"
    assert payloads[0]["workflow_lifecycle"]["completion_recorded"] is False
    assert sleep.call_count == int(surface == "watch")
    assert query.call_count == (4 if surface == "watch" else 2)
    if surface == "watch":
        assert payloads[-1]["status"] == "SUCCEEDED"


@pytest.mark.parametrize("mode", ["cached", "error", "unknown", "exception", "malformed"])
def test_handoff_cannot_hide_unavailable_live_verification(observed_status, mocker, mode):
    resolution, jobs = observed_status
    resolution.runtime_state.update(updated_at="2001-01-01T00:00:00Z",
                                    waves=[_wave("prepare", "11", "succeeded")])
    if mode == "error":
        jobs.side_effect = None
        jobs.return_value = SimpleNamespace(status="SUCCEEDED", error="permission denied")
    elif mode == "unknown":
        jobs.side_effect = None
        jobs.return_value = SimpleNamespace(status="UNKNOWN", error="")
    elif mode == "exception":
        jobs.side_effect = RuntimeError("malformed response token=synthetic-private")
    elif mode == "malformed":
        resolution.manifest["steps"] = [{"state": "prepare", "status": "succeeded"}]
        jobs.side_effect = None
        jobs.return_value = SimpleNamespace(status="invalid", error="")
    payload = _durable_workflow_status("run-test", cached=mode == "cached")
    assert payload["status"] == ("CACHED" if mode == "cached" else "VERIFICATION_UNAVAILABLE")
    assert payload["automation_may_trust_state"] is False
    assert payload["stages"]["prepare"]["managed_job_id"] == "11"
    assert "synthetic-private" not in json.dumps(payload)
    if mode == "cached":
        jobs.assert_not_called()
    else:
        assert payload["live_verification"]["reason"]


@pytest.mark.parametrize("live_status", ["FAILED", "CANCELLED"])
def test_handoff_retains_conflicting_scheduler_outcome(observed_status, live_status):
    resolution, jobs = observed_status
    resolution.runtime_state["waves"] = [_wave("prepare", "11", "succeeded")]
    jobs.side_effect = None
    jobs.return_value = SimpleNamespace(status=live_status, error="")
    payload = _durable_workflow_status("run-test")
    assert payload["status"] == "UNKNOWN"
    assert payload["stages"]["prepare"]["outcome_conflict"] is True
    assert payload["stages"]["prepare"]["raw_scheduler_state"] == live_status
    assert "workflow_lifecycle" not in payload


@pytest.mark.parametrize("evidence", ["missing-id", "conflicting-id", "malformed-wave"])
def test_handoff_requires_exact_observed_job_evidence(observed_status, evidence):
    resolution, _ = observed_status
    resolution.runtime_state["waves"] = [_wave("prepare", "11", "succeeded")]
    if evidence == "missing-id":
        resolution.runtime_state["waves"][0]["job_id"] = ""
    elif evidence == "conflicting-id":
        resolution.runtime_state["waves"].append(_wave("prepare", "12", "succeeded"))
    else:
        resolution.runtime_state["waves"].append("malformed")
    payload = _durable_workflow_status("run-test")
    assert payload["status"] == "VERIFICATION_UNAVAILABLE"
    assert payload["automation_may_trust_state"] is False


def test_handoff_preserves_real_stage_progress_time(observed_status):
    resolution, _ = observed_status
    progress = "2026-01-02T03:04:05Z"
    resolution.runtime_state.update(updated_at="2026-01-02T03:04:06Z",
        waves=[_wave("prepare", "11", "succeeded")],
        stages=[{"stage": "prepare", "attempt": 1, "last_heartbeat_at": progress}])
    payload = _durable_workflow_status("run-test")
    assert payload["last_heartbeat_at"] == progress
    assert payload["stages"]["prepare"]["last_progress_at"] == progress
    assert payload["last_known"]["observed_at"] == "2026-01-02T03:04:06Z"
    assert payload["workflow_lifecycle"]["driver_liveness"] == "unknown"


@pytest.mark.parametrize(("manifest_status", "runtime_status", "exit_code"), [
    ("running", "failed", 1), ("running", "cancelled", 1),
    ("failed", "running", 1), ("cancelled", "running", 1),
])
def test_handoff_cli_preserves_terminal_failure_exit(observed_status, mocker, manifest_status, runtime_status, exit_code):
    resolution, _ = observed_status
    resolution.manifest["status"] = manifest_status
    resolution.runtime_state.update(status=runtime_status, waves=[_wave("prepare", "11", "succeeded")])
    result = _status_cli(mocker, "--json")
    assert result.exit_code == exit_code
    payload = json.loads(result.stdout)
    assert payload["status"] in {"FAILED", "CANCELLED"}
    assert payload["verification_status"] == "VERIFIED"


@pytest.mark.parametrize("completed", [False, True])
def test_missing_lifecycle_time_is_not_replaced_with_poll_time(observed_status, completed):
    resolution, _ = observed_status
    resolution.manifest.pop("updated_at", None)
    resolution.manifest["status"] = "succeeded" if completed else "running"
    resolution.runtime_state["waves"] = [_wave("prepare", "11", "succeeded")]
    payload = _durable_workflow_status("run-test")
    assert payload["workflow_lifecycle"]["updated_at"] == ""
    assert payload["last_known"]["observed_at"] == ""
    assert payload["last_heartbeat_at"] == ""
    assert payload["workflow_lifecycle"]["completion_recorded"] is completed


@pytest.mark.parametrize("watch", [False, True])
@pytest.mark.parametrize(("manifest_status", "runtime_status"), [
    ("succeeded", "failed"), ("failed", "succeeded"), ("cancelled", "failed"),
])
def test_terminal_evidence_conflict_exits_without_watch_retry(observed_status, mocker, watch, manifest_status, runtime_status):
    resolution, _ = observed_status
    resolution.manifest["status"] = manifest_status
    resolution.runtime_state.update(status=runtime_status, waves=[_wave("prepare", "11", "succeeded")])
    sleep = mocker.patch("npa.cli.workbench.workflow.time.sleep", side_effect=AssertionError("must not retry"))
    result = _status_cli(mocker, "--json", *(["--watch"] if watch else []))
    assert result.exit_code == 2, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "EVIDENCE_INCONSISTENT"
    assert payload["workflow_lifecycle"]["completion_recorded"] is False
    sleep.assert_not_called()


@pytest.mark.parametrize("manifest_status", [None, "", "unknown", "invalid", {"status": "running"}])
def test_invalid_manifest_status_is_not_normalized_to_healthy_handoff(observed_status, manifest_status):
    resolution, _ = observed_status
    resolution.manifest["status"] = manifest_status
    resolution.runtime_state["waves"] = [_wave("prepare", "11", "succeeded")]
    payload = _durable_workflow_status("run-test")
    assert payload["status"] == "VERIFICATION_UNAVAILABLE"
    assert payload["automation_may_trust_state"] is False


@pytest.fixture()
def completed_interpreter_run(observed_status, tmp_path):
    from npa.orchestration.npa_workflow.interpreter import run_workflow
    from npa.orchestration.npa_workflow.spec import load_spec

    resolution, jobs = observed_status
    spec_path = tmp_path / "workflow.yaml"
    spec_path.write_text("""apiVersion: npa.workflow/v0.0.1
kind: Workflow
metadata:
  name: completion-fixture
config: {}
initial: finalize
states:
  finalize:
    run:
      argv: [echo, fixture]
    terminal: true
""")
    objects = {}
    store = RunStateStore(
        bucket="bucket", prefix="run-test",
        writer=lambda bucket, key, body: objects.__setitem__((bucket, key), body),
        reader=lambda bucket, key: objects[(bucket, key)].decode(),
    )
    executor = SimpleNamespace(execute=lambda step: {"state": step.state, "status": "ok", "job_id": "11"})
    run_workflow(load_spec(spec_path), run_id="run-test", execute=True, state_store=store, step_executor=executor)
    raw = objects[("bucket", "run-test/npa-workflow/manifest.json")]
    resolution.manifest = json.loads(raw)
    assert resolution.manifest["status"] == "completed"
    assert resolution.manifest["steps"][0]["status"] == "ok"
    resolution.runtime_state.update(status="succeeded", updated_at="2026-01-02T03:04:06Z",
                                    waves=[_wave("finalize", "11", "succeeded")])
    return resolution, jobs, raw


@pytest.mark.parametrize("runtime_status", ["running", "succeeded"])
@pytest.mark.parametrize("watch", [False, True])
@pytest.mark.parametrize("json_output", [False, True])
def test_real_interpreter_completion_succeeds_in_status_and_watch(
    completed_interpreter_run, mocker, runtime_status, watch, json_output,
):
    resolution, jobs, raw = completed_interpreter_run
    resolution.runtime_state["status"] = runtime_status
    sleep = mocker.patch("npa.cli.workbench.workflow.time.sleep", side_effect=AssertionError("Unexpected second poll"))
    args = ["--watch"] if watch else []
    if json_output:
        args.append("--json")
    result = _status_cli(mocker, *args)
    assert result.exit_code == 0, result.output
    assert [call.args[0] for call in jobs.call_args_list] == ["11"]
    sleep.assert_not_called()
    assert resolution.manifest == json.loads(raw)
    if json_output:
        payload = json.loads(result.stdout)
        assert payload["status"] == "SUCCEEDED"
        assert payload["verification_status"] == "VERIFIED"
        assert payload["workflow_lifecycle"]["manifest_evidence"] == {
            "status": "completed", "updated_at": json.loads(raw)["updated_at"], "source": "authoritative_manifest",
        }
    else:
        assert "status: SUCCEEDED" in result.stdout


@pytest.mark.parametrize("runtime_status", ["failed", "cancelled"])
def test_completed_producer_manifest_conflicts_with_runtime_failure(completed_interpreter_run, runtime_status):
    resolution, _, raw = completed_interpreter_run
    resolution.runtime_state["status"] = runtime_status
    payload = _durable_workflow_status("run-test")
    assert payload["status"] == "EVIDENCE_INCONSISTENT"
    assert payload["workflow_lifecycle"]["manifest_evidence"]["status"] == "completed"
    assert resolution.manifest == json.loads(raw)


@pytest.mark.parametrize("failed_status", ["failed", "cancelled"])
@pytest.mark.parametrize("watch", [False, True])
def test_completed_producer_manifest_cannot_hide_latest_failed_wave(
    completed_interpreter_run, mocker, failed_status, watch,
):
    resolution, jobs, _ = completed_interpreter_run
    resolution.runtime_state["waves"] = [_wave("finalize", "11", failed_status)]
    sleep = mocker.patch("npa.cli.workbench.workflow.time.sleep", side_effect=AssertionError("Unexpected second poll"))
    result = _status_cli(mocker, "--json", *(["--watch"] if watch else []))
    assert result.exit_code == 2, result.output
    assert json.loads(result.stdout)["status"] == "EVIDENCE_INCONSISTENT"
    assert [call.args[0] for call in jobs.call_args_list] == ["11"]
    sleep.assert_not_called()


@pytest.mark.parametrize("invalid", [None, "", "unknown", "complete", "done", {"status": "completed"}])
def test_real_completion_does_not_admit_invalid_manifest_states(completed_interpreter_run, invalid):
    resolution, _, _ = completed_interpreter_run
    resolution.manifest["status"] = invalid
    payload = _durable_workflow_status("run-test")
    assert payload["status"] == "VERIFICATION_UNAVAILABLE"
    assert payload["automation_may_trust_state"] is False


@pytest.mark.parametrize("invalid", ["completed", "COMPLETED"])
def test_runtime_completed_remains_invalid_with_real_manifest(completed_interpreter_run, invalid):
    resolution, _, _ = completed_interpreter_run
    resolution.runtime_state["status"] = invalid
    payload = _durable_workflow_status("run-test")
    assert payload["status"] == "VERIFICATION_UNAVAILABLE"
    assert payload["automation_may_trust_state"] is False


def test_real_completion_cannot_hide_live_query_failure(completed_interpreter_run):
    _, jobs, _ = completed_interpreter_run
    jobs.side_effect = None
    jobs.return_value = SimpleNamespace(status="SUCCEEDED", error="synthetic authentication denied")
    payload = _durable_workflow_status("run-test")
    assert payload["status"] == "VERIFICATION_UNAVAILABLE"
    assert "authentication denied" in payload["live_verification"]["reason"]
    assert payload["automation_may_trust_state"] is False


def test_real_completion_cached_mode_stays_non_authoritative(completed_interpreter_run):
    _, jobs, _ = completed_interpreter_run
    payload = _durable_workflow_status("run-test", cached=True)
    assert payload["status"] == "CACHED"
    assert payload["automation_may_trust_state"] is False
    jobs.assert_not_called()


@pytest.mark.parametrize("invalid", [False, True])
def test_real_completion_controls_final_artifact_loading(completed_interpreter_run, mocker, invalid):
    resolution, _, _ = completed_interpreter_run
    if invalid:
        resolution.runtime_state["waves"] = [_wave("finalize", "11", "failed")]
    loader = mocker.patch("npa.cli.workbench.workflow._load_paidf_artifact", return_value={"status": "verified", "verified": True})
    result = CliRunner().invoke(app, ["workbench", "workflow", "load-artifact", "run-test", "--project", "test", "--json"])
    if invalid:
        assert result.exit_code == 1
        assert "EVIDENCE_INCONSISTENT" in result.output
        loader.assert_not_called()
    else:
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)["verified"] is True
        loader.assert_called_once_with(project="test", run_id="run-test", run_prefix_uri="s3://bucket/run-test", s3_endpoint="", agent_name="")


def test_real_completion_cannot_hide_runtime_identity_mismatch(completed_interpreter_run, mocker):
    from npa.orchestration.npa_workflow.run_resolution import _attach_runtime_state

    resolution, jobs, _ = completed_interpreter_run
    wrong = {**resolution.runtime_state, "run_id": "different-run",
             "waves": [_wave("foreign", "99", "succeeded")]}
    jobs.side_effect = None
    jobs.return_value = SimpleNamespace(status="SUCCEEDED", error="")
    mocker.patch("npa.orchestration.skypilot.workflow_state.get_json", return_value=wrong)
    resolution.runtime_state = {}
    _attach_runtime_state(resolution, resolution.state)
    payload = _durable_workflow_status("run-test")
    assert payload["status"] == "VERIFICATION_UNAVAILABLE"
    assert "does not match" in payload["live_verification"]["reason"]
    assert payload["automation_may_trust_state"] is False
    assert [call.args[0] for call in jobs.call_args_list] == ["11"]


def test_real_completion_retains_success_after_historical_failed_attempt(completed_interpreter_run):
    resolution, jobs, _ = completed_interpreter_run
    resolution.runtime_state["waves"] = [
        _wave("finalize", "10", "failed"), _wave("finalize", "11", "succeeded", attempt=2),
    ]
    payload = _durable_workflow_status("run-test")
    assert payload["status"] == "SUCCEEDED"
    assert [call.args[0] for call in jobs.call_args_list] == ["11"]
    assert len(payload["stages"]["finalize"]["managed_job_attempts"]) == 2


def test_real_completion_still_requires_actual_final_artifact(completed_interpreter_run, mocker):
    head = mocker.Mock(side_effect=KeyError("synthetic missing artifact"))
    paginator = SimpleNamespace(paginate=lambda **kwargs: [{"Contents": []}])
    client = SimpleNamespace(s3=SimpleNamespace(head_object=head, get_paginator=lambda name: paginator))
    mocker.patch("npa.orchestration.npa_workflow.src_staging._storage_client", return_value=client)
    mocker.patch("npa.orchestration.npa_workflow.submission_state.update_submission_state")
    agents = mocker.patch("npa.cli.agent.resolve_project_agents")
    result = CliRunner().invoke(app, ["workbench", "workflow", "load-artifact", "run-test", "--project", "test", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "partial" and payload["verified"] is False
    assert "no .rrd artifact exists" in payload["detail"]
    head.assert_called_once_with(Bucket="bucket", Key="run-test/reports/sim2real.rrd")
    agents.assert_not_called()
