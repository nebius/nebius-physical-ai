"""Live status must include jobs written after the initial runtime manifest."""

import json
from types import SimpleNamespace

from typer.testing import CliRunner

from npa.cli.main import app

import pytest

from npa.cli.workbench.workflow import _durable_workflow_status
from npa.orchestration.npa_workflow.run_resolution import RunResolution
from npa.orchestration.npa_workflow.run_state import RunManifest, runtime_manifest_view
from npa.orchestration.skypilot.workflow_state import WorkflowS3Config


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


@pytest.mark.parametrize("mode", ["cached", "error", "unknown", "exception", "malformed", "regressed"])
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
    elif mode in {"malformed", "regressed"}:
        resolution.manifest["steps"] = [{"state": "prepare", "status": "succeeded"}]
        jobs.side_effect = None
        jobs.return_value = SimpleNamespace(status="invalid" if mode == "malformed" else "RUNNING", error="")
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
