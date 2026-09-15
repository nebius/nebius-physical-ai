"""Live status must include jobs written after the initial runtime manifest."""

from types import SimpleNamespace

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

    assert payload["status"] == "VERIFICATION_UNAVAILABLE"
    assert payload["last_known"]["state"] == runtime_status.upper()
    assert payload["stages"]["prepare"]["state"] == "SUCCEEDED"
    assert "completion is not recorded" in payload["live_verification"]["reason"]


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
