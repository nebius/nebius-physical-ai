"""Offline unit regressions for live-submit waiting; no provider calls or live proof.

The live entrypoints are called directly with unit-only CLI, storage, status,
artifact and cancellation doubles. Their wait selection, polling and argv
builders remain real; the virtual clock never sleeps.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from npa.orchestration.npa_workflow.submit_matrix import (
    SUBMIT_LIVE_MATRIX,
    SubmitLiveCase,
)

WAIT_ENV = "NPA_E2E_NPA_WORKFLOW_SUBMIT_MAX_WAIT_SECONDS"
FINITE_WAITS = [(None, 3600), ("2400", 2400), ("7200", 7200)]


@pytest.fixture
def live_submit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    monkeypatch.delenv(WAIT_ENV, raising=False)
    return importlib.import_module("tests.e2e.test_npa_workflow_submit_live_e2e")


@pytest.fixture(params=["nurec-colmap-reconstruct.yaml", "nurec-reconstruct.yaml"])
def case(request):
    return next(case for case in SUBMIT_LIVE_MATRIX if case.spec == request.param)


def _cli_result(payload):
    return SimpleNamespace(exit_code=0, output=json.dumps(payload))


def _status(status):
    return SimpleNamespace(status=status, stdout="", stderr="")


@pytest.fixture
def harness(live_submit, case, monkeypatch, tmp_path):
    """Replace external boundaries only, including timeout cleanup and S3 checks."""
    mod = live_submit
    monkeypatch.setenv("NPA_E2E_NPA_WORKFLOW_SUBMIT_CANCEL_ON_TIMEOUT", "1")
    monkeypatch.setenv("NPA_E2E_NPA_WORKFLOW_SUBMIT_POLL_SECONDS", "30")
    monkeypatch.setattr(mod, "live_bucket", Mock(return_value="unit-bucket"))
    monkeypatch.setattr(mod, "_run_id_for", Mock(return_value="unit-run"))
    monkeypatch.setattr(
        mod, "materialize_live_spec", Mock(return_value=tmp_path / case.spec)
    )
    for name in ("seed_live_workflow_inputs", "write_runtime_evidence"):
        monkeypatch.setattr(mod, name, Mock())
    for name in ("_image_args", "_secret_env_args", "_skypilot_config_args"):
        monkeypatch.setattr(mod, name, Mock(return_value=[]))
    artifacts = Mock()
    monkeypatch.setattr(mod, "assert_nurec_colmap_live_outputs", artifacts)
    invoke = Mock(
        side_effect=[
            _cli_result({"status": "PLANNED", "steps": 1}),
            _cli_result({"status": "SUBMITTED", "job_id": "unit-job"}),
        ]
    )
    monkeypatch.setattr(mod, "RUNNER", SimpleNamespace(invoke=invoke))
    status = Mock(side_effect=map(_status, ["PENDING", "RUNNING", "SUCCEEDED"]))
    monkeypatch.setattr(mod, "workflow_status", status)
    clock = SimpleNamespace(now=0)

    def advance(seconds):
        clock.now += seconds

    sleep = Mock(side_effect=advance)
    monkeypatch.setattr(
        mod, "time", SimpleNamespace(monotonic=lambda: clock.now, sleep=sleep)
    )
    cancel = Mock()
    monkeypatch.setattr(
        "npa.orchestration.skypilot.workflow_state.cancel_workflow_job", cancel
    )
    monkeypatch.setattr(
        "npa.orchestration.skypilot._bin.resolve_config",
        Mock(return_value=SimpleNamespace(sky_bin="unit-sky")),
    )
    return SimpleNamespace(
        mod=mod,
        invoke=invoke,
        status=status,
        clock=clock,
        sleep=sleep,
        cancel=cancel,
        artifacts=artifacts,
        args=dict(
            case=case,
            tmp_path=tmp_path,
            e2e_project=None,
            e2e_registry="registry.example/workbench",
            forbidden_markers=[],
        ),
    )


@pytest.mark.parametrize(
    "declared,env_value,expected",
    [
        (0, None, 3600),
        (5400, None, 5400),
        (0, "2400", 2400),
        (5400, "2400", 5400),
        (0, "7200", 7200),
        (5400, "7200", 5400),
        (0, "0", 0),
        (5400, "0", 0),
        (7200, "0", 0),
    ],
)
def test_case_wait_precedence(live_submit, monkeypatch, declared, env_value, expected):
    if env_value is not None:
        monkeypatch.setenv(WAIT_ENV, env_value)
    case = SubmitLiveCase("unit-wait.yaml", "gpu", max_wait_seconds=declared)
    assert live_submit._case_max_wait(case) == expected


def test_explicit_zero_polls_past_declared_wait_until_success(
    harness, case, monkeypatch
):
    monkeypatch.setenv(WAIT_ENV, "0")
    # Virtual time crosses both the 3600s default and NuRec's 5400s declaration.
    monkeypatch.setenv("NPA_E2E_NPA_WORKFLOW_SUBMIT_POLL_SECONDS", "6000")

    harness.mod.test_npa_workflow_submit_live_reaches_terminal(**harness.args)

    assert harness.status.call_args_list == [call("unit-job")] * 3
    assert harness.sleep.call_args_list == [call(6000)] * 2
    assert harness.clock.now == 12000
    harness.cancel.assert_not_called()
    if case.spec == "nurec-colmap-reconstruct.yaml":
        harness.artifacts.assert_called_once_with(
            bucket="unit-bucket", run_id="unit-run", e2e_project=None
        )


@pytest.mark.parametrize("env_value,fallback", FINITE_WAITS)
@pytest.mark.parametrize("succeeds", [True, False], ids=["success", "timeout"])
def test_finite_wait_keeps_polling_and_timeout_cleanup(
    harness, case, monkeypatch, env_value, fallback, succeeds
):
    if env_value is not None:
        monkeypatch.setenv(WAIT_ENV, env_value)
    expected = 5400 if case.spec == "nurec-reconstruct.yaml" else fallback
    interval = expected // 3
    monkeypatch.setenv("NPA_E2E_NPA_WORKFLOW_SUBMIT_POLL_SECONDS", str(interval))

    if succeeds:
        harness.mod.test_npa_workflow_submit_live_reaches_terminal(**harness.args)
        harness.cancel.assert_not_called()
        assert harness.sleep.call_args_list == [call(interval)] * 2
    else:
        harness.status.side_effect = None
        harness.status.return_value = _status("PENDING")
        with pytest.raises(pytest.fail.Exception, match=f"within {expected}s"):
            harness.mod.test_npa_workflow_submit_live_reaches_terminal(**harness.args)
        assert harness.clock.now == expected
        assert harness.sleep.call_args_list == [call(interval)] * 3
        harness.cancel.assert_called_once_with(
            sky_bin="unit-sky", job_id="unit-job", run_id="unit-run", cluster="unit-run"
        )
        harness.artifacts.assert_not_called()
    assert harness.status.call_args_list == [call("unit-job")] * 3


@pytest.mark.parametrize("env_value,fallback", [*FINITE_WAITS, ("0", 0)])
def test_runtime_entrypoint_forwards_selected_wait(
    harness, case, monkeypatch, env_value, fallback
):
    if env_value is not None:
        monkeypatch.setenv(WAIT_ENV, env_value)
    monkeypatch.setenv("NPA_E2E_NPA_WORKFLOW_RUNTIME", "1")
    harness.invoke.side_effect = None
    harness.invoke.return_value = _cli_result({"status": "succeeded", "waves": [{}]})

    harness.mod.test_npa_workflow_runtime_live_reaches_terminal(**harness.args)

    harness.invoke.assert_called_once()
    argv = harness.invoke.call_args.args[1]
    expected = (
        5400 if case.spec == "nurec-reconstruct.yaml" and env_value != "0" else fallback
    )
    assert "--runtime" in argv
    assert argv[argv.index("--max-wait-seconds") + 1] == str(expected)


def test_explicit_zero_still_reports_terminal_failure(harness, monkeypatch):
    monkeypatch.setenv(WAIT_ENV, "0")
    harness.status.side_effect = map(_status, ["PENDING", "RUNNING", "FAILED_RUNTIME"])

    with pytest.raises(
        pytest.fail.Exception, match="terminal failure status=FAILED_RUNTIME"
    ):
        harness.mod.test_npa_workflow_submit_live_reaches_terminal(**harness.args)

    assert harness.status.call_count == 3
    harness.cancel.assert_not_called()
    harness.artifacts.assert_not_called()
