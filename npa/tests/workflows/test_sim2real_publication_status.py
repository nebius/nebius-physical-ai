"""Actual status/watch regression for known publication unavailability."""

from unittest.mock import Mock

import pytest

from npa.workflows.sim2real import monitor
from npa.workflows.sim2real.publication import PublicationConflict


@pytest.mark.parametrize(
    "state, expected", [("publishing", "RUNNING"), ("unavailable", "UNKNOWN")]
)
def test_successful_worker_does_not_hide_publication(monkeypatch, state, expected):
    stages = {
        "report": {
            "state": "PENDING",
            "publication_state": state,
            "error_code": "publication_conflict",
        }
    }
    monkeypatch.setattr(monitor, "_stage_states", lambda **_: stages)
    monkeypatch.setattr(
        monitor,
        "load_operator_config",
        lambda: monitor.OperatorConfig("unit", "https://storage.example", "", "unit"),
    )
    monkeypatch.setattr(
        monitor,
        "_k8s_orchestrator_status",
        lambda **_: {"found": True, "phase": "SUCCEEDED"},
    )
    monkeypatch.setattr(monitor, "_k8s_sibling_summary", lambda **_: [])
    monkeypatch.setattr(monitor.StorageClient, "from_environment", lambda **_: Mock())
    monkeypatch.setattr(monitor, "_load_workflow_state", lambda *_: {})
    monkeypatch.setattr(monitor, "_extract_eval_metrics", lambda **_: {})
    result = monitor.watch_sim2real_status(
        "run-a", watch=False, json_output=True, kubeconfig="/fixture/kubeconfig"
    )
    assert result["status"] == expected
    assert not monitor.status_is_terminal(result["status"])
    assert result["publication_state"] == state


def test_late_metrics_publication_race_revokes_success(monkeypatch):
    stages = {"report": {"state": "SUCCEEDED", "publication_state": "available"}}
    monkeypatch.setattr(monitor, "_stage_states", lambda **_: stages)
    monkeypatch.setattr(
        monitor,
        "load_operator_config",
        lambda: monitor.OperatorConfig("unit", "https://storage.example", "", ""),
    )
    monkeypatch.setattr(monitor.StorageClient, "from_environment", lambda **_: Mock())
    monkeypatch.setattr(monitor, "_load_workflow_state", lambda *_: {})

    def metrics(**kwargs):
        raise PublicationConflict("fixture race", publication_state="publishing")

    monkeypatch.setattr(monitor, "_extract_eval_metrics", metrics)
    result = monitor.watch_sim2real_status("run-a", watch=False, json_output=True)
    assert result["status"] == "RUNNING"
    assert result["stages"]["report"]["state"] != "SUCCEEDED"
    assert result["eval_metrics"] == {}


def test_watch_waits_for_publication_even_after_worker_success(monkeypatch):
    pending = {
        "report": {
            "state": "PENDING",
            "publication_state": "publishing",
            "error_code": "publication_conflict",
        }
    }
    ready = {"report": {"state": "SUCCEEDED", "publication_state": "available"}}
    states = iter([pending, ready])
    monkeypatch.setattr(monitor, "_stage_states", lambda **_: next(states))
    monkeypatch.setattr(
        monitor,
        "load_operator_config",
        lambda: monitor.OperatorConfig("unit", "https://storage.example", "", "unit"),
    )
    monkeypatch.setattr(
        monitor,
        "_k8s_orchestrator_status",
        lambda **_: {"found": True, "phase": "SUCCEEDED"},
    )
    monkeypatch.setattr(monitor, "_k8s_sibling_summary", lambda **_: [])
    monkeypatch.setattr(monitor.StorageClient, "from_environment", lambda **_: Mock())
    monkeypatch.setattr(monitor, "_load_workflow_state", lambda *_: {})
    monkeypatch.setattr(monitor, "_extract_eval_metrics", lambda **_: {})
    sleeps = []
    monkeypatch.setattr("time.sleep", sleeps.append)
    result = monitor.watch_sim2real_status(
        "run-a",
        watch=True,
        interval=0.1,
        json_output=True,
        kubeconfig="/fixture/kubeconfig",
    )
    assert sleeps == [0.1]
    assert result["status"] == "SUCCEEDED"
    assert result["publication_state"] == "available"


@pytest.mark.parametrize(
    "code, expected", [("404", "absent"), ("AccessDenied", "unavailable")]
)
def test_legacy_report_absence_is_distinct_from_denial(monkeypatch, code, expected):
    from botocore.exceptions import ClientError
    from test_sim2real_monitor import _mock_s3_client

    client = _mock_s3_client({})
    original = client._s3.head_object.side_effect

    def head(**kwargs):
        if kwargs["Key"].endswith("reports/sim2real-report.json"):
            raise ClientError({"Error": {"Code": code}}, "HeadObject")
        return original(**kwargs)

    client._s3.head_object.side_effect = head
    monkeypatch.setattr(monitor.StorageClient, "from_environment", lambda **_: client)
    stages = monitor._stage_states(
        bucket="unit",
        run_id="run-a",
        s3_prefix="runs",
        endpoint="https://storage.example",
        outer_iterations=1,
    )
    assert stages["report"]["publication_state"] == expected
    assert stages["report"]["state"] == "PENDING"
    assert bool(stages["report"].get("error_code")) == (code != "404")
