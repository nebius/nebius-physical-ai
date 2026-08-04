"""`detection-training train` must be able to wait, and to carry a label map.

The BDD100K pipeline template's train task did three things a `toolRef` could not reach:

* POSTed ``/train`` and then **polled ``/status`` in bash** until the run completed, failing
  on ``failed`` or timeout, and finally asserting every epoch ran and a checkpoint pattern
  exists. Without that wait, ``train`` returns while training is still running and the
  eval stage evaluates a checkpoint that does not exist yet.
* sent a ``label_map`` in the request body — a real ``TrainRequest`` field that had no CLI
  flag and is not an accepted ``--override`` key, so it was unreachable from any spec.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from typer.testing import CliRunner

from npa.cli.workbench import detection_training as dt

runner = CliRunner()


# --------------------------------------------------------------------------- label map


def test_label_map_accepts_the_templates_json_spelling() -> None:
    parsed = dt.parse_label_map('{"person":0,"rider":1,"traffic light":8}')

    assert parsed == {"person": 0, "rider": 1, "traffic light": 8}


def test_label_map_accepts_name_equals_index_pairs() -> None:
    assert dt.parse_label_map(" person=0 , rider=1 ") == {"person": 0, "rider": 1}


def test_label_map_is_absent_when_not_requested() -> None:
    assert dt.parse_label_map("") is None
    assert dt.parse_label_map("   ") is None


@pytest.mark.parametrize(
    "raw",
    [
        '{"person": 0,',  # invalid JSON
        '["person"]',  # JSON, but not an object
        "person",  # missing "=index"
        "=0",  # missing name
        "person=one",  # non-integer index
    ],
)
def test_label_map_rejects_junk(raw: str) -> None:
    from typer import Exit

    with pytest.raises(Exit):
        dt.parse_label_map(raw)


# ------------------------------------------------------------------------------- wait


def _status_sequence(*payloads: dict[str, Any]):
    """Return a `request_json` stand-in that walks `payloads` on successive calls."""

    calls: list[tuple[str, str]] = []
    remaining = list(payloads)

    def fake_request_json(method, endpoint, path, **kwargs):
        calls.append((method, path))
        if path == "/train":
            return {"run_id": "train-rider", "status": "running"}
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    return fake_request_json, calls


COMPLETE = {
    "status": "completed",
    "epochs_completed": 2,
    "total_epochs": 2,
    "checkpoint_uri_pattern": "s3://b/p/checkpoints/epoch_{epoch}.pt",
}


def test_wait_returns_the_terminal_status_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    fake, calls = _status_sequence({"status": "running", "epochs_completed": 1}, COMPLETE)
    monkeypatch.setattr(dt, "request_json", fake)

    result = dt.wait_for_training_run(
        "train-rider",
        endpoint="http://mock",
        token_env="T",
        poll_seconds=0,
        timeout_seconds=60,
    )

    assert result == COMPLETE
    # Polled until terminal rather than trusting the first answer.
    assert calls == [("GET", "/status"), ("GET", "/status")]


def test_wait_fails_when_the_run_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    from typer import Exit

    fake, _ = _status_sequence({"status": "failed", "error": "cuda oom"})
    monkeypatch.setattr(dt, "request_json", fake)

    with pytest.raises(Exit):
        dt.wait_for_training_run(
            "train-rider", endpoint="http://mock", token_env="T", poll_seconds=0, timeout_seconds=60
        )


def test_wait_fails_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    from typer import Exit

    fake, calls = _status_sequence({"status": "running"})
    monkeypatch.setattr(dt, "request_json", fake)

    with pytest.raises(Exit):
        dt.wait_for_training_run(
            "train-rider", endpoint="http://mock", token_env="T", poll_seconds=0, timeout_seconds=0
        )

    # One poll, then the deadline is already past.
    assert calls == [("GET", "/status")]


def test_wait_fails_without_a_run_id(monkeypatch: pytest.MonkeyPatch) -> None:
    from typer import Exit

    monkeypatch.setattr(dt, "request_json", lambda *a, **k: pytest.fail("must not poll"))

    with pytest.raises(Exit):
        dt.wait_for_training_run(
            "", endpoint="http://mock", token_env="T", poll_seconds=0, timeout_seconds=60
        )


@pytest.mark.parametrize(
    "payload",
    [
        # The template's closing `jq -e`: all epochs ran ...
        {**COMPLETE, "epochs_completed": 1},
        # ... and a checkpoint pattern exists.
        {**COMPLETE, "checkpoint_uri_pattern": ""},
    ],
)
def test_wait_enforces_the_templates_completion_assertion(
    monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]
) -> None:
    from typer import Exit

    fake, _ = _status_sequence(payload)
    monkeypatch.setattr(dt, "request_json", fake)

    with pytest.raises(Exit):
        dt.wait_for_training_run(
            "train-rider", endpoint="http://mock", token_env="T", poll_seconds=0, timeout_seconds=60
        )


# ------------------------------------------------------------------- end to end (CLI)


def _train_argv(*extra: str) -> list[str]:
    return [
        "train",
        "--view",
        "bdd100k_rider_train",
        "--lance-uri",
        "s3://bucket/run/lancedb/",
        "--output-uri",
        "s3://bucket/run/training/bdd100k_rider_train",
        "--service",
        "--endpoint",
        "http://mock",
        *extra,
    ]


def test_train_waits_and_reports_the_final_status(monkeypatch: pytest.MonkeyPatch) -> None:
    fake, calls = _status_sequence({"status": "running"}, COMPLETE)
    monkeypatch.setattr(dt, "request_json", fake)

    result = runner.invoke(dt.app, _train_argv("--wait", "--poll-seconds", "0"))

    assert result.exit_code == 0, result.output
    assert calls[0] == ("POST", "/train")
    assert ("GET", "/status") in calls
    assert json.loads(result.stdout)["status"] == "completed"


def test_train_does_not_poll_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--wait` is opt-in: the existing fire-and-forget behaviour is unchanged."""

    fake, calls = _status_sequence(COMPLETE)
    monkeypatch.setattr(dt, "request_json", fake)

    result = runner.invoke(dt.app, _train_argv())

    assert result.exit_code == 0, result.output
    assert calls == [("POST", "/train")]


def test_train_sends_the_label_map(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: dict[str, Any] = {}

    def fake_request_json(method, endpoint, path, *, payload=None, **kwargs):
        sent.update(payload or {})
        return {"run_id": "train-rider", "status": "running"}

    monkeypatch.setattr(dt, "request_json", fake_request_json)

    result = runner.invoke(dt.app, _train_argv("--label-map", '{"person":0,"rider":1}'))

    assert result.exit_code == 0, result.output
    assert sent["label_map"] == {"person": 0, "rider": 1}


def test_train_omits_the_label_map_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: dict[str, Any] = {}

    def fake_request_json(method, endpoint, path, *, payload=None, **kwargs):
        sent.update(payload or {})
        return {"run_id": "train-rider", "status": "running"}

    monkeypatch.setattr(dt, "request_json", fake_request_json)

    result = runner.invoke(dt.app, _train_argv())

    assert result.exit_code == 0, result.output
    assert sent["label_map"] is None
