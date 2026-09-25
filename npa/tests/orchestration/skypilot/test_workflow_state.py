"""Cause-aware durable workflow-state reads."""

from __future__ import annotations

from io import BytesIO

from botocore.exceptions import ClientError
import pytest

from npa.orchestration.skypilot import workflow_state
from npa.orchestration.skypilot.workflow_state import (
    WorkflowS3Config,
    WorkflowStateError,
    read_stage_status,
)


class _ObjectClient:
    def __init__(self, *, body: bytes = b"{}", failure: BaseException | None = None):
        self.body = body
        self.failure = failure

    def get_object(self, **_kwargs):
        if self.failure is not None:
            raise self.failure
        return {"Body": BytesIO(self.body)}


def _provider_error(code: str) -> ClientError:
    return ClientError(
        {"Error": {"Code": code, "Message": "synthetic provider failure"}},
        "GetObject",
    )


def _state(monkeypatch: pytest.MonkeyPatch, client: _ObjectClient) -> WorkflowS3Config:
    monkeypatch.setattr(workflow_state.boto3, "client", lambda *args, **kwargs: client)
    return WorkflowS3Config(
        bucket="synthetic-bucket",
        prefix="synthetic-run",
        endpoint_url="https://storage.example.invalid",
    )


@pytest.mark.parametrize("code", ["NoSuchKey", "404"])
def test_read_stage_status_treats_only_missing_provider_codes_as_optional(
    monkeypatch: pytest.MonkeyPatch,
    code: str,
) -> None:
    state = _state(monkeypatch, _ObjectClient(failure=_provider_error(code)))

    assert read_stage_status(state, "train") is None


@pytest.mark.parametrize(
    "failure",
    [FileNotFoundError("missing"), KeyError("missing")],
    ids=["file-not-found", "key-error"],
)
def test_read_stage_status_preserves_local_missing_semantics(
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
) -> None:
    state = _state(monkeypatch, _ObjectClient(failure=failure))

    assert read_stage_status(state, "train") is None


@pytest.mark.parametrize(
    "failure",
    [
        _provider_error("AccessDenied"),
        _provider_error("SlowDown"),
        _provider_error("NoSuchBucket"),
        TimeoutError("synthetic timeout"),
    ],
    ids=["access-denied", "throttled", "missing-bucket", "transport"],
)
def test_read_stage_status_raises_when_exact_state_is_unreadable(
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
) -> None:
    state = _state(monkeypatch, _ObjectClient(failure=failure))

    with pytest.raises(WorkflowStateError) as exc_info:
        read_stage_status(state, "train")

    assert exc_info.value.__cause__ is failure


def test_read_stage_status_rejects_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(monkeypatch, _ObjectClient(body=b"{"))

    with pytest.raises(WorkflowStateError, match="Invalid JSON"):
        read_stage_status(state, "train")


def test_read_stage_status_returns_valid_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(
        monkeypatch,
        _ObjectClient(body=b'{"state": "SUCCEEDED", "job_id": "42"}'),
    )

    assert read_stage_status(state, "train") == {
        "state": "SUCCEEDED",
        "job_id": "42",
    }


@pytest.mark.parametrize(
    "failure",
    [KeyError("credential"), FileNotFoundError("credential"), _provider_error("404")],
    ids=["configuration-key", "credential-file", "credential-provider-404"],
)
def test_client_setup_failure_never_proves_object_absence(monkeypatch, failure):
    state = _state(monkeypatch, _ObjectClient())

    def fail_setup(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(workflow_state.boto3, "client", fail_setup)
    with pytest.raises(WorkflowStateError) as caught:
        read_stage_status(state, "train")
    assert caught.value.__cause__ is failure
    assert not workflow_state.workflow_state_error_is_missing(caught.value)


@pytest.mark.parametrize("response", [{}, None, {"Body": None}])
def test_malformed_object_response_is_unreadable(monkeypatch, response):
    client = _ObjectClient()
    monkeypatch.setattr(client, "get_object", lambda **_kwargs: response)
    state = _state(monkeypatch, client)

    with pytest.raises(WorkflowStateError, match="response unreadable") as caught:
        read_stage_status(state, "train")
    assert not workflow_state.workflow_state_error_is_missing(caught.value)


@pytest.mark.parametrize(
    "failure",
    [
        KeyError("body"),
        FileNotFoundError("body"),
        _provider_error("404"),
        TimeoutError(),
    ],
    ids=["key-error", "file-not-found", "provider-404", "timeout"],
)
def test_body_read_failure_is_unreadable_and_closes_body(monkeypatch, failure):
    body = BytesIO(b"{}")

    def fail_read():
        raise failure

    monkeypatch.setattr(body, "read", fail_read)
    client = _ObjectClient()
    monkeypatch.setattr(client, "get_object", lambda **_kwargs: {"Body": body})
    state = _state(monkeypatch, client)

    with pytest.raises(WorkflowStateError, match="response unreadable") as caught:
        read_stage_status(state, "train")
    assert caught.value.__cause__ is failure
    assert not workflow_state.workflow_state_error_is_missing(caught.value)
    assert body.closed


def test_body_close_failure_is_unreadable(monkeypatch):
    body = BytesIO(b"{}")

    def fail_close():
        raise KeyError("cleanup")

    monkeypatch.setattr(body, "close", fail_close)
    client = _ObjectClient()
    monkeypatch.setattr(client, "get_object", lambda **_kwargs: {"Body": body})
    state = _state(monkeypatch, client)

    with pytest.raises(WorkflowStateError, match="response unreadable") as caught:
        read_stage_status(state, "train")
    assert not workflow_state.workflow_state_error_is_missing(caught.value)
