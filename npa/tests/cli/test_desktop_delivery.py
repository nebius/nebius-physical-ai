"""Verify durable submission receipts across slow owners, retries, and restarts."""

import threading
import uuid
from unittest.mock import Mock

import pytest

from npa.tools.desktop.chat_delivery import Deliveries


@pytest.fixture
def delivery(tmp_path, monkeypatch):
    journal = Deliveries(tmp_path / "deliveries.sqlite")
    finished = threading.Event()
    complete = journal._complete

    def finish(*args):
        try:
            complete(*args)
        finally:
            finished.set()

    monkeypatch.setattr(journal, "_complete", finish)
    return journal, finished


def message():
    return {
        "id": "existing-thread",
        "text": "Continue the task",
        "clientUserMessageId": str(uuid.uuid4()),
    }


def test_slow_submission_returns_receipt_and_concurrent_retry_does_not_replay(delivery):
    journal, finished = delivery
    body = message()
    identifier = body["clientUserMessageId"]
    release = threading.Event()
    action = Mock(side_effect=lambda: (release.wait(), {"accepted": True})[1])
    try:
        assert journal.submit(identifier, body, action)["state"] == "pending"
        assert journal.submit(identifier, body, action)["state"] == "pending"
        with pytest.raises(ValueError, match="another message"):
            journal.submit(identifier, {**body, "text": "Different"}, action)
        assert journal.status(identifier)["threadId"] == body["id"]
    finally:
        release.set()
        assert finished.wait(5)
    assert action.call_count == 1
    assert journal.status(identifier) == {
        "state": "complete",
        "result": {"accepted": True},
    }


def test_completed_async_receipt_survives_restart(delivery, tmp_path):
    journal, finished = delivery
    body = message()
    action = Mock(return_value={"accepted": True})
    identifier = body["clientUserMessageId"]
    journal.submit(identifier, body, action)
    assert finished.wait(5)
    restarted = Deliveries(tmp_path / "deliveries.sqlite")
    assert restarted.submit(identifier, body, action)["state"] == "complete"
    assert action.call_count == 1
    assert (tmp_path / "deliveries.sqlite").stat().st_mode & 0o777 == 0o600


def test_failed_owner_reply_stays_uncertain_until_confirmed(delivery, tmp_path):
    journal, finished = delivery
    body = message()
    identifier = body["clientUserMessageId"]
    action = Mock(side_effect=RuntimeError("Owner acknowledgment timed out"))
    journal.submit(identifier, body, action)
    assert finished.wait(5)
    assert journal.status(identifier)["state"] == "uncertain"
    assert "timed out" in journal.status(identifier)["error"]
    restarted = Deliveries(tmp_path / "deliveries.sqlite")
    assert restarted.submit(identifier, body, action)["state"] == "uncertain"
    assert action.call_count == 1
    restarted.confirm(identifier, {"clientUserMessageId": identifier})
    assert restarted.status(identifier)["state"] == "complete"


def test_confirmation_is_preserved_if_owner_later_reports_timeout(delivery):
    journal, finished = delivery
    body = message()
    identifier = body["clientUserMessageId"]
    release = threading.Event()

    def action():
        release.wait()
        raise RuntimeError("Owner acknowledgment timed out")

    try:
        journal.submit(identifier, body, action)
        journal.confirm(identifier, {"observed": True})
    finally:
        release.set()
        assert finished.wait(5)
    assert journal.status(identifier) == {
        "state": "complete",
        "result": {"observed": True},
    }


def test_existing_uncertain_journal_is_not_replayed_by_async_upgrade(tmp_path):
    journal = Deliveries(tmp_path / "deliveries.sqlite")
    body = message()
    identifier = body["clientUserMessageId"]
    action = Mock(side_effect=RuntimeError("Connection lost"))
    with pytest.raises(RuntimeError):
        journal.execute(identifier, body, action)
    upgraded = Deliveries(tmp_path / "deliveries.sqlite")
    receipt = upgraded.submit(identifier, body, action)
    assert receipt["state"] == "uncertain"
    assert receipt["threadId"] == body["id"]
    assert action.call_count == 1


def test_missing_and_invalid_receipts_never_invoke_an_action(delivery):
    journal, _ = delivery
    assert journal.status(str(uuid.uuid4())) == {"state": "missing"}
    with pytest.raises(ValueError):
        journal.status("invalid")
