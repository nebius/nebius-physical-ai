"""Verify send reservations and outcomes survive abrupt process loss and restore."""

import multiprocessing
import sqlite3
import threading
import time
import uuid
from unittest.mock import Mock

import pytest

from npa.tools.desktop.chat_delivery import Deliveries


def _sender(path, body, completed, connection):
    journal = Deliveries(path)
    identifier = body["clientUserMessageId"]

    def action():
        if completed:
            return {"accepted": True}
        connection.send("accepted")
        threading.Event().wait()

    journal.submit(identifier, body, action)
    if completed:
        while journal.status(identifier)["state"] != "complete":
            time.sleep(0.01)
        connection.send("complete")
    threading.Event().wait()


@pytest.mark.parametrize("completed", [False, True])
def test_killed_sender_retains_journal_without_replaying(tmp_path, completed):
    path = tmp_path / "deliveries.sqlite"
    body = {
        "id": "saved-chat",
        "text": "Continue",
        "clientUserMessageId": str(uuid.uuid4()),
    }
    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(target=_sender, args=(path, body, completed, writer))
    process.start()
    try:
        assert reader.poll(10), "The isolated sender did not reach its durable state"
        assert reader.recv() == ("complete" if completed else "accepted")
    finally:
        process.kill()
        process.join(10)
        reader.close()
        writer.close()
    assert not process.is_alive()
    journal = Deliveries(path)
    assert journal.database.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    action = Mock()
    receipt = journal.submit(body["clientUserMessageId"], body, action)
    assert receipt["state"] == ("complete" if completed else "uncertain")
    action.assert_not_called()
    journal.database.close()


def test_sqlite_backup_restore_retains_confirmed_identity(tmp_path):
    body = {
        "id": "saved-chat",
        "text": "Continue",
        "clientUserMessageId": str(uuid.uuid4()),
    }
    journal = Deliveries(tmp_path / "original.sqlite")
    journal.execute(body["clientUserMessageId"], body, lambda: {"accepted": True})
    restored_path = tmp_path / "restored.sqlite"
    with sqlite3.connect(restored_path) as backup:
        journal.database.backup(backup)
    restored = Deliveries(restored_path)
    action = Mock()
    assert restored.submit(body["clientUserMessageId"], body, action) == {
        "state": "complete",
        "result": {"accepted": True},
    }
    action.assert_not_called()
    restored.database.close()
    journal.database.close()
