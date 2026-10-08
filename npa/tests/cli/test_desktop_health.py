"""Check real local health responses without exposing credentials or changing history."""

import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
import threading
from unittest.mock import Mock

import pytest

from npa.tools.desktop import remote


def _health_handler(responses):
    expected = "Basic " + base64.b64encode(b"example:private-test-password").decode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.headers["Authorization"] == expected
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(responses[self.path]).encode())

        def log_message(self, *_args):
            pass

    return Handler


@pytest.fixture
def health_service(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    password = tmp_path / "password"
    password.write_text("private-test-password")
    responses = {
        "/chat/api/state": {"connected": True},
        "/chat/api/models": {"data": [{"model": "example"}]},
    }
    server = ThreadingHTTPServer(("127.0.0.1", 0), _health_handler(responses))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "port": server.server_port,
                "username": "example",
                "password_file": str(password),
                "session_secret": "private-signing-key",
            }
        )
    )
    try:
        yield config, responses
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_health_checks_authenticated_runtime_not_just_running_service(health_service):
    config, responses = health_service
    health = remote._chat_health(config)
    assert health == {
        "installed": True,
        "connected": True,
        "model_count": 1,
        "history": {"verified": False},
    }
    responses["/chat/api/state"] = {"connected": False}
    assert remote._chat_health(config)["connected"] is False


@pytest.mark.parametrize(
    "path,payload",
    [
        ("/chat/api/state", None),
        ("/chat/api/models", {"data": "private-test-password"}),
        ("/chat/api/models", None),
    ],
)
def test_malformed_runtime_response_fails_without_private_details(
    health_service, path, payload
):
    config, responses = health_service
    responses[path] = payload
    result = remote._chat_health(config)
    assert result["probe_error"] == "Chat readiness could not be verified."
    assert result["model_count"] is None
    assert "private-" not in json.dumps(result)


def test_history_counts_missing_files_without_modifying_database(tmp_path):
    present = tmp_path / "saved.jsonl"
    present.write_text("saved transcript")
    database = tmp_path / "history with spaces.sqlite"
    with sqlite3.connect(database) as db:
        db.execute(
            "CREATE TABLE threads (rollout_path TEXT, source TEXT, archived INTEGER)"
        )
        db.executemany(
            "INSERT INTO threads VALUES (?, ?, ?)",
            [
                (str(present), "vscode", 0),
                (str(tmp_path / "absent"), "vscode", 0),
                (None, "vscode", 1),
                (None, "cli", 0),
            ],
        )
    before = database.read_bytes()
    assert remote._chat_history_health(database) == {
        "verified": True,
        "records": 2,
        "missing_transcripts": 1,
    }
    assert database.read_bytes() == before


def test_uninstalled_health_does_not_create_files(tmp_path):
    assert remote._chat_health(tmp_path / "missing") == {"installed": False}
    assert remote._chat_history_health(tmp_path / "database") == {"verified": False}
    assert not list(tmp_path.iterdir())


def test_remote_status_reports_capacity_before_desktop_install(monkeypatch, tmp_path):
    monkeypatch.setattr(remote, "_STATE", tmp_path / "uninstalled")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(
        remote.subprocess, "run", Mock(return_value=Mock(stdout="inactive"))
    )
    result = remote._status()
    assert result["chat_health"] == {"installed": False}
    assert result["disk"]["total_bytes"] > result["disk"]["free_bytes"] > 0
    assert not (tmp_path / "uninstalled").exists()
