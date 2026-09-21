"""Exercise authenticated chat boundaries and shared Codex controls over real HTTP."""

from http.server import ThreadingHTTPServer
import threading
from unittest.mock import Mock

import httpx
import pytest

from npa.tools.desktop.chat_server import ChatHandler, _answer_result
from npa.tools.desktop.chat_history import owned_elsewhere


@pytest.fixture
def chat(tmp_path):
    password = tmp_path / "password"
    password.write_text("unit-test-only-password")
    server = ThreadingHTTPServer(("127.0.0.1", 0), ChatHandler)
    server.config = {
        "password_file": str(password),
        "username": "developer",
        "origin": "https://desktop.example.test",
        "cwd": str(tmp_path),
    }
    server.rpc = Mock()
    server.attached = set()
    server.created = {}
    server.config["socket"] = str(tmp_path / "codex.sock")
    server.rpc.call.side_effect = lambda method, params: (
        {"thread": {"id": "existing-thread", "status": {"type": "idle"}}}
        if method == "thread/read"
        else {"accepted": True}
    )
    server.rpc.pending.return_value = []
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    with httpx.Client(
        base_url=f"http://127.0.0.1:{server.server_port}",
        auth=("developer", password.read_text()),
        headers={"Origin": server.config["origin"]},
        trust_env=False,
    ) as client:
        yield client, server.rpc
    server.shutdown()
    server.server_close()
    worker.join()


@pytest.mark.parametrize(
    "path",
    [
        "/chat/",
        "/chat/chat.js",
        "/chat/chat.css",
        "/chat/api/threads",
        "/chat/api/events",
    ],
)
def test_every_chat_route_requires_login(chat, path):
    client, rpc = chat
    response = client.get(path, auth=None)
    assert response.status_code == 401
    assert "WWW-Authenticate" in response.headers
    rpc.call.assert_not_called()


def test_wrong_password_cannot_read_threads(chat):
    client, rpc = chat
    assert (
        client.get("/chat/api/threads", auth=("developer", "wrong")).status_code == 401
    )
    rpc.call.assert_not_called()


@pytest.mark.parametrize("origin", [None, "https://hostile.example.test", "null"])
def test_cross_origin_mutations_are_rejected(chat, origin):
    client, rpc = chat
    request = client.build_request(
        "POST", "/chat/api/send", json={"id": "thread", "text": "hello"}
    )
    if origin is None:
        del request.headers["Origin"]
    else:
        request.headers["Origin"] = origin
    assert client.send(request).status_code == 403
    rpc.call.assert_not_called()


def test_static_page_has_no_inline_script_or_cache(chat):
    client, _ = chat
    response = client.get("/chat/")
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
    assert '<script src="/chat/chat.js" defer>' in response.text
    assert "unit-test-only-password" not in response.text


def test_list_includes_all_providers_and_source_kinds(chat):
    client, rpc = chat
    assert (
        client.get("/chat/api/threads?search=example&archived=true").status_code == 200
    )
    method, params = rpc.call.call_args.args
    assert method == "thread/list"
    assert params["modelProviders"] == []
    assert {"vscode", "cli", "appServer", "exec"} <= set(params["sourceKinds"])
    assert params["archived"] is True
    assert params["searchTerm"] == "example"


def test_resume_preserves_existing_thread_and_permissions(chat):
    client, rpc = chat
    assert (
        client.post("/chat/api/resume", json={"id": "existing-thread"}).status_code
        == 200
    )
    rpc.call.assert_called_with(
        "thread/resume", {"threadId": "existing-thread", "excludeTurns": True}
    )


def test_followup_steers_exact_active_turn(chat):
    client, rpc = chat
    response = client.post(
        "/chat/api/send",
        json={"id": "thread", "text": "Focus on tests", "turnId": "active-turn"},
    )
    assert response.status_code == 200
    rpc.call.assert_called_with(
        "turn/steer",
        {
            "threadId": "thread",
            "expectedTurnId": "active-turn",
            "input": [{"type": "text", "text": "Focus on tests"}],
        },
    )


def test_stop_targets_one_turn_without_stopping_runtime(chat):
    client, rpc = chat
    assert (
        client.post(
            "/chat/api/stop", json={"id": "thread", "turnId": "turn"}
        ).status_code
        == 200
    )
    rpc.call.assert_called_once_with(
        "turn/interrupt", {"threadId": "thread", "turnId": "turn"}
    )


@pytest.mark.parametrize(
    "path", ["/chat/api/config/write", "/chat/api/command/exec", "/chat/api/rpc"]
)
def test_arbitrary_codex_methods_are_not_exposed(chat, path):
    client, rpc = chat
    assert client.post(path, json={"method": "account/read"}).status_code == 400
    rpc.call.assert_not_called()


def test_resolved_request_cannot_be_approved_again(chat):
    client, rpc = chat
    response = client.post(
        "/chat/api/answer", json={"requestId": 7, "decision": "accept"}
    )
    assert response.status_code == 400
    rpc.answer.assert_not_called()


def test_permission_approval_only_grants_requested_permissions():
    request = {
        "method": "item/permissions/requestApproval",
        "params": {
            "permissions": {"network": {"enabled": True}},
        },
    }
    result = _answer_result(
        request, {"decision": "accept", "permissions": {"filesystem": "all"}}
    )
    assert result == {"scope": "turn", "permissions": {"network": {"enabled": True}}}
    assert _answer_result(request, {"decision": "decline"})["permissions"] == {}


def test_question_answers_must_match_pending_questions():
    request = {
        "method": "item/tool/requestUserInput",
        "params": {"questions": [{"id": "choice"}]},
    }
    with pytest.raises(ValueError, match="every question"):
        _answer_result(request, {"answers": {"unrelated": "yes"}})
    assert _answer_result(request, {"answers": {"choice": "yes"}}) == {
        "answers": {"choice": {"answers": ["yes"]}}
    }


def test_invalid_json_shape_is_rejected(chat):
    client, rpc = chat
    assert client.post("/chat/api/send", json=[]).status_code == 400
    rpc.call.assert_not_called()


def test_legacy_client_cannot_create_a_second_writer(tmp_path):
    rollout = tmp_path / "rollout.jsonl"
    rollout.write_text("")
    process = tmp_path / "proc/123"
    (process / "fd").mkdir(parents=True)
    (process / "comm").write_text("codex\n")
    (process / "cmdline").write_bytes(b"codex\0app-server\0")
    (process / "fd/7").symlink_to(rollout)
    thread = {"path": str(rollout), "status": {"type": "notLoaded"}}
    assert owned_elsewhere(thread, "/private/shared.sock", tmp_path / "proc")
    (process / "cmdline").write_bytes(
        b"codex\0app-server\0--listen\0unix:///private/shared.sock\0"
    )
    assert not owned_elsewhere(thread, "/private/shared.sock", tmp_path / "proc")


def test_new_thread_works_before_codex_materializes_its_history(chat):
    client, rpc = chat

    def upstream(method, params):
        if method == "thread/start":
            return {"thread": {"id": "new-thread", "status": {"type": "idle"}}}
        if method in {"thread/read", "thread/resume", "thread/turns/list"}:
            raise AssertionError("New thread has no persisted history yet")
        return {"turn": {"id": "first-turn", "status": "inProgress"}}

    rpc.call.side_effect = upstream
    assert client.post("/chat/api/new", json={}).status_code == 200
    assert client.post("/chat/api/resume", json={"id": "new-thread"}).status_code == 200
    assert client.get("/chat/api/turns?id=new-thread").json()["data"] == []
    assert (
        client.post(
            "/chat/api/send", json={"id": "new-thread", "text": "hello"}
        ).status_code
        == 200
    )
