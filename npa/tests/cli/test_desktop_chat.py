"""Exercise authenticated chat boundaries and shared Codex controls over real HTTP."""

from collections import deque
from http.server import ThreadingHTTPServer
import threading
import uuid
from unittest.mock import Mock

import httpx
import pytest

from npa.tools.desktop.chat_server import (
    ChatHandler,
    _answer_result,
    _update_created_threads,
)
from npa.tools.desktop.chat_rpc import CodexConnection
from npa.tools.desktop.chat_history import owned_elsewhere
from npa.tools.desktop.chat_models import available_models, model_selection
from npa.tools.desktop.chat_delivery import Deliveries

_MODEL = {
    "model": "example-model",
    "displayName": "Example model",
    "defaultReasoningEffort": "medium",
    "supportedReasoningEfforts": [
        {"reasoningEffort": "low", "description": "Quick"},
        {"reasoningEffort": "medium", "description": "Balanced"},
    ],
}


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
    server.deliveries = Deliveries(tmp_path / "deliveries.sqlite")
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
        "/chat/api/models",
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


def test_send_identity_is_forwarded_and_replayed_without_a_second_turn(chat):
    client, rpc = chat
    body = {
        "id": "existing-thread",
        "text": "one send",
        "clientUserMessageId": str(uuid.uuid4()),
    }
    assert client.post("/chat/api/send", json=body).status_code == 200
    assert client.post("/chat/api/send", json=body).status_code == 200
    sends = [call for call in rpc.call.call_args_list if call.args[0] == "turn/start"]
    assert len(sends) == 1
    assert sends[0].args[1]["clientUserMessageId"] == body["clientUserMessageId"]


def test_image_only_prompt_preserves_image_input(chat):
    client, rpc = chat
    image = "data:image/png;base64,aGVsbG8="
    response = client.post(
        "/chat/api/send", json={"id": "existing-thread", "images": [image]}
    )
    assert response.status_code == 200
    assert rpc.call.call_args.args[1]["input"][-1] == {"type": "image", "url": image}


@pytest.mark.parametrize(
    "image",
    ["https://example.test/private.png", "data:image/svg+xml;base64,PHN2Zz4=", {}],
)
def test_images_cannot_fetch_remote_resources_or_execute_svg(chat, image):
    client, rpc = chat
    response = client.post(
        "/chat/api/send",
        json={"id": "existing-thread", "text": "image", "images": [image]},
    )
    assert response.status_code == 400
    assert all(call.args[0] != "turn/start" for call in rpc.call.call_args_list)


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
    assert '<script src="./chat.js" defer>' in response.text
    assert "unit-test-only-password" not in response.text


def test_list_includes_all_providers_and_source_kinds(chat):
    client, rpc = chat
    rpc.call.side_effect = None
    rpc.call.return_value = {"data": [], "nextCursor": None}
    assert (
        client.get("/chat/api/threads?search=example&archived=true").status_code == 200
    )
    method, params = rpc.call.call_args.args
    assert method == "thread/list"
    assert params["modelProviders"] == []
    assert {"vscode", "cli", "appServer", "exec"} <= set(params["sourceKinds"])
    assert params["archived"] is True
    assert params["searchTerm"] == "example"


def test_session_list_bounds_labels_without_changing_conversation_history(chat):
    client, rpc = chat
    original = {
        "id": "existing-thread",
        "name": "Long prompt " * 4000,
        "preview": "Detailed context " * 4000,
        "cwd": "/workspace/project",
        "status": {"type": "active"},
        "updatedAt": 100,
    }
    rpc.call.side_effect = lambda method, params: (
        {"data": [original], "nextCursor": "next-page"}
        if method == "thread/list"
        else {"thread": original}
    )
    response = client.get("/chat/api/threads?cursor=previous-page")
    assert response.status_code == 200
    page = response.json()
    entry = page["data"][0]
    assert page["nextCursor"] == "next-page"
    assert rpc.call.call_args.args[1]["cursor"] == "previous-page"
    for field in ("name", "preview"):
        assert len(entry[field]) == 240
        assert entry[field].endswith("…")
    for field in ("id", "cwd", "status", "updatedAt"):
        assert entry[field] == original[field]
    assert len(response.content) < 2000
    assert client.get("/chat/api/thread?id=existing-thread").json()["thread"] == {
        **original,
        "archived": False,
    }
    assert len(original["name"]) > 240


def test_session_list_preserves_short_and_missing_labels(chat):
    client, rpc = chat
    threads = [{"id": "one", "name": None, "preview": "Short preview"}, {"id": "two"}]
    rpc.call.side_effect = None
    rpc.call.return_value = {"data": threads, "nextCursor": None}
    assert client.get("/chat/api/threads").json()["data"] == [
        {**thread, "archived": False} for thread in threads
    ]


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


def test_model_catalog_reads_all_pages_without_hardcoded_model_names():
    rpc = Mock()
    rpc.call.side_effect = [
        {"data": [_MODEL], "nextCursor": "next-page"},
        {"data": [{**_MODEL, "model": "another-model"}], "nextCursor": None},
    ]
    assert [model["model"] for model in available_models(rpc)] == [
        "example-model",
        "another-model",
    ]
    rpc.call.assert_called_with(
        "model/list", {"includeHidden": False, "cursor": "next-page"}
    )


@pytest.mark.parametrize(
    "body, message",
    [
        ({"model": "unavailable"}, "available"),
        ({"model": "example-model", "effort": "unsupported"}, "effort"),
        ({"model": "example-model", "approvalPolicy": "never"}, "Only model"),
        ({"model": "example-model", "cwd": "/different/project"}, "Only model"),
    ],
)
def test_model_selection_rejects_unsupported_or_unrelated_overrides(body, message):
    rpc = Mock()
    rpc.call.return_value = {"data": [_MODEL]}
    with pytest.raises(ValueError, match=message):
        model_selection(rpc, {"id": "thread", **body}, {})


def test_model_change_uses_supported_default_when_previous_effort_is_incompatible():
    rpc = Mock()
    rpc.call.return_value = {"data": [_MODEL]}
    assert model_selection(
        rpc, {"id": "thread", "model": "example-model"}, {"reasoningEffort": "ultra"}
    ) == {"threadId": "thread", "model": "example-model", "effort": "medium"}


def test_settings_change_updates_the_same_thread_without_starting_a_turn(chat):
    client, rpc = chat
    thread = {"id": "thread", "status": {"type": "idle"}, "model": "example-model"}

    def upstream(method, params):
        if method == "model/list":
            return {"data": [_MODEL]}
        if method in {"thread/read", "thread/resume"}:
            return {"thread": thread}
        return {}

    rpc.call.side_effect = upstream
    response = client.post(
        "/chat/api/settings",
        json={"id": "thread", "model": "example-model", "effort": "low"},
    )
    assert response.status_code == 200
    rpc.call.assert_called_with(
        "thread/settings/update",
        {"threadId": "thread", "model": "example-model", "effort": "low"},
    )
    assert not {"turn/start", "thread/start", "thread/fork"} & {
        call.args[0] for call in rpc.call.call_args_list
    }


def test_settings_change_rejects_a_foreign_origin(chat):
    client, rpc = chat
    response = client.post(
        "/chat/api/settings",
        json={"id": "thread", "model": "example-model"},
        headers={"Origin": "https://hostile.example.test"},
    )
    assert response.status_code == 403
    rpc.call.assert_not_called()


def test_native_first_turn_invalidates_unmaterialized_mobile_history():
    created = {"new-thread": {"id": "new-thread", "model": "old-model"}}
    rpc = object.__new__(CodexConnection)
    rpc.condition = threading.Condition()
    rpc.events = deque()
    rpc.sequence = 0
    rpc.on_notification = lambda message: _update_created_threads(created, message)
    rpc._receive(
        {
            "method": "thread/settings/updated",
            "params": {
                "threadId": "new-thread",
                "threadSettings": {
                    "model": "updated-model",
                    "effort": "low",
                },
            },
        }
    )
    assert created["new-thread"]["model"] == "updated-model"
    assert created["new-thread"]["reasoningEffort"] == "low"
    rpc._receive(
        {
            "method": "turn/started",
            "params": {"threadId": "new-thread", "turn": {"id": "native-turn"}},
        }
    )
    assert "new-thread" not in created
    assert rpc.sequence == 2
    assert rpc.events[-1][1]["method"] == "turn/started"


@pytest.mark.parametrize(
    "name", ["", "  ", "x" * 201, "two\nlines", "bad\x00name", None, []]
)
def test_rename_rejects_invalid_names_without_mutating(chat, name):
    client, rpc = chat
    response = client.post(
        "/chat/api/rename", json={"id": "existing-thread", "name": name}
    )
    assert response.status_code == 400
    rpc.call.assert_not_called()


def test_rename_preserves_identity_and_does_not_resume(chat):
    client, rpc = chat
    response = client.post(
        "/chat/api/rename", json={"id": "existing-thread", "name": "  New name  "}
    )
    assert response.status_code == 200
    assert [call.args for call in rpc.call.call_args_list] == [
        ("thread/read", {"threadId": "existing-thread", "includeTurns": False}),
        ("thread/name/set", {"threadId": "existing-thread", "name": "New name"}),
    ]


@pytest.mark.parametrize("action", ["rename", "archive", "unarchive"])
def test_management_requires_authentication_and_same_origin(chat, action):
    client, rpc = chat
    body = {"id": "existing-thread", "name": "New name"}
    assert client.post("/chat/api/" + action, json=body, auth=None).status_code == 401
    assert (
        client.post(
            "/chat/api/" + action,
            json=body,
            headers={"Origin": "https://foreign.example.test"},
        ).status_code
        == 403
    )
    rpc.call.assert_not_called()


def test_archive_refuses_a_running_turn(chat):
    client, rpc = chat
    rpc.call.side_effect = lambda method, params: {
        "thread": {"id": "existing-thread", "status": {"type": "active"}}
    }
    response = client.post("/chat/api/archive", json={"id": "existing-thread"})
    assert response.status_code == 400
    assert "finish" in response.json()["error"]
    assert rpc.call.call_count == 1


@pytest.mark.parametrize("action", ["archive", "unarchive"])
def test_archival_routes_only_the_requested_thread(chat, action):
    client, rpc = chat
    assert (
        client.post("/chat/api/" + action, json={"id": "existing-thread"}).status_code
        == 200
    )
    rpc.call.assert_called_with("thread/" + action, {"threadId": "existing-thread"})
    assert all(call.args[0] != "thread/resume" for call in rpc.call.call_args_list)


@pytest.mark.parametrize("action", ["send", "settings", "rename", "resume"])
def test_archived_chats_remain_read_only_until_restored(chat, action):
    client, rpc = chat
    rpc.call.side_effect = lambda method, params: {
        "thread": {
            "id": "existing-thread",
            "path": "/workspace/.codex/archived_sessions/saved.jsonl",
            "status": {"type": "notLoaded"},
        }
    }
    response = client.post(
        "/chat/api/" + action,
        json={"id": "existing-thread", "name": "Name", "text": "Hello"},
    )
    assert response.status_code == (200 if action == "resume" else 400)
    assert rpc.call.call_count == 1
    if action == "resume":
        assert response.json()["thread"]["archived"] is True
