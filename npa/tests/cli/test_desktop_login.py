"""Exercise persistent browser login and public install assets over real HTTP."""

import base64
import json
from http.server import ThreadingHTTPServer
from pathlib import Path
import secrets
import threading
from unittest.mock import Mock

import httpx
import pytest

from npa.tools.desktop import chat_setup, remote
from npa.tools.desktop.chat_auth import LoginHandler
from npa.tools.desktop.chat_server import ChatHandler
from npa.tools.desktop.chat_session import mobile_session, session_cookie, session_key


@pytest.fixture
def login(tmp_path):
    password = tmp_path / "password"
    password.write_text("unit-test-only-password")
    server = ThreadingHTTPServer(("127.0.0.1", 0), LoginHandler)
    server.config = {
        "origin": "https://desktop.example.test",
        "username": "developer",
        "password_file": str(password),
        "session_secret": secrets.token_urlsafe(32),
        "auth_port": 6092,
    }
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    with httpx.Client(
        base_url=f"http://127.0.0.1:{server.server_port}",
        trust_env=False,
        headers={"Origin": server.config["origin"]},
    ) as client:
        yield client, server.config
    server.shutdown()
    server.server_close()
    worker.join()


def _signin(client, **values):
    return client.post(
        "/chat/login",
        data={
            "username": "developer",
            "password": "unit-test-only-password",
            "next": "/chat/",
            **values,
        },
    )


def test_one_login_authorizes_desktop_and_chat_and_revokes_on_password_change(login):
    client, config = login
    assert client.get("/auth/check").status_code == 401
    response = _signin(client)
    assert response.status_code == 303
    assert response.headers["Location"] == "/chat/"
    cookie = response.headers["Set-Cookie"]
    assert all(
        flag in cookie
        for flag in ("Secure", "HttpOnly", "SameSite=Lax", "Max-Age=2592000", "Path=/")
    )
    assert "unit-test-only-password" not in cookie
    # An explicit header models the HTTPS gateway forwarding its secure cookie.
    client.headers["Cookie"] = cookie.split(";", 1)[0]
    assert client.get("/auth/check").status_code == 204
    assert mobile_session(cookie, session_key(config, "unit-test-only-password"))
    Path(config["password_file"]).write_text("changed-test-password")
    assert client.get("/auth/check").status_code == 401


@pytest.mark.parametrize("origin", ["https://foreign.example.test", "null", None])
def test_login_rejects_cross_origin_or_missing_origin(login, origin):
    client, _ = login
    if origin is None:
        del client.headers["Origin"]
    else:
        client.headers["Origin"] = origin
    response = _signin(client)
    assert response.status_code == 403
    assert "Set-Cookie" not in response.headers


def test_wrong_login_is_retriable_without_basic_auth_popup(login):
    client, _ = login
    response = _signin(client, password="wrong")
    assert response.status_code == 401
    assert "Set-Cookie" not in response.headers
    assert "WWW-Authenticate" not in response.headers
    assert _signin(client).status_code == 303


@pytest.mark.parametrize(
    "destination",
    [
        "https://foreign.example.test",
        "//foreign.example.test",
        "/\\foreign.example.test",
        "/chat/\r\nInjected: yes",
        "/chat/login",
        "/%2f%2fforeign.example.test",
        "/private",
    ],
)
def test_login_return_destination_cannot_escape_known_pages(login, destination):
    client, _ = login
    assert _signin(client, next=destination).headers["Location"] == "/chat/"


def test_explicit_desktop_choice_and_chat_deep_link_survive_login(login):
    client, _ = login
    for destination in ("/desktop.html?desktop=1", "/chat/#session-id"):
        assert _signin(client, next=destination).headers["Location"] == destination


def test_basic_clients_remain_supported(login):
    client, _ = login
    auth = ("developer", "unit-test-only-password")
    assert client.get("/auth/check", auth=auth).status_code == 204
    assert client.get("/auth/check", auth=("developer", "wrong")).status_code == 401


def test_installer_can_read_manifest_and_png_icons_without_private_content(login):
    client, _ = login
    manifest = client.get("/chat/manifest.webmanifest").json()
    assert manifest["display"] == "standalone"
    assert manifest["start_url"] == "./"
    assert manifest["scope"] == "./"
    for size in (180, 192, 512):
        response = client.get(f"/chat/icon-{size}.png")
        assert response.headers["Content-Type"] == "image/png"
        assert response.content.startswith(b"\x89PNG\r\n\x1a\n")
        assert int.from_bytes(response.content[16:20], "big") == size
    for path in ("/chat/api/threads", "/desktop-credentials.json", "/config.json"):
        assert client.get(path).status_code == 404


def test_login_page_uses_password_manager_and_standalone_metadata(login):
    client, _ = login
    response = client.get("/chat/login")
    assert 'autocomplete="username"' in response.text
    assert 'autocomplete="current-password"' in response.text
    assert 'rel="apple-touch-icon"' in response.text
    assert 'rel="manifest"' in response.text
    assert "no-store" in response.headers["Cache-Control"]
    assert "form-action 'self'" in response.headers["Content-Security-Policy"]


def test_tampered_expired_and_non_object_cookies_are_rejected(monkeypatch):
    from npa.tools.desktop import chat_session

    key = secrets.token_urlsafe(32)
    cookie = session_cookie(key)
    assert mobile_session(cookie, key)
    assert not mobile_session(cookie, "different-key")
    assert not mobile_session(cookie.replace("codex_session=", "codex_session=x"), key)
    current = chat_session.time.time()
    monkeypatch.setattr(chat_session.time, "time", lambda: current + 31 * 86400)
    assert not mobile_session(cookie, key)
    assert not mobile_session(
        "codex_session=" + base64.b64encode(b"[]").decode() + ".invalid", key
    )


def test_cloud_cookie_authenticates_chat_without_basic_headers(login):
    client, config = login
    cookie = _signin(client).headers["Set-Cookie"].split(";", 1)[0]
    server = ThreadingHTTPServer(("127.0.0.1", 0), ChatHandler)
    server.config = config
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/chat/"
        assert (
            httpx.get(url, headers={"Cookie": cookie}, trust_env=False).status_code
            == 200
        )
        unauthorized = httpx.get(url, trust_env=False)
        assert unauthorized.status_code == 401
        assert "WWW-Authenticate" not in unauthorized.headers
        manifest = httpx.get(url + "manifest.webmanifest", trust_env=False)
        assert manifest.status_code == 200
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def test_repeated_auth_install_preserves_key_and_never_restarts_codex(
    monkeypatch, tmp_path
):
    config = {"session_secret": secrets.token_urlsafe(32), "auth_port": 6092}
    original = dict(config)
    monkeypatch.setattr(chat_setup, "_ROOT", tmp_path / "runtime")
    monkeypatch.setattr(chat_setup, "_UNITS", tmp_path / "units")
    run = Mock()
    monkeypatch.setattr(chat_setup, "_run", run)
    monkeypatch.setattr(chat_setup, "_wait_for_auth", Mock())
    chat_setup._authentication(config)
    chat_setup._authentication(config)
    assert config == original
    assert json.loads((tmp_path / "runtime/config.json").read_text()) == original
    assert all("codex" not in str(call) for call in run.call_args_list)
    unit = (tmp_path / "units/npa-desktop-auth.service").read_text()
    assert "UMask=0077" in unit
    assert "app-server" not in unit


def test_auth_readiness_uses_local_listener_without_environment_proxy(
    login, monkeypatch
):
    client, _ = login
    for variable in ("http_proxy", "HTTP_PROXY"):
        monkeypatch.setenv(variable, "http://proxy.example.test:1")
    monkeypatch.setenv("no_proxy", "")
    monkeypatch.setenv("NO_PROXY", "")
    chat_setup._wait_for_auth({"auth_port": client.base_url.port})


@pytest.mark.parametrize("status", [302, 503])
def test_auth_readiness_rejects_redirects_and_unhealthy_status(
    login, monkeypatch, status
):
    client, _ = login
    paths = []

    def reply(handler):
        paths.append(handler.path)
        handler.send_response(status if handler.path == "/chat/login" else 200)
        handler.send_header("Location", "/unexpected-readiness-target")
        handler.end_headers()

    monkeypatch.setattr(LoginHandler, "do_GET", reply)
    monkeypatch.setattr(chat_setup.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="did not become ready"):
        chat_setup._wait_for_auth({"auth_port": client.base_url.port})
    assert paths == ["/chat/login"] * 30


def test_auth_readiness_recovers_after_startup_failure(login, monkeypatch):
    client, _ = login
    statuses = iter([503, 200])
    sleeps = []

    def reply(handler):
        handler.send_response(next(statuses))
        handler.end_headers()

    monkeypatch.setattr(LoginHandler, "do_GET", reply)
    monkeypatch.setattr(chat_setup.time, "sleep", sleeps.append)
    chat_setup._wait_for_auth({"auth_port": client.base_url.port})
    assert sleeps == [0.2]


def test_auth_readiness_closes_failed_connection(monkeypatch):
    connection = Mock()
    connection.request.side_effect = ConnectionRefusedError
    monkeypatch.setattr(chat_setup, "HTTPConnection", Mock(return_value=connection))
    assert not chat_setup._auth_ready(6092)
    connection.close.assert_called_once()


def test_gateway_enables_form_login_only_after_auth_service_is_ready(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(remote, "_STATE", tmp_path)
    config = {"public_ip": "203.0.113.10", "https_port": 8443}
    (tmp_path / "chat.json").write_text(json.dumps({"url": "private"}))
    assert 'auth_basic "Development desktop"' in remote._gateway_config(
        config, tls=True
    )
    (tmp_path / "chat.json").write_text(json.dumps({"browser_login": True}))
    gateway = remote._gateway_config(config, tls=True)
    assert "auth_request /_desktop_auth" in gateway
    assert "internal;" in gateway
    assert 'auth_basic "Development desktop"' not in gateway
    assert "limit_req zone=npa_login" in gateway
    assert "return 302 $npa_entry" in gateway
    assert "if ($npa_mobile_chat) { return 302 /chat/; }" in gateway
    assert '"1:" 1;' in gateway
