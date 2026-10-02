"""Verify local installation boundaries, gateway preservation, and durable sends."""

import json
import base64
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import subprocess
import threading
from unittest.mock import Mock
import uuid

import pytest

from npa.tools.desktop import gateway_remote, local_runtime, local_service
from npa.tools.desktop.chat_delivery import Deliveries
from npa.tools.desktop.chat_history import owned_elsewhere
from npa.tools.desktop.chat_session import mobile_session


def test_local_dry_run_never_connects_or_writes(monkeypatch, tmp_path):
    monkeypatch.setattr(local_runtime.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(local_runtime, "_root", lambda: tmp_path / "absent")
    monkeypatch.setattr(
        local_runtime, "gateway_operation", Mock(side_effect=AssertionError)
    )
    result = local_runtime.setup_local(
        gateway_host="gateway.example.test", dry_run=True
    )
    assert result["runtime"] == "local"
    assert not (tmp_path / "absent").exists()


def test_local_setup_rejects_other_platform_before_writing(monkeypatch):
    monkeypatch.setattr(local_runtime.platform, "system", lambda: "Linux")
    with pytest.raises(ValueError, match="macOS"):
        local_runtime.setup_local(dry_run=True)


def test_reverse_tunnel_is_outbound_and_loopback_only():
    command = local_service.tunnel_command("gateway.example.test", 7001, 7002)
    assert command[command.index("-R") + 1] == "127.0.0.1:7002:127.0.0.1:7001"
    assert "ExitOnForwardFailure=yes" in command
    assert command[-2:] == ["--", "gateway.example.test"]
    assert "-L" not in command


def test_repeated_local_setup_preserves_credentials(monkeypatch, tmp_path):
    config = {
        "gateway_host": "example.test",
        "username": "existing",
        "password_file": "private",
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(local_runtime, "_root", lambda: tmp_path)
    monkeypatch.setattr(
        local_runtime, "gateway_operation", Mock(side_effect=AssertionError)
    )
    assert local_runtime._configuration("example.test", 7001, 7002) == config
    with pytest.raises(ValueError, match="another gateway"):
        local_runtime._configuration("different.example.test", 7001, 7002)


@pytest.mark.parametrize("suffix", ["", " {\n flush_interval -1\n}"])
def test_existing_mobile_route_is_preserved(monkeypatch, suffix):
    original = "example.test {\n reverse_proxy 127.0.0.1:8787" + suffix + "\n}\n"
    monkeypatch.setattr(gateway_remote, "_read_root", lambda _: original)
    monkeypatch.setattr(gateway_remote, "_install", Mock())
    monkeypatch.setattr(gateway_remote.subprocess, "run", Mock())
    replace = Mock()
    monkeypatch.setattr(gateway_remote, "_replace_and_reload", replace)
    gateway_remote._configure_mobile(7002)
    candidate = replace.call_args.args[2]
    assert "handle /chat/*" in candidate
    assert "reverse_proxy 127.0.0.1:7002" in candidate
    assert "handle {\nreverse_proxy 127.0.0.1:8787" + suffix + "\n}" in candidate
    assert "@npa_chat_entry path / /index.html /chat" in candidate
    assert "redir * /chat/?{query} 302" in candidate
    assert "handle /legacy/ {\nrewrite * /\n" in candidate


def test_existing_mobile_bookmark_routes_upgrade_once(monkeypatch):
    original = (
        "example.test {\n# npa-local-chat\nhandle /chat/* {\n"
        "reverse_proxy 127.0.0.1:7002\n}\n"
        "handle {\nreverse_proxy 127.0.0.1:8787\n}\n}\n"
    )
    monkeypatch.setattr(gateway_remote, "_read_root", lambda _: original)
    monkeypatch.setattr(gateway_remote, "_install", Mock())
    monkeypatch.setattr(gateway_remote, "_ensure_restart", Mock())
    run = Mock()
    monkeypatch.setattr(gateway_remote.subprocess, "run", run)
    replace = Mock()
    monkeypatch.setattr(gateway_remote, "_replace_and_reload", replace)
    gateway_remote._configure_mobile(7002)
    candidate = replace.call_args.args[2]
    assert candidate.count("# npa-local-chat-entry") == 1
    assert "handle /chat/* {\nreverse_proxy 127.0.0.1:7002\n}" in candidate
    assert "validate" in run.call_args.args[0]
    monkeypatch.setattr(gateway_remote, "_read_root", lambda _: candidate)
    gateway_remote._configure_mobile(7002)
    assert replace.call_count == 1
    assert run.call_count == 1
    with pytest.raises(RuntimeError, match="another port"):
        gateway_remote._configure_mobile(7003)


def test_linux_chat_stays_on_its_existing_route(monkeypatch):
    original = "location = /desktop-credentials.json {}\nlocation /chat/ { proxy_pass http://127.0.0.1:6090; }"
    monkeypatch.setattr(gateway_remote, "_read_root", lambda _: original)
    monkeypatch.setattr(gateway_remote, "_install", Mock())
    monkeypatch.setattr(gateway_remote.subprocess, "run", Mock())
    replace = Mock()
    monkeypatch.setattr(gateway_remote, "_replace_and_reload", replace)
    gateway_remote._configure_desktop(7002)
    candidate = replace.call_args.args[2]
    assert "location /local-chat/" in candidate
    assert "http://127.0.0.1:7002/chat/" in candidate
    assert "location /chat/ { proxy_pass http://127.0.0.1:6090; }" in candidate


def test_gateway_reload_failure_restores_original(monkeypatch):
    install = Mock()
    monkeypatch.setattr(gateway_remote, "_install", install)
    run = Mock(side_effect=[subprocess.CalledProcessError(1, "reload"), None])
    monkeypatch.setattr(gateway_remote.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        gateway_remote._replace_and_reload(
            "/private/config", "original", "new", "gateway"
        )
    assert install.call_args.args == ("/private/config", "original")
    assert run.call_count == 2


def test_delivery_survives_service_restart_without_replaying(tmp_path):
    path = tmp_path / "deliveries.sqlite"
    identifier = str(uuid.uuid4())
    action = Mock(return_value={"accepted": True})
    assert Deliveries(path).execute(identifier, {"text": "hello"}, action) == {
        "accepted": True
    }
    assert Deliveries(path).execute(identifier, {"text": "hello"}, action) == {
        "accepted": True
    }
    assert action.call_count == 1
    assert path.stat().st_mode & 0o777 == 0o600


def test_uncertain_send_is_never_replayed_after_restart(tmp_path):
    path = tmp_path / "deliveries.sqlite"
    identifier = str(uuid.uuid4())
    action = Mock(side_effect=ConnectionError("response lost"))
    with pytest.raises(ConnectionError):
        Deliveries(path).execute(identifier, {"text": "hello"}, action)
    with pytest.raises(RuntimeError, match="uncertain"):
        Deliveries(path).execute(identifier, {"text": "hello"}, action)
    assert action.call_count == 1
    with pytest.raises(ValueError, match="another message"):
        Deliveries(path).execute(identifier, {"text": "different"}, action)


def test_missing_process_filesystem_fails_closed(tmp_path):
    with pytest.raises(OSError):
        owned_elsewhere(
            {"status": {"type": "notLoaded"}, "path": str(tmp_path / "rollout")},
            "test.sock",
            tmp_path / "missing-proc",
        )


def test_local_status_does_not_include_password(monkeypatch, tmp_path):
    monkeypatch.setattr(local_runtime, "_root", lambda: tmp_path)
    monkeypatch.setattr(local_runtime, "service_running", lambda _: True)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "url": "https://example.test/chat/",
                "username": "example",
                "password": "not-for-output",
            }
        )
    )
    status = local_runtime.local_status()
    assert "not-for-output" not in json.dumps(status)
    assert status["service_running"] is True


def test_existing_gateway_cookie_requires_signature_and_expiry(monkeypatch):
    from npa.tools.desktop import chat_session

    monkeypatch.setattr(chat_session.time, "time", lambda: 100)
    secret = "test-session-key"
    payload = (
        base64.urlsafe_b64encode(json.dumps({"expires": 200000}).encode())
        .decode()
        .rstrip("=")
    )
    signature = (
        base64.urlsafe_b64encode(
            hmac.new(secret.encode(), payload.encode(), hashlib.sha256).digest()
        )
        .decode()
        .rstrip("=")
    )
    cookie = f"codex_session={payload}.{signature}"
    assert mobile_session(cookie, secret)
    assert not mobile_session(cookie, "wrong-key")
    assert not mobile_session(cookie + ".extra", secret)
    monkeypatch.setattr(chat_session.time, "time", lambda: 300)
    assert not mobile_session(cookie, secret)


def test_active_mobile_turn_blocks_runtime_upgrade(monkeypatch, tmp_path):
    monkeypatch.setattr(local_runtime, "service_running", lambda _: True)
    monkeypatch.setattr(
        local_runtime,
        "_local_state",
        lambda *args: {"runtime": {"ownedActive": True}},
    )
    password = tmp_path / "password"
    password.write_text("unit-test-password")
    config = {
        "runtime_path": "old",
        "username": "example",
        "password_file": str(password),
        "port": 7001,
    }
    with pytest.raises(RuntimeError, match="finish"):
        local_runtime._check_upgrade(config, tmp_path / "new")


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--local", "--ssh-host", "example.test"],
        ["--ssh-host", "example.test", "--gateway-ssh-host", "gateway.example.test"],
        ["--local", "--connect-vscode"],
    ],
)
def test_local_cli_rejects_ambiguous_targets(arguments):
    from typer.testing import CliRunner
    from npa.cli.tools import app

    result = CliRunner().invoke(app, ["desktop", "chat-setup", *arguments, "--dry-run"])
    assert result.exit_code != 0


def test_adopting_gateway_does_not_replace_running_server_password(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(local_runtime, "_root", lambda: tmp_path)
    monkeypatch.setattr(local_runtime, "_check_upgrade", Mock())
    monkeypatch.setattr(
        local_runtime,
        "gateway_operation",
        lambda *args: {
            "origin": "https://example.test",
            "username": "new-user",
            "password": "new-test-password",
            "kind": "desktop",
        },
    )
    password = tmp_path / "password"
    password.write_text("original-test-password")
    config = local_runtime._adopt_gateway(
        {"password_file": str(password)}, "gateway.example.test"
    )
    assert password.read_text() == "original-test-password"
    assert config["password_file"] != str(password)
    assert config["url"].endswith("/local-chat/")


@pytest.mark.parametrize("address", ["0.0.0.0:7002", "*:7002", "[::]:7002"])
def test_gateway_rejects_public_tunnel_listener(monkeypatch, address):
    monkeypatch.setattr(
        gateway_remote,
        "_local_state",
        lambda *args: {"installationId": "test-installation"},
    )
    monkeypatch.setattr(
        gateway_remote.subprocess,
        "check_output",
        lambda *args, **kwargs: f"LISTEN 0 128 {address} 0.0.0.0:*\n",
    )
    with pytest.raises(RuntimeError, match="loopback"):
        gateway_remote._verify_tunnel(
            {
                "port": 7002,
                "username": "test",
                "password": "unit-test-password",
                "installation_id": "test-installation",
            }
        )


def test_gateway_rejects_another_installation_before_routing(monkeypatch):
    monkeypatch.setattr(
        gateway_remote,
        "_local_state",
        lambda *args: {"installationId": "different-installation"},
    )
    with pytest.raises(RuntimeError, match="another installation"):
        gateway_remote._verify_tunnel(
            {
                "port": 7002,
                "username": "test",
                "password": "unit-test-password",
                "installation_id": "test-installation",
            }
        )


def test_launchd_bootstrap_retries_pending_removal(monkeypatch, tmp_path):
    failed = subprocess.CompletedProcess(["launchctl"], 5, b"", b"removal pending")
    success = subprocess.CompletedProcess(["launchctl"], 0, b"", b"")
    run = Mock(side_effect=[failed, success])
    monkeypatch.setattr(local_service.subprocess, "run", run)
    monkeypatch.setattr(local_service.time, "sleep", Mock())
    local_service._bootstrap(tmp_path / "service.plist")
    assert run.call_count == 2


def test_launchd_bootstrap_does_not_retry_permission_failure(monkeypatch, tmp_path):
    run = Mock(
        return_value=subprocess.CompletedProcess(["launchctl"], 1, b"", b"not allowed")
    )
    monkeypatch.setattr(local_service.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        local_service._bootstrap(tmp_path / "service.plist")
    assert run.call_count == 1


def test_gateway_failure_removes_unverified_tunnel(monkeypatch, tmp_path):
    password = tmp_path / "password"
    password.write_text("unit-test-password")
    monkeypatch.setattr(local_runtime, "install_service", Mock())
    remove = Mock()
    monkeypatch.setattr(local_runtime, "remove_service", remove)
    monkeypatch.setattr(
        local_runtime,
        "gateway_operation",
        Mock(side_effect=RuntimeError("verification failed")),
    )
    config = {
        "gateway_host": "gateway.example.test",
        "port": 7001,
        "gateway_port": 7002,
        "origin": "https://example.test",
        "installation_id": "unit-installation",
        "username": "unit-user",
        "password_file": str(password),
    }
    with pytest.raises(RuntimeError, match="verification failed"):
        local_runtime._connect_gateway(config, tmp_path)
    remove.assert_called_once_with("com.nebius.codex-chat.tunnel")


def test_unverified_tunnel_does_not_restart_after_login(monkeypatch, tmp_path):
    monkeypatch.setattr(local_service.Path, "home", lambda: tmp_path)
    path = tmp_path / "Library/LaunchAgents/com.nebius.codex-chat.tunnel.plist"
    path.parent.mkdir(parents=True)
    path.write_text("owned service")
    run = Mock(return_value=subprocess.CompletedProcess(["launchctl"], 0))
    monkeypatch.setattr(local_service.subprocess, "run", run)
    local_service.remove_service("com.nebius.codex-chat.tunnel")
    assert not path.exists()
    assert run.call_args.args[0][1] == "bootout"


def test_desktop_gateway_origin_excludes_viewer_path(monkeypatch, tmp_path):
    monkeypatch.setattr(gateway_remote.Path, "home", lambda: tmp_path)
    root = tmp_path / ".local/share/nebius-desktop"
    root.mkdir(parents=True)
    password = root / "public-password"
    password.write_text("unit-test-password")
    (root / "public-access.json").write_text(
        json.dumps(
            {
                "url": "https://example.test:8443/desktop.html",
                "username": "developer",
                "password_file": str(password),
            }
        )
    )
    info = gateway_remote._inspect()
    assert info["origin"] == "https://example.test:8443"
    assert info["kind"] == "desktop"


@pytest.mark.parametrize(
    "url", ["http://example.test/", "https://user:password@example.test/"]
)
def test_managed_gateway_rejects_plaintext_or_embedded_login(url):
    with pytest.raises(ValueError, match="HTTPS"):
        gateway_remote._origin(url)


@pytest.fixture
def state_probe_server():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append((self.path, self.headers.get("Authorization")))
            status = self.server.probe_status if self.path == "/chat/api/state" else 200
            self.send_response(status)
            self.send_header(
                "Location", f"http://127.0.0.1:{self.server.server_port}/redirected"
            )
            self.end_headers()
            self.wfile.write(b'{"installationId":"test-installation"}')

        def log_message(self, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield server, requests
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


@pytest.mark.parametrize("status", [200, 302, 307, 401, 500])
def test_state_probe_keeps_credentials_on_loopback(
    monkeypatch, state_probe_server, status
):
    server, requests = state_probe_server
    server.probe_status = status
    for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:1")
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(name, "")
    if status == 200:
        assert gateway_remote._local_state(
            server.server_port, "test", "test-password"
        ) == {"installationId": "test-installation"}
    else:
        with pytest.raises(RuntimeError, match="authenticate"):
            gateway_remote._local_state(server.server_port, "test", "test-password")
    expected = "Basic " + base64.b64encode(b"test:test-password").decode()
    assert requests == [("/chat/api/state", expected)]


@pytest.mark.parametrize("port", [True, 80, 65536, "7001/redirect"])
def test_state_probe_rejects_invalid_port_before_connecting(monkeypatch, port):
    connection = Mock(side_effect=AssertionError("must not connect"))
    monkeypatch.setattr(gateway_remote, "HTTPConnection", connection)
    with pytest.raises(ValueError, match="port"):
        gateway_remote._local_state(port, "test", "test-password")
    connection.assert_not_called()
