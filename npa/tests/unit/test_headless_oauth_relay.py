"""Verify the skill's private callback input and direct loopback delivery."""

from __future__ import annotations

import importlib.util
import io
import os
import pty
import select
import signal
import subprocess
import sys
import termios
import threading
import warnings
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import Mock
from urllib.parse import urlencode

import pytest


_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "skills/atomic/nebius-headless-oauth/scripts/relay_callback.py"
)
_SPEC = importlib.util.spec_from_file_location("headless_oauth_relay", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_RELAY = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RELAY)


def _authorization(redirect="http://127.0.0.1:54321", **overrides):
    fields = {
        "client_id": "nebius-cli",
        "response_type": "code",
        "code_challenge_method": "S256",
        "code_challenge": "synthetic-challenge",
        "state": "synthetic-state",
        "redirect_uri": redirect,
    }
    fields.update(overrides)
    return "https://auth.nebius.com/oauth2/authorize?" + urlencode(fields)


def _callback(port=54321, **overrides):
    fields = {"code": "synthetic-code", "state": "synthetic-state"}
    fields.update(overrides)
    return f"http://127.0.0.1:{port}/?" + urlencode(fields)


@pytest.fixture
def _private_input(monkeypatch):
    monkeypatch.setattr(_RELAY.sys, "argv", [str(_SCRIPT)])
    monkeypatch.setattr(_RELAY.sys, "stdin", Mock(isatty=lambda: True))
    monkeypatch.setattr(
        _RELAY.getpass, "getpass", Mock(side_effect=[_authorization(), _callback()])
    )
    delivery = Mock()
    monkeypatch.setattr(_RELAY, "_deliver", delivery)
    return delivery


@pytest.fixture
def _listener():
    requests = []
    settings = {"status": 200}

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            if settings.get("malformed"):
                self.wfile.write(b"synthetic-invalid-status\r\n\r\n")
                return
            self.send_response(settings["status"])
            self.send_header("Location", "/must-not-follow")
            self.end_headers()
            self.wfile.write(b"synthetic-secret-response")

        def log_message(self, *_arguments):
            pass

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port, requests, settings
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("trailing_slash", ["", "/"])
def test_callback_preserves_encoded_code_and_matches_state(trailing_slash):
    """Accept the CLI's root redirect with or without its trailing slash.

    Args:
        trailing_slash: Root redirect spelling.
    Returns:
        None.
    Raises:
        AssertionError: Callback parameters changed during validation.
    """
    authorization = _authorization("http://127.0.0.1:54321" + trailing_slash)
    callback = _callback(code="synthetic+code/with=encoding")
    assert _RELAY._callback_target(authorization, callback) == (
        54321,
        "/?" + callback.split("?", 1)[1],
    )


@pytest.mark.parametrize(
    "redirect",
    [
        "http://localhost:54321/",
        "http://[::1]:54321/",
        "https://127.0.0.1:54321/",
        "http://192.0.2.1:54321/",
        "http://127.0.0.1/",
        "http://127.0.0.1:0/",
        "http://127.0.0.1:65536/",
        "http://127.0.0.1:54321/other",
        "http://user@127.0.0.1:54321/",
        "http://127.0.0.1:54321/?extra=value",
        "http://127.0.0.1:54321/#fragment",
        "http://127.0.0.1:54321/?",
    ],
)
def test_rejects_unexpected_original_redirect(redirect):
    """Reject an original link whose listener cannot be pinned safely.

    Args:
        redirect: Unsupported redirect URL.
    Returns:
        None.
    Raises:
        AssertionError: An unsupported redirect was accepted.
    """
    with pytest.raises(ValueError):
        _RELAY._callback_target(_authorization(redirect), _callback())


@pytest.mark.parametrize(
    "authorization",
    [
        _authorization().replace("auth.nebius.com", "auth.nebius.com.invalid.example"),
        _authorization().replace("https:", "http:"),
        _authorization().replace("/oauth2/authorize", "/other"),
        _authorization() + "&state=duplicate",
        _authorization() + "#fragment",
        _authorization() + "\n",
        _authorization(state=""),
        _authorization(code_challenge_method="plain"),
        _authorization(response_type="token"),
        _authorization(code_challenge=""),
        _authorization(client_id="another-client"),
        _authorization(access_token="synthetic-token"),
    ],
)
def test_rejects_ambiguous_or_unexpected_authorization(authorization):
    """Require an unambiguous authorization-code request from the official host.

    Args:
        authorization: Unsupported authorization link.
    Returns:
        None.
    Raises:
        AssertionError: An unsupported link was accepted.
    """
    with pytest.raises(ValueError):
        _RELAY._callback_target(authorization, _callback())


@pytest.mark.parametrize(
    "callback",
    [
        _callback(54322),
        _callback(state="different-attempt"),
        _callback(code=""),
        _callback(state=""),
        _callback() + "&code=duplicate",
        _callback() + "&state=duplicate",
        _callback() + "#fragment",
        _callback() + "\r\n",
        _callback(error="access_denied"),
        _callback(access_token="synthetic-token"),
        _callback(next="https://invalid.example"),
        _callback().replace("127.0.0.1", "localhost"),
        _callback().replace("http:", "https:"),
        _callback().replace("/?", "/other?"),
        _callback().replace("127.0.0.1", "user@127.0.0.1"),
        "http://127.0.0.1:54321/?state=synthetic-state",
    ],
)
def test_invalid_callback_causes_no_connection(callback, monkeypatch):
    """Reject mismatches before opening a socket.

    Args:
        callback: Unsupported callback response.
        monkeypatch: Isolates connection creation.
    Returns:
        None.
    Raises:
        AssertionError: Invalid input reached a socket.
    """
    connection = Mock()
    monkeypatch.setattr(_RELAY.http.client, "HTTPConnection", connection)
    with pytest.raises(ValueError):
        _RELAY._deliver(_authorization(), callback)
    connection.assert_not_called()


@pytest.mark.parametrize("status", [200, 302, 400, 500])
def test_real_loopback_delivery_ignores_proxies_and_redirects(
    status, _listener, monkeypatch, capsys
):
    """Send exactly one local request and never disclose its response body.

    Args:
        status: Listener response status.
        _listener: Isolated loopback HTTP server.
        monkeypatch: Sets an unusable proxy.
        capsys: Captures public output.
    Returns:
        None.
    Raises:
        AssertionError: Delivery followed a redirect or exposed output.
    """
    port, requests, settings = _listener
    settings["status"] = status
    for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:1")
    authorization = _authorization(f"http://127.0.0.1:{port}")
    if status == 200:
        _RELAY._deliver(authorization, _callback(port))
    else:
        with pytest.raises(ValueError):
            _RELAY._deliver(authorization, _callback(port))
    assert requests == ["/?code=synthetic-code&state=synthetic-state"]
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize(
    "failure",
    [
        None,
        ValueError("synthetic-secret"),
        OSError("synthetic-secret"),
        _RELAY.http.client.HTTPException("synthetic-secret"),
        EOFError(),
        KeyboardInterrupt(),
        _RELAY.getpass.GetPassWarning("synthetic-secret"),
    ],
)
def test_private_prompt_reports_only_generic_status(failure, _private_input, capsys):
    """Hide URLs and exception text in success, failure, and cancellation output.

    Args:
        failure: Simulated delivery failure.
        _private_input: Hidden prompt and delivery fixtures.
        capsys: Captures public output.
    Returns:
        None.
    Raises:
        AssertionError: The helper disclosed private input.
    """
    _private_input.side_effect = failure
    assert _RELAY._main() == (0 if failure is None else 1)
    output = capsys.readouterr()
    assert "synthetic" not in output.out + output.err
    assert "http" not in output.out + output.err


@pytest.mark.parametrize("mode", ["arguments", "pipe", "echo-fallback"])
def test_refuses_input_that_could_expose_the_callback(
    mode, _private_input, monkeypatch
):
    """Reject argument, pipe, and getpass echo-fallback input paths.

    Args:
        mode: Unsafe input mechanism.
        _private_input: Hidden prompt and delivery fixtures.
        monkeypatch: Substitutes terminal properties.
    Returns:
        None.
    Raises:
        AssertionError: Unsafe input was used for delivery.
    """
    if mode == "arguments":
        monkeypatch.setattr(_RELAY.sys, "argv", [str(_SCRIPT), _callback()])
    elif mode == "pipe":
        monkeypatch.setattr(_RELAY.sys, "stdin", io.StringIO(_callback()))
    else:

        def _echo_fallback(_prompt):
            warnings.warn("synthetic-secret", _RELAY.getpass.GetPassWarning)

        monkeypatch.setattr(_RELAY.getpass, "getpass", _echo_fallback)
    assert _RELAY._main() == 1
    _private_input.assert_not_called()


@pytest.fixture
def _terminal(tmp_path):
    master, slave = pty.openpty()
    process = subprocess.Popen(
        [sys.executable, str(_SCRIPT)],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        cwd=tmp_path,
        env={"PATH": os.defpath, "PYTHONDONTWRITEBYTECODE": "1"},
        start_new_session=True,
    )
    terminal = {
        "process": process,
        "master": master,
        "slave": slave,
        "output": bytearray(),
    }
    try:
        yield terminal
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        os.close(master)
        os.close(slave)


def _expect_hidden_prompt(terminal, prompt):
    while prompt not in terminal["output"]:
        readable, _, _ = select.select([terminal["master"]], [], [], 10)
        assert readable, "The helper did not display the expected private prompt"
        terminal["output"].extend(os.read(terminal["master"], 65536))
    assert not termios.tcgetattr(terminal["slave"])[3] & termios.ECHO


def _enter_authorization(terminal, value):
    _expect_hidden_prompt(terminal, b"Original CLI authorization URL (hidden): ")
    os.write(terminal["master"], value.encode() + b"\n")


def _finish_terminal(terminal, tmp_path, expected_status):
    assert terminal["process"].wait(timeout=10) == expected_status
    while select.select([terminal["master"]], [], [], 0)[0]:
        terminal["output"].extend(os.read(terminal["master"], 65536))
    output = terminal["output"]
    assert b"synthetic" not in output
    assert b"http" not in output
    assert b"Traceback" not in output
    assert terminal["process"].args == [sys.executable, str(_SCRIPT)]
    assert list(tmp_path.iterdir()) == []
    assert termios.tcgetattr(terminal["slave"])[3] & termios.ECHO


@pytest.mark.parametrize(
    "scenario",
    [
        "success",
        "wrong-state",
        "duplicate-code",
        "redirect",
        "server-error",
        "malformed",
    ],
)
def test_private_terminal_callback_flow(scenario, _terminal, _listener, tmp_path):
    """Exercise actual hidden input, callback delivery, output, and cleanup.

    Args:
        scenario: Success or a callback failure case.
        _terminal: Real child process with its own terminal.
        _listener: Isolated HTTP listener and request capture.
        tmp_path: Empty working directory to check for persisted input.
    Returns:
        None.
    Raises:
        AssertionError: Input leaked, delivery differed, or cleanup failed.
    """
    port, requests, settings = _listener
    settings["status"] = {"redirect": 302, "server-error": 500}.get(scenario, 200)
    settings["malformed"] = scenario == "malformed"
    authorization = _authorization(f"http://127.0.0.1:{port}")
    callback = _callback(port)
    if scenario == "wrong-state":
        callback = _callback(port, state="synthetic-other-attempt")
    elif scenario == "duplicate-code":
        callback += "&code=synthetic-second-code"
    _enter_authorization(_terminal, authorization)
    _expect_hidden_prompt(_terminal, b"Final browser callback URL (hidden): ")
    os.write(_terminal["master"], callback.encode() + b"\n")
    _finish_terminal(_terminal, tmp_path, 0 if scenario == "success" else 1)
    assert len(requests) == (0 if scenario in {"wrong-state", "duplicate-code"} else 1)
    assert (b"Callback delivered." in _terminal["output"]) == (scenario == "success")


@pytest.mark.parametrize("cancel", ["eof", "interrupt"])
def test_private_terminal_cancellation(cancel, _terminal, _listener, tmp_path):
    """Cancel an actual prompt without leaking input or leaving a child alive.

    Args:
        cancel: EOF or process interrupt.
        _terminal: Real child process with its own terminal.
        _listener: Isolated HTTP listener and request capture.
        tmp_path: Empty working directory to check for persisted input.
    Returns:
        None.
    Raises:
        AssertionError: Cancellation sent a callback or exposed input.
    """
    port, requests, _ = _listener
    _enter_authorization(_terminal, _authorization(f"http://127.0.0.1:{port}"))
    _expect_hidden_prompt(_terminal, b"Final browser callback URL (hidden): ")
    if cancel == "eof":
        os.write(_terminal["master"], b"\x04")
    else:
        _terminal["process"].send_signal(signal.SIGINT)
    _finish_terminal(_terminal, tmp_path, 1)
    assert not requests
    assert b"Cancelled." in _terminal["output"]


def test_private_terminal_rejects_original_link_before_callback(_terminal, tmp_path):
    """Refuse an invalid original link without asking for a callback code.

    Args:
        _terminal: Real child process with its own terminal.
        tmp_path: Empty working directory to check for persisted input.
    Returns:
        None.
    Raises:
        AssertionError: Invalid original input advanced to the callback prompt.
    """
    _enter_authorization(_terminal, _authorization(state=""))
    _finish_terminal(_terminal, tmp_path, 1)
    assert b"Final browser callback" not in _terminal["output"]


def test_real_process_refuses_piped_callback(tmp_path):
    """Reject a piped callback without reproducing it in either output stream.

    Args:
        tmp_path: Empty working directory to check for persisted input.
    Returns:
        None.
    Raises:
        AssertionError: Non-terminal input was accepted or exposed.
    """
    result = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        input=_callback().encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=tmp_path,
        timeout=10,
    )
    assert result.returncode == 1
    assert b"synthetic" not in result.stdout + result.stderr
    assert b"http" not in result.stdout + result.stderr
    assert list(tmp_path.iterdir()) == []
