"""Deliver a privately entered OAuth callback to one Nebius CLI loopback listener."""

from __future__ import annotations

import getpass
import hmac
import http.client
import sys
import warnings
from urllib.parse import SplitResult, parse_qs, urlencode, urlsplit


def _parse_url(value: str) -> SplitResult:
    if not value or any(
        character.isspace() or ord(character) < 32 for character in value
    ):
        raise ValueError("Invalid URL")
    parsed = urlsplit(value)
    if parsed.username is not None or parsed.password is not None or "#" in value:
        raise ValueError("Invalid URL")
    return parsed


def _query(value: str) -> dict[str, str]:
    fields = parse_qs(value, keep_blank_values=True, strict_parsing=True)
    if any(len(values) != 1 or not values[0] for values in fields.values()):
        raise ValueError("Ambiguous query")
    return {key: values[0] for key, values in fields.items()}


def _loopback_port(value: str) -> int:
    parsed = _parse_url(value)
    port = parsed.port
    if (
        parsed.scheme != "http"
        or port is None
        or not 1 <= port <= 65535
        or parsed.netloc != f"127.0.0.1:{port}"
        or parsed.path not in {"", "/"}
        or parsed.query
        or "?" in value
    ):
        raise ValueError("Invalid loopback redirect")
    return port


def _authorization(value: str) -> tuple[int, str]:
    parsed = _parse_url(value)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "auth.nebius.com"
        or parsed.path != "/oauth2/authorize"
    ):
        raise ValueError("Unexpected authorization endpoint")
    fields = _query(parsed.query)
    if (
        fields.get("client_id") != "nebius-cli"
        or fields.get("response_type") != "code"
        or fields.get("code_challenge_method") != "S256"
        or not fields.get("code_challenge")
        or not fields.get("state")
        or not fields.get("redirect_uri")
        or {"code", "error", "access_token", "refresh_token", "id_token"}
        & fields.keys()
    ):
        raise ValueError("Invalid authorization request")
    return _loopback_port(fields["redirect_uri"]), fields["state"]


def _callback_target(authorization_url: str, callback_url: str) -> tuple[int, str]:
    port, state = _authorization(authorization_url)
    parsed = _parse_url(callback_url)
    callback_port = _loopback_port(callback_url.split("?", 1)[0])
    if callback_port != port:
        raise ValueError("Callback port mismatch")
    fields = _query(parsed.query)
    if (
        fields.keys() - {"code", "state", "session_state", "iss"}
        or not fields.get("code")
        or not fields.get("state")
        or not hmac.compare_digest(fields["state"].encode(), state.encode())
    ):
        raise ValueError("Invalid callback response")
    return port, "/?" + urlencode(fields)


def _deliver(authorization_url: str, callback_url: str) -> None:
    port, target = _callback_target(authorization_url, callback_url)
    # A direct connection avoids proxies and URL openers that follow redirects.
    connection = http.client.HTTPConnection("127.0.0.1", port)
    try:
        connection.request("GET", target)
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError("Callback was not accepted")
    finally:
        connection.close()


def _read_hidden(prompt: str) -> str:
    # getpass normally falls back to echoing stdin when no terminal is available.
    with warnings.catch_warnings():
        warnings.simplefilter("error", getpass.GetPassWarning)
        return getpass.getpass(prompt)


def _main() -> int:
    if len(sys.argv) != 1 or not sys.stdin.isatty():
        print("Run without arguments in a private terminal on the CLI machine.")
        return 1
    try:
        authorization_url = _read_hidden("Original CLI authorization URL (hidden): ")
        _authorization(authorization_url)
        callback_url = _read_hidden("Final browser callback URL (hidden): ")
        _deliver(authorization_url, callback_url)
    except (EOFError, KeyboardInterrupt):
        print("Cancelled. Stop the waiting CLI login process.")
        return 1
    except (ValueError, OSError, http.client.HTTPException, getpass.GetPassWarning):
        print("Callback failed. Stop the waiting CLI login and begin a fresh attempt.")
        return 1
    print("Callback delivered. Wait for CLI success, then verify the selected profile.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
