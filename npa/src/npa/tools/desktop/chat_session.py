"""Validate an existing mobile gateway's signed cookie without moving Codex tokens."""

import base64
import binascii
import hashlib
import hmac
from http.cookies import CookieError, SimpleCookie
import json
import time


def authorized(header, username, password):
    """Compare a Basic credential without exposing it in logs.

    Args:
        header: Authorization header.
        username: Configured login name.
        password: Current password read from private state.
    Returns:
        Whether the credential matches.
    Raises:
        None.
    """
    if not header.startswith("Basic "):
        return False
    try:
        supplied = base64.b64decode(header[6:], validate=True)
    except (ValueError, binascii.Error):
        return False
    return hmac.compare_digest(supplied, f"{username}:{password}".encode())


def session_key(config, password):
    """Bind cloud sessions to the current login while retaining legacy Mac cookies.

    Args:
        config: Private service configuration.
        password: Current desktop password.
    Returns:
        Signing key, or None when cookie authentication is not configured.
    Raises:
        None.
    """
    secret = config.get("session_secret")
    if not secret or not config.get("auth_port"):
        return secret
    identity = f"{config['username']}:{password}".encode()
    return hmac.new(secret.encode(), identity, hashlib.sha256).hexdigest()


def session_cookie(secret):
    """Issue a persistent HTTPS-only cookie using the shared gateway format.

    Args:
        secret: Private signing key.
    Returns:
        Set-Cookie header for a thirty-day session.
    Raises:
        None.
    """
    lifetime = 30 * 24 * 60 * 60
    value = json.dumps({"expires": (time.time() + lifetime) * 1000}).encode()
    payload = base64.urlsafe_b64encode(value).decode().rstrip("=")
    signature = (
        base64.urlsafe_b64encode(
            hmac.new(secret.encode(), payload.encode(), hashlib.sha256).digest()
        )
        .decode()
        .rstrip("=")
    )
    return (
        f"codex_session={payload}.{signature}; Path=/; Max-Age={lifetime}; "
        "Secure; HttpOnly; SameSite=Lax"
    )


def mobile_session(header, secret):
    """Accept only an unexpired cookie signed by the selected private gateway.

    Args:
        header: Cookie header from the HTTPS browser request.
        secret: Existing gateway session key, retained in private configuration.
    Returns:
        Whether the cookie is valid.
    Raises:
        None.
    """
    if not secret:
        return False
    try:
        cookies = SimpleCookie(header)
        cookie = cookies.get("codex_session")
        if cookie is None:
            return False
        payload, signature = cookie.value.split(".")
        expected = (
            base64.urlsafe_b64encode(
                hmac.new(secret.encode(), payload.encode(), hashlib.sha256).digest()
            )
            .decode()
            .rstrip("=")
        )
        if not hmac.compare_digest(signature, expected):
            return False
        value = json.loads(
            base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
        )
        return isinstance(value, dict) and (
            isinstance(value.get("expires"), (int, float))
            and value["expires"] > time.time() * 1000
        )
    except (CookieError, ValueError, TypeError, UnicodeDecodeError):
        return False
