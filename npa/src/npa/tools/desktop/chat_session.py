"""Validate an existing mobile gateway's signed cookie without moving Codex tokens."""

import base64
import hashlib
import hmac
from http.cookies import CookieError, SimpleCookie
import json
import time


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
        return (
            isinstance(value.get("expires"), (int, float))
            and value["expires"] > time.time() * 1000
        )
    except (CookieError, ValueError, TypeError, UnicodeDecodeError):
        return False
